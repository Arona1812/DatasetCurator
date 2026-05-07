import os
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

import logging
import re
import io
import csv
import json
import time
import math
import base64
import hashlib
import shutil
import threading
import traceback
import warnings
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple


HF_HUB_UNAUTH_WARNING = "You are sending unauthenticated requests to the HF Hub"


class _SuppressHfHubUnauthWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return HF_HUB_UNAUTH_WARNING not in record.getMessage()


warnings.filterwarnings(
    "ignore",
    message=r".*You are sending unauthenticated requests to the HF Hub.*",
    category=UserWarning,
)
logging.getLogger("huggingface_hub.utils._http").addFilter(_SuppressHfHubUnauthWarning())

import requests
import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

try:
    import mediapipe as mp
    HAVE_MP = True
except ImportError:
    HAVE_MP = False

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

try:
    import open_clip
    HAVE_CLIP = HAVE_TORCH  # CLIP needs torch; open_clip alone is meaningless
except ImportError:
    HAVE_CLIP = False

try:
    # InsightFace + ONNX Runtime fuer ArcFace-basierten Identitaets-Konsistenz-Check.
    # Beides optional: ohne diese Libraries wird der Check komplett uebersprungen.
    # Lizenz-Hinweis: insightface-Code ist MIT, die vortrainierten Modelle
    # (buffalo_l/buffalo_s/antelopev2) sind nur fuer non-commercial research use freigegeben.
    # Siehe https://github.com/deepinsight/insightface fuer kommerzielle Lizenzierung.
    import insightface  # type: ignore
    import onnxruntime  # type: ignore
    HAVE_INSIGHTFACE = True
except ImportError:
    HAVE_INSIGHTFACE = False


#+#+#+#+############################################################
# 1) KONFIGURATION
#+#+#+#+############################################################

# IMPORTANT:
# API keys must NOT be hardcoded in this file.
# Precedence:
#   1) UI override via _ui_config.json (API_KEY)
#   2) Environment variable OPENAI_API_KEY
# If neither is set, the script will error when the first API call is attempted.
API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Hauptmodell für Bildaudit + Triggerwortprüfung
# gpt-5.4-mini liefert deutlich treffsicherere Audits als nano:
# - erkennt Filter-induzierte Ueberbelichtung (Wachshaut/blown highlights)
# - bewertet Body-Camera-Winkel realistischer
# - vergibt seltener vorsichtiges 'review' fuer harmlose Bilder
# Im Gegenzug ist mini ~5-10x teurer pro Audit als nano. Bei typischen
# Datasetgroessen (<200 Bildern) ist das vernachlaessigbar.
AI_MODEL = "gpt-5.4-mini"
TRIGGER_CHECK_MODEL = "gpt-5.4-mini"

# Optionale Eskalation für schwierige Fälle:
# Erstes Audit läuft mit AI_MODEL. Falls ein Bild im Grenzbereich liegt,
# ein Review ist oder lokale und AI-Heuristik widersprüchlich sind,
# kann optional ein zweites, stärkeres Modell entscheiden.
USE_REVIEW_ESCALATION = False
REVIEW_ESCALATION_MODEL = ""
REVIEW_ESCALATION_SCORE_MIN = 50
REVIEW_ESCALATION_SCORE_MAX = 58
ESCALATE_ON_REVIEW_STATUS = True
ESCALATE_ON_STATUS_CONFLICT = True
ESCALATE_SMART_CROP_CLOSE_CALLS = True
SMART_CROP_ESCALATION_MAX_DELTA = 10.0

# Eindeutiges Triggerwort für das spätere LoRA-Training.
TRIGGER_WORD = ""
INPUT_FOLDER = r""

# Zielgröße des finalen Datensatzes. Das Skript versucht diese Zahl zu erreichen,
# notfalls auch mit guten Reservebildern aus Review-Kandidaten.
TARGET_DATASET_SIZE = 30

# Zielverteilung
RATIO_HEADSHOT = 0.50
RATIO_MEDIUM = 0.35
RATIO_FULL_BODY = 0.15

# Harte lokale Vorfilter
HARD_MIN_SIDE_PX = 768
API_MAX_IMAGE_SIDE = 1024
API_IMAGE_DETAIL = "high"

# ── GROSSDATENSATZ-VORFILTER (alle ohne API-Call, vor Pass 1) ─────────────
# Empfohlen ab ~500 Bildern. Jeder aktive Filter spart API-Kosten.

# Minimale Dateigröße in KB. Sehr kleine JPEGs sind meist stark komprimiert
# und liefern schlechte Trainingsdaten trotz ausreichender Pixelzahl.
USE_MIN_FILESIZE_FILTER   = True
HARD_MIN_FILESIZE_KB      = 80        # Unter 80 KB -> reject

# Unschärfe-Erkennung per Laplacian-Varianz (OpenCV).
# Zweistufig:
#   Stufe 1 (vor API): laxer Full-Image-Check auf 512px normiert,
#     faengt nur Totalausfaelle ab (spart API-Calls).
#   Stufe 2 (nach API + Face-Detection): strenger Check auf der
#     Face-Bbox (das ist fuer LoRA entscheidend), mit Fallback auf
#     Gesamtbild wenn kein Gesicht da ist.
#
# Normierung: Bilder werden vor der Messung auf BLUR_NORMALIZE_LONG_EDGE
# Pixel (laengste Seite) resized. Damit ist die Varianz ueber Datasets
# mit gemischten Aufloesungen vergleichbar. Ohne Normierung liefern
# kleine Bilder systematisch hoehere Werte als grosse.
USE_BLUR_FILTER            = True
BLUR_NORMALIZE_LONG_EDGE   = 512       # Zielgroesse fuer Blur-Messung (px, laengste Seite)
HARD_MIN_BLUR_VARIANCE     = 25.0      # Stufe 1: Totalausfall-Schwelle auf Gesamtbild (laxer Vorfilter)
# Stufe 2 (Face-Bbox): Typische Wertbereiche nach Normierung auf 512px:
#   scharfe Fotos mit guter Beleuchtung:  120-400+
#   normale Handy-Selfies:                 60-150
#   Beauty-Filter-Selfies (Skin-Smoothing): 20-60
#   klar verwackelte Gesichter:            <20
# Default 45 ist ein Kompromiss: trifft klar verwackelte Gesichter, kann aber
# stark weichgezeichnete Beauty-Filter-Selfies fangen. Wenn zu viele Bilder
# faelschlich gerejectet werden: in UI auf 25-30 runterdrehen und spaeter die
# geloggte face_blur_variance pro Bild in der Report-Auswertung ansehen.
FACE_MIN_BLUR_VARIANCE     = 45.0
# Shot-type-spezifische Schwellen fuer den Face-Blur-Check.
# Hintergrund: die Laplacian-Variance ueber die ganze Face-Bbox misst die
# Detail-Dichte. Bei Closeups (Headshots) ist die Bbox sehr gross und die
# glatten Wangenflaechen drueken die Variance mit, selbst wenn das Bild
# perfekt scharf ist. Daher braucht der Headshot-Threshold eine niedrigere
# Schwelle als full_body, wo das Gesicht klein und detailreich ist.
# Werte 0 oder negativ deaktivieren den shot-type-spezifischen Pfad und
# fallen auf FACE_MIN_BLUR_VARIANCE zurueck.
FACE_MIN_BLUR_VARIANCE_HEADSHOT  = 25.0
FACE_MIN_BLUR_VARIANCE_MEDIUM    = 35.0
FACE_MIN_BLUR_VARIANCE_FULL_BODY = 45.0
FACE_BLUR_PADDING_FACTOR   = 0.15      # Face-Bbox um diesen Faktor erweitern vor Blur-Messung

# Belichtungs-Check per Histogramm-Median (PIL, kein OpenCV nötig).
# Zu dunkel: Median < DARK_THRESHOLD. Zu hell: Median > BRIGHT_THRESHOLD.
USE_EXPOSURE_FILTER       = False
HARD_MAX_DARK_MEDIAN      = 20        # Unter 30/255 -> zu dunkel
HARD_MIN_BRIGHT_MEDIAN    = 255       # Über 225/255 -> überbelichtet

# pHash-Vorfilter VOR der API: berechnet alle Hashes lokal und wirft
# nur nahezu identische Bilder raus, bevor ein einziger API-Call gemacht wird.
# Wichtig: Early-Dedup ist absichtlich strenger als der spaetere Pass-2-Filter,
# damit aehnliche Varianten (Pose, Mimik, kleine Perspektivwechsel) nicht schon
# vor der eigentlichen Analyse komplett verschwinden.
USE_EARLY_PHASH_DEDUP     = True
EARLY_PHASH_HAMMING_THRESHOLD = 4
EARLY_PHASH_KEEP_PER_GROUP = 2
# Optional two-pass early pHash filtering, controlled by the UI:
#   Loop 1: exact/near-exact duplicates, keep only the best one.
#   Loop 2: bulk/video-frame near-duplicates, keep a little variation.
# The legacy EARLY_PHASH_* values above remain as non-UI defaults/fallbacks.
USE_EARLY_PHASH_LOOP1 = True
EARLY_PHASH_HAMMING_THRESHOLD_1 = 1
EARLY_PHASH_KEEP_PER_GROUP_1 = 1
# Bei Loop 1 (exact duplicates, threshold=1) wird der Survivor strikt nach
# Auflösung/Größe gewählt: höchste Megapixel gewinnen, Dateigröße als
# zweitwichtiges Kriterium, Schärfe nur als Tie-Breaker.
# Begründung: bei nahezu pixelidentischen Bildern dominiert die technische
# Variante (Original > Kompressionskopie > Resize) über minimale Schärfe-
# Schwankungen durch JPEG-Recompression.
# Loop 2 (Bulk-Frames, threshold=4) bleibt bei der Quality-First-Logik.
EARLY_PHASH_LOOP1_PREFER_RESOLUTION = True
USE_EARLY_PHASH_LOOP2 = True
EARLY_PHASH_HAMMING_THRESHOLD_2 = 4
EARLY_PHASH_KEEP_PER_GROUP_2 = 2

# Qualitätsschwellen (0-100, nach interner ×10-Normalisierung)
# Bilder unter diesem Wert werden von "keep" auf "review" herabgestuft.
KEEP_SCORE_MIN = 55
# Bilder unter diesem Wert werden direkt als "reject" markiert.
HARD_REJECT_SCORE = 30
# Unterhalb dieses Werts kann lokal direkt Reject erfolgen. 0 = deaktiviert.
REVIEW_SCORE_MIN = 0

# Lokale Mindest-Gesichtsgrößen (Gesichtsfläche / Gesamtbildfläche)
MIN_FACE_RATIO = {
    "headshot": 0.050,
    "medium": 0.015,
    "full_body": 0.004,
    }

# --------------------------------
# Multiple-People-Behandlung
# --------------------------------
# Wenn die API multiple_people=True meldet, gibt es zwei Strategien:
#
# 1. ALWAYS_DOWNGRADE_TO_REVIEW=True (Default, empfohlen):
#    Jedes Bild mit API multiple_people=True wird auf review degradiert
#    statt rejected - du sichtest manuell. Hintergrund: MediaPipe als
#    lokaler Cross-Check ist auf Brillenträger-Selfies und Innenräumen
#    unzuverlässig (Phantom-Gesichter durch Reflexionen, Hintergrund-
#    Details), und auch die secondary_face_area_ratio reflektiert dann
#    nur, wie groß der größte Phantom-Detect ist. In der Praxis hat
#    sich gezeigt: lieber 20-30 Bilder einmal manuell sichten als
#    systematische Falsch-Rejects bei Einzelpersonen-Bildern.
#
# 2. ALWAYS_DOWNGRADE_TO_REVIEW=False:
#    Klassischer Pfad mit Dominance-Check. Wenn das lokale Hauptgesicht
#    klar dominiert (sec_ratio klein), wird das Bild auf review degradiert,
#    sonst rejected. Greift nur wenn:
#      - lokal mindestens 2 Gesichter erkannt wurden
#      - secondary_face_area_ratio < CO_FACE_AREA_RATIO_THRESHOLD
#      - quality_total >= MULTIPLE_PEOPLE_SOFT_SCORE_MIN
#
# Beide Pfade ergänzen den Pick-Score nicht - die Behandlung erfolgt
# rein über den Status (keep/review/reject).
ENABLE_MULTIPLE_PEOPLE_DOMINANCE_OVERRIDE = True
MULTIPLE_PEOPLE_ALWAYS_DOWNGRADE_TO_REVIEW = True   # Empfohlen: True
# Wenn ALWAYS_DOWNGRADE aktiv ist, gibt es trotzdem einen Hard-Reject-Pfad
# fuer eindeutig echte Mehrpersonen-Bilder: ein zweites lokal erkanntes
# Gesicht, das nicht klein gegenueber dem Hauptgesicht ist (Halluzinations-
# Verdachts-Schwelle), bedeutet, dass tatsaechlich zwei Personen prominent
# im Frame sind. Solche Bilder sind objektiv unbrauchbar fuers Training -
# direkt rejecten statt unnoetig im Review-Bucket landen lassen.
MULTIPLE_PEOPLE_HARD_REJECT_SECONDARY_FACE_RATIO = 0.30  # 0.0 = hard reject deaktiviert
CO_FACE_AREA_RATIO_THRESHOLD = 0.25                 # nur relevant wenn ALWAYS_DOWNGRADE=False
MULTIPLE_PEOPLE_SOFT_SCORE_MIN = 75                 # nur relevant wenn ALWAYS_DOWNGRADE=False

# --------------------------------
# Body-Visibility-Bonus (LoRA-Body-Learning)
# --------------------------------
# Bei der Final-Auswahl wird Bildern mit gut sichtbarem Koerper ein Bonus
# auf den Pick-Score gegeben - bei sonst gleicher Bildqualitaet gewinnen
# die Body-Shots, die dem LoRA mehr Koerperinformation liefern (Bikini,
# Tank Top + Shorts, Sportkleidung etc.).
#
# WICHTIG: Wirkt nur auf adjusted_pick_score (Final-Auswahl), NICHT auf
# quality_total/keep/review/reject. Bilder mit viel Kleidung werden NICHT
# bestraft - sie bekommen nur weniger Bonus.
#
# Geltungsbereich nach shot_type:
#   - full_body: voller Bonus
#   - medium:    halber Bonus (Torso teilweise sichtbar)
#   - headshot:  0 (Koerper nicht im Frame, body_skin_visibility=n_a)
ENABLE_BODY_VISIBILITY_BONUS = True
BODY_VISIBILITY_BONUS_FULLBODY_HIGH = 6.0
BODY_VISIBILITY_BONUS_FULLBODY_MEDIUM = 2.0
BODY_VISIBILITY_BONUS_MEDIUM_SHOT_HIGH = 3.0
BODY_VISIBILITY_BONUS_MEDIUM_SHOT_MEDIUM = 1.0

# --------------------------------
# Face-Orientation-Penalty (Anatomie im Bildrahmen)
# --------------------------------
# Bewertet, wie das Gesicht im 2D-FRAME orientiert ist - nicht die Pose
# der Person im 3D-Raum. Ein liegendes Selfie kann 'upright' sein, wenn
# das Foto so gehalten wurde dass die Augen weiterhin oben sind. Wenn
# die Kamera dagegen aus extremer Unter-/Aufsicht aufgenommen wurde
# und im Frame die Augen UNTER dem Mund liegen, lernt das LoRA die
# Anatomie umgekehrt - das ist toxisch fuer's Training.
#
# Werte (aus dem Audit):
#   upright   : Augen klar ueber Mund, Kopf vertikal (Rotation bis ~30°)
#   tilted    : Schraege ~30-60°, Augen noch im oberen Gesichtsbereich
#   sideways  : ~60-120° rotiert, Augen seitlich neben dem Mund
#   inverted  : Augen UNTER dem Mund, Frame quasi auf-dem-Kopf (>~120°)
#   n_a       : Kein Gesicht im Frame (Rueckansicht, Occlusion)
#
# Wirkung:
#   - Pick-Score-Penalty (sofort, nur Final-Auswahl)
#   - Status-Downgrade keep -> review fuer 'inverted' immer,
#     fuer 'sideways' nur wenn quality_composition < 70
ENABLE_FACE_ORIENTATION_PENALTY = True
FACE_ORIENTATION_PENALTY_TILTED = 3.0
FACE_ORIENTATION_PENALTY_SIDEWAYS = 10.0
FACE_ORIENTATION_PENALTY_INVERTED = 20.0
FACE_ORIENTATION_DOWNGRADE_INVERTED_TO_REVIEW = True
FACE_ORIENTATION_DOWNGRADE_SIDEWAYS_TO_REVIEW = True
FACE_ORIENTATION_SIDEWAYS_DOWNGRADE_COMPOSITION_MAX = 70
# Bei 'tilted' ist die Schraege moderat - Downgrade nur wenn die
# Komposition zusaetzlich schwach ist. Setzt damit eine doppelte
# Schwelle: Schraege + schlechte Komposition = wahrscheinlich
# untrainierbarer Untersicht-/Aufsicht-Shot. Liegt auf einer
# strikten Skala unter dem sideways-Threshold von 70.
FACE_ORIENTATION_DOWNGRADE_TILTED_TO_REVIEW = True
FACE_ORIENTATION_TILTED_DOWNGRADE_COMPOSITION_MAX = 65

# --------------------------------
# Triggerwort-Prüfung
# --------------------------------
USE_AI_TRIGGERWORD_CHECK = False  # Prüft das Triggerwort per KI auf Kollisionen / problematische Namensähnlichkeit

# --------------------------------
# Near-Duplicate Optionen
# --------------------------------
USE_CLIP_DUPLICATE_SCORING = True  # Erkennt semantisch sehr ähnliche Bilder per CLIP
USE_PHASH_DUPLICATE_SCORING = True  # Erkennt visuell nahezu identische Bilder per pHash

PHASH_HAMMING_THRESHOLD = 8

# Für semantische Dubletten konservativ halten
CLIP_COSINE_THRESHOLD = 0.985

# CLIP Setup – ViT-L-14 ist deutlich besser für Person-Similarity als ViT-B-32
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "laion2b_s32b_b82k"
CLIP_DEVICE = "cuda" if HAVE_TORCH and torch.cuda.is_available() else "cpu"

# --------------------------------
# Session-/Outfit-Clusterung
# --------------------------------
USE_SESSION_OUTFIT_CLUSTERING = True  # Begrenzt zu viele ähnliche Bilder derselben Session / desselben Outfits
MAX_PER_OUTFIT_CLUSTER = 4  # Maximalzahl pro Outfit-Cluster im finalen Datensatz
MAX_PER_SESSION_CLUSTER = 5  # Maximalzahl pro Session-Cluster im finalen Datensatz
ENABLE_DIVERSITY_PENALTIES = True  # Bestraft zu ähnliche Kandidaten bei der Endauswahl

# --------------------------------
# Pose-Bucket-Diversity (ueber API ermittelt)
# --------------------------------
# Bestraft zu viele Bilder mit gleicher Kopfpose bei der Endauswahl.
# Pose-Bucket wird vom API-Audit als head_pose_bucket geliefert und
# fliesst in diversity_penalty() ein. Kein Hard-Reject - nur Punktabzug,
# damit bei vielen frontalen Aufnahmen automatisch 3/4-Profile bevorzugt
# werden, sofern qualitativ vergleichbar.
ENABLE_POSE_DIVERSITY = True
# Erlaubte Anzahl pro Pose-Bucket bevor Penalty einsetzt (gleiche Logik
# wie expression_count): pose_count > 2 → Penalty.
POSE_DIVERSITY_SOFT_LIMIT = 2
# Penalty-Gewicht pro ueberzaehligem Bild im selben Bucket. Sitzt zwischen
# Outfit (5.0) und Lighting (2.5), weil Pose wichtiger fuer Generalisierung
# ist als Licht, aber weniger kritisch als Outfit-Wiederholung.
POSE_DIVERSITY_PENALTY_WEIGHT = 4.0

# --------------------------------
# ArcFace Identitaets-Konsistenz-Check (nach Final-Pick)
# --------------------------------
# Berechnet pro Bild im Final-Set ein ArcFace-Embedding und vergleicht
# es mit einem outlier-getrimmten Centroid des Sets. Bilder mit grosser
# Distanz zur Set-Identitaet werden geflaggt:
#   Hard-Flag (sim < HARD_THRESHOLD): wahrscheinlich andere Person -
#     Bild wird aus 01_train_ready entfernt und mit Praefix in
#     06_needs_manual_review kopiert. Captions bleiben unangetastet.
#   Soft-Flag (HARD_THRESHOLD <= sim < SOFT_THRESHOLD): Grenzfall -
#     Bild bleibt im Train-Set, wird aber im Markdown-Report markiert.
#
# Wertbereiche der ArcFace-Cosine-Similarity zum Centroid:
#   gleiche Person, normale Variation:  0.65 - 0.95
#   gleiche Person, Beauty-Filter/Maske: 0.50 - 0.70
#   andere Person, aehnlich aussehend:  0.30 - 0.55
#   eindeutig andere Person:            < 0.40
USE_ARCFACE_IDENTITY_CHECK = True
ARCFACE_HARD_THRESHOLD = 0.50      # unter diesem Wert -> Hard-Flag (raus aus Train-Set)
ARCFACE_SOFT_THRESHOLD = 0.65      # zwischen Hard und Soft -> Markierung im Report
ARCFACE_TRIM_FRACTION  = 0.10      # 10% schlechteste Embeddings vor Centroid-Neuberechnung verwerfen
ARCFACE_MIN_FACES_FOR_CENTROID = 5 # weniger als 5 Gesichter -> Check skippen (nicht aussagekraeftig)
ARCFACE_MODEL_PACK = "buffalo_l"   # buffalo_l (genauer) oder buffalo_s (schneller)
ARCFACE_DET_SIZE = 640             # Detection-Eingabegroesse (kleinere Werte = schneller, weniger genau)
ARCFACE_USE_CUDA = False           # Aus = CPU erzwingen. Verhindert ONNXRuntime-CUDA-DLL-Fehler bei fehlendem CUDA/cuDNN.

# --------------------------------
# Crop-Profile
# --------------------------------
USE_AI_TOOLKIT_CROP_PROFILES = True  # Nutzt bucket-taugliche Crop-Profile für das spätere Training

# --------------------------------
# Retry / Resume
# --------------------------------
ENABLE_CACHE = True  # Nutzt vorhandene API-/Analyse-Ergebnisse wieder, spart Zeit und Kosten
MAX_RETRIES = 8
RETRY_BASE_SECONDS = 5.0
SLEEP_BETWEEN_CALLS = 1.0
HEARTBEAT_INTERVAL_SECONDS = 15.0

# --------------------------------
# Export
# --------------------------------
EXPORT_REVIEW_IMAGES = True  # Exportiert Review-Bilder zusätzlich in einen separaten Ordner
EXPORT_REJECT_IMAGES = False  # Exportiert Reject-Bilder physisch mit; oft aus = spart Platz
EXPORT_SMART_CROP_COMPARISON = False  # Exportiert Vergleichspaare (Original vs. Headshot-Crop) in 08_smart_crop_pairs

# --------------------------------
# Ausgabeordner
# --------------------------------
OUTPUT_ROOT = os.path.join(INPUT_FOLDER, f"curated_{TRIGGER_WORD}")
TRAIN_READY_DIR = os.path.join(OUTPUT_ROOT, "01_train_ready")
KEEP_UNUSED_DIR = os.path.join(OUTPUT_ROOT, "02_keep_unused")
CAPTION_REMOVE_DIR = os.path.join(OUTPUT_ROOT, "03_caption_remove")
REVIEW_DIR = os.path.join(OUTPUT_ROOT, "04_review")
REJECT_DIR = os.path.join(OUTPUT_ROOT, "05_reject")
MANUAL_REVIEW_DIR = os.path.join(OUTPUT_ROOT, "06_needs_manual_review")
CACHE_DIR = os.path.join(OUTPUT_ROOT, "_cache")
CLIP_CACHE_DIR = os.path.join(CACHE_DIR, "clip")
ARCFACE_CACHE_DIR = os.path.join(CACHE_DIR, "arcface")
TRIGGER_CACHE_DIR = os.path.join(CACHE_DIR, "trigger")
SMART_CROP_COMPARISON_DIR = os.path.join(OUTPUT_ROOT, "08_smart_crop_pairs")
IG_FRAME_CROP_DIR = os.path.join(CACHE_DIR, "ig_frame_crops")

# Caption-Regeln
CAPTION_PROFILE = "ernie"  # "ernie" | "z_image_base" | "custom"
CAPTION_POLICY = {
    "include_gender_class": True,
    "include_skin_tone": True, 
    "include_body_build": True,
    "include_tattoos": True,
    "include_glasses": True,
    "include_piercings": True,
    "include_makeup": True,
    "include_background": True,
    "include_lighting": True,
    "include_gaze": True,
    "include_expression": True,
    "include_hair_always": True,   
    "include_hair_when_variable": True,
    "include_beard_always": False,
    "include_beard_when_variable": True,
    "include_mirror_selfie_marker": True,
    "include_eye_color": True,        # ← NEU: Augenfarbe (siehe unten)
}

# Bilder mit Text / Wasserzeichen bei Bedarf separat ausgeben
SEND_TEXT_IMAGES_TO_CAPTION_REMOVE = True  # Bilder mit sichtbarem Text/Watermark -> 03_caption_remove statt train_ready
# Bug 1 fix: INTERACTIVE_CAPTION_OVERRIDE wurde komplett entfernt. Console-basiertes
# Override per input() ist im UI-Subprocess-Modus nicht moeglich. Overrides erfolgen
# ueber den Subject-Profile-Tab in der UI bzw. ueber _profile_override.json.

# ============================================================
# SUBJECT PROFILE PIPELINE (Phase 2/3)
# ============================================================
# Pipeline-Modus:
#   "single_pass"          : Audit -> direkt Caption (klassisches Verhalten,
#                            Phase 2 wendet Profile automatisch an, ohne UI)
#   "profile_then_caption" : Audit -> Profile-Build -> UI-Pause -> Caption
#                            (in Phase 3 vom UI gesetzt)
PIPELINE_MODE = "single_pass"

# Phase 3: Wenn True, wird kein neues Audit ausgefuehrt. Das Skript laedt
# den zuvor gespeicherten Caption-Stage-Zustand und exportiert nur Bilder +
# Captions mit dem aktuell bestaetigten _subject_profile.json.
CONTINUE_FROM_PROFILE = False

# Dateiname fuer den pausierten Zustand zwischen Profile-Build und Caption-Export.
CAPTION_STAGE_FILENAME = "_caption_stage.json"

# Subject-Profile-Cache (zentral, pro Trigger-Word)
SUBJECT_PROFILE_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".dataset_curator", "profiles"
)

# Stratified Sampling fuer Profile-Normalizer:
# Wenn die Anzahl gueltiger Audits > PROFILE_SAMPLE_THRESHOLD ist, wird ein
# stratifiziertes Sample von PROFILE_SAMPLE_SIZE Bildern verwendet, um den
# Normalizer-Context-Window nicht zu sprengen. Sonst gehen alle rein.
PROFILE_SAMPLE_THRESHOLD = 100   # ueberschreibbar via UI / _ui_config.json
PROFILE_SAMPLE_SIZE = 80         # ueberschreibbar via UI / _ui_config.json

# UI-Modus-Schwelle: bei N <= dieser Schwelle zeigt die UI Per-Bild-Dropdowns,
# sonst die aggregierte Spot-Check-Sicht.
PROFILE_UI_PER_IMAGE_THRESHOLD = 30   # ueberschreibbar via UI

# Profile-Builder verwendet welche Buckets als Input:
PROFILE_INPUT_BUCKETS = ["train_ready", "keep_unused"]   # rejects/reviews aus

# Normalizer-Modell (gpt-5.4-mini empfohlen wegen Context-Window)
PROFILE_NORMALIZER_MODEL = "gpt-5.4-mini"

# Cache-Version fuer zentrale Subject-Profile. Bei Aenderungen am
# Profile-Schema oder an der Normalizer-Logik inkrementieren.
#   v1: initial (Phase 2)
#   v2: confidence ist jetzt ein Objekt {level, reasoning, outliers}
#       statt nur ein String. Alte Cache-Eintraege werden invalidiert.
PROFILE_CACHE_SCHEMA_VERSION = "v2"

# ── SMART PRE-CROP (Post-API Headshot-Zoom) ────────────────────────────────────────────────
# Nach dem API-Audit des Originals: wenn das Bild groß ist und das Gesicht klein,
# wird ein enger Headshot-Crop erzeugt und SEPARAT zur API geschickt.
# Beide Versionen (Original + Crop) werden bewertet; die bessere gewinnt das Dataset.
ENABLE_SMART_PRECROP = True                # Pre-Crop aktivieren
SMART_PRECROP_MIN_FACE_PX = 120            # Mindest-Pixelgröße des Gesichts (min(fw, fh)) für Pre-Crop. Unter diesem Wert zu klein.
SMART_PRECROP_TRIGGER_RATIO = 0.07         # Pre-Crop nur wenn Gesicht < 7% des Gesamtbildes. Größere Gesichter brauchen kein Zoom.
SMART_PRECROP_PADDING_FACTOR = 0.6         # Padding pro Seite als Faktor der Gesichtsgroesse. 0.6 -> Gesamtbreite ~2.2x Gesicht (Gesicht + Haare + obere Schultern). Werte 0.4-0.8 sind sinnvoll; ueber 1.0 wird der Crop weit und naehert sich Halbkoerper-Bildaufbau an.
SMART_PRECROP_MIN_GAIN = 8.0               # Mindestvorsprung des Crop-Scores gegenüber dem Original, damit der Crop übernommen wird
SMART_PRECROP_ALLOW_DATASET_DUPLICATES = False  # False = Original und Crop dürfen NICHT beide ins finale Dataset

# ── INSTAGRAM-FRAME AUTO-CROP ──────────────────────────────────────────────────
# Erkennt und entfernt automatisch Instagram-Story-Rahmen (farbige Balken
# links/rechts, ggf. oben/unten) BEVOR das Bild zur API geht.
# Das gecropte Bild ersetzt das Original für alle weiteren Pipeline-Schritte.
ENABLE_IG_FRAME_CROP = True                # IG-Frame-Erkennung aktivieren
IG_FRAME_MIN_BORDER_PX = 30               # Mindestbreite eines Rahmens in Pixeln, um als Frame zu gelten
IG_FRAME_MIN_CONTENT_PX = 400             # Mindestbreite/-höhe des verbleibenden Inhalts nach Frame-Crop
# Zweistufige Bar-Detection (fuer Android-Nav-Bars, Drop-Shadow-Gradienten oben/unten):
# Erkennt uniforme Bloecke am oberen/unteren Rand, tolerant gegenueber UI-Icons
# (Nav-Buttons, Textfelder). Triggert nur wenn bereits Seitenframe gefunden wurde
# (verhindert False-Positives bei normalen dunklen Bildelementen wie Kissen oder
# dunklem Hintergrund). Ausschalten wenn unerwartete Crops auftreten.
IG_FRAME_TWO_STAGE_BAR_DETECT = True
# Cache-Version fuer IG-Frame-Crops. Jede Aenderung an der Detection-Logik,
# die andere Crop-Ergebnisse liefert, erfordert ein Increment dieser Version,
# damit vorhandene Caches neu berechnet werden.
# v1 = Original (nur Seiten + simple Top/Bottom-Gradienten)
# v2 = + Zweistufige Bar-Detection (Android-Nav-Bars, Drop-Shadows)
IG_FRAME_CACHE_VERSION = 2


# ── SUBJECT-SANITY-CHECK (Gliedmassen-/Winkel-Filter) ──────────────────────────
# Verwirft Bilder wie "nur Fuesse am Strand" oder "nur Haare + Hand",
# die zwar technisch ok sind, aber fuer Person-LoRAs nutzlos:
# kein Torso, kein Gesicht, kein Wiedererkennungsmerkmal.
# Loest NUR aus, wenn face_visible == False (aus API- oder lokaler Erkennung).
# Sichtbare Gesichter sind per Definition verwertbar und werden nie
# durch diesen Filter gekillt. Rueckenansichten mit klar erkennbarem
# Torso (mind. 2 von 4 Schulter/Hueft-Landmarks) bleiben erhalten.
ENABLE_SUBJECT_SANITY_CHECK = True
# Wie viele der 4 Torso-Landmarks (2 Schultern + 2 Hueften) mit ausreichender
# Sichtbarkeit vorhanden sein muessen, damit ein faceless-Bild als valider
# Koerper gilt. Bei < diesem Wert -> reject als "no_torso_no_face".
SUBJECT_MIN_TORSO_LANDMARKS = 2
# Mindest-Sichtbarkeit pro Landmark (MediaPipe-Visibility, 0..1)
SUBJECT_LANDMARK_VIS_MIN = 0.55


# ── UI-Config Override ────────────────────────────────────────────────────────
# Wird von dataset_curator_ui.py geschrieben. Überschreibt die Standardwerte
# oben mit den Werten aus der Weboberfläche. Ohne UI wird dieser Block ignoriert.
_UI_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ui_config.json")
# Interne Konstanten, die nie von der UI ueberschrieben werden duerfen.
# IG_FRAME_CACHE_VERSION insbesondere: wenn der User eine alte UI-Config mit
# einer veralteten Version auf den Curator losliesse, wuerden alte Caches
# faelschlich wiederverwendet. Gleiche Logik gilt fuer PROFILE_CACHE_SCHEMA_VERSION
# und AUDIT_CACHE_SCHEMA_VERSION - letzteres ist besonders kritisch, weil ein
# falscher Wert dazu fuehrt, dass alte Audits aus inkompatiblen Schemas
# wiederverwendet werden, statt neu erhoben zu werden (genau das passiert,
# wenn alte v6-Caches mit einer v7-Schema-Logik gelesen werden).
# Diese Liste wachst mit jedem internen Feld, das aus strukturellen Gruenden
# keine UI-Kontrolle haben soll.
_UI_PROTECTED_KEYS = {
    "IG_FRAME_CACHE_VERSION",
    "PROFILE_CACHE_SCHEMA_VERSION",
    "AUDIT_CACHE_SCHEMA_VERSION",
}
if os.path.exists(_UI_CONFIG_PATH):
    try:
        with open(_UI_CONFIG_PATH, "r", encoding="utf-8") as _f:
            _ui_cfg = json.load(_f)
        for _k, _v in _ui_cfg.items():
            # CAPTION_POLICY separat mergen, nicht komplett ersetzen
            if _k == "CAPTION_POLICY":
                continue
            if _k in _UI_PROTECTED_KEYS:
                continue
            if _k in globals() and not _k.startswith("_"):
                globals()[_k] = _v
        # CAPTION_POLICY: Defaults beibehalten, nur gesetzte Keys ueberschreiben
        if "CAPTION_POLICY" in _ui_cfg and isinstance(_ui_cfg["CAPTION_POLICY"], dict):
            CAPTION_POLICY.update(_ui_cfg["CAPTION_POLICY"])
    except Exception as _e:
        print(f"⚠️ Failed to load UI config: {_e}")

    # Abgeleitete Pfade muessen nach dem Override neu berechnet werden,
    # da INPUT_FOLDER und TRIGGER_WORD sich geaendert haben koennten.
    OUTPUT_ROOT = os.path.join(INPUT_FOLDER, f"curated_{TRIGGER_WORD}")
    TRAIN_READY_DIR = os.path.join(OUTPUT_ROOT, "01_train_ready")
    KEEP_UNUSED_DIR = os.path.join(OUTPUT_ROOT, "02_keep_unused")
    CAPTION_REMOVE_DIR = os.path.join(OUTPUT_ROOT, "03_caption_remove")
    REVIEW_DIR = os.path.join(OUTPUT_ROOT, "04_review")
    REJECT_DIR = os.path.join(OUTPUT_ROOT, "05_reject")
    MANUAL_REVIEW_DIR = os.path.join(OUTPUT_ROOT, "06_needs_manual_review")
    CACHE_DIR = os.path.join(OUTPUT_ROOT, "_cache")
    CLIP_CACHE_DIR = os.path.join(CACHE_DIR, "clip")
    ARCFACE_CACHE_DIR = os.path.join(CACHE_DIR, "arcface")
    TRIGGER_CACHE_DIR = os.path.join(CACHE_DIR, "trigger")
    SMART_CROP_COMPARISON_DIR = os.path.join(OUTPUT_ROOT, "08_smart_crop_pairs")
    IG_FRAME_CROP_DIR = os.path.join(CACHE_DIR, "ig_frame_crops")

# Keep environment and in-script config consistent (also helps if other libs/tools
# look at OPENAI_API_KEY).
if API_KEY:
    os.environ["OPENAI_API_KEY"] = API_KEY


# ============================================================
# 2) INITIALISIERUNG
# ============================================================

for folder in [
    OUTPUT_ROOT,
    TRAIN_READY_DIR,
    KEEP_UNUSED_DIR,
    CAPTION_REMOVE_DIR,
    REVIEW_DIR,
    CACHE_DIR,
    CLIP_CACHE_DIR,
    ARCFACE_CACHE_DIR,
    TRIGGER_CACHE_DIR,
]:
    os.makedirs(folder, exist_ok=True)

if EXPORT_REJECT_IMAGES:
    os.makedirs(REJECT_DIR, exist_ok=True)
os.makedirs(MANUAL_REVIEW_DIR, exist_ok=True)

if EXPORT_SMART_CROP_COMPARISON:
    os.makedirs(SMART_CROP_COMPARISON_DIR, exist_ok=True)

if ENABLE_IG_FRAME_CROP:
    os.makedirs(IG_FRAME_CROP_DIR, exist_ok=True)

MP_FACE = None
MP_POSE = None
if HAVE_MP:
    try:
        MP_FACE = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        MP_POSE = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
    except Exception:
        MP_FACE = None
        MP_POSE = None

HAAR_CASCADE = None
if HAVE_CV2:
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        HAAR_CASCADE = cv2.CascadeClassifier(cascade_path)
    except Exception:
        HAAR_CASCADE = None

CLIP_MODEL = None
CLIP_PREPROCESS = None
if USE_CLIP_DUPLICATE_SCORING and HAVE_CLIP:
    try:
        CLIP_MODEL, _, CLIP_PREPROCESS = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
            device=CLIP_DEVICE,
        )
        CLIP_MODEL.eval()
    except Exception:
        CLIP_MODEL = None
        CLIP_PREPROCESS = None

# ── ArcFace-Modell (lazy init) ────────────────────────────────────────
# Wird erst beim ersten Gebrauch initialisiert, um Startup-Zeit zu sparen
# wenn das Feature deaktiviert ist oder gar keine ArcFace-Library da ist.
# Definition der Init-Funktion folgt weiter unten (nach safe_print).
ARCFACE_APP = None
ARCFACE_INIT_ATTEMPTED = False


# ============================================================
# 3) HILFSFUNKTIONEN
# ============================================================

def safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("utf-8", errors="replace").decode("utf-8"))


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rest = seconds - (minutes * 60)
    return f"{minutes}m {rest:.1f}s"


def start_phase_heartbeat(label: str, interval: float = HEARTBEAT_INTERVAL_SECONDS):
    started_at = time.time()
    stop_event = threading.Event()

    safe_print(f"   ⏱️  START {label}")

    def _heartbeat() -> None:
        while not stop_event.wait(interval):
            elapsed = format_elapsed(time.time() - started_at)
            safe_print(f"   ⏳ still working: {label} | elapsed={elapsed}")

    thread = threading.Thread(target=_heartbeat, daemon=True)
    thread.start()
    return started_at, stop_event, thread


def stop_phase_heartbeat(
    label: str,
    started_at: float,
    stop_event: threading.Event,
    thread: threading.Thread,
    success: bool = True,
) -> None:
    stop_event.set()
    if thread.is_alive():
        thread.join(timeout=0.2)
    elapsed = format_elapsed(time.time() - started_at)
    status = "DONE" if success else "FAILED"
    icon = "✅" if success else "❌"
    safe_print(f"   {icon} {status} {label} | elapsed={elapsed}")


def run_with_heartbeat(label: str, func: Callable[..., Any], *args, **kwargs) -> Any:
    started_at, stop_event, thread = start_phase_heartbeat(label)
    try:
        result = func(*args, **kwargs)
        stop_phase_heartbeat(label, started_at, stop_event, thread, success=True)
        return result
    except Exception:
        stop_phase_heartbeat(label, started_at, stop_event, thread, success=False)
        raise


def slugify_filename(text: str) -> str:
    text = re.sub(r"[^\w\-]+", "_", text.strip(), flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "subject"


SAFE_TRIGGER = slugify_filename(TRIGGER_WORD)


def file_sha1(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    v = value.strip().lower()
    v = re.sub(r"\s+", " ", v)
    return v


def normalize_compact_text(value: Optional[str]) -> str:
    v = normalize_text(value)
    if not v:
        return ""
    v = re.sub(r"[,;:]+", " ", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v


# ============================================================
# 5b) VOCABULARY & NORMALIZATION
# ============================================================
# Diese Sektion zentralisiert das Vokabular fuer Caption-Felder
# und stellt deterministische Normalisierungs-Helfer bereit, die
# Muelltext (Hedge-Phrasen, "none visible", "moderate or no") aus
# Audit-Antworten herausfiltern.
#
# Phase 1 verwendet diese Helfer nur defensiv (also: was reinkommt,
# wird gesaeubert). Phase 2 baut darauf den Profile-Normalizer auf,
# der per LLM-Call die Per-Image-Audits zu kanonischen Tokens
# konsolidiert.
# ============================================================

# --- Hedge-Phrasen (werden vor Verwertung aus Strings entfernt) ----------
# Wenn ein Audit "possibly blue eyes" liefert, soll die Caption
# "blue eyes" sagen, nicht "possibly blue eyes". Diese Phrasen werden
# als Substring entfernt, der Rest des Strings bleibt erhalten.
HEDGE_PHRASES: List[str] = [
    "possibly", "perhaps", "maybe", "appears to be", "appears",
    "looks like", "looks to be", "kind of", "sort of",
    "somewhat", "slightly", "approximately", "roughly",
    "presumably", "likely", "probably", "seemingly",
]

# --- Verbotene Trait-Phrasen ---------------------------------------------
# Wenn ein Feldwert NUR aus einer dieser Phrasen besteht (oder klar damit
# beginnt), wird das Feld auf Leerstring gesetzt. Damit landet
# "none visible" nicht in Captions.
INVALID_TRAIT_PHRASES: set = {
    "none", "no", "n/a", "na", "unknown", "not visible", "not applicable",
    "none visible", "no visible", "not clearly visible", "not clearly",
    "minimal or no", "moderate or no", "subtle or no",
    "nothing visible", "nothing", "no makeup", "no piercings",
    "no glasses", "no tattoos", "no beard", "minimal or",
    "subtle or", "moderate or", "slight or", "light or",
}

# --- Prioritaets-Mapping fuer "X or Y"-Aufloesung ------------------------
# Wenn das Audit "moderate or full makeup" liefert, splitten wir an " or "
# und behalten den Wert mit hoeherer Prioritaet (intensiver/spezifischer).
# Default ist die rechte Seite, ausser eine Seite ist in dieser Map mit
# hoeherem Score eingetragen.
OR_PRIORITY_MAP: Dict[str, int] = {
    # Makeup
    "none": 0, "minimal": 1, "subtle": 1, "light": 1, "natural": 2,
    "moderate": 3, "defined": 4, "full": 5, "heavy": 5, "dramatic": 6,
    "bold": 6,
    # Generic intensity
    "slight": 1, "soft": 1, "medium": 3, "strong": 5,
}

# --- Kanonische Vokabular-Buckets fuer LLM-Normalizer (Phase 2) ----------
# Diese Listen werden im Audit-Prompt als Hinweise mitgegeben (nicht als
# strikte ENUMs - der User wollte Freitext mit nachgelagerter LLM-Norm).
HAIR_FORM_VOCAB: List[str] = [
    "loose_straight", "loose_wavy", "loose_curly", "loose_coily",
    "afro_natural", "ponytail", "pigtails", "two_braids", "single_braid",
    "box_braids", "knotless_braids", "cornrows", "bun", "updo",
    "half_up", "pulled_back", "short_cut",
]

HAIR_COLOR_VOCAB: List[str] = [
    "black", "dark_brown", "brown", "light_brown", "blonde", "platinum",
    "red", "auburn", "burgundy", "gray", "white", "dyed_other",
]

EYE_COLOR_VOCAB: List[str] = [
    "blue", "green", "hazel", "brown", "dark_brown", "gray", "amber",
]

SKIN_TONE_VOCAB: List[str] = [
    "fair", "light", "medium", "tan", "olive", "dark", "deep",
]

BODY_BUILD_VOCAB: List[str] = [
    "slim", "average", "athletic", "curvy", "plus_size", "muscular",
]

MAKEUP_INTENSITY_VOCAB: List[str] = [
    "none", "minimal", "natural", "defined", "full", "dramatic",
]

LIGHTING_TYPE_VOCAB: List[str] = [
    "studio_softbox", "studio_ringlight", "studio_other",
    "natural_outdoor_sun", "natural_outdoor_overcast",
    "natural_indoor_window", "indoor_artificial", "mixed", "low_light",
]

BACKGROUND_TYPE_VOCAB: List[str] = [
    "studio_plain", "studio_textured", "indoor_room", "indoor_bathroom",
    "outdoor_urban", "outdoor_nature", "outdoor_beach", "outdoor_other",
    "vehicle_interior", "transparent_or_isolated", "other",
]

GLASSES_FRAME_SHAPE_VOCAB: List[str] = [
    "round", "square", "rectangular", "oval", "aviator", "cat_eye",
    "oversized", "rimless", "browline", "geometric", "other",
]

# --- Tattoo-Locations als kontrolliertes ENUM ----------------------------
# Wird im Audit-Schema als Strict-ENUM verwendet, weil Tattoo-Lokationen
# fuer die Inventar-Deduplizierung in Phase 2 deterministisch gleich
# benannt sein muessen.
TATTOO_LOCATION_ENUM: List[str] = [
    "forearm_left", "forearm_right",
    "upper_arm_left", "upper_arm_right",
    "hand_left", "hand_right",
    "wrist_left", "wrist_right",
    "shoulder_left", "shoulder_right",
    "neck_left", "neck_right", "neck_back",
    "chest_upper", "chest_sternum",
    "collarbone_left", "collarbone_right",
    "ribcage_left", "ribcage_right",
    "abdomen", "back_upper", "back_lower",
    "thigh_left", "thigh_right",
    "calf_left", "calf_right",
    "ankle_left", "ankle_right",
    "foot_left", "foot_right",
    "finger_left", "finger_right",
    "behind_ear_left", "behind_ear_right",
    "face", "scalp",
    "other",
]

PIERCING_LOCATION_ENUM: List[str] = [
    "ear_lobe_left", "ear_lobe_right",
    "ear_helix_left", "ear_helix_right",
    "ear_tragus_left", "ear_tragus_right",
    "ear_gauge_left", "ear_gauge_right",
    "nose_left", "nose_right", "nose_septum",
    "nose_bridge", "eyebrow_left", "eyebrow_right",
    "lip_upper", "lip_lower", "lip_corner_left", "lip_corner_right",
    "tongue", "navel", "other",
]


def strip_hedge_phrases(text: str) -> str:
    """Entfernt Hedge-Woerter wie 'possibly', 'appears to be' aus einem String.
    Der Rest des Strings bleibt unveraendert. Mehrfach-Whitespace wird
    kollabiert.
    """
    if not text:
        return ""
    out = " " + text.lower() + " "
    for hedge in HEDGE_PHRASES:
        # Wort-Boundary-aehnliche Ersetzung (mit Leerzeichen drumherum)
        out = out.replace(f" {hedge} ", " ")
    out = re.sub(r"\s+", " ", out).strip(" ,.;:")
    return out


def is_invalid_trait_value(text: str) -> bool:
    """True, wenn der String nur aus 'none visible', 'moderate or no',
    'not applicable' o.ae. besteht und damit als Feldwert wertlos ist.
    """
    if not text:
        return True
    t = text.strip().lower().rstrip(".,;:")
    if t in INVALID_TRAIT_PHRASES:
        return True
    # Phrasen, die mit einem verbotenen Praefix beginnen, abfangen:
    # "none visible, minimal or no makeup" -> True
    for prefix in ("none visible", "not visible", "no visible",
                   "minimal or no", "moderate or no", "subtle or no"):
        if t.startswith(prefix):
            # ... aber nur wenn nichts Substanzielles drauf folgt,
            # was selbst gueltig waere (z.B. "none visible, light makeup"
            # -> sollte NICHT verworfen werden, weil "light makeup" gueltig ist).
            tail = t[len(prefix):].lstrip(",; .")
            if not tail or is_invalid_trait_value(tail):
                return True
    return False


def resolve_or_phrase(text: str) -> str:
    """Loest 'X or Y'-Phrasen auf, indem die Seite mit hoeherer
    Intensitaets-Prioritaet behalten wird.

    'moderate or full makeup' -> 'full makeup'
    'minimal or no makeup'    -> '' (beide Seiten ungueltig oder leer)
    'blue or green eyes'      -> 'blue or green eyes' (kein klarer Sieger -> Original)
    """
    if not text or " or " not in text.lower():
        return text

    parts = re.split(r"\s+or\s+", text, flags=re.IGNORECASE)
    if len(parts) != 2:
        return text

    left, right = parts[0].strip(), parts[1].strip()

    # Erstes Token jeder Seite extrahieren (das ist der Intensitaets-Indikator)
    def first_token(s: str) -> str:
        m = re.match(r"\s*([a-zA-Z\-]+)", s)
        return m.group(1).lower() if m else ""

    left_token = first_token(left)
    right_token = first_token(right)

    left_score = OR_PRIORITY_MAP.get(left_token, -1)
    right_score = OR_PRIORITY_MAP.get(right_token, -1)

    # Wenn keine Seite in der Map ist, Original behalten
    if left_score < 0 and right_score < 0:
        return text

    # Bei Gleichstand: rechten Teil nehmen (haeufiger das spezifischere)
    if right_score >= left_score:
        # Den linken Token durch den rechten ersetzen, Rest behalten
        # "moderate or full makeup" -> "full makeup"
        # Wir nehmen den rechten Teil samt seinem Suffix (das ist der vollstaendige Begriff)
        return right
    else:
        return left


def clean_audit_string(text: Optional[str]) -> str:
    """Vollstaendige Saeuberung eines Audit-Freitext-Strings.
    Reihenfolge:
      1. Strip + Lowercase fuer Vergleiche (Original-Casing wird beibehalten)
      2. Hedge-Woerter entfernen
      3. 'X or Y'-Aufloesung
      4. Invalid-Phrase-Check -> ggf. Leerstring
      5. Whitespace normalisieren
    """
    if not text:
        return ""
    t = str(text).strip()
    if not t:
        return ""

    # 1. Hedge-Woerter raus (Funktion arbeitet auf lowercase-Buffer,
    #    gibt aber lowercase zurueck - was fuer Trait-Tokens okay ist).
    t = strip_hedge_phrases(t)

    # 2. 'X or Y'-Aufloesung
    t = resolve_or_phrase(t)

    # 3. Invalid-Check
    if is_invalid_trait_value(t):
        return ""

    # 4. Restliche Reinigung
    t = re.sub(r"\s+", " ", t).strip(" ,.;:")
    return t


# ============================================================
# 5b-end) END VOCABULARY & NORMALIZATION
# ============================================================


def normalize_caption_profile(value: Optional[str]) -> str:
    v = normalize_text(value)
    if v in {"ernie", "shared_compact"}:
        return "shared_compact"
    if v in {"z_image_base", "custom"}:
        return v
    return "shared_compact"


def enforce_caption_policy_profile(profile: Optional[str], policy: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure profile-specific caption fields stay enabled."""
    normalized = normalize_caption_profile(profile)
    if normalized in {"ernie", "shared_compact"}:
        policy["include_body_build"] = True
        policy["include_tattoos"] = True
        policy["include_eye_color"] = True
    return policy


def coarse_key(value: Optional[str], max_words: int = 5) -> str:
    v = normalize_text(value)
    if not v:
        return "unknown"
    words = re.findall(r"[a-zA-Z0-9äöüÄÖÜß\-]+", v)
    return " ".join(words[:max_words]) if words else "unknown"


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))


def iter_input_images(root: str) -> List[str]:
    paths = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isfile(p) and is_image_file(name):
            paths.append(p)
    return sorted(paths)


def resize_and_encode_for_api(image_path: str, max_side: int = API_MAX_IMAGE_SIDE) -> str:
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=88)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_dimensions(path: str) -> Tuple[int, int]:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return img.size


def compute_phash(path: str) -> int:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32)

    if HAVE_CV2:
        dct = cv2.dct(arr)
        low = dct[:8, :8]
        med = np.median(low[1:, 1:])
        bits = low > med
    else:
        med = np.median(arr)
        bits = arr[:8, :8] > med

    result = 0
    for bit in bits.flatten():
        result = (result << 1) | int(bool(bit))
    return result


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def bbox_area_ratio(bbox: Optional[List[int]], w: int, h: int) -> float:
    if not bbox or w <= 0 or h <= 0:
        return 0.0
    x, y, bw, bh = bbox
    return max(0, bw) * max(0, bh) / float(w * h)


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def get_file_mtime_bucket(path: str, seconds_bucket: int = 6 * 3600) -> str:
    try:
        ts = int(os.path.getmtime(path))
        return str(ts // seconds_bucket)
    except Exception:
        return "unknown"


def generate_headshot_crop(
    image_path: str,
    ai_face_bbox_abs: List[int],
    img_w: int,
    img_h: int,
) -> Optional[str]:
    """
    Erzeugt einen eng zugeschnittenen Headshot-Crop rund um die AI-erkannte
    Gesichts-BBox (in absoluten Pixel-Koordinaten des Originalbilds).
    Gibt einen Temp-Dateipfad zurueck. Caller muss die Datei via try/finally loeschen.

    Geometrie ist identisch zur Smart-Crop-Branch in body_aware_crop():
    Crop-Groesse = face + 2 * SMART_PRECROP_PADDING_FACTOR pro Seite.
    Default 0.6 -> 2.2x face_size (echter Headshot mit Haaren + obere
    Schultern). Damit bewerten wir API-seitig genau das Bild, das spaeter
    auch in 01_train_ready landet, ohne zwei verschiedene Croppings.
    """
    if not ENABLE_SMART_PRECROP:
        return None
    try:
        import tempfile
        fx, fy, fw, fh = ai_face_bbox_abs
        face_size = max(int(fw), int(fh))

        size = int(round(face_size * (1.0 + 2.0 * SMART_PRECROP_PADDING_FACTOR)))
        min_size = int(round(face_size * 1.5))
        max_size = int(round(min(img_w, img_h) * 0.80))
        size = max(min(size, max_size), min(min_size, max_size))

        cx = int(fx) + int(fw) // 2
        cy = int(fy) + int(fh) // 2
        zoom_ratio = size / max(1, face_size)
        v_offset_factor = max(0.35, min(0.50, 0.35 + (zoom_ratio - 1.5) * 0.10))

        sq_x1 = max(0, min(cx - size // 2, img_w - size))
        sq_y1 = max(0, min(cy - int(size * v_offset_factor), img_h - size))
        x1 = sq_x1
        y1 = sq_y1
        x2 = sq_x1 + size
        y2 = sq_y1 + size

        with Image.open(image_path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
            cropped = pil_img.crop((x1, y1, x2, y2))
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="headshot_crop_")
            os.close(tmp_fd)
            cropped.save(tmp_path, "JPEG", quality=100)
            return tmp_path
    except Exception:
        return None


def local_blur_variance(image_path: str) -> float:
    """
    Berechnet die Laplacian-Varianz als Schaerfemass.
    Niedrige Werte = unscharf/verwackelt. Benoetigt OpenCV.

    WICHTIG: Das Bild wird VOR der Messung auf BLUR_NORMALIZE_LONG_EDGE
    (laengste Seite) heruntergerechnet. Ohne diese Normierung liefern
    kleine Bilder systematisch hoehere Varianzen als grosse, was jeden
    festen Threshold unbrauchbar macht. Nach Normierung sind die Werte
    ueber unterschiedliche Aufloesungen hinweg vergleichbar.

    Gibt -1.0 zurueck wenn OpenCV nicht verfuegbar ist (Filter ueberspringen).
    """
    if not HAVE_CV2:
        return -1.0
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return -1.0
        h, w = img.shape[:2]
        long_edge = max(h, w)
        if long_edge > BLUR_NORMALIZE_LONG_EDGE:
            scale = BLUR_NORMALIZE_LONG_EDGE / float(long_edge)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return -1.0


def local_blur_variance_in_face(image_path: str, face_bbox: Optional[List[int]]) -> float:
    """
    Misst die Laplacian-Varianz nur innerhalb der Face-Bbox (leicht erweitert),
    normiert auf BLUR_NORMALIZE_LONG_EDGE. Fuer LoRA-Training ist entscheidend,
    dass das Gesicht scharf ist, nicht der Hintergrund.

    face_bbox: [x, y, w, h] in Pixeln relativ zum Originalbild, oder None.
    Gibt bei None oder Fehler -1.0 zurueck (Filter ueberspringen).
    """
    if not HAVE_CV2 or not face_bbox:
        return -1.0
    try:
        fx, fy, fw, fh = [int(v) for v in face_bbox]
        if fw <= 0 or fh <= 0:
            return -1.0
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return -1.0
        H, W = img.shape[:2]
        # Bbox um FACE_BLUR_PADDING_FACTOR erweitern, damit Kanten
        # (Kieferlinie, Haaransatz) mit in die Messung einfliessen.
        pad_x = int(round(fw * FACE_BLUR_PADDING_FACTOR))
        pad_y = int(round(fh * FACE_BLUR_PADDING_FACTOR))
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y)
        x2 = min(W, fx + fw + pad_x)
        y2 = min(H, fy + fh + pad_y)
        if x2 <= x1 or y2 <= y1:
            return -1.0
        crop = img[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        long_edge = max(ch, cw)
        # Normierung: Face-Crop auf dieselbe Zielgroesse wie Vollbild-Messung.
        # So sind Face-Werte und Full-Image-Werte direkt vergleichbar.
        if long_edge != BLUR_NORMALIZE_LONG_EDGE:
            scale = BLUR_NORMALIZE_LONG_EDGE / float(long_edge)
            new_w = max(8, int(round(cw * scale)))
            new_h = max(8, int(round(ch * scale)))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            crop = cv2.resize(crop, (new_w, new_h), interpolation=interp)
        return float(cv2.Laplacian(crop, cv2.CV_64F).var())
    except Exception:
        return -1.0


def subject_torso_landmark_count(image_path: str) -> int:
    """
    Zaehlt wie viele der 4 Kern-Torso-Landmarks (linke/rechte Schulter,
    linke/rechte Huefte) mit ausreichender Sichtbarkeit (>= SUBJECT_LANDMARK_VIS_MIN)
    erkannt werden. Nutzt MediaPipe Pose.

    Gibt einen Wert zwischen 0 und 4 zurueck. -1 wenn MediaPipe nicht
    verfuegbar ist (dann soll der Caller den Check ueberspringen, nicht
    verwerfen).

    Gedacht als Sanity-Check: wenn ein Bild KEIN Gesicht zeigt und auch
    keinen erkennbaren Torso, dann sind es vermutlich nur isolierte
    Gliedmassen (Fuesse, Haende) und fuer Person-LoRAs nutzlos.
    """
    if MP_POSE is None or not HAVE_CV2:
        return -1
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return -1
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pose_result = MP_POSE.process(rgb)
        if not pose_result or not pose_result.pose_landmarks:
            return 0
        # MediaPipe PoseLandmark-Indizes:
        # 11 = LEFT_SHOULDER, 12 = RIGHT_SHOULDER
        # 23 = LEFT_HIP,      24 = RIGHT_HIP
        torso_idx = (11, 12, 23, 24)
        lms = pose_result.pose_landmarks.landmark
        count = 0
        for idx in torso_idx:
            if idx >= len(lms):
                continue
            lm = lms[idx]
            if (lm.visibility >= SUBJECT_LANDMARK_VIS_MIN
                    and 0.0 <= lm.x <= 1.0
                    and 0.0 <= lm.y <= 1.0):
                count += 1
        return count
    except Exception:
        return -1


def local_exposure_median(image_path: str) -> float:
    """
    Berechnet den Helligkeits-Median des Graustufenbilds (0-255).
    Sehr niedrig = unterbelichtet, sehr hoch = überbelichtet.
    Gibt 128.0 zurück bei Fehler (neutraler Wert, kein Filter).
    """
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert("L")
            arr = np.asarray(img, dtype=np.uint8)
            return float(np.median(arr))
    except Exception:
        return 128.0

def early_duplicate_pick_score(image_path: str) -> Tuple[float, Dict[str, float]]:
    """
    Schneller, deterministischer Lokalscore für Early-pHash-Gruppen.
    Bevorzugt scharfe, hochauflösende Bilder mit klar erkennbarem Hauptgesicht.
    Dateigröße dient nur als schwacher Tie-Breaker.
    """
    width, height = image_dimensions(image_path)
    pixel_count = max(1.0, float(width * height))
    megapixels = pixel_count / 1_000_000.0
    blur_variance = local_blur_variance(image_path)
    blur_score = math.log1p(max(0.0, blur_variance))
    filesize_kb = local_filesize_kb(image_path)

    main_face_ratio = 0.0
    face_count = 0
    pose_ratio = 0.0

    try:
        metrics = local_subject_metrics(image_path, phash_cache=None)
        main_face_ratio = float(metrics.get("main_face_ratio") or 0.0)
        face_count = int(metrics.get("face_count_local") or 0)
        pose_ratio = bbox_area_ratio(metrics.get("pose_bbox"), width, height)
    except Exception:
        pass

    score = (
        blur_score * 4.0
        + megapixels * 1.5
        + min(main_face_ratio, 0.35) * 18.0
        + min(pose_ratio, 0.85) * 2.0
        - max(0, face_count - 1) * 1.5
        + min(filesize_kb / 1024.0, 20.0) * 0.15
    )

    return score, {
        "blur_variance": blur_variance,
        "megapixels": megapixels,
        "main_face_ratio": main_face_ratio,
        "face_count": float(face_count),
        "pose_ratio": pose_ratio,
        "filesize_kb": filesize_kb,
    }


def early_duplicate_pick_score_resolution_strict(image_path: str) -> Tuple[float, Dict[str, float]]:
    """
    Strikte Auswahl-Logik fuer Loop 1 (exact duplicates, threshold=1).

    Bei nahezu pixelidentischen Bildern dominiert die technische Variante
    (Original > Kompressionskopie > Resize) ueber minimale Schaerfe-
    Schwankungen durch JPEG-Recompression. Daher:

      1. Megapixel  - Hauptkriterium (klar dominant)
      2. Dateigroesse in KB - bei gleicher Aufloesung gewinnt die
         technisch unkomprimiertere Version
      3. Schaerfe (Laplacian-Varianz) - reiner Tie-Breaker

    Score so kalibriert, dass jeder Schritt eine eigene Groessenordnung
    bekommt: Megapixel-Term ist immer groesser als der maximale
    Filesize-Term, der wiederum immer groesser als der Schaerfe-Term ist.
    """
    width, height = image_dimensions(image_path)
    pixel_count = max(1.0, float(width * height))
    megapixels = pixel_count / 1_000_000.0
    blur_variance = local_blur_variance(image_path)
    blur_score = math.log1p(max(0.0, blur_variance))
    filesize_kb = local_filesize_kb(image_path)

    # Hierarchische Gewichtung:
    #   Megapixel (×1000)   -> dominanter Term, jeder MP ist ~1000 Punkte
    #   Filesize KB (×0.1)  -> bei 10 MB ~1024 Punkte, immer kleiner als 1 MP
    #   Blur (×1.0)         -> log-Skala, Werte typisch 3-7, reiner Tie-Breaker
    score = (
        megapixels * 1000.0
        + min(filesize_kb, 50_000.0) * 0.1
        + blur_score * 1.0
    )

    # main_face_ratio wird nicht aktiv ins Scoring einbezogen, aber fuer
    # die Sekundaer-Sortierung im Pass mitgeliefert (siehe _early_phash_dedup_pass).
    main_face_ratio = 0.0
    try:
        metrics = local_subject_metrics(image_path, phash_cache=None)
        main_face_ratio = float(metrics.get("main_face_ratio") or 0.0)
    except Exception:
        pass

    return score, {
        "blur_variance": blur_variance,
        "megapixels": megapixels,
        "main_face_ratio": main_face_ratio,
        "face_count": 0.0,
        "pose_ratio": 0.0,
        "filesize_kb": filesize_kb,
    }



def local_filesize_kb(image_path: str) -> float:
    try:
        return os.path.getsize(image_path) / 1024.0
    except Exception:
        return 9999.0


# ============================================================
# Instagram-Frame Auto-Crop
# ============================================================

def detect_and_crop_ig_frame(image_path: str) -> Optional[str]:
    """
    Erkennt Instagram-Story-Rahmen (farbige Balken, Blur-Hintergründe,
    Gradient-Verläufe links/rechts und ggf. oben/unten) und schneidet sie weg.

    Zweistufige Erkennung:
    1. Frame-Indikator: Prüft ob die äußeren ~15% pro Seite ein Frame-Pattern
       haben (median_row_std < 15 = jede Zeile im Strip ist nahezu einfarbig,
       auch wenn sich die Farbe von Zeile zu Zeile ändert → Gradient/Blur/Solid).
    2. Kanten-Lokalisierung: Findet die genaue Grenze zwischen Frame und Foto
       über horizontale Farbgradienten mit Symmetrie-Fallback.

    Gibt den Pfad der permanent gespeicherten, gecroppten Datei zurück
    (in IG_FRAME_CROP_DIR), oder None wenn kein Frame erkannt wurde.
    Bei wiederholtem Aufruf wird das existierende Ergebnis wiederverwendet.
    """
    if not ENABLE_IG_FRAME_CROP:
        return None

    try:
        from scipy.ndimage import uniform_filter1d

        # Cache-Pfad basierend auf Datei-Hash
        src_hash = file_sha1(image_path)
        cached_path = os.path.join(IG_FRAME_CROP_DIR, f"{src_hash}_ig_cropped_v{IG_FRAME_CACHE_VERSION}.jpg")
        if os.path.exists(cached_path):
            return cached_path

        pil_img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        img = np.array(pil_img, dtype=np.float32)
        h, w = img.shape[:2]

        if w < 400 or h < 400:
            return None

        # ── STUFE 1: Frame-Indikator via Zeilen-Uniformität ──
        # Echte IG-Rahmen (solid, blur, gradient) haben pro Zeile fast
        # identische Pixelwerte innerhalb des Randstreifens.
        # median_row_std < 15 = Frame-Pattern, >= 15 = normaler Bildinhalt.
        # Wir testen mehrere Probe-Breiten (schmal → breit), weil ein
        # zu breiter Probe-Strip bei schmalen Rahmen in das Foto hineinragt
        # und fälschlicherweise hohe Varianz zeigt.
        # Mindestens 2 von 4 Breiten müssen Frame-Pattern bestätigen, damit
        # ein einzelner Grenzwert-Treffer bei der schmalsten Probe kein
        # False Positive auslöst.

        def is_frame_side(side: str) -> bool:
            hits = 0
            for divisor in [20, 14, 10, 7]:
                pw = max(20, w // divisor)
                if side == "left":
                    strip = img[:, :pw, :]
                else:
                    strip = img[:, w - pw:, :]
                row_stds = strip.std(axis=1).mean(axis=1)
                if float(np.median(row_stds)) < 15.0:
                    hits += 1
            return hits >= 3

        left_is_frame = is_frame_side("left")
        right_is_frame = is_frame_side("right")

        if not left_is_frame and not right_is_frame:
            return None

        # ── STUFE 2: Exakte Kanten-Lokalisierung ──
        h_grad = np.abs(np.diff(img, axis=1)).mean(axis=2)  # (h, w-1)
        col_score_strict = uniform_filter1d(
            (h_grad > 20).sum(axis=0) / h, size=3
        )
        col_score_relaxed = uniform_filter1d(
            (h_grad > 10).sum(axis=0) / h, size=3
        )

        # Linke Kante suchen (nur wenn links als Frame erkannt)
        left_edge = 0
        if left_is_frame:
            left_zone = col_score_strict[: w // 3]
            left_cands = np.where(left_zone > 0.15)[0]
            if len(left_cands) > 0:
                best = left_cands[np.argmax(left_zone[left_cands])]
                left_edge = int(best) + 1
            else:
                # Gradient ist so weich dass keine scharfe Kante existiert.
                # Fallback: Zeile-für-Zeile row_std scannen und finden wo
                # der Inhalt beginnt (row_std springt über 15).
                for col in range(max(10, w // 20), w // 3):
                    strip = img[:, col:col + 5, :]
                    if float(np.median(strip.std(axis=1).mean(axis=1))) >= 15.0:
                        left_edge = col
                        break

        # Rechte Kante suchen (nur wenn rechts als Frame erkannt)
        right_edge = w
        if right_is_frame:
            r_off = 2 * w // 3
            right_zone = col_score_strict[r_off:]
            right_cands = np.where(right_zone > 0.15)[0]
            if len(right_cands) > 0:
                best = right_cands[np.argmax(right_zone[right_cands])]
                right_edge = int(r_off + best)
            else:
                for col in range(w - max(10, w // 20), 2 * w // 3, -1):
                    strip = img[:, col - 5:col, :]
                    if float(np.median(strip.std(axis=1).mean(axis=1))) >= 15.0:
                        right_edge = col
                        break

        left_border = left_edge
        right_border = w - right_edge

        # Symmetrie-Fallback: wenn nur eine Seite per Stufe-1 erkannt wurde,
        # aber die andere Seite eine schwächere Kante hat
        if left_is_frame and not right_is_frame and left_border >= IG_FRAME_MIN_BORDER_PX:
            sym = w - left_border
            for col in range(max(0, sym - 25), min(len(col_score_relaxed), sym + 26)):
                if col_score_relaxed[col] > 0.12:
                    right_edge = col
                    right_border = w - right_edge
                    break
        elif right_is_frame and not left_is_frame and right_border >= IG_FRAME_MIN_BORDER_PX:
            sym = right_border
            for col in range(max(0, sym - 25), min(w // 3, sym + 26)):
                if col_score_relaxed[col] > 0.12:
                    left_edge = col + 1
                    left_border = left_edge
                    break

        # Mindestens eine Seite muss signifikanten Rand haben
        has_frame = (
            left_border >= IG_FRAME_MIN_BORDER_PX
            or right_border >= IG_FRAME_MIN_BORDER_PX
        )
        if not has_frame:
            return None

        # ── False-Positive-Filter ──
        # Kein einzelner Rand breiter als 30% der Bildbreite
        max_border = max(left_border, right_border)
        if max_border / w > 0.30:
            return None

        # ── Vertikale Kanten (oben/unten) ──
        v_grad = np.abs(np.diff(img, axis=0)).mean(axis=2)
        row_score = uniform_filter1d(
            (v_grad > 20).sum(axis=1) / w, size=3
        )

        top_zone = row_score[: int(h * 0.4)]
        top_cands = np.where(top_zone > 0.25)[0]
        top_edge = int(top_cands[np.argmax(top_zone[top_cands])] + 1) if len(top_cands) > 0 else 0

        bot_off = int(h * 0.7)
        bot_zone = row_score[bot_off:]
        bot_cands = np.where(bot_zone > 0.20)[0]
        bottom_edge = int(bot_off + bot_cands[np.argmax(bot_zone[bot_cands])]) if len(bot_cands) > 0 else h

        # ── Zweistufige Bar-Detection (fuer Android-Nav-Bars, IG-Shadow-Frames) ──
        # Die gradienten-basierte Suche oben verpasst zwei haeufige Faelle:
        #   1) Grosse schwarze Android-Nav-Bar, die weit ueber der 70%-Marke
        #      beginnt (Suchzone ist dann komplett innerhalb der Bar → kein Gradient).
        #   2) Weiche Schatten-Gradienten oben/unten (Drop-Shadows um innere Fotos),
        #      die der row_score>0.25-Schwelle nicht genuegen.
        # Diese Zusatz-Detection triggert NUR, wenn bereits ein Seitenrahmen gefunden
        # wurde. Damit wird verhindert, dass dunkle Kopfkissen o.ae. fuer eine Bar
        # gehalten werden.
        def _detect_bar_two_stage(side: str) -> int:
            """
            Zweistufige Erkennung einer uniformen Bar am oberen/unteren Rand.
            Stufe A: Row-std < 15 -> fast einfarbige Zeile.
            Stufe B: Ab Ende von Stufe A weiter suchen, wenn die Bar eine
            typische dunkle (<60) oder helle (>200) Farbe hat — auch wenn
            die Zeile UI-Elemente (Icons, Buttons) enthaelt, solange die
            dominante Farbe dieselbe bleibt (>55% Pixel). Fuer Android-
            Nav-Bars mit schwarzem Hintergrund + weisse Nav-Icons.
            """
            max_rows = int(h * 0.5)
            if side == "bottom":
                rows_region = img[h - max_rows:, :, :][::-1]  # von unten
            else:
                rows_region = img[:max_rows, :, :]
            row_stds_local = rows_region.std(axis=1).mean(axis=1)

            stage_a = 0
            gap_a = 0
            for i, std_v in enumerate(row_stds_local):
                if std_v < 15.0:
                    stage_a = i + 1
                    gap_a = 0
                else:
                    gap_a += 1
                    if gap_a > 20:
                        break
            if stage_a == 0:
                return 0

            ref_mean = float(rows_region[:stage_a].mean())
            is_dark_bar = ref_mean < 60.0
            is_bright_bar = ref_mean > 200.0
            if not (is_dark_bar or is_bright_bar):
                # Uniforme aber "mittelhelle" Zone (z.B. bunter IG-Frame ohne
                # UI-Overlays): Stufe B uebspringen, Stage-A-Laenge zurueckgeben.
                return stage_a

            stage_b = stage_a
            gap_b = 0
            for i in range(stage_a, len(row_stds_local)):
                row_px = rows_region[i]
                if is_dark_bar:
                    dominant_mask = (row_px < 40).all(axis=-1)
                else:
                    dominant_mask = (row_px > 220).all(axis=-1)
                dominant_ratio = float(dominant_mask.sum()) / float(row_px.shape[0])
                if dominant_ratio > 0.55:
                    stage_b = i + 1
                    gap_b = 0
                else:
                    gap_b += 1
                    if gap_b > 15:
                        break
            return stage_b

        # Nur anwenden, wenn mindestens eine Seite als Frame erkannt wurde
        # (sonst False-Positives bei normalen dunklen Bildelementen wie Kissen,
        #  Haaren, dunklen Hintergruenden).
        if IG_FRAME_TWO_STAGE_BAR_DETECT and (left_is_frame or right_is_frame):
            bar_top = _detect_bar_two_stage("top")
            bar_bot = _detect_bar_two_stage("bottom")
            # Die bereits gefundene Kante nur erweitern, nicht verengen
            if bar_top > top_edge:
                top_edge = bar_top
            if bar_bot > 0 and (h - bar_bot) < bottom_edge:
                bottom_edge = h - bar_bot

        # ── UI-Elemente / Captions entfernen ──
        inner = img[top_edge:bottom_edge, left_edge:right_edge, :]
        inner_h = inner.shape[0]

        content_top = 0
        for r in range(0, min(inner_h // 3, 300), 2):
            if inner[r, :, :].var() > 300:
                content_top = r
                break

        content_bottom = inner_h
        for r in range(inner_h - 1, max(2 * inner_h // 3, inner_h - 300), -2):
            if inner[r, :, :].var() > 300:
                content_bottom = r + 1
                break

        final_top = top_edge + content_top
        final_bottom = top_edge + content_bottom

        # ── Ergebnis-Validierung ──
        content_w = right_edge - left_edge
        content_h = final_bottom - final_top

        if content_w < IG_FRAME_MIN_CONTENT_PX or content_h < IG_FRAME_MIN_CONTENT_PX:
            return None

        total_removed = left_border + right_border + final_top + (h - final_bottom)
        if total_removed < 40:
            return None

        cropped = pil_img.crop((left_edge, final_top, right_edge, final_bottom))
        cropped.save(cached_path, "JPEG", quality=100)
        return cached_path

    except Exception:
        return None


def local_quick_reject(image_path: str, width: int, height: int) -> Optional[str]:
    """
    Legacy-Wrapper: fuehrt ALLE aktivierten Vorfilter durch (Filesize + Blur +
    Exposure). Wird nicht mehr vom Haupt-Pipelineflow aufgerufen (Pipeline
    nutzt local_quick_reject_pre_crop + local_quick_reject_post_crop), aber
    fuer Abwaertskompatibilitaet beibehalten.
    """
    if USE_MIN_FILESIZE_FILTER:
        kb = local_filesize_kb(image_path)
        if kb < HARD_MIN_FILESIZE_KB:
            return f"filesize_too_small_{kb:.0f}kb"

    if USE_BLUR_FILTER:
        variance = local_blur_variance(image_path)
        if variance >= 0 and variance < HARD_MIN_BLUR_VARIANCE:
            return f"blur_variance_too_low_{variance:.1f}"

    if USE_EXPOSURE_FILTER:
        median = local_exposure_median(image_path)
        if median < HARD_MAX_DARK_MEDIAN:
            return f"image_too_dark_median_{median:.0f}"
        if median > HARD_MIN_BRIGHT_MEDIAN:
            return f"image_overexposed_median_{median:.0f}"

    return None


def local_quick_reject_post_crop(image_path: str, width: int, height: int) -> Optional[str]:
    """
    Vorfilter, die NACH dem IG-Frame-Crop laufen sollen: Blur und Exposure.
    Dateigroesse wurde schon vor dem IG-Crop geprueft (dort ist sie noch
    die Original-Filesize).

    Der Blur-Check arbeitet mit der auflösungs-normierten Laplacian-Varianz
    (Stufe 1); der eigentliche Face-Bbox-Check (Stufe 2) laeuft spaeter in
    local_status_override nach der Face-Detection.
    """
    if USE_BLUR_FILTER:
        variance = local_blur_variance(image_path)
        if variance >= 0 and variance < HARD_MIN_BLUR_VARIANCE:
            return f"blur_variance_too_low_{variance:.1f}"

    if USE_EXPOSURE_FILTER:
        median = local_exposure_median(image_path)
        if median < HARD_MAX_DARK_MEDIAN:
            return f"image_too_dark_median_{median:.0f}"
        if median > HARD_MIN_BRIGHT_MEDIAN:
            return f"image_overexposed_median_{median:.0f}"

    return None


def _early_phash_dedup_pass(
    image_paths: List[str],
    phash_cache: Dict[str, int],
    threshold: int,
    keep_per_group: int,
    label: str,
    prefer_resolution_strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """Run one deterministic early pHash grouping pass on already hashed images.

    prefer_resolution_strict: wenn True, wird bei der Survivor-Auswahl strikt
    nach Aufloesung/Dateigroesse priorisiert (siehe
    early_duplicate_pick_score_resolution_strict). Gedacht fuer Loop 1
    (exakte Duplikate), wo der Bildinhalt nahezu identisch ist und die
    technische Variante das einzig sinnvolle Unterscheidungsmerkmal ist.
    """
    survivor_set = set()
    duplicate_set = set()
    no_hash_paths = [path for path in image_paths if path not in phash_cache]

    hashed_items = sorted(
        [(path, phash_cache[path]) for path in image_paths if path in phash_cache],
        key=lambda x: os.path.basename(x[0]).lower(),
    )

    groups: List[Dict[str, Any]] = []
    for path, phash in hashed_items:
        assigned = False
        for group in groups:
            anchor_hash = group["anchor_hash"]
            if hamming_distance(phash, anchor_hash) <= threshold:
                group["members"].append((path, phash))
                assigned = True
                break
        if not assigned:
            groups.append({"anchor_hash": phash, "members": [(path, phash)]})

    survivors = list(no_hash_paths)
    score_cache: Dict[str, Tuple[float, Dict[str, float]]] = {}
    keep_n = max(1, int(keep_per_group))
    pick_fn = (
        early_duplicate_pick_score_resolution_strict
        if prefer_resolution_strict
        else early_duplicate_pick_score
    )
    for group in groups:
        members: List[Tuple[str, int]] = group["members"]
        ranked_members = []
        for member_path, _ in members:
            score_cache[member_path] = pick_fn(member_path)
            ranked_members.append((member_path, *score_cache[member_path]))

        if prefer_resolution_strict:
            # Strikte Reihenfolge: Megapixel zuerst, dann Filesize, dann Schaerfe.
            # Der primaere score enthaelt die Hierarchie bereits, aber wir
            # legen die Einzelfelder nochmal als Tie-Breaker dahinter,
            # damit identische Scores deterministisch aufloesen.
            ranked_members.sort(
                key=lambda item: (
                    item[1],
                    item[2].get("megapixels", 0.0),
                    item[2].get("filesize_kb", 0.0),
                    item[2].get("blur_variance", -1.0),
                    item[2].get("main_face_ratio", 0.0),
                    item[0].lower(),
                ),
                reverse=True,
            )
        else:
            ranked_members.sort(
                key=lambda item: (
                    item[1],
                    item[2].get("main_face_ratio", 0.0),
                    item[2].get("blur_variance", -1.0),
                    item[2].get("megapixels", 0.0),
                    item[2].get("filesize_kb", 0.0),
                    item[0].lower(),
                ),
                reverse=True,
            )

        kept_members = ranked_members[:keep_n]
        removed_members = ranked_members[keep_n:]

        for member_path, _, _ in kept_members:
            survivor_set.add(member_path)
            survivors.append(member_path)
        for member_path, _, _ in removed_members:
            duplicate_set.add(member_path)

    survivors = [p for p in image_paths if p in survivor_set or (p in no_hash_paths and p not in duplicate_set)]
    duplicates = [p for p in image_paths if p in duplicate_set]
    mode_label = "resolution-priority" if prefer_resolution_strict else "quality-priority"
    safe_print(
        f"   ↳ {label}: kept {len(survivors)}, removed {len(duplicates)} duplicates "
        f"(threshold={threshold}, keep/group={keep_n}, mode={mode_label})"
    )
    return survivors, duplicates


def early_phash_dedup(image_paths: List[str]) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Berechnet pHash für alle Bilder und entfernt nur nahezu identische,
    pixelnahe Duplikate
    BEVOR die API aufgerufen wird. Gibt (survivors, duplicates, phash_cache) zurück.
    phash_cache: {absoluter_pfad: phash_int} für Wiederverwendung in local_subject_metrics.
    Gewinner werden pro Duplikat-Gruppe anhand eines deterministischen,
    lokalen Qualitätsscores gewählt; Dateigröße ist nur Tie-Breaker.

    Unterschiede zur spaeteren Pass-2-Deduplikation:
    - strengere Schwelle (EARLY_PHASH_HAMMING_THRESHOLD)
    - Gruppierung nur gegen einen Anchor je Gruppe, um Kettenbildung zu vermeiden
    - mehrere Bilder pro Gruppe koennen erhalten bleiben
    """
    if not USE_EARLY_PHASH_DEDUP or not USE_PHASH_DUPLICATE_SCORING:
        return image_paths, [], {}

    safe_print(f"\n🔍 Early pHash dedup: locally comparing {len(image_paths)} images...")
    hashes: List[Tuple[str, Optional[int]]] = []
    phash_cache: Dict[str, int] = {}
    for p in image_paths:
        try:
            h = compute_phash(p)
            phash_cache[p] = h
        except Exception:
            h = None
        hashes.append((p, h))

    survivors = list(image_paths)
    all_duplicates: List[str] = []

    if bool(globals().get("USE_EARLY_PHASH_LOOP1", True)):
        survivors, duplicates = _early_phash_dedup_pass(
            survivors,
            phash_cache,
            int(globals().get("EARLY_PHASH_HAMMING_THRESHOLD_1", 1)),
            int(globals().get("EARLY_PHASH_KEEP_PER_GROUP_1", 1)),
            "Early pHash loop 1 (exact duplicates)",
            prefer_resolution_strict=bool(globals().get("EARLY_PHASH_LOOP1_PREFER_RESOLUTION", True)),
        )
        all_duplicates.extend(duplicates)

    if bool(globals().get("USE_EARLY_PHASH_LOOP2", True)):
        survivors, duplicates = _early_phash_dedup_pass(
            survivors,
            phash_cache,
            int(globals().get("EARLY_PHASH_HAMMING_THRESHOLD_2", EARLY_PHASH_HAMMING_THRESHOLD)),
            int(globals().get("EARLY_PHASH_KEEP_PER_GROUP_2", EARLY_PHASH_KEEP_PER_GROUP)),
            "Early pHash loop 2 (bulk near-duplicates)",
        )
        all_duplicates.extend(duplicates)

    # Backward-compatible fallback if both UI loops are disabled but the legacy
    # master switch is enabled: run the old single-pass configuration.
    if not bool(globals().get("USE_EARLY_PHASH_LOOP1", True)) and not bool(globals().get("USE_EARLY_PHASH_LOOP2", True)):
        survivors, duplicates = _early_phash_dedup_pass(
            survivors,
            phash_cache,
            int(EARLY_PHASH_HAMMING_THRESHOLD),
            int(EARLY_PHASH_KEEP_PER_GROUP),
            "Early pHash legacy pass",
        )
        all_duplicates.extend(duplicates)

    duplicate_seen = set()
    unique_duplicates = []
    for p in all_duplicates:
        if p not in duplicate_seen:
            duplicate_seen.add(p)
            unique_duplicates.append(p)

    safe_print(f"   ↳ Early pHash total: kept {len(survivors)}, removed {len(unique_duplicates)} duplicates\n")
    return survivors, unique_duplicates, phash_cache


def local_subject_metrics(image_path: str, phash_cache: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    width, height = image_dimensions(image_path)
    metrics: Dict[str, Any] = {
        "width": width,
        "height": height,
        "file_size_mb": round(file_size_mb(image_path), 3),
        "face_count_local": 0,
        "main_face_bbox": None,
        "main_face_ratio": 0.0,
        "secondary_face_area_ratio": 0.0,  # 2.-grösstes Gesicht / grösstes Gesicht (0..1). 0.0 = nur ein Gesicht.
        "pose_bbox": None,
        "torso_landmark_count": -1,  # -1 = MediaPipe nicht gelaufen / nicht verfuegbar
        "phash": None,
        "mtime_bucket": get_file_mtime_bucket(image_path),
    }

    if USE_PHASH_DUPLICATE_SCORING:
        # Aus Early-Dedup-Cache wiederverwenden wenn vorhanden
        if phash_cache and image_path in phash_cache:
            metrics["phash"] = phash_cache[image_path]
        else:
            try:
                metrics["phash"] = compute_phash(image_path)
            except Exception:
                metrics["phash"] = None

    if not HAVE_CV2:
        return metrics

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return metrics

    h, w = img_bgr.shape[:2]

    if MP_FACE is not None:
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            face_result = MP_FACE.process(rgb)
            if face_result and face_result.detections:
                boxes = []
                for det in face_result.detections:
                    bbox = det.location_data.relative_bounding_box
                    x = clamp_int(int(bbox.xmin * w), 0, w - 1)
                    y = clamp_int(int(bbox.ymin * h), 0, h - 1)
                    bw = clamp_int(int(bbox.width * w), 1, w)
                    bh = clamp_int(int(bbox.height * h), 1, h)
                    boxes.append((x, y, bw, bh, float(det.score[0])))
                metrics["face_count_local"] = len(boxes)
                # Boxes nach Area sortieren (groesstes zuerst), dann Verhaeltnis 2./1. berechnen.
                # Genutzt von local_status_override fuer dominance-aware multiple_people-Override.
                sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
                best = sorted_boxes[0]
                metrics["main_face_bbox"] = [best[0], best[1], best[2], best[3]]
                metrics["main_face_ratio"] = bbox_area_ratio(metrics["main_face_bbox"], w, h)
                if len(sorted_boxes) >= 2:
                    main_area = max(1, sorted_boxes[0][2] * sorted_boxes[0][3])
                    sec_area = sorted_boxes[1][2] * sorted_boxes[1][3]
                    metrics["secondary_face_area_ratio"] = round(sec_area / main_area, 4)
        except Exception:
            pass

    if metrics["face_count_local"] == 0 and HAAR_CASCADE is not None:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = HAAR_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                metrics["face_count_local"] = len(faces)
                sorted_faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, bw, bh = sorted_faces[0]
                metrics["main_face_bbox"] = [int(x), int(y), int(bw), int(bh)]
                metrics["main_face_ratio"] = bbox_area_ratio(metrics["main_face_bbox"], w, h)
                if len(sorted_faces) >= 2:
                    main_area = max(1, sorted_faces[0][2] * sorted_faces[0][3])
                    sec_area = sorted_faces[1][2] * sorted_faces[1][3]
                    metrics["secondary_face_area_ratio"] = round(sec_area / main_area, 4)
        except Exception:
            pass

    if MP_POSE is not None:
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pose_result = MP_POSE.process(rgb)
            if pose_result and pose_result.pose_landmarks:
                xs, ys = [], []
                for lm in pose_result.pose_landmarks.landmark:
                    if lm.visibility >= 0.45 and 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0:
                        xs.append(int(lm.x * w))
                        ys.append(int(lm.y * h))
                if len(xs) >= 8:
                    x1, x2 = max(0, min(xs)), min(w, max(xs))
                    y1, y2 = max(0, min(ys)), min(h, max(ys))
                    if x2 > x1 and y2 > y1:
                        metrics["pose_bbox"] = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

                # Torso-Landmark-Count mitberechnen (vermeidet zweiten
                # MediaPipe-Call spaeter in subject_torso_landmark_count).
                # Indizes: 11/12 = Schultern, 23/24 = Hueften.
                torso_idx = (11, 12, 23, 24)
                lms = pose_result.pose_landmarks.landmark
                torso_count = 0
                for idx in torso_idx:
                    if idx >= len(lms):
                        continue
                    lm = lms[idx]
                    if (lm.visibility >= SUBJECT_LANDMARK_VIS_MIN
                            and 0.0 <= lm.x <= 1.0
                            and 0.0 <= lm.y <= 1.0):
                        torso_count += 1
                metrics["torso_landmark_count"] = torso_count
            else:
                # Pose-Detection hat nichts gefunden -> 0 Landmarks
                metrics["torso_landmark_count"] = 0
        except Exception:
            pass

    return metrics


# ============================================================
# 4) CACHE
# ============================================================

def cache_path_for_file(file_hash: str) -> str:
    return os.path.join(CACHE_DIR, f"{file_hash}.json")


# Versions-Tag fuer Audit-Caches. Wird in den Cache-Key eingewoben, damit
# Caches aus inkompatiblen frueheren Versionen automatisch verworfen werden.
# History:
#   v1: implizit, Audit auf gemischter 0-10/0-100-Skala mit Heuristik in
#       normalize_audit_scores. Konnte zu inkonsistenten Scores fuehren
#       (z.B. quality_total = 321), ausserdem alte Smart-Crop-Geometrie
#       (Padding wurde doppelt aufgeschlagen).
#   v2: Audit explizit auf 0-10 (json_schema-strict erzwungen), interne
#       Hochskalierung deterministisch *10. Smart-Crop-Geometrie neu:
#       SMART_PRECROP_PADDING_FACTOR ist Padding-pro-Seite; Crop ist jetzt
#       echt eng (~2.2x Gesicht statt ~5x Gesicht). Caches inkompatibel.
#   v3: Phase 1 - Schema um kategoriale Aux-Felder erweitert (lighting_type,
#       background_type, hair_texture, makeup_intensity, has_glasses_now,
#       glasses_frame_shape) und strukturierte Inventur-Listen
#       (tattoo_inventory_now, piercing_inventory_now). Anti-Hedge-Regeln
#       im Audit-Prompt erzwingen sauberere Trait-Werte. Caches inkompatibel.
#   v4: Body-Build-Bias-Hotfix: Audit-Prompt zwingt body_build auf "" bei
#       Headshots und draengt das Modell, Curvy/Plus_size/Muscular nicht
#       weichzuspuelen. Aenderung der Antwortverteilung -> Cache-Bump,
#       damit alte 'slim'-Antworten auf Headshots neu erhoben werden.
#   v5: Schema um body_skin_visibility erweitert (low/medium/high/n_a).
#       Neues Pflichtfeld im Audit, der Pick-Score nutzt es fuer einen
#       Body-Shot-Bonus zugunsten von Bildern mit gut sichtbarem Koerper
#       (LoRA-Body-Learning). Caches inkompatibel - alle Audits werden
#       neu erhoben. Kein Heuristik-Fallback aus clothing_description.
#   v6: Schema um face_orientation_in_frame erweitert
#       (upright/tilted/sideways/inverted/n_a). Bewertet die Orientierung
#       des Gesichts im 2D-Bildrahmen, nicht die Pose im 3D-Raum: ein
#       liegendes Selfie aus Aug-Hoehe ist 'upright'; ein liegendes
#       Selfie aus extremer Untersicht (Augen unter dem Mund im Frame)
#       ist 'inverted' und fuer LoRA-Training toxisch, weil das Modell
#       die Anatomie umgekehrt lernt. Caches inkompatibel.
#   v7: Audit-Prompt fuer 'issues' geschaerft - explizite Anweisungen
#       fuer 'strong_filter' (Filter-induzierter Hauttextur-Verlust,
#       Wachshaut, blown highlights auf Wangen) und 'extreme_angle'
#       (Worm's-Eye / Bird's-Eye / Selfie-from-below mit verzerrten
#       Koerperproportionen). Erkennung von Bildern wie ueberbelichtete
#       Filter-Selfies und Untersicht-Bett-Selfies wird systematischer.
#       Default-Modell von gpt-5.4-nano auf gpt-5.4-mini gewechselt -
#       nano hat Filter-Hauttextur und extreme Kamerawinkel zuverlaessig
#       falsch bewertet. Caches inkompatibel.
#   v8: Audit-Prompt fuer 'prominent_readable_text' und
#       'watermark_or_overlay' geschaerft. prominent_readable_text wird
#       nun nur fuer GROSSEN, dominanten Text vergeben - kleine Logos
#       auf Kleidung oder winzige Schilder im Hintergrund triggern es
#       nicht mehr. watermark_or_overlay bleibt fuer trainings-toxische
#       Faelle reserviert (Datumsstempel, Wasserzeichen, harte Overlays).
#       Trigger-Logik fuer caption_remove parallel angepasst:
#       prominent_readable_text alleine triggert nicht mehr, nur
#       watermark_or_overlay oder mirror_selfie. Caches inkompatibel.
#   v9: Schema um image_medium erweitert (photograph/illustration/
#       painting/3d_render/screenshot/mixed). Filtert AI-generierte
#       Bilder, Anime/Manga-Fanart, Cartoons, gemalte Portraits und
#       Screenshots heraus, die bisher nur per Freitext-Glueck im
#       short_reason erkannt wurden. Alles ausser 'photograph' fuehrt
#       zu hard reject mit Reason 'non_photographic_medium'. Caches
#       inkompatibel.
AUDIT_CACHE_SCHEMA_VERSION = "v9"


def audit_cache_key(base_hash: str, model: str, variant: str = "audit") -> str:
    raw = f"{AUDIT_CACHE_SCHEMA_VERSION}|{variant}|{base_hash}|{(model or '').strip().lower()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_cached_audit(file_hash: str) -> Optional[Dict[str, Any]]:
    path = cache_path_for_file(file_hash)
    if not ENABLE_CACHE or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_cached_audit(file_hash: str, payload: Dict[str, Any]) -> None:
    if not ENABLE_CACHE:
        return
    path = cache_path_for_file(file_hash)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def audit_cache_payload(audit: Dict[str, Any], model: str, variant: str) -> Dict[str, Any]:
    return {
        "audit": audit,
        "model": model,
        "variant": variant,
        "schema_version": AUDIT_CACHE_SCHEMA_VERSION,
    }


def trigger_cache_path(trigger_word: str) -> str:
    key = slugify_filename(trigger_word.lower())
    return os.path.join(TRIGGER_CACHE_DIR, f"{key}.json")


def load_cached_trigger_check(trigger_word: str) -> Optional[Dict[str, Any]]:
    path = trigger_cache_path(trigger_word)
    if not ENABLE_CACHE or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_cached_trigger_check(trigger_word: str, payload: Dict[str, Any]) -> None:
    if not ENABLE_CACHE:
        return
    path = trigger_cache_path(trigger_word)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ============================================================
# 5) CLIP
# ============================================================

def get_clip_cache_path(file_hash: str) -> str:
    return os.path.join(CLIP_CACHE_DIR, f"{file_hash}.npy")


def load_clip_embedding_cached(file_hash: str) -> Optional[np.ndarray]:
    path = get_clip_cache_path(file_hash)
    if not ENABLE_CACHE or not os.path.exists(path):
        return None
    try:
        vec = np.load(path)
        return vec.astype(np.float32)
    except Exception:
        return None


def save_clip_embedding_cached(file_hash: str, vec: np.ndarray) -> None:
    if not ENABLE_CACHE:
        return
    path = get_clip_cache_path(file_hash)
    np.save(path, vec.astype(np.float32))


def compute_clip_embedding(image_path: str, file_hash: str) -> Optional[np.ndarray]:
    if not USE_CLIP_DUPLICATE_SCORING or not HAVE_CLIP or CLIP_MODEL is None or CLIP_PREPROCESS is None:
        return None

    cached = load_clip_embedding_cached(file_hash)
    if cached is not None:
        return cached

    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        tensor = CLIP_PREPROCESS(img).unsqueeze(0).to(CLIP_DEVICE)

        with torch.no_grad():
            features = CLIP_MODEL.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            vec = features[0].detach().cpu().numpy().astype(np.float32)

        save_clip_embedding_cached(file_hash, vec)
        return vec
    except Exception:
        return None


def clip_cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return -1.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return -1.0
    return float(np.dot(a, b) / denom)


# ============================================================
# 5b) ARCFACE IDENTITAETS-EMBEDDING
# ============================================================

def _init_arcface_app():
    """
    Lazy-Initialisierung der InsightFace FaceAnalysis App. Idempotent.
    Wird erst beim ersten Aufruf von compute_arcface_embedding() angetriggert,
    damit Datasets ohne aktivierten Identity-Check nicht beim Start blockiert
    werden (Modell-Download kann beim allerersten Lauf ~250 MB sein).
    """
    global ARCFACE_APP, ARCFACE_INIT_ATTEMPTED
    if ARCFACE_INIT_ATTEMPTED:
        return ARCFACE_APP
    ARCFACE_INIT_ATTEMPTED = True
    if not (USE_ARCFACE_IDENTITY_CHECK and HAVE_INSIGHTFACE):
        return None
    try:
        providers = ["CPUExecutionProvider"]
        if ARCFACE_USE_CUDA and HAVE_TORCH and torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        from insightface.app import FaceAnalysis  # type: ignore
        app = FaceAnalysis(name=ARCFACE_MODEL_PACK, providers=providers)
        app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1,
                    det_size=(ARCFACE_DET_SIZE, ARCFACE_DET_SIZE))
        ARCFACE_APP = app
        safe_print(f"   ArcFace ready ({ARCFACE_MODEL_PACK}, providers={providers[0]})")
    except Exception as e:
        safe_print(f"   ⚠️ ArcFace init failed ({e}); identity check disabled.")
        ARCFACE_APP = None
    return ARCFACE_APP


def get_arcface_cache_path(file_hash: str) -> str:
    return os.path.join(ARCFACE_CACHE_DIR, f"{file_hash}.npy")


def load_arcface_embedding_cached(file_hash: str) -> Optional[np.ndarray]:
    path = get_arcface_cache_path(file_hash)
    if not ENABLE_CACHE or not os.path.exists(path):
        return None
    try:
        vec = np.load(path)
        return vec.astype(np.float32)
    except Exception:
        return None


def save_arcface_embedding_cached(file_hash: str, vec: np.ndarray) -> None:
    if not ENABLE_CACHE:
        return
    path = get_arcface_cache_path(file_hash)
    np.save(path, vec.astype(np.float32))


def compute_arcface_embedding(image_path: str, file_hash: str) -> Optional[np.ndarray]:
    """
    Berechnet ein 512-dimensionales ArcFace-Embedding fuer das groesste
    erkannte Gesicht im Bild. Liefert None, wenn:
      - InsightFace/onnxruntime nicht installiert ist
      - das Feature deaktiviert ist
      - die Modell-Init fehlschlaegt
      - kein Gesicht erkannt wurde
    """
    if not USE_ARCFACE_IDENTITY_CHECK or not HAVE_INSIGHTFACE:
        return None

    cached = load_arcface_embedding_cached(file_hash)
    if cached is not None:
        return cached

    app = _init_arcface_app()
    if app is None:
        return None

    try:
        # InsightFace erwartet BGR (cv2-Konvention). Wir laden via PIL und
        # konvertieren, um konsistent mit dem Rest des Tools zu bleiben.
        with Image.open(image_path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
            rgb_np = np.array(pil_img)
        bgr_np = rgb_np[..., ::-1].copy()

        faces = app.get(bgr_np)
        if not faces:
            return None

        # Groesstes erkanntes Gesicht waehlen (Bbox-Flaeche)
        def _bbox_area(face):
            x1, y1, x2, y2 = face.bbox
            return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

        main_face = max(faces, key=_bbox_area)

        emb = getattr(main_face, "normed_embedding", None)
        if emb is None:
            emb = getattr(main_face, "embedding", None)
            if emb is None:
                return None
            emb = np.asarray(emb, dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            if norm <= 0:
                return None
            emb = emb / norm
        else:
            emb = np.asarray(emb, dtype=np.float32)

        save_arcface_embedding_cached(file_hash, emb)
        return emb
    except Exception:
        return None


def arcface_cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Cosine-Similarity zwischen zwei Embeddings. -1.0 wenn ungueltig."""
    if a is None or b is None:
        return -1.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def compute_trimmed_centroid(
    embeddings: List[np.ndarray],
    trim_fraction: float = ARCFACE_TRIM_FRACTION,
) -> Optional[np.ndarray]:
    """
    Berechnet einen outlier-getrimmten Centroid:
      1. Initialer Centroid = Mittelwert aller Embeddings
      2. Cosine-Distanz zum Initial-Centroid pro Embedding
      3. Schlechteste trim_fraction (z.B. 10%) verwerfen
      4. Centroid auf den verbleibenden Embeddings neu berechnen

    Damit zieht ein einzelnes "Schwester-Bild" den finalen Centroid nicht
    in Richtung der falschen Identitaet, sondern wird beim Trimming entfernt.

    Liefert ein L2-normalisiertes Centroid-Embedding oder None.
    """
    if not embeddings:
        return None

    arr = np.stack(embeddings, axis=0).astype(np.float32)

    # Initialer Centroid + L2-Normierung
    init_centroid = arr.mean(axis=0)
    init_norm = float(np.linalg.norm(init_centroid))
    if init_norm <= 0:
        return None
    init_centroid = init_centroid / init_norm

    # Distanzen zum initialen Centroid
    sims = arr @ init_centroid  # weil arr und init_centroid normiert sind
    n = len(embeddings)
    keep_n = max(1, int(round(n * (1.0 - trim_fraction))))

    # Bei sehr kleinen Sets das Trimming deaktivieren - sonst koennte ein
    # einziger Outlier 100% seines Einflusses ueber die verbleibenden Bilder
    # ausueben, und das Ergebnis waere nicht stabiler als ohne Trimming.
    if n < 8:
        keep_n = n

    # Top keep_n Embeddings nach Similarity behalten
    keep_idx = np.argsort(-sims)[:keep_n]
    trimmed = arr[keep_idx]

    final_centroid = trimmed.mean(axis=0)
    final_norm = float(np.linalg.norm(final_centroid))
    if final_norm <= 0:
        return None
    return final_centroid / final_norm


def run_identity_consistency_check(
    selected_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Berechnet Identitaets-Konsistenz fuer die finalen Bilder:
      - Embedding pro Bild (mit Cache)
      - Outlier-getrimmter Centroid
      - Cosine-Similarity jedes Embeddings zum Centroid
      - Klassifikation in hard / soft / ok

    Schreibt direkt in jeden Row:
      arcface_distance_to_centroid: float (-1.0 wenn nicht berechenbar)
      arcface_flag: "hard" | "soft" | "ok" | "no_face" | "skipped"

    Gibt einen Summary-Dict zurueck mit Counts und der Liste der hard-flags.
    Wenn das Feature aus oder die Library nicht verfuegbar ist, wird ein
    "skipped"-Status auf alle Rows gesetzt und ein leerer Summary geliefert.
    """
    summary: Dict[str, Any] = {
        "enabled": False,
        "centroid_present": False,
        "n_with_face": 0,
        "n_no_face": 0,
        "n_hard": 0,
        "n_soft": 0,
        "n_ok": 0,
        "hard_flagged": [],   # filenames mit hard flag
        "soft_flagged": [],   # filenames mit soft flag
        "skipped_reason": "",
    }

    # Voraussetzungen pruefen
    if not USE_ARCFACE_IDENTITY_CHECK:
        summary["skipped_reason"] = "feature_disabled"
        for r in selected_rows:
            r["arcface_flag"] = "skipped"
            r["arcface_distance_to_centroid"] = -1.0
        return summary

    if not HAVE_INSIGHTFACE:
        summary["skipped_reason"] = "insightface_not_installed"
        for r in selected_rows:
            r["arcface_flag"] = "skipped"
            r["arcface_distance_to_centroid"] = -1.0
        return summary

    summary["enabled"] = True
    safe_print("\n🪪 Identity consistency check (ArcFace):")

    # Embeddings sammeln
    embeddings: List[np.ndarray] = []
    rows_with_emb: List[Tuple[Dict[str, Any], np.ndarray]] = []
    for row in selected_rows:
        # Originalpfad oder gecropter Pfad - body_aware_crop wird hier nicht
        # benutzt, weil ArcFace selber Face-Detection macht und das Original
        # mehr Kontext bietet (Hintergrund schadet ArcFace nicht).
        path = row.get("original_path", "")
        file_hash = row.get("file_hash") or (file_sha1(path) if path and os.path.exists(path) else "")
        if not file_hash or not os.path.exists(path):
            row["arcface_flag"] = "no_face"
            row["arcface_distance_to_centroid"] = -1.0
            summary["n_no_face"] += 1
            continue

        emb = compute_arcface_embedding(path, file_hash)
        if emb is None:
            row["arcface_flag"] = "no_face"
            row["arcface_distance_to_centroid"] = -1.0
            summary["n_no_face"] += 1
            continue

        embeddings.append(emb)
        rows_with_emb.append((row, emb))

    summary["n_with_face"] = len(embeddings)

    if len(embeddings) < ARCFACE_MIN_FACES_FOR_CENTROID:
        summary["skipped_reason"] = (
            f"too_few_faces_{len(embeddings)}_lt_{ARCFACE_MIN_FACES_FOR_CENTROID}"
        )
        safe_print(
            f"   ⚠️ Only {len(embeddings)} faces detected; "
            f"need at least {ARCFACE_MIN_FACES_FOR_CENTROID} for a meaningful centroid. "
            f"Skipping consistency classification."
        )
        for row, _ in rows_with_emb:
            row["arcface_flag"] = "skipped"
            row["arcface_distance_to_centroid"] = -1.0
        return summary

    # Outlier-getrimmten Centroid berechnen
    centroid = compute_trimmed_centroid(embeddings, ARCFACE_TRIM_FRACTION)
    if centroid is None:
        summary["skipped_reason"] = "centroid_computation_failed"
        for row, _ in rows_with_emb:
            row["arcface_flag"] = "skipped"
            row["arcface_distance_to_centroid"] = -1.0
        return summary

    summary["centroid_present"] = True

    # Klassifikation pro Row
    for row, emb in rows_with_emb:
        sim = arcface_cosine(emb, centroid)
        row["arcface_distance_to_centroid"] = round(sim, 4)
        if sim < ARCFACE_HARD_THRESHOLD:
            row["arcface_flag"] = "hard"
            summary["n_hard"] += 1
            summary["hard_flagged"].append(row.get("original_filename", ""))
        elif sim < ARCFACE_SOFT_THRESHOLD:
            row["arcface_flag"] = "soft"
            summary["n_soft"] += 1
            summary["soft_flagged"].append(row.get("original_filename", ""))
        else:
            row["arcface_flag"] = "ok"
            summary["n_ok"] += 1

    safe_print(
        f"   {summary['n_ok']} ok | {summary['n_soft']} soft-flag | "
        f"{summary['n_hard']} hard-flag | {summary['n_no_face']} no face detected"
    )
    if summary["n_hard"]:
        safe_print(
            f"   ⚠️ Hard-flagged (likely different person, will be moved out of train_ready):"
        )
        for fn in summary["hard_flagged"]:
            row = next((r for r in selected_rows if r.get("original_filename") == fn), None)
            sim_str = f"sim={row['arcface_distance_to_centroid']:.3f}" if row else ""
            safe_print(f"      - {fn} ({sim_str})")
    if summary["n_soft"]:
        safe_print(
            f"   ℹ️ Soft-flagged (borderline, kept in train_ready, see report):"
        )
        for fn in summary["soft_flagged"]:
            row = next((r for r in selected_rows if r.get("original_filename") == fn), None)
            sim_str = f"sim={row['arcface_distance_to_centroid']:.3f}" if row else ""
            safe_print(f"      - {fn} ({sim_str})")

    return summary


# ============================================================
# 6) OPENAI / RESPONSES API
# ============================================================

def extract_response_text(response_json: Dict[str, Any]) -> str:
    if response_json.get("NSFW_BLOCKED"):
        return '{"NSFW_BLOCKED": True}'

    for item in response_json.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text" and part.get("text"):
                    return part["text"]
    raise ValueError("Kein output_text in Responses-Antwort gefunden.")


def responses_api_call(model: str, payload: Dict[str, Any], phase_label: str = "responses_api") -> Dict[str, Any]:
    if not API_KEY or not str(API_KEY).strip():
        raise RuntimeError(
            "OpenAI API key fehlt. Bitte in der UI im Feld 'OpenAI API Key' eintragen "
            "oder die Umgebungsvariable OPENAI_API_KEY setzen."
        )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        attempt_label = f"{phase_label} | model={model} | attempt={attempt}/{MAX_RETRIES}"
        started_at, stop_event, thread = start_phase_heartbeat(attempt_label)
        try:
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                json={"model": model, **payload},
                timeout=180,
            )
            if response.status_code >= 400:
                try:
                    err = response.json()
                except Exception:
                    err = {"error": {"message": response.text}}
                message = err.get("error", {}).get("message", f"HTTP {response.status_code}")
                raise RuntimeError(message)
            stop_phase_heartbeat(attempt_label, started_at, stop_event, thread, success=True)
            safe_print(f"   ↳ API response ok: status={response.status_code} | phase={phase_label}")
            return response.json()
        except Exception as e:
            stop_phase_heartbeat(attempt_label, started_at, stop_event, thread, success=False)
            last_error = e
            if attempt >= MAX_RETRIES:
                break
            sleep_s = RETRY_BASE_SECONDS * attempt
            safe_print(
                f"   ↳ API error in {phase_label}, retry {attempt}/{MAX_RETRIES} in {sleep_s:.1f}s: {e}"
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"Responses-API fehlgeschlagen: {last_error}")


def triggerword_check_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
            "is_potentially_problematic": {"type": "boolean"},
            "reason": {"type": "string"},
            "suggested_trigger": {"type": "string"},
        },
        "required": ["risk_level", "is_potentially_problematic", "reason", "suggested_trigger"],
        "additionalProperties": False,
    }


def check_trigger_word_via_ai(trigger_word: str) -> Dict[str, Any]:
    cached = load_cached_trigger_check(trigger_word)
    if cached:
        return cached

    instructions = """
You are evaluating whether a LoRA trigger word is too generic, too name-like, or likely to collide with preexisting associations in a base image model.
Be practical and conservative.
"""

    payload = {
        "instructions": instructions,
        "input": [{
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": (
                    f"Evaluate this trigger word for a person LoRA: '{trigger_word}'. "
                    f"Return whether it is potentially problematic and suggest a safer alternative if needed."
                )
            }]
        }],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "triggerword_check",
                "schema": triggerword_check_schema(),
                "strict": True,
            }
        },
        "max_output_tokens": 300,
        "store": False,
        "temperature": 0.1,
    }

    data = responses_api_call(TRIGGER_CHECK_MODEL, payload)
    text = extract_response_text(data)
    parsed = json.loads(text)
    save_cached_trigger_check(trigger_word, parsed)
    return parsed


def build_api_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "gender_class": {"type": "string", "enum": ["man", "woman", "boy", "girl", "person"]},
            "shot_type": {"type": "string", "enum": ["headshot", "medium", "full_body"]},
            "multiple_people": {"type": "boolean"},
            "main_subject_clear": {"type": "boolean"},
            "face_visible": {"type": "boolean"},
            "face_bbox_ai": {
                "type": "array",
                "description": "Bounding box of the main person's face as [xmin, ymin, width, height] using relative coords (0.0 to 1.0). If no face is visible, return empty array.",
                "items": {"type": "number"}
            },
            "face_occlusion": {"type": "string", "enum": ["none", "minor", "major"]},
            "watermark_or_overlay": {"type": "boolean"},
            "image_medium": {
                "type": "string",
                "enum": ["photograph", "illustration", "painting", "3d_render", "screenshot", "mixed"],
                "description": "Medium of the image. Use 'photograph' only for real camera photos of real people. Anything else (drawings, anime, paintings, AI-generated illustrations, video game screenshots, app screenshots, mixed photo+overlay) is non-photographic and unsuitable for identity training."
            },
            "prominent_readable_text": {"type": "boolean"},
            "mirror_selfie": {"type": "boolean"},
            "hair_description": {"type": "string"},
            "beard_description": {"type": "string"},
            "glasses_description": {"type": "string"},
            "piercings_description": {"type": "string"},
            "makeup_description": {"type": "string"},
            "skin_tone": {"type": "string"},
            "eye_color": {
                "type": "string",
                "description": "Eye color of the main subject, e.g. 'blue', 'green', 'gray-green', 'brown', 'dark brown'. Empty string if not visible."
            },
            "body_build": {"type": "string"},
            "body_skin_visibility": {
                "type": "string",
                "enum": ["low", "medium", "high", "n_a"],
                "description": "Fraction of bare skin visible on the body (excluding face and neck). See prompt for criteria. Use 'n_a' for headshots where the body is not in frame."
            },
            "face_orientation_in_frame": {
                "type": "string",
                "enum": ["upright", "tilted", "sideways", "inverted", "n_a"],
                "description": "Orientation of the face within the 2D image frame, NOT the person's pose in 3D space. Judge what a viewer sees in the frame. See prompt for criteria. Use 'n_a' if no face is in the frame."
            },
            "tattoos_visible": {"type": "boolean"},
            "tattoos_description": {"type": "string"},
            "clothing_description": {"type": "string"},
            "pose_description": {"type": "string"},
            "expression": {"type": "string"},
            "gaze_direction": {"type": "string"},
            "head_pose_bucket": {
                "type": "string",
                "enum": [
                    "frontal",
                    "three_quarter_left",
                    "three_quarter_right",
                    "profile_left",
                    "profile_right",
                    "looking_up",
                    "looking_down",
                    "back",
                    "unknown"
                ],
                "description": "Coarse classification of the main subject's head orientation. 'frontal' = facing camera, 'three_quarter' = ~30-60 degrees yaw, 'profile' = ~90 degrees yaw, 'looking_up'/'looking_down' = significant pitch, 'back' = head turned away, 'unknown' = not determinable."
            },
            "background_description": {"type": "string"},
            "lighting_description": {"type": "string"},

            # --- NEU (Phase 1): kategoriale Aux-Felder fuer Profile-Stage ---
            "lighting_type": {
                "type": "string",
                "description": (
                    "Categorical lighting label. Allowed values: studio_softbox, "
                    "studio_ringlight, studio_other, natural_outdoor_sun, "
                    "natural_outdoor_overcast, natural_indoor_window, "
                    "indoor_artificial, mixed, low_light. "
                    "Use empty string only if truly indeterminable. "
                    "This is critical for studio-bias correction in skin-tone profiling."
                )
            },
            "background_type": {
                "type": "string",
                "description": (
                    "Categorical background label. Allowed values: studio_plain, "
                    "studio_textured, indoor_room, indoor_bathroom, outdoor_urban, "
                    "outdoor_nature, outdoor_beach, outdoor_other, vehicle_interior, "
                    "transparent_or_isolated, other. Empty string only if no background visible."
                )
            },
            "hair_texture": {
                "type": "string",
                "description": (
                    "Hair texture (separate from style). Use one of: straight, wavy, "
                    "curly, coily, afro_textured. If hair is in protective styling "
                    "(braids, locs) use the underlying natural texture if discernible, "
                    "else empty string."
                )
            },
            "makeup_intensity": {
                "type": "string",
                "description": (
                    "Makeup intensity classification. Use exactly one of: none, "
                    "minimal, natural, defined, full, dramatic. "
                    "NEVER use 'or'-phrases like 'minimal or no'. If unclear, pick the "
                    "closest single value."
                )
            },
            "has_glasses_now": {
                "type": "boolean",
                "description": "True if eyeglasses are visible in this image."
            },
            "glasses_frame_shape": {
                "type": "string",
                "description": (
                    "If has_glasses_now is true: shape of the frame. One of: round, "
                    "square, rectangular, oval, aviator, cat_eye, oversized, rimless, "
                    "browline, geometric, other. Empty string if no glasses."
                )
            },
            "tattoo_inventory_now": {
                "type": "array",
                "description": (
                    "Structured list of tattoos VISIBLE in this image. Each entry "
                    "has a controlled location and a freetext description. Only "
                    "include tattoos actually visible; do NOT speculate about hidden ones."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "enum": TATTOO_LOCATION_ENUM},
                        "description": {
                            "type": "string",
                            "description": "Short freetext description, e.g. 'rose tattoo', 'script tattoo', 'small heart'."
                        }
                    },
                    "required": ["location", "description"],
                    "additionalProperties": False,
                }
            },
            "piercing_inventory_now": {
                "type": "array",
                "description": "Structured list of piercings VISIBLE in this image.",
                "items": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "enum": PIERCING_LOCATION_ENUM},
                        "description": {
                            "type": "string",
                            "description": "Short description like 'small hoop', 'stud', 'plug/gauge'."
                        }
                    },
                    "required": ["location", "description"],
                    "additionalProperties": False,
                }
            },
            "quality_sharpness": {"type": "number", "minimum": 0, "maximum": 10},
            "quality_lighting": {"type": "number", "minimum": 0, "maximum": 10},
            "quality_composition": {"type": "number", "minimum": 0, "maximum": 10},
            "quality_identity_usefulness": {"type": "number", "minimum": 0, "maximum": 10},
            "quality_total": {"type": "number", "minimum": 0, "maximum": 10},
            "issues": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "none", "motion_blur", "soft_focus", "heavy_noise", 
                        "overexposed", "underexposed", "harsh_flash", "extreme_angle", 
                        "small_face", "sunglasses", "heavy_occlusion", "strong_filter", 
                        "cropped_limbs", "busy_background", "text_overlay", "watermark", "other"
                    ]
                }
            },
            "suggested_status": {"type": "string", "enum": ["keep", "review", "reject"]},
            "short_reason": {"type": "string"},
        },
        "required": [
            "gender_class",
            "shot_type",
            "multiple_people",
            "main_subject_clear",
            "face_visible",
            "face_bbox_ai",
            "face_occlusion",
            "watermark_or_overlay",
            "image_medium",
            "prominent_readable_text",
            "mirror_selfie",
            "hair_description",
            "beard_description",
            "glasses_description",
            "piercings_description",
            "makeup_description",
            "skin_tone",
            "eye_color",
            "body_build",
            "body_skin_visibility",
            "face_orientation_in_frame",
            "tattoos_visible",
            "tattoos_description",
            "clothing_description",
            "pose_description",
            "expression",
            "gaze_direction",
            "head_pose_bucket",
            "background_description",
            "lighting_description",
            "lighting_type",
            "background_type",
            "hair_texture",
            "makeup_intensity",
            "has_glasses_now",
            "glasses_frame_shape",
            "tattoo_inventory_now",
            "piercing_inventory_now",
            "quality_sharpness",
            "quality_lighting",
            "quality_composition",
            "quality_identity_usefulness",
            "quality_total",
            "issues",
            "suggested_status",
            "short_reason"
        ],
        "additionalProperties": False,
    }


def openai_audit_image(
    image_path: str,
    local_meta: Dict[str, Any],
    model: Optional[str] = None,
    phase_label: Optional[str] = None,
) -> Dict[str, Any]:
    schema = build_api_schema()
    image_b64 = resize_and_encode_for_api(image_path)
    chosen_model = (model or AI_MODEL).strip() or AI_MODEL

    instructions = f"""
You are auditing a single image for a person LoRA training dataset for a realistic image model.
Trigger word: "{TRIGGER_WORD}".

Return only raw visible facts about THIS ONE IMAGE.
Do not compare against a dataset.
Do not write a final caption.
Do not speculate.

CRITICAL FACE DETECTION TASK:
You must locate the main subject's face if visible.
Provide `face_bbox_ai` as an array of 4 floats: [xmin, ymin, width, height] using relative coordinates from 0.0 to 1.0 (where 0.0, 0.0 is the top-left corner of the image).
Example for a face in the center: [0.4, 0.4, 0.2, 0.2]
If the face is completely hidden or looking away so no facial features are visible, set `face_visible` to false and return an empty array [].

Quality rules:
- For headshot: the face must be sharp, clear, and useful for identity learning.
- For medium/full_body/landscape: overall subject readability, body proportions, and training usefulness matter more than pore-level face detail.
- If it's a full_body shot from behind (face NOT visible): Score the body shape, posture, and clothing! Do NOT penalize or reject just because the face is hidden.
- Use "keep" for any image that is good or great. Use "review" ONLY if there are major flaws (e.g. heavy blur, bad occlusion).
- Use "reject" when the image is clearly harmful or useless for training.

SCORING SYSTEM:
You MUST score every quality dimension on a strict 0.00 to 10.00 scale.
Use decimals for fine-grained scoring (e.g. 7.50 or 8.20). Do NOT use a 0-100 scale.
- quality_sharpness: 0.00 to 10.00 (decimals required for nuance)
- quality_lighting: 0.00 to 10.00 (decimals required for nuance)
- quality_composition: 0.00 to 10.00 (decimals required for nuance)
- quality_identity_usefulness: 0.00 to 10.00 (decimals required for nuance)
- quality_total: weighted internal field; you may set it to the simple average of the 4 scores above (also 0.00 to 10.00). The host system will recompute the canonical weighted score.

Important:
- TEXT/WATERMARK/OVERLAY DETECTION (critical for LoRA training cleanliness):
  Two separate fields with DIFFERENT thresholds. Read carefully.

  watermark_or_overlay: set True ONLY when there is a TRAINING-TOXIC overlay
  burned into or laid over the image. These are elements that did not exist
  in the original scene and would be reproduced by the LoRA as part of the
  person's "look" if not flagged. Trigger cases:
    * Visible date stamps (e.g. "'21 09 24" in a corner)
    * Photographer/site watermarks (e.g. "© Photographer", "shutterstock")
    * App/filter overlays (Snapchat date, Instagram-style stickers, GIF text)
    * Heavy frame borders, polaroid-style frames added in post
  Do NOT trigger for: text on physical objects in the scene (shirt prints,
  helmet logos, signs in the background), text that is part of the photo
  content rather than added on top of it.

  prominent_readable_text: set True ONLY when text is LARGE and DOMINANT
  enough to occupy a noticeable portion of the frame and would meaningfully
  compete with the subject for visual attention. Threshold: text must be at
  least 8-10% of the frame area, OR centrally placed and clearly legible at
  a glance, OR repeated multiple times in the frame. Trigger cases:
    * Large slogan or text on the front of a shirt/sweater filling much of
      the chest
    * Big neon/advertising signs prominently behind the subject
    * Large book/magazine/poster covers held up by the subject
  Do NOT trigger for: small brand logos under ~5% of frame (helmet logos,
  small embroidery on jackets, tiny clothing tags), distant signage in the
  background, license plates or street signs not central to composition,
  faint reflections, blurred background text. When in doubt, DO NOT flag -
  this field has been over-triggered in the past and needs a high bar.

  Both fields independent: a date stamp is watermark_or_overlay=True even
  if it is small. A huge shirt slogan is prominent_readable_text=True even
  if it is not an overlay. They can both be True simultaneously.

- IMAGE MEDIUM CLASSIFICATION (critical, hard filter):
  Determine what TYPE of image this is. The training pipeline can only use
  real photographs of real people - anything else teaches the model wrong
  visual statistics. Use exactly one value:
    * "photograph": a real camera photo of a real human being. This is the
      ONLY value that allows the image into the training set. Includes
      selfies, portraits, candid shots, professional photography, photos
      with light filters/grading, scanned analog photos.
    * "illustration": drawings, line art, anime/manga style, cartoon,
      stylized digital art, fanart, comic-book style. Even highly detailed
      digital illustrations belong here, NOT in 'photograph'.
    * "painting": traditional or digital paintings (oil, watercolor,
      acrylic style, painterly digital art that imitates traditional media).
    * "3d_render": CGI, 3D-rendered characters, video game screenshots,
      Pixar-style renders, Daz3D, Blender renders, virtual avatars.
    * "screenshot": app interface screenshots, social media UI screenshots
      (TikTok Live, Instagram, Discord), video calls, anything that shows
      a software interface or chrome around the actual content. A photo
      that just happens to have a small UI element (timestamp, chat bubble)
      should be 'photograph' with watermark_or_overlay=True instead.
    * "mixed": composite images that combine a photograph with significant
      illustrative or graphic-design elements (Instagram Story art layered
      over a selfie, photo + drawn-on stickers/text covering large parts,
      heavily photoshopped fanart of a real person).
  When in doubt between 'photograph' and 'mixed': if removing the graphic
  layer would still leave a recognizable, usable photograph, use
  'photograph' + watermark_or_overlay. If the graphic layer is integral to
  the image and dominates significantly, use 'mixed'.
  When in doubt between 'photograph' and 'illustration': look for skin
  pore detail, realistic hair strands, natural lighting falloff. If those
  are absent and replaced by stylized smooth shading or line art, it is
  'illustration' regardless of how realistic the proportions are.
  Be strict. False classification of an illustration as 'photograph'
  poisons the training set. False classification of a photograph as
  'illustration' is a low-cost false positive (the image goes to review).

- Flag multiple prominent people.
- Ignore brand names and exact text content. Just flag the presence.
- Describe visible tattoos only as a raw fact.
- Describe hair color, length, and texture PRECISELY (e.g. "long wavy blonde hair", "short dark brown curly hair"). Never return empty or vague values like "brown".
- Describe eye color PRECISELY if visible (e.g. "blue", "green", "gray-green", "hazel"). Return empty string only if eyes are not visible.
- Describe skin_tone as a neutral factual value (e.g. "fair", "light", "medium", "olive", "dark"). Never return empty.
- Describe beard/glasses/piercings/makeup only as visible raw facts.
- body_build: ONLY judge body build when the body is actually visible.
    * On HEADSHOTS (only head and shoulders visible): body_build MUST be empty string "". Do not guess.
    * On medium shots: only fill body_build if torso shape is clearly readable.
    * On full_body shots: judge accurately.
    * Resist the tendency to default to "slim" or "average". Use "curvy", "plus_size",
      "athletic", "muscular" when the body actually shows those traits. Do not soften.
    * Allowed values: slim | average | athletic | curvy | plus_size | muscular | "" (empty for headshots).
- body_skin_visibility: how much bare skin (body only, EXCLUDING face and neck)
  is visible. Use exactly one of these values:
    * "low": long sleeves, long pants/skirt below the knee, body almost fully
      covered (winter coat, hoodie + jeans, full-length dress, business suit).
    * "medium": short sleeves OR knee-length bottoms, forearms or lower legs
      visible but not both extremities prominently bare (t-shirt + jeans,
      polo + chinos, blouse + midi skirt).
    * "high": tank top / sleeveless top / spaghetti straps, OR shorts above the
      knee, OR swimwear (bikini, swimsuit, trunks), OR sportswear with
      significant bare skin (athletic crop top, running shorts).
    * "n_a": headshot where the body is not in frame, OR body fully obscured
      (e.g. wrapped in a blanket, only silhouette visible, framing too tight).
  Decide based on what is visible in THIS image only. Do not soften toward
  "low" out of caution. This is a neutral factual classification.
- face_orientation_in_frame: orientation of the face within the 2D IMAGE
  FRAME as a viewer sees it. This is NOT the person's pose in 3D space - a
  person lying on a bed can still appear "upright" in the frame if the
  photo was taken so the eyes are above the mouth in the picture. Judge
  the rendered image, NOT what you imagine the real-world scene looks like.
  CRITICAL: do not mentally rotate the image to "fix" it. If a viewer
  scrolling on a phone would see the face upside-down without rotating
  their device, classify it as 'inverted'. Use exactly one value:
    * "upright": eyes clearly above mouth in the frame, head roughly
      vertical (rotation up to ~30 degrees from vertical). Standard
      portraits, normal selfies, walking shots etc.
    * "tilted": noticeable rotation ~30-60 degrees, head visibly leaning
      but eyes still in the upper region of the face area in the frame.
    * "sideways": face rotated ~60-120 degrees in the frame, eyes appear
      LEFT or RIGHT of the mouth rather than above it. Typical for
      selfies of someone lying on their side where the camera is held
      level with the body.
    * "inverted": face is upside-down in the frame, eyes appear BELOW the
      mouth. Typical for selfies of someone lying on their back where the
      camera is held above and pointed down toward their feet, or for
      photos that were taken upside-down and not corrected.
    * "n_a": no face is in the frame at all (back of head visible, face
      fully occluded by an object).
  This classification is critical for LoRA training: 'sideways' and
  'inverted' images teach the model wrong anatomy unless rotated first.
- ISSUES TAGGING (critical for training data quality):
  Be aggressive about tagging the following issues - missing them
  pollutes the training set. The 'issues' array should contain ALL
  applicable values, not just one:
    * "strong_filter": apply this whenever the subject's skin or face
      shows clear signs of beauty-filter processing - poreless or
      wax-like skin, blown-out highlights on cheeks/forehead/nose
      such that natural skin texture is lost, unnaturally smooth or
      glowing complexion, plastic-looking face. Do NOT use 'strong_filter'
      only for color filters or vintage looks - it is specifically for
      skin-smoothing/whitening artifacts that would teach the model wrong
      facial anatomy. When in doubt about whether skin is filter-smoothed
      or just well-lit: if you cannot see realistic pore structure on the
      cheeks at viewing distance, it is filter-smoothed - tag it.
    * "extreme_angle": apply this whenever the camera angle SEVERELY
      distorts body proportions in a way that would teach wrong anatomy.
      Trigger cases include: extreme worm's-eye view (camera below feet
      pointing up so legs look enormously long, head looks tiny), extreme
      bird's-eye view (camera above head pointing straight down so the
      torso is foreshortened beyond recognition), and selfie-from-below
      shots where the body parts closest to camera (knees/legs/torso)
      dwarf the face/head in the frame. A normal selfie at arm's length
      with slight angle is NOT extreme_angle. Only use this when the
      proportions in the rendered frame are clearly anatomically wrong
      relative to a standing portrait.
    * "overexposed": general scene overexposure (background blown out,
      not specifically the face/skin). Use 'strong_filter' instead when
      the issue is skin-specific.
    * Other issues from the enum follow their plain meaning.
  When unsure between 'strong_filter' and 'overexposed' for face
  skin: pick 'strong_filter' if the skin looks unnaturally smooth,
  pick 'overexposed' if highlights are blown but texture is still
  visible.
- Classify head_pose_bucket based on the main subject's head orientation:
    'frontal' = directly facing camera (yaw < ~15 degrees);
    'three_quarter_left' / 'three_quarter_right' = yaw between ~15 and ~75 degrees, named for which side of the face is more visible to camera;
    'profile_left' / 'profile_right' = pure side view (yaw ~90 degrees);
    'looking_up' / 'looking_down' = significant pitch (head clearly tilted up/down) regardless of yaw;
    'back' = head fully turned away (face not visible);
    'unknown' = head pose cannot be determined.

============================================================
CONTROLLED VOCABULARY (Phase 1)
============================================================
For the categorical aux fields (lighting_type, background_type, hair_texture,
makeup_intensity, glasses_frame_shape), use ONLY these values:

lighting_type:
  studio_softbox | studio_ringlight | studio_other |
  natural_outdoor_sun | natural_outdoor_overcast |
  natural_indoor_window | indoor_artificial | mixed | low_light

background_type:
  studio_plain | studio_textured | indoor_room | indoor_bathroom |
  outdoor_urban | outdoor_nature | outdoor_beach | outdoor_other |
  vehicle_interior | transparent_or_isolated | other

hair_texture (the natural texture of the hair, separate from style):
  straight | wavy | curly | coily | afro_textured

makeup_intensity (pick exactly ONE):
  none | minimal | natural | defined | full | dramatic

glasses_frame_shape (only if has_glasses_now is true):
  round | square | rectangular | oval | aviator | cat_eye |
  oversized | rimless | browline | geometric | other

If a value truly does not fit any of the above, use empty string "" for the
auxiliary field, but still fill the freetext field (e.g. lighting_description).

============================================================
ANTI-HEDGE RULES — STRICT
============================================================
NEVER use any of the following phrases anywhere in your output:
  - "possibly", "perhaps", "maybe", "appears to be", "looks like"
  - "kind of", "sort of", "somewhat", "approximately"
  - "X or Y" constructions like "moderate or full makeup", "minimal or no makeup",
    "blue or green eyes". Pick ONE value. If you cannot decide, pick the more
    intense / specific one.
  - "none visible", "not visible", "minimal or no", "moderate or no" as the
    ENTIRE value of any descriptive field. If a feature is absent, return an
    empty string "" instead.

Examples of WRONG vs RIGHT:
  WRONG: makeup_description = "minimal or no makeup"
  RIGHT: makeup_description = "minimal makeup with subtle lip color"
         (or "" if truly no makeup is visible)

  WRONG: eye_color = "possibly blue"
  RIGHT: eye_color = "blue"

  WRONG: piercings_description = "none visible"
  RIGHT: piercings_description = ""    (and piercing_inventory_now = [])

============================================================
STUDIO LIGHTING & SKIN-TONE GUIDANCE
============================================================
Studio softbox / ringlight illumination tends to lighten the perceived skin
tone of dark-skinned subjects by one or two perceptual steps. When labeling
skin_tone, judge by the actual pigmentation visible in the SHADOWED side of
the face (under the chin, in the neck), NOT by the brightest highlight.

Provide lighting_type accurately so downstream profile-building can correct
for studio-induced lightening.

============================================================
TATTOO & PIERCING INVENTORY
============================================================
Fill tattoo_inventory_now and piercing_inventory_now ONLY with items VISIBLE
in this image. Do not speculate about hidden ones. Use the controlled
location enum exactly. If a tattoo crosses two zones, pick the dominant one.
If you cannot place it precisely, use "other".

For each tattoo, give a short freetext description ("rose tattoo", "small
script", "linework florals on forearm"). Avoid repeating the location in the
description.

If no tattoos are visible: tattoo_inventory_now = [].
If no piercings are visible: piercing_inventory_now = [].
"""

    local_hint = (
        f"Local hints: width={local_meta.get('width')}, height={local_meta.get('height')}, "
        f"face_count_local={local_meta.get('face_count_local')}, "
        f"main_face_ratio={local_meta.get('main_face_ratio', 0):.4f}, "
        f"file_size_mb={local_meta.get('file_size_mb', 0):.2f}. "
        f"Use them only as weak hints, not as ground truth."
    )

    payload = {
        "instructions": instructions,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Audit this image for dataset curation.\n" + local_hint},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": API_IMAGE_DETAIL
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "image_audit",
                "schema": schema,
                "strict": True,
            }
        },
        "max_output_tokens": 1800,  # leicht erhoeht wegen neuer Felder
        "store": False,
        "temperature": 0.1,
    }

    data = responses_api_call(
        chosen_model,
        payload,
        phase_label=phase_label or f"audit:{os.path.basename(image_path)}",
    )
    if data.get("NSFW_BLOCKED"):
        return {"NSFW_BLOCKED": True}
    text = extract_response_text(data)
    return json.loads(text)


def normalize_audit_scores(audit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Skaliert die KI-Bewertungen deterministisch von der API-Skala 0.00-10.00
    auf die interne Anzeige-/Filter-Skala 0.0-100.0.

    Hintergrund: ChatGPT-basierte Bewertungs-APIs sind aus dem Training
    stark auf 0-10-Skalen konditioniert und produzieren auch bei explizit
    abweichender Vorgabe gerne wieder 0-10. Statt mit Heuristiken zu
    erraten, auf welcher Skala die Antwort kam, geben wir 0-10 als
    expliziten Schema-Constraint vor und multiplizieren intern fest mit 10.
    Damit sind Score-Outlier wie 321 mathematisch ausgeschlossen.

    Werte ausserhalb [0, 10] werden defensiv behandelt. Werte >10 werden als
    bereits normalisierte 0-100-Cache-Werte interpretiert und auf 0-10
    zurueckgerechnet; danach wird geclampt. Dadurch ist die Funktion auch fuer
    Cache-Hits idempotent und alte normalisierte Caches werden nicht auf 100
    hochgezogen.

    quality_total wird neu berechnet als gewichtete Summe (intern auf
    0-100), unabhaengig davon was die KI selbst dort einsetzt. So ist
    quality_total konsistent mit allen Schwellenwerten (KEEP_SCORE_MIN,
    HARD_REJECT_SCORE etc.), die historisch in der 0-100-Skala definiert
    sind.

    Gewichte fuer quality_total:
      sharpness: 4.0   (kritisch fuer LoRA-Training)
      lighting:  2.5
      composition: 2.0
      identity:  1.5
    Summe der Gewichte = 10.0 -> max. quality_total = 10 * 10.0 = 100.0
    """

    def _to_unit(v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        # Defensive: API-Skala 0-10. Cache-Hits koennen bereits auf 0-100
        # normalisiert sein; dann zurueck auf 0-10 rechnen, damit erneutes
        # Normalisieren idempotent bleibt.
        if f > 10.0:
            f = f / 10.0
        # Negative Werte sind nicht definiert; auf 0 clampen. Extreme Ausreisser
        # werden nach der optionalen Rueckrechnung weiterhin auf 10 begrenzt.
        return max(0.0, min(10.0, f))

    qs10 = _to_unit(audit.get("quality_sharpness", 0))
    ql10 = _to_unit(audit.get("quality_lighting", 0))
    qc10 = _to_unit(audit.get("quality_composition", 0))
    qi10 = _to_unit(audit.get("quality_identity_usefulness", 0))

    # Auf interne 0-100-Skala hochskalieren (1 Dezimalstelle, wie bisher)
    audit["quality_sharpness"] = round(qs10 * 10.0, 1)
    audit["quality_lighting"] = round(ql10 * 10.0, 1)
    audit["quality_composition"] = round(qc10 * 10.0, 1)
    audit["quality_identity_usefulness"] = round(qi10 * 10.0, 1)

    # Gewichtete Summe direkt auf den 0-10-Werten (einmal *10 indirekt
    # ueber Gewichte). Ergebnis liegt garantiert in [0.0, 100.0].
    weighted = (qs10 * 4.0) + (ql10 * 2.5) + (qc10 * 2.0) + (qi10 * 1.5)
    audit["quality_total"] = round(min(100.0, max(0.0, weighted)), 1)

    return audit


def should_use_review_escalation() -> bool:
    return bool(USE_REVIEW_ESCALATION and str(REVIEW_ESCALATION_MODEL or "").strip())


def should_escalate_audit(api_status: str, local_status: str, score: float) -> bool:
    if not should_use_review_escalation():
        return False
    if ESCALATE_ON_REVIEW_STATUS and (api_status == "review" or local_status == "review"):
        return True
    if ESCALATE_ON_STATUS_CONFLICT and api_status != local_status:
        return True
    return REVIEW_ESCALATION_SCORE_MIN <= score <= REVIEW_ESCALATION_SCORE_MAX


# ============================================================
# 7) FEATURE-NORMALISIERUNG / REGELN
# ============================================================

def normalize_feature_value(val: Optional[str]) -> str:
    """Normalisiert einen API-Audit-Feldwert auf einen sauberen, captionierbaren
    String. Filtert 'none visible', 'moderate or no makeup' und Hedge-Woerter.
    Gibt Leerstring zurueck, wenn der Wert wertlos ist.
    """
    v = normalize_text(val)
    if not v:
        return ""
    # Volle Saeuberung durch das Vokabular-Modul
    v = clean_audit_string(v)
    return v


# ============================================================
# Caption-Cleanup-Helpers
# ============================================================

# Bekannte Kleidungs-Substantive die in der Phrase "wearing X" als Hauptnomen
# auftauchen koennen. Liste ist defensiv gewaehlt - es schadet nicht, wenn
# ein Substantiv auf der Liste ist auch wenn die Caption in seltenen Faellen
# legitim ohne Artikel auskommen koennte (z.B. uncountable nouns).
_CLOTHING_NOUNS = {
    "top", "shirt", "blouse", "tee", "t-shirt", "tshirt", "sweater", "hoodie",
    "pullover", "cardigan", "jacket", "coat", "blazer", "vest", "tank",
    "dress", "skirt", "pants", "trousers", "jeans", "shorts", "leggings",
    "robe", "kimono", "scarf", "shawl", "poncho", "cape", "outfit",
    "uniform", "suit", "jumpsuit", "romper", "bodysuit", "swimsuit",
    "bikini", "lingerie", "bra", "underwear", "pajamas", "nightgown",
    "tunic", "kaftan", "saree", "sari",
}

# Kleine Adjektiv-Liste fuer Vokal-Erkennung beim Artikel ("a" vs "an").
# Wir entscheiden basierend auf dem ersten Wort der Phrase.
_VOWEL_SOUNDS = ("a", "e", "i", "o", "u")


def _ensure_article(phrase: str) -> str:
    """
    Fuegt vor einer Kleidungs-Phrase einen Artikel ein, falls einer fehlt.
    'dark sleeveless top' -> 'a dark sleeveless top'
    'orange jumpsuit' -> 'an orange jumpsuit'
    'a black blazer' -> 'a black blazer' (unveraendert)
    'jeans' -> 'jeans' (uncountable / plural, kein Artikel)
    'brown cardigan over a white top and blue jeans' ->
        'a brown cardigan over a white top and blue jeans'
        (erste Phrase bekommt Artikel, weitere bleiben wie sie sind)
    """
    p = phrase.strip()
    if not p:
        return p

    first_word = p.split()[0].lower().rstrip(",.")
    # Bereits Artikel oder Possessiv vorhanden?
    if first_word in {"a", "an", "the", "her", "his", "their", "my"}:
        return p

    # Plural-Endungen oder uncountable -> kein Artikel noetig.
    # WICHTIG: Wir schauen das ERSTE Substantiv an (vor dem ersten 'over',
    # 'with', 'and', Komma), nicht das letzte Wort der gesamten Phrase.
    # Sonst greifen wir nicht bei 'cardigan over a top and jeans' weil das
    # letzte Wort 'jeans' (plural) ist.
    plural_or_uncount = {"jeans", "trousers", "pants", "shorts", "leggings",
                         "tights", "stockings", "pajamas", "scrubs", "sweats",
                         "underwear", "lingerie"}
    # Splitte nur die erste Phrase (bis zum ersten Konnektor)
    first_segment = re.split(r"\s+(?:over|with|and|under|above|on|in|,)\s+", p, maxsplit=1)[0]
    first_segment_words = first_segment.split()
    if not first_segment_words:
        return p
    first_segment_last_word = first_segment_words[-1].lower().rstrip(",.")
    if first_segment_last_word in plural_or_uncount:
        return p

    # Listen-Aufzaehlung wie "blue dress and white sneakers" -> Artikel vor erster Phrase
    # Wir checken ob ein Kleidungs-Substantiv im Phrase vorkommt, sonst sicherer Skip
    has_clothing_noun = any(
        w.lower().rstrip(",.;") in _CLOTHING_NOUNS
        for w in p.split()
    )
    if not has_clothing_noun:
        return p
    article = "an" if first_word.startswith(_VOWEL_SOUNDS) else "a"
    return f"{article} {p}"


def _clean_expression(expr: str) -> str:
    """
    Stellt sicher, dass der Expression-Wert eine grammatikalisch sinnvolle
    Phrase ist. Bringt drei Faelle in saubere Form:

    1. Single-Adjektiv: 'neutral' -> 'neutral expression'
    2. Mehrfach-Adjektive: 'neutral, confident' -> 'neutral and confident expression'
    3. Augen-Beschreibung als Expression: 'eyes closed' -> '' (leer, weil
       'eyes closed' kein Gesichtsausdruck ist sondern Augen-Eigenschaft;
       wird in build_caption getrennt als 'with eyes closed' angehaengt)

    Phrasen mit Substantiv ('slight smile', 'wide-eyed playful expression')
    bleiben unveraendert.

    Behebt Bugs:
    - 'with a neutral, looking at camera' (Bug 1, Adjektiv ohne Substantiv)
    - 'with a neutral, confident, toward camera' (Bug B, Doppel-Adjektiv)
    - 'with a eyes closed with relaxed lips' (Bug A, eyes-closed in Expression)
    """
    e = expr.strip().rstrip(",.;").strip()
    if not e:
        return ""

    # Sonderfall: 'eyes closed' ist kein Expression-Adjektiv. Verwerfen,
    # damit der Caption-Builder das ueber den eigenen Pfad anhaengt.
    if re.search(r"\beyes closed\b", e, re.IGNORECASE):
        # Falls die Phrase NUR 'eyes closed' enthaelt, leer zurueckgeben.
        # Falls die Phrase 'eyes closed with relaxed lips' o.ae. enthaelt,
        # extrahiere den Teil nach dem 'with' (das ist der echte Ausdruck).
        m = re.search(r"eyes closed\s+with\s+(.+)$", e, re.IGNORECASE)
        if m:
            # Rekursiver Cleanup auf den Rest. Falls Rest mit 'a ' / 'an '
            # beginnt (z.B. 'a calm, posed expression'), den Artikel
            # strippen damit der Caption-Builder nicht 'with a a calm...'
            # produziert.
            cleaned = _clean_expression(m.group(1))
            cleaned = re.sub(r"^(an?|the)\s+", "", cleaned, flags=re.IGNORECASE)
            return cleaned
        # Reines 'eyes closed' oder 'eyes closed, relaxed lips' etc.
        return ""

    # Bekannte Substantive die signalisieren: Phrase ist schon vollstaendig
    EXPRESSION_NOUNS = {
        "expression", "look", "smile", "smirk", "frown", "grin", "pout",
        "stare", "gaze", "glance", "face", "demeanor", "mood",
    }
    words = [w.lower().rstrip(",.;") for w in e.split()]
    if any(w in EXPRESSION_NOUNS for w in words):
        return e

    # Komma-getrennte Mehrfach-Adjektive zusammenfuehren
    if "," in e:
        parts = [p.strip() for p in e.split(",") if p.strip()]
        # Filter: nur Adjektiv-aehnliche Teile (1-3 Worte ohne Substantive)
        adj_parts = []
        for p in parts:
            p_words = [w.lower() for w in p.split()]
            if any(w in EXPRESSION_NOUNS for w in p_words):
                # Wenn ein Teil schon ein Substantiv enthaelt, nimm diesen Teil
                # alleine - er ist die saubere Phrase
                return p
            if len(p_words) <= 3:
                adj_parts.append(p)
        if len(adj_parts) >= 2:
            return f"{' and '.join(adj_parts)} expression"
        elif len(adj_parts) == 1:
            return f"{adj_parts[0]} expression"
        return ""

    # Single-Adjektiv-Phrase -> 'expression' anhaengen
    return f"{e} expression"


def _clean_pose_phrase(pose: str) -> str:
    """
    Saeubert die pose_description-Phrase von haeufigen KI-Output-Bugs:
    - 'front-facing selfie seated in a car' -> 'seated in a car' (entfernt
       redundanten Compound-Modifier am Anfang wenn er mit einem inkompatiblen
       Hauptverb kollidiert)
    - 'close-up selfie with one hand' -> 'with one hand' (entfernt
       Shot-Type-Doublung)

    Heuristik: Wenn die Phrase mit einem Adjektiv-Compound startet
    ('front-facing', 'side-profile', 'close-up', 'head-tilted') und
    danach ein neuer Subjekt-Verb-Block kommt ('seated', 'sitting', 'standing',
    'lying', 'with'), dann verwirft sie den Adjektiv-Compound.
    """
    p = pose.strip()
    if not p:
        return ""

    # Compound-Modifier die typisch falsch verschmolzen werden
    redundant_starters = {
        "front-facing", "side-profile", "side-facing", "close-up", "head-tilted",
        "back-facing", "three-quarter", "frontal",
    }
    incompatible_continuations = {
        "selfie", "shot", "portrait", "view",
    }
    follow_verbs = {
        "seated", "sitting", "standing", "lying", "laying", "leaning",
        "kneeling", "crouching", "with",
    }

    words = p.split()
    if len(words) < 4:
        return p

    first = words[0].lower().rstrip(",")
    # Pattern: "<modifier> <noun> <verb>" -> nimm "<verb>..." wenn modifier+noun zur Falle wird
    if first in redundant_starters and words[1].lower().rstrip(",") in incompatible_continuations:
        # Suche nach erstem follow-verb ab Position 2
        for i in range(2, len(words)):
            if words[i].lower().rstrip(",") in follow_verbs:
                return " ".join(words[i:])
    return p


def _normalize_glasses_token(text: str) -> str:
    """
    Ersetzt 'eyeglasses' durch 'glasses' in der gesamten Caption.
    Behaelt 'sunglasses' bei (sind ein eigenes Wort).
    """
    if not text:
        return text
    # Erst sunglasses schuetzen, dann ersetzen, dann zurueckmappen
    text = text.replace("sunglasses", "\x00SUNGLASSES\x00")
    text = re.sub(r"\beyeglasses\b", "glasses", text, flags=re.IGNORECASE)
    text = text.replace("\x00SUNGLASSES\x00", "sunglasses")
    return text


def _simplify_or_phrase(text: str) -> str:
    """
    Reduziert Phrasen mit KI-Unentschiedenheit ('X or Y Z' oder 'X/Y Z') auf
    das eindeutige Substantiv ('Z'). Wenn die KI sich nicht zwischen zwei
    Beschreibungs-Optionen entscheiden kann ('small hoop or stud nose
    piercing', 'small floral/script tattoo'), wird die uneindeutige
    Adjektiv-Auswahl entfernt.

    Beispiele:
    - 'small hoop or stud earring' -> 'small earring'
    - 'small hoop or stud nose piercing' -> 'small nose piercing'
    - 'small floral/script tattoo' -> 'small tattoo'
    - 'two or more' -> 'two or more' (kein Substantiv-Trigger)

    Behaelt feste Phrasen die genuin 'or' enthalten ('two or more') unangetastet,
    weil dort kein Adjektiv-Auswahl-Pattern vorliegt.
    """
    if not text:
        return text

    def replace(m: re.Match) -> str:
        prefix = m.group(1) or ""
        adj1 = m.group(2)
        adj2 = m.group(3)
        noun_part = m.group(4)
        if adj1.lower() == adj2.lower():
            return f"{prefix}{adj1} {noun_part}"
        return f"{prefix}{noun_part}".strip()

    # Pattern 1: 'X or Y Z' (Whitespace-Trennung um 'or')
    pattern_or = re.compile(
        r"\b((?:small |large |big |tiny |medium |short |long )?)"
        r"([a-z]+) or ([a-z]+) "
        r"((?:[a-z]+(?:\s+[a-z]+){0,2}))",
        re.IGNORECASE,
    )
    text = pattern_or.sub(replace, text)

    # Pattern 2: 'X/Y Z' (Slash-Trennung ohne Whitespace)
    pattern_slash = re.compile(
        r"\b((?:small |large |big |tiny |medium |short |long )?)"
        r"([a-z]+)/([a-z]+) "
        r"((?:[a-z]+(?:\s+[a-z]+){0,2}))",
        re.IGNORECASE,
    )
    text = pattern_slash.sub(replace, text)

    return text


def _dedupe_phrase_list(phrases: List[str]) -> List[str]:
    """
    Entfernt Doppleinträge in einer Liste von kurzen Beschreibungs-Phrasen.
    Behandelt 'small hoop earring' und 'small hoop' als gleichwertig
    (Substring-Match), behaelt aber den laengeren/spezifischeren Eintrag.

    Behebt den Earring-Doublette-Bug: 'small hoop earring, small hoop'.
    """
    if not phrases:
        return phrases
    cleaned: List[str] = []
    seen_normalized: List[str] = []
    for p in phrases:
        p_clean = p.strip().lower().rstrip(",.;")
        if not p_clean:
            continue
        # Ist dieser Eintrag in einem bereits aufgenommenen enthalten?
        if any(p_clean in s or s in p_clean for s in seen_normalized):
            # Wenn der neue Eintrag laenger ist als ein bereits aufgenommener,
            # ersetze ihn statt zu skippen
            replaced = False
            for i, s in enumerate(seen_normalized):
                if s in p_clean and len(p_clean) > len(s):
                    cleaned[i] = p
                    seen_normalized[i] = p_clean
                    replaced = True
                    break
            if not replaced:
                continue
        else:
            cleaned.append(p)
            seen_normalized.append(p_clean)
    return cleaned


def compute_global_rules(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    rules: Dict[str, Any] = {}

    def mode_info(field: str, min_fraction_for_stable: float = 0.80) -> Dict[str, Any]:
        values = [normalize_feature_value(i.get(field)) for i in items]
        values = [v for v in values if v and v not in {"none", "unknown", "n/a", "none visible"}]
        if not values:
            return {"mode": "", "stable": False, "variable": True, "override_candidates": [], "counts": {}}

        counts = Counter(values)
        mode_val, mode_count = counts.most_common(1)[0]
        total = max(1, len(values))
        frac = mode_count / total

        stable = frac >= min_fraction_for_stable
        # If not stable, show the top 5 variants so the user can see why it's fragmented
        override_candidates = [v for v, c in counts.most_common(5)] if not stable else []

        return {
            "mode": mode_val,
            "stable": stable,
            "variable": not stable,
            "override_candidates": override_candidates,
            "counts": dict(counts.most_common(10)),
        }

    # Wir berechnen globale Regeln NUR für Features, bei denen es die "when_variable" Logik gibt!
    # Brillen, Tattoos etc. sind fest durch CAPTION_POLICY geregelt und brauchen keine Mehrheitsentscheidung.
    rules["hair_description"] = mode_info("hair_description", 0.85)
    rules["beard_description"] = mode_info("beard_description", 0.85)

    return rules

def get_caption_rule_overview(global_rules: Dict[str, Any]) -> Dict[str, Any]:
    fixed = {}
    override = {}
    for key, info in global_rules.items():
        if not isinstance(info, dict):
            continue
        if info.get("stable"):
            fixed[key] = {
                "mode": info.get("mode", ""),
                "counts": info.get("counts", {}),
            }
        else:
            override[key] = {
                "mode": info.get("mode", ""),
                "counts": info.get("counts", {}),
                "candidates": info.get("override_candidates", []),
            }
    return {"fixed": fixed, "override": override}


# ============================================================
# 7b) SUBJECT PROFILE BUILDER (Phase 2)
# ============================================================

def subject_profile_cache_path(trigger_word: Optional[str] = None) -> str:
    safe = slugify_filename((trigger_word or TRIGGER_WORD or "subject").strip())
    return os.path.join(SUBJECT_PROFILE_CACHE_DIR, f"{safe}.profile.json")


def output_subject_profile_path() -> str:
    return os.path.join(OUTPUT_ROOT, "_subject_profile.json")


def output_profile_override_path() -> str:
    return os.path.join(OUTPUT_ROOT, "_profile_override.json")


def output_caption_stage_path() -> str:
    return os.path.join(OUTPUT_ROOT, CAPTION_STAGE_FILENAME)


def profile_image_id(row: Dict[str, Any]) -> str:
    """Stabiler Bild-Key fuer Subject-Profile und per-image Tokens."""
    h = str(row.get("file_hash") or "").strip()
    if h:
        return h
    src = str(row.get("original_path") or row.get("original_filename") or "")
    return hashlib.sha1(src.encode("utf-8", errors="ignore")).hexdigest()


def _profile_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "ja", "y"}


def profile_input_hash(rows: List[Dict[str, Any]]) -> str:
    """Hash ueber die relevanten Audit-Felder, nicht ueber Captions/Outputnamen."""
    relevant: List[Dict[str, Any]] = []
    for row in rows:
        relevant.append({
            "image_id": profile_image_id(row),
            "file_hash": row.get("file_hash", ""),
            "original_filename": row.get("original_filename", ""),
            "base_status": row.get("base_status", ""),
            "shot_type": row.get("shot_type", ""),
            "quality_total": row.get("quality_total", ""),
            "gender_class": row.get("gender_class", ""),
            "skin_tone": row.get("skin_tone", ""),
            "eye_color": row.get("eye_color", ""),
            "body_build": row.get("body_build", ""),
            "hair_description": row.get("hair_description", ""),
            "hair_texture": row.get("hair_texture", ""),
            "beard_description": row.get("beard_description", ""),
            "glasses_description": row.get("glasses_description", ""),
            "has_glasses_now": row.get("has_glasses_now", False),
            "glasses_frame_shape": row.get("glasses_frame_shape", ""),
            "makeup_description": row.get("makeup_description", ""),
            "makeup_intensity": row.get("makeup_intensity", ""),
            "tattoos_visible": row.get("tattoos_visible", False),
            "tattoos_description": row.get("tattoos_description", ""),
            "tattoo_inventory_now": row.get("tattoo_inventory_now", []),
            "piercings_description": row.get("piercings_description", ""),
            "piercing_inventory_now": row.get("piercing_inventory_now", []),
            "lighting_description": row.get("lighting_description", ""),
            "lighting_type": row.get("lighting_type", ""),
            "background_description": row.get("background_description", ""),
            "background_type": row.get("background_type", ""),
            "head_pose_bucket": row.get("head_pose_bucket", ""),
        })
    relevant.sort(key=lambda x: (str(x.get("image_id", "")), str(x.get("original_filename", ""))))
    payload = {
        "schema": PROFILE_CACHE_SCHEMA_VERSION,
        "trigger": SAFE_TRIGGER,
        "items": relevant,
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _quality_tier(row: Dict[str, Any]) -> str:
    try:
        q = float(row.get("quality_total", 0))
    except Exception:
        q = 0.0
    if q >= 75:
        return "high"
    if q >= 55:
        return "mid"
    return "low"


def stratified_sample_for_profile(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministisches Sample fuer grosse Datasets.
    Strata: lighting_type × shot_type × quality-tier.
    """
    if len(rows) <= int(PROFILE_SAMPLE_THRESHOLD):
        return list(rows)

    target = max(1, int(PROFILE_SAMPLE_SIZE))
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            normalize_text(row.get("lighting_type")) or "unknown_lighting",
            normalize_text(row.get("shot_type")) or "unknown_shot",
            _quality_tier(row),
        )
        groups[key].append(row)

    for key in groups:
        groups[key].sort(key=lambda r: profile_image_id(r))

    selected: List[Dict[str, Any]] = []
    for key in sorted(groups.keys()):
        if len(selected) >= target:
            break
        if groups[key]:
            selected.append(groups[key].pop(0))

    while len(selected) < target:
        progressed = False
        for key in sorted(groups.keys()):
            if len(selected) >= target:
                break
            if groups[key]:
                selected.append(groups[key].pop(0))
                progressed = True
        if not progressed:
            break

    return selected


def _confidence_field_schema() -> Dict[str, Any]:
    """Schema fuer einen Confidence-Eintrag (per Stable-Trait).

    Bug 3 fix (additiv, nicht-breaking): zusaetzlich zu 'level' werden
    'reasoning' und 'outliers' (image_ids) optional erfasst, damit die UI
    spaeter Outlier-Listen anzeigen kann ('Welche Bilder weichen vom Mode
    ab?'). Alte Profile mit string-only Confidence werden in der UI
    transparent als {level: <string>} interpretiert.
    """
    return {
        "type": "object",
        "properties": {
            "level": {
                "type": "string",
                "description": "Confidence label, e.g. high | medium | low | fallback.",
            },
            "reasoning": {
                "type": "string",
                "description": "One short sentence explaining the verdict. May be empty.",
            },
            "outliers": {
                "type": "array",
                "description": "image_ids that disagreed with the chosen value. Empty array if none.",
                "items": {"type": "string"},
            },
        },
        "required": ["level", "reasoning", "outliers"],
        "additionalProperties": False,
    }


def subject_profile_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "subject_id": {"type": "string"},
            "stable_identity": {
                "type": "object",
                "properties": {
                    "gender": {"type": "string"},
                    "skin_tone": {"type": "string"},
                    "eye_color": {"type": "string"},
                    "hair_texture": {"type": "string"},
                    "body_build": {"type": "string"},
                },
                "required": ["gender", "skin_tone", "eye_color", "hair_texture", "body_build"],
                "additionalProperties": False,
            },
            "confidence": {
                "type": "object",
                "description": (
                    "Per-field confidence info. Each entry is an object with "
                    "the canonical level plus an optional reasoning string and "
                    "outlier image_ids. Backward compatible: legacy profiles "
                    "where the value is just a string are still accepted by "
                    "the UI, which falls back to {level: <string>}."
                ),
                "properties": {
                    "gender":       _confidence_field_schema(),
                    "skin_tone":    _confidence_field_schema(),
                    "eye_color":    _confidence_field_schema(),
                    "hair_texture": _confidence_field_schema(),
                    "body_build":   _confidence_field_schema(),
                },
                "required": ["gender", "skin_tone", "eye_color", "hair_texture", "body_build"],
                "additionalProperties": False,
            },
            "identity_markers": {
                "type": "object",
                "properties": {
                    "glasses": {
                        "type": "object",
                        "properties": {
                            "wears_regularly": {"type": "boolean"},
                            "canonical_description": {"type": "string"},
                            "frequency": {"type": "string"},
                        },
                        "required": ["wears_regularly", "canonical_description", "frequency"],
                        "additionalProperties": False,
                    },
                    "tattoo_inventory": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "canonical_description": {"type": "string"},
                                "frequency": {"type": "string"},
                            },
                            "required": ["location", "canonical_description", "frequency"],
                            "additionalProperties": False,
                        },
                    },
                    "piercing_baseline": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "canonical_description": {"type": "string"},
                                "frequency": {"type": "string"},
                            },
                            "required": ["location", "canonical_description", "frequency"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["glasses", "tattoo_inventory", "piercing_baseline"],
                "additionalProperties": False,
            },
            "normalizer_notes": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["subject_id", "stable_identity", "confidence", "identity_markers", "normalizer_notes"],
        "additionalProperties": False,
    }


def _profile_sample_payload(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload_rows: List[Dict[str, Any]] = []
    for row in rows:
        payload_rows.append({
            "image_id": profile_image_id(row),
            "filename": row.get("original_filename", ""),
            "quality_total": row.get("quality_total", 0),
            "shot_type": row.get("shot_type", ""),
            "head_pose_bucket": row.get("head_pose_bucket", ""),
            "lighting_type": row.get("lighting_type", ""),
            "background_type": row.get("background_type", ""),
            "raw": {
                "gender_class": row.get("gender_class", ""),
                "skin_tone": row.get("skin_tone", ""),
                "eye_color": row.get("eye_color", ""),
                "body_build": row.get("body_build", ""),
                "hair_description": row.get("hair_description", ""),
                "hair_texture": row.get("hair_texture", ""),
                "glasses_description": row.get("glasses_description", ""),
                "has_glasses_now": row.get("has_glasses_now", False),
                "glasses_frame_shape": row.get("glasses_frame_shape", ""),
                "makeup_description": row.get("makeup_description", ""),
                "makeup_intensity": row.get("makeup_intensity", ""),
                "tattoo_inventory_now": row.get("tattoo_inventory_now", []),
                "piercing_inventory_now": row.get("piercing_inventory_now", []),
                "lighting_description": row.get("lighting_description", ""),
                "background_description": row.get("background_description", ""),
            },
        })
    return payload_rows


def call_subject_profile_normalizer(rows: List[Dict[str, Any]], input_hash: str, total_count: int) -> Dict[str, Any]:
    instructions = """
You consolidate raw per-image audits into one Subject Identity Profile for a person LoRA dataset.
All input images are intended to show the same subject. Some outliers may exist.

Important:
- Stable identity traits must be canonical and consistent across captions.
- Use single, clean tokens or short phrases. No hedge words, no 'or'-phrases, no 'none visible'.
- For skin tone, account for studio-lighting bias: studio or ring-light images can make darker skin read lighter.
- For eye color, treat mirror selfies, filters, and extreme lighting as possible outliers.
- Body build is unreliable on headshots. If less than ~30% of input images are medium/full_body,
  set body_build to "" (empty string) and confidence.body_build.level = "low" with reasoning
  "few full-body observations". Vision models tend to over-label women as 'slim' on headshots
  due to RLHF politeness bias - resist this tendency.
- Glasses are regular only if visible in at least about 60% of sampled usable images.
- Piercing baseline includes locations visible in at least about 40% of sampled usable images.
- Tattoo inventory is the union of visible tattoos, grouped by location. Mention only visible markers later.
- Force-only-when-visible policy: markers like glasses, tattoos and piercings must not be captioned in images where they are not visible.

Confidence object format (REQUIRED for each stable trait):
  {
    "level":     "high" | "medium" | "low" | "fallback",
    "reasoning": "<one short sentence; may be empty>",
    "outliers":  ["<image_id>", ...]   // image_ids that disagreed with the chosen value; [] if none
  }

Return JSON only.
"""

    user_payload = {
        "trigger_word": TRIGGER_WORD,
        "safe_trigger": SAFE_TRIGGER,
        "total_usable_images": total_count,
        "sampled_images": len(rows),
        "input_hash": input_hash,
        "vocab_hints": {
            "skin_tone": SKIN_TONE_VOCAB,
            "eye_color": EYE_COLOR_VOCAB,
            "hair_texture": ["straight", "wavy", "curly", "coily", "afro_textured"],
            "body_build": BODY_BUILD_VOCAB,
            "makeup_intensity": MAKEUP_INTENSITY_VOCAB,
            "hair_form": HAIR_FORM_VOCAB,
            "hair_color": HAIR_COLOR_VOCAB,
            "tattoo_locations": TATTOO_LOCATION_ENUM,
            "piercing_locations": PIERCING_LOCATION_ENUM,
        },
        "images": _profile_sample_payload(rows),
    }

    payload = {
        "instructions": instructions,
        "input": [{
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": json.dumps(user_payload, ensure_ascii=False),
            }],
        }],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "subject_profile",
                "schema": subject_profile_schema(),
                "strict": True,
            }
        },
        "max_output_tokens": 2400,
        "store": False,
        "temperature": 0.1,
    }

    data = responses_api_call(PROFILE_NORMALIZER_MODEL, payload)
    parsed = json.loads(extract_response_text(data))
    parsed["profile_schema_version"] = PROFILE_CACHE_SCHEMA_VERSION
    parsed["input_hash"] = input_hash
    parsed["normalizer_model"] = PROFILE_NORMALIZER_MODEL
    parsed["sample_size"] = len(rows)
    parsed["total_usable_images"] = total_count
    parsed["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return parsed


def _mode_clean(rows: List[Dict[str, Any]], field: str) -> str:
    vals = [compact_trait(r.get(field)) for r in rows]
    vals = [v for v in vals if v]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def fallback_subject_profile(rows: List[Dict[str, Any]], input_hash: str, reason: str = "") -> Dict[str, Any]:
    """Deterministischer Fallback, falls der Normalizer-Call fehlschlaegt."""
    n = max(1, len(rows))

    glasses_rows = [r for r in rows if _profile_bool(r.get("has_glasses_now")) or compact_trait(r.get("glasses_description"))]
    glasses_descs = [compact_trait(r.get("glasses_description")) for r in glasses_rows]
    glasses_descs = [d for d in glasses_descs if d]
    glasses_mode = Counter(glasses_descs).most_common(1)[0][0] if glasses_descs else ""

    tattoos_by_loc: Dict[str, List[str]] = defaultdict(list)
    piercings_by_loc: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        for t in row.get("tattoo_inventory_now") or []:
            loc = normalize_text(t.get("location")) or "other"
            desc = compact_trait(t.get("description")) or "tattoo"
            tattoos_by_loc[loc].append(desc)
        for p in row.get("piercing_inventory_now") or []:
            loc = normalize_text(p.get("location")) or "other"
            desc = compact_trait(p.get("description")) or "piercing"
            piercings_by_loc[loc].append(desc)

    def inv_items(grouped: Dict[str, List[str]], min_fraction: float = 0.0) -> List[Dict[str, str]]:
        out = []
        for loc, descs in sorted(grouped.items()):
            if (len(descs) / n) < min_fraction:
                continue
            c = Counter(descs)
            desc = max(c.keys(), key=lambda s: (c[s], len(s)))
            out.append({
                "location": loc,
                "canonical_description": desc,
                "frequency": f"{len(descs)}/{n}",
            })
        return out

    # Body-Build-Demotion: Wenn der Anteil aussagekraeftiger Shots
    # (medium / full_body) zu klein ist, ist body_build unzuverlaessig.
    body_eligible = sum(1 for r in rows if normalize_text(r.get("shot_type")) in {"medium", "full_body"})
    body_eligible_fraction = body_eligible / n
    body_build_value = _mode_clean(rows, "body_build")
    body_build_reason = ""
    if body_eligible_fraction < 0.30:
        body_build_value = ""  # Headshot-Dominanz: lieber leer als raten
        body_build_reason = (
            f"Only {body_eligible}/{n} medium-or-full-body shots; body_build "
            f"unreliable on headshots (fallback)."
        )

    profile = {
        "subject_id": SAFE_TRIGGER,
        "profile_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "input_hash": input_hash,
        "normalizer_model": "fallback_local",
        "sample_size": len(rows),
        "total_usable_images": len(rows),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stable_identity": {
            "gender": _mode_clean(rows, "gender_class") or "person",
            "skin_tone": _mode_clean(rows, "skin_tone"),
            "eye_color": _mode_clean(rows, "eye_color"),
            "hair_texture": _mode_clean(rows, "hair_texture"),
            "body_build": body_build_value,
        },
        "confidence": {
            "gender":       {"level": "fallback", "reasoning": "", "outliers": []},
            "skin_tone":    {"level": "fallback", "reasoning": "", "outliers": []},
            "eye_color":    {"level": "fallback", "reasoning": "", "outliers": []},
            "hair_texture": {"level": "fallback", "reasoning": "", "outliers": []},
            "body_build":   {
                "level": "low" if body_eligible_fraction < 0.30 else "fallback",
                "reasoning": body_build_reason,
                "outliers": [],
            },
        },
        "identity_markers": {
            "glasses": {
                "wears_regularly": (len(glasses_rows) / n) >= 0.60,
                "canonical_description": glasses_mode,
                "frequency": f"{len(glasses_rows)}/{n}",
            },
            "tattoo_inventory": inv_items(tattoos_by_loc, min_fraction=0.0),
            "piercing_baseline": inv_items(piercings_by_loc, min_fraction=0.40),
        },
        "normalizer_notes": [f"Fallback profile used. {reason}".strip()],
    }
    return profile


def _contains_any(text: str, needles: List[str]) -> bool:
    t = text.lower()
    return any(n in t for n in needles)


def canonical_hair_color(row: Dict[str, Any]) -> str:
    text = normalize_text(" ".join([
        str(row.get("hair_description", "")),
        str(row.get("hair_texture", "")),
    ]))
    if not text:
        return ""
    if _contains_any(text, ["platinum", "white-blonde", "white blonde", "very light blonde", "very light ash", "silver blonde"]):
        return "platinum"
    if _contains_any(text, ["burgundy", "wine-red", "wine red", "deep red", "dark red"]):
        return "burgundy"
    if _contains_any(text, ["auburn", "copper"]):
        return "auburn"
    if _contains_any(text, ["red hair", "red-haired", "reddish"]):
        return "red"
    if _contains_any(text, ["black hair", "jet black", "raven", "black braided", "black braids", "dark black"]):
        return "black"
    if _contains_any(text, ["dark brown", "deep brown", "brunette"]):
        return "dark_brown"
    if _contains_any(text, ["light brown", "dirty blonde"]):
        return "light_brown"
    if _contains_any(text, ["blonde", "blond", "ash-blonde", "ash blonde"]):
        return "blonde"
    if _contains_any(text, ["brown hair", "brown wavy", "brown straight"]):
        return "brown"
    if _contains_any(text, ["gray", "grey"]):
        return "gray"
    if "white" in text:
        return "white"
    return ""


def canonical_hair_form(row: Dict[str, Any]) -> str:
    text = normalize_text(" ".join([
        str(row.get("hair_description", "")),
        str(row.get("hair_texture", "")),
    ]))
    if not text:
        return ""
    if _contains_any(text, ["knotless braid", "knotless braids"]):
        return "knotless_braids"
    if _contains_any(text, ["box braid", "box braids", "individual braid", "individual braids", "small braids", "rope-like braid"]):
        return "box_braids"
    if "cornrow" in text:
        return "cornrows"
    if _contains_any(text, ["two braids", "pigtail braids", "double braids"]):
        return "two_braids"
    if _contains_any(text, ["single braid", "one braid"]):
        return "single_braid"
    if "pigtail" in text:
        return "pigtails"
    if "ponytail" in text:
        return "ponytail"
    if _contains_any(text, ["bun", "top knot", "chignon"]):
        return "bun"
    if _contains_any(text, ["updo", "up-do"]):
        return "updo"
    if _contains_any(text, ["half up", "half-up"]):
        return "half_up"
    if _contains_any(text, ["pulled back", "tied back", "slicked back"]):
        return "pulled_back"
    if _contains_any(text, ["short hair", "pixie", "short cut"]):
        return "short_cut"
    if _contains_any(text, ["afro", "rounded shape", "voluminous rounded", "afro-textured"]):
        return "afro_natural"
    if _contains_any(text, ["coily"]):
        return "loose_coily"
    if _contains_any(text, ["curly", "ringlet"]):
        return "loose_curly"
    if _contains_any(text, ["wavy", "wave"]):
        return "loose_wavy"
    if _contains_any(text, ["straight"]):
        return "loose_straight"
    return ""


def canonical_makeup_intensity(row: Dict[str, Any]) -> str:
    explicit = normalize_text(row.get("makeup_intensity"))
    if explicit in MAKEUP_INTENSITY_VOCAB:
        return explicit
    text = normalize_text(row.get("makeup_description"))
    if not text:
        return ""
    if _contains_any(text, ["dramatic", "bold", "heavy glam"]):
        return "dramatic"
    if _contains_any(text, ["full makeup", "heavy makeup", "glam makeup"]):
        return "full"
    if _contains_any(text, ["defined", "eyeliner", "eyeshadow", "contour", "bold eye"]):
        return "defined"
    if _contains_any(text, ["natural", "soft makeup"]):
        return "natural"
    if _contains_any(text, ["minimal", "light makeup", "subtle"]):
        return "minimal"
    if _contains_any(text, ["no makeup", "none"]):
        return "none"
    return ""


def _phrase_from_token(token: str) -> str:
    return (token or "").replace("_", " ").strip()


def profile_hair_caption(hair_color: str, hair_form: str) -> str:
    """Build a grammatical hair phrase from normalized profile tokens.

    Phase 2 originally produced artifacts such as "blonde pulled back" because
    style tokens like pulled_back were concatenated without the word "hair".
    This helper keeps compact LoRA-friendly tokens, but always returns a phrase
    that can safely follow "with" in the first caption sentence.
    """
    color = _phrase_from_token(hair_color)
    form_token = normalize_text(hair_form)
    form = _phrase_from_token(hair_form)

    if not color and not form_token:
        return ""

    def color_prefix() -> str:
        return (color + " ") if color else ""

    if form_token.startswith("loose_"):
        texture = _phrase_from_token(form_token.replace("loose_", ""))
        return " ".join([p for p in [color, texture, "hair"] if p]).strip()

    phrase_map = {
        "pulled_back": f"{color_prefix()}hair pulled back",
        "half_up": f"{color_prefix()}hair in a half-up style",
        "ponytail": f"{color_prefix()}hair in a ponytail",
        "pigtails": f"{color_prefix()}hair in pigtails",
        "bun": f"{color_prefix()}hair in a bun",
        "updo": f"{color_prefix()}hair in an updo",
        "two_braids": f"{color_prefix()}hair in two braids",
        "single_braid": f"{color_prefix()}hair in a single braid",
        "box_braids": f"{color_prefix()}box braids",
        "knotless_braids": f"{color_prefix()}knotless braids",
        "cornrows": f"{color_prefix()}cornrows",
        "short_cut": f"short {color_prefix()}hair".strip(),
        "afro_natural": f"{color_prefix()}natural afro-textured hair",
    }
    if form_token in phrase_map:
        return re.sub(r"\s+", " ", phrase_map[form_token]).strip()

    if form:
        # Unknown profile token: keep it, but make the phrase grammatical.
        return " ".join([p for p in [color, form, "hair"] if p]).strip()

    return " ".join([p for p in [color, "hair"] if p]).strip()


def _inventory_map(profile: Dict[str, Any], marker_key: str) -> Dict[str, str]:
    markers = (profile or {}).get("identity_markers", {})
    if marker_key == "tattoos":
        items = markers.get("tattoo_inventory", [])
    elif marker_key == "piercings":
        items = markers.get("piercing_baseline", [])
    else:
        items = []
    out: Dict[str, str] = {}
    for item in items or []:
        loc = normalize_text(item.get("location"))
        desc = compact_trait(item.get("canonical_description"))
        if loc and desc:
            out[loc] = desc
    return out


def per_image_profile_traits(row: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    tattoos_visible = []
    for t in row.get("tattoo_inventory_now") or []:
        loc = normalize_text(t.get("location"))
        if loc:
            tattoos_visible.append(loc)

    piercings_visible = []
    for p in row.get("piercing_inventory_now") or []:
        loc = normalize_text(p.get("location"))
        if loc:
            piercings_visible.append(loc)

    return {
        "hair_color_base": canonical_hair_color(row),
        "hair_form": canonical_hair_form(row),
        "makeup_intensity": canonical_makeup_intensity(row),
        "glasses_visible": _profile_bool(row.get("has_glasses_now")) or bool(compact_trait(row.get("glasses_description"))),
        "tattoo_locations_visible": sorted(set(tattoos_visible)),
        "piercing_locations_visible": sorted(set(piercings_visible)),
    }


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_profile_override() -> Dict[str, Any]:
    path = output_profile_override_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        safe_print(f"   ⚠️ Could not read profile override file: {e}")
        return {}


def save_subject_profile(profile: Dict[str, Any]) -> None:
    os.makedirs(SUBJECT_PROFILE_CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    with open(subject_profile_cache_path(), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    with open(output_subject_profile_path(), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    example_path = os.path.join(OUTPUT_ROOT, "_profile_override.example.json")
    if not os.path.exists(example_path):
        example = {
            "stable_identity": {
                "skin_tone": "",
                "eye_color": "",
                "hair_texture": "",
                "body_build": "",
            },
            "identity_markers": {
                "glasses": {
                    "wears_regularly": False,
                    "canonical_description": "",
                }
            },
            "per_image_traits": {}
        }
        with open(example_path, "w", encoding="utf-8") as f:
            json.dump(example, f, ensure_ascii=False, indent=2)


def load_subject_profile_cache(input_hash: str) -> Optional[Dict[str, Any]]:
    path = subject_profile_cache_path()
    if not ENABLE_CACHE or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        if (
            profile.get("profile_schema_version") == PROFILE_CACHE_SCHEMA_VERSION
            and profile.get("input_hash") == input_hash
        ):
            return profile
    except Exception:
        return None
    return None


def build_subject_profile(profile_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Baut/lädt das zentrale Subject-Profile und erzeugt per-image Tokens.
    Reject- und Review-Bilder sollen upstream ausgeschlossen werden.
    """
    rows = [r for r in profile_rows if r.get("base_status") == "keep" and r.get("arcface_flag") != "hard"]
    if not rows:
        safe_print("   ⚠️ Subject profile skipped: no usable keep rows.")
        return {}

    input_hash = profile_input_hash(rows)
    cached = load_subject_profile_cache(input_hash)
    if cached:
        profile = cached
        safe_print(
            f"   🧬 Subject profile cache used: {SAFE_TRIGGER} "
            f"({len(rows)} usable images)"
        )
    else:
        sample = stratified_sample_for_profile(rows)
        safe_print(
            f"   🧬 Building subject profile with {PROFILE_NORMALIZER_MODEL}: "
            f"{len(sample)}/{len(rows)} sampled usable images"
        )
        try:
            profile = call_subject_profile_normalizer(sample, input_hash, total_count=len(rows))
        except Exception as e:
            safe_print(f"   ⚠️ Subject profile normalizer failed; using local fallback: {e}")
            profile = fallback_subject_profile(sample if sample else rows, input_hash, reason=str(e))
            profile["total_usable_images"] = len(rows)

    # Body-Build-Bias-Wachposten: auch wenn der Normalizer einen Wert liefert,
    # ueberpruefen wir lokal die Datenbasis. Bei <30% medium/full_body-Shots
    # ist body_build unzuverlaessig (Vision-Modelle defaulten auf 'slim'/'average'
    # bei Headshots wegen RLHF-Bias).
    body_eligible = sum(1 for r in rows if normalize_text(r.get("shot_type")) in {"medium", "full_body"})
    body_eligible_fraction = body_eligible / max(1, len(rows))
    if body_eligible_fraction < 0.30:
        stable = profile.setdefault("stable_identity", {})
        prev_body = stable.get("body_build", "")
        if prev_body:
            stable["body_build"] = ""
            conf = profile.setdefault("confidence", {})
            existing = conf.get("body_build", {})
            if not isinstance(existing, dict):
                existing = {"level": str(existing or ""), "reasoning": "", "outliers": []}
            existing["level"] = "low"
            existing["reasoning"] = (
                f"Demoted: only {body_eligible}/{len(rows)} medium-or-full-body shots; "
                f"normalizer suggested '{prev_body}' but headshots are unreliable for body build."
            )
            existing.setdefault("outliers", [])
            conf["body_build"] = existing
            profile.setdefault("normalizer_notes", []).append(
                f"Body build demoted to empty (was '{prev_body}'): only "
                f"{body_eligible}/{len(rows)} medium/full-body images. Override in UI if known."
            )

    per_image: Dict[str, Any] = {}
    for row in rows:
        image_id = profile_image_id(row)
        row["profile_image_id"] = image_id
        per_image[image_id] = per_image_profile_traits(row, profile)

    profile["per_image_traits"] = per_image
    profile["input_hash"] = input_hash
    profile["profile_schema_version"] = PROFILE_CACHE_SCHEMA_VERSION
    profile["subject_id"] = profile.get("subject_id") or SAFE_TRIGGER
    profile["force_only_when_visible"] = True

    override = load_profile_override()
    if override:
        profile = deep_merge_dict(profile, override)
        profile.setdefault("normalizer_notes", []).append("Local _profile_override.json was applied.")

    save_subject_profile(profile)

    stable = profile.get("stable_identity", {})
    safe_print(
        "   🧬 Subject profile ready: "
        f"skin={stable.get('skin_tone','') or '-'} | "
        f"eyes={stable.get('eye_color','') or '-'} | "
        f"hair_texture={stable.get('hair_texture','') or '-'} | "
        f"body={stable.get('body_build','') or '-'}"
    )
    return profile


def subject_profile_report_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    if not profile:
        return {}
    return {
        "subject_id": profile.get("subject_id", ""),
        "profile_schema_version": profile.get("profile_schema_version", ""),
        "normalizer_model": profile.get("normalizer_model", ""),
        "sample_size": profile.get("sample_size", 0),
        "total_usable_images": profile.get("total_usable_images", 0),
        "force_only_when_visible": profile.get("force_only_when_visible", True),
        "stable_identity": profile.get("stable_identity", {}),
        "confidence": profile.get("confidence", {}),
        "identity_markers": profile.get("identity_markers", {}),
        "normalizer_notes": profile.get("normalizer_notes", []),
    }




# ============================================================
# 7c) SUBJECT PROFILE UI-GATE / CAPTION STAGE (Phase 3)
# ============================================================

def make_json_safe(value: Any) -> Any:
    """Konvertiert Row-/Report-Daten so, dass sie in _caption_stage.json
    gespeichert werden koennen. Grosse Embeddings werden bewusst entfernt.
    """
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in {"clip_embedding", "arcface_embedding"}:
                continue
            out[str(k)] = make_json_safe(v)
        return out
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    return value


def save_caption_stage(
    *,
    all_rows: List[Dict[str, Any]],
    selected_sorted: List[Dict[str, Any]],
    review_items: List[Dict[str, Any]],
    unselected_keep: List[Dict[str, Any]],
    reject_items: List[Dict[str, Any]],
    global_rules: Dict[str, Any],
    subject_profile: Dict[str, Any],
    identity_summary: Dict[str, Any],
    warnings: List[str],
    valid_candidate_count: int,
) -> None:
    """Speichert den Zustand nach Audit + Profil-Build, aber vor Caption-Export.

    Phase 3 nutzt diese Datei, damit der User das Profil in der UI bearbeiten
    kann und danach nur der Caption-/Bildexport laeuft, ohne neues Audit.
    """
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    stage = {
        "stage_schema_version": "v1",
        "trigger_word": TRIGGER_WORD,
        "safe_trigger": SAFE_TRIGGER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "all_rows": all_rows,
        "selected_sorted": selected_sorted,
        "review_items": review_items,
        "unselected_keep": unselected_keep,
        "reject_items": reject_items,
        "global_rules": global_rules,
        "subject_profile": subject_profile,
        "identity_summary": identity_summary,
        "warnings": warnings,
        "valid_candidate_count": valid_candidate_count,
    }
    with open(output_caption_stage_path(), "w", encoding="utf-8") as f:
        json.dump(make_json_safe(stage), f, ensure_ascii=False, indent=2)


def load_caption_stage() -> Dict[str, Any]:
    path = output_caption_stage_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Caption stage file not found: {path}. Run profile_then_caption first."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Invalid caption stage file: root is not an object.")
    return data


def load_confirmed_subject_profile(stage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Laedt das vom User bestaetigte/bearbeitete Profil aus dem Output-Ordner.
    Falls es fehlt, wird das im Stage-File gespeicherte Profil verwendet.
    """
    path = output_subject_profile_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data["force_only_when_visible"] = True
                return data
        except Exception as e:
            safe_print(f"   ⚠️ Could not read confirmed subject profile: {e}")
    profile = (stage or {}).get("subject_profile", {}) if isinstance(stage, dict) else {}
    return profile if isinstance(profile, dict) else {}


def clean_caption_output_dirs() -> None:
    """Entfernt alte Bild-/Caption-Exports vor dem Continue-Export.
    Cache, Profile und Audit-Zwischenstaende bleiben erhalten.
    """
    for folder in [TRAIN_READY_DIR, KEEP_UNUSED_DIR, CAPTION_REMOVE_DIR, REVIEW_DIR]:
        os.makedirs(folder, exist_ok=True)
        for name in os.listdir(folder):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".txt")):
                try:
                    os.remove(os.path.join(folder, name))
                except Exception:
                    pass


def _sync_row_update(row_index: Dict[str, Dict[str, Any]], row: Dict[str, Any]) -> None:
    key = row.get("original_filename")
    if key and key in row_index:
        row_index[key].update({
            "selected": row.get("selected", row_index[key].get("selected")),
            "output_bucket": row.get("output_bucket", ""),
            "new_basename": row.get("new_basename", ""),
            "final_caption": row.get("final_caption", ""),
        })


def _write_captioned_image(row: Dict[str, Any], out_dir: str, new_basename: str, global_rules: Dict[str, Any], subject_profile: Dict[str, Any]) -> None:
    row["new_basename"] = new_basename
    row["final_caption"] = build_caption(row, global_rules, subject_profile)
    cropped = body_aware_crop(row["original_path"], row)
    img_out = os.path.join(out_dir, f"{new_basename}.jpg")
    txt_out = os.path.join(out_dir, f"{new_basename}.txt")
    cropped.save(img_out, "JPEG", quality=100)
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(row["final_caption"])


def build_reject_reason_string(row: Dict[str, Any]) -> str:
    """
    Baut den vollstaendigen REJECTED REASON-String aus allen verfuegbaren
    Reject-Quellen einer Row. Wird sowohl im Single-Pass-Modus (main()) als
    auch im Profile-Then-Caption-Modus (continue_caption_from_profile)
    verwendet, damit beide Pfade konsistente .txt-Dateien produzieren.

    Reihenfolge der Quellen:
      1. local_override_reasons (Liste oder String aus CSV-Roundtrip)
      2. status_notes (Duplikat-Marker, Smart-Crop-Marker etc.)
      3. short_reason (hart vergebener Grund: too_small, NSFW, script_error,
         oder bei API-Reject die Audit-Beschreibung)
      4. duplicate_method/duplicate_of explizit
      5. API suggested_status=reject mit short_reason als api_reject:
    """
    reason_parts: List[str] = []

    lor = row.get("local_override_reasons", [])
    if isinstance(lor, str):
        lor = [x.strip() for x in lor.split(",") if x.strip()]
    reason_parts.extend(lor)

    sn = row.get("status_notes", [])
    if isinstance(sn, str):
        sn = [x.strip() for x in sn.split(",") if x.strip()]
    for note in sn:
        if note not in reason_parts:
            reason_parts.append(note)

    sr = row.get("short_reason", "")
    if sr and sr not in reason_parts:
        reason_parts.append(sr)

    dup_method = row.get("duplicate_method", "")
    dup_of = row.get("duplicate_of", "")
    if dup_method and dup_of:
        dup_info = f"duplicate_of:{dup_of} (method:{dup_method})"
        if dup_info not in reason_parts:
            reason_parts.append(dup_info)

    api_status = row.get("suggested_status", "")
    api_reason = row.get("short_reason", "")
    if api_status == "reject" and api_reason:
        api_label = f"api_reject: {api_reason}"
        if api_label not in reason_parts:
            reason_parts.append(api_label)

    return ", ".join(reason_parts) if reason_parts else "unknown"


def needs_caption_remove(row: Dict[str, Any]) -> bool:
    """
    Entscheidet, ob ein Bild in den 03_caption_remove-Bucket gehoert.

    Trigger-Logik (ab v8 Update 2):
      - watermark_or_overlay=True: trainings-toxische Overlays (Datumsstempel,
        Wasserzeichen, App-Filter-Stickers, eingebrannte Texte). Immer
        caption_remove.

    NICHT mehr Trigger:
      - mirror_selfie=True: hat sich in der Praxis als zu aggressiv erwiesen.
        Mirror-Selfies sind meistens harmlose Outfit-Shots ohne lesbare
        Spiegelschrift. Wenn doch echte Spiegelschrift auf Kleidung
        prominent zu sehen ist, faengt das geschaerfte
        prominent_readable_text-Kriterium - falls die KI das uebersieht,
        bleibt 04_review als Korrektur-Pfad. 16+ Bilder pro typischem
        Datensatz waren False-Positives.
      - prominent_readable_text=True: alleine NICHT Trigger - das Feld
        wurde in der Praxis zu aggressiv vergeben (kleine Helmlogos,
        Bootsnamen im Hintergrund). Im neuen v8-Audit-Prompt ist das Feld
        deutlich strenger definiert (8-10% Frame-Anteil oder zentral
        platziert), aber wir verlassen uns nicht alleine darauf.

    Diese Funktion ist die SINGLE SOURCE OF TRUTH fuer caption_remove-
    Entscheidungen - alle vier Output-Pfade (Single-Pass main + Smart-Crop,
    Profile-Then-Caption + Smart-Crop) rufen sie auf, damit die Logik
    nicht divergieren kann.
    """
    if bool(row.get("watermark_or_overlay")):
        return True
    return False


def write_caption_stage_reports(
    *,
    all_rows: List[Dict[str, Any]],
    selected_sorted: List[Dict[str, Any]],
    review_items: List[Dict[str, Any]],
    unselected_keep: List[Dict[str, Any]],
    reject_items: List[Dict[str, Any]],
    global_rules: Dict[str, Any],
    subject_profile: Dict[str, Any],
    identity_summary: Dict[str, Any],
    warnings: List[str],
    valid_candidate_count: int,
) -> None:
    csv_fields = [
        "original_filename", "base_status", "selected", "output_bucket", "new_basename",
        "quality_total", "grundscore", "score_nach_eskalation", "quality_sharpness",
        "quality_lighting", "quality_composition", "quality_identity_usefulness", "shot_type",
        "gender_class", "face_visible", "face_occlusion", "multiple_people",
        "main_subject_clear", "watermark_or_overlay", "prominent_readable_text",
        "image_medium",
        "mirror_selfie", "hair_description", "beard_description", "glasses_description",
        "piercings_description", "makeup_description", "skin_tone", "eye_color", "body_build",
        "body_skin_visibility",
        "face_orientation_in_frame",
        "tattoos_visible", "tattoos_description", "clothing_description", "pose_description",
        "expression", "gaze_direction", "head_pose_bucket", "background_description",
        "lighting_description", "lighting_type", "background_type", "hair_texture",
        "makeup_intensity", "has_glasses_now", "glasses_frame_shape", "issues",
        "short_reason", "local_override_reasons", "duplicate_of", "duplicate_method",
        "duplicate_distance", "main_face_ratio", "secondary_face_area_ratio",
        "face_count_local", "width", "height",
        "file_size_mb", "arcface_distance_to_centroid", "arcface_flag", "final_caption",
    ]

    csv_path = os.path.join(OUTPUT_ROOT, f"dataset_audit_{SAFE_TRIGGER}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            row_copy = dict(row)
            row_copy["issues"] = ", ".join(row_copy.get("issues", [])) if isinstance(row_copy.get("issues"), list) else row_copy.get("issues", "")
            row_copy["local_override_reasons"] = ", ".join(row_copy.get("local_override_reasons", [])) if isinstance(row_copy.get("local_override_reasons"), list) else row_copy.get("local_override_reasons", "")
            writer.writerow(row_copy)

    jsonl_path = os.path.join(OUTPUT_ROOT, f"dataset_audit_{SAFE_TRIGGER}.jsonl")
    write_jsonl(jsonl_path, make_json_safe(all_rows))

    summary = {
        "input_images": len(all_rows),
        "kept_clean_candidates_before_selection": valid_candidate_count,
        "review_candidates": len(review_items),
        "keep_unused_overflow": len(unselected_keep),
        "rejected": len(reject_items),
        "selected_total": len(selected_sorted),
        "selected_train_ready": sum(1 for r in selected_sorted if r.get("output_bucket") == "train_ready"),
        "selected_caption_remove": sum(1 for r in selected_sorted if r.get("output_bucket") == "caption_remove"),
        "selected_headshots": sum(1 for r in selected_sorted if r.get("shot_type") == "headshot"),
        "selected_medium": sum(1 for r in selected_sorted if r.get("shot_type") == "medium"),
        "selected_full_body": sum(1 for r in selected_sorted if r.get("shot_type") == "full_body"),
        "smart_crop_pairs_evaluated": 0,
        "smart_crop_pairs_accepted": 0,
        "smart_crop_pairs_won": 0,
        "identity_check_enabled": identity_summary.get("enabled", False),
        "identity_check_centroid_present": identity_summary.get("centroid_present", False),
        "identity_check_n_with_face": identity_summary.get("n_with_face", 0),
        "identity_check_n_no_face": identity_summary.get("n_no_face", 0),
        "identity_check_n_ok": identity_summary.get("n_ok", 0),
        "identity_check_n_soft_flagged": identity_summary.get("n_soft", 0),
        "identity_check_n_hard_flagged_removed": identity_summary.get("n_hard", 0),
        "subject_profile_enabled": bool(subject_profile),
        "subject_profile_normalizer_model": (subject_profile or {}).get("normalizer_model", ""),
        "subject_profile_sample_size": (subject_profile or {}).get("sample_size", 0),
        "subject_profile_total_usable_images": (subject_profile or {}).get("total_usable_images", 0),
        "caption_stage_continued_from_profile": True,
    }

    report = {
        "summary": summary,
        "warnings": warnings,
        "global_rules": global_rules,
        "identity_check": identity_summary,
        "subject_profile": subject_profile_report_summary(subject_profile),
    }

    md_path = os.path.join(OUTPUT_ROOT, f"dataset_report_{SAFE_TRIGGER}.md")
    save_report_md(md_path, report)

    safe_print("")
    safe_print("=" * 70)
    safe_print(f"DONE: {TRIGGER_WORD}")
    safe_print("=" * 70)
    for k, v in summary.items():
        safe_print(f"{k}: {v}")
    safe_print("-" * 70)
    if warnings:
        safe_print("WARNINGS:")
        for w in warnings:
            safe_print(f" - {w}")
        safe_print("-" * 70)
    safe_print(f"CSV:   {csv_path}")
    safe_print(f"JSONL: {jsonl_path}")
    safe_print(f"MD:    {md_path}")
    safe_print(f"Train-ready:     {TRAIN_READY_DIR}")
    if unselected_keep:
        safe_print(f"Keep-unused:     {KEEP_UNUSED_DIR} ({len(unselected_keep)} overflow)")
    safe_print(f"Caption-remove:  {CAPTION_REMOVE_DIR}")
    if EXPORT_REVIEW_IMAGES:
        safe_print(f"Review:          {REVIEW_DIR}")
    safe_print("=" * 70)


def continue_caption_from_profile() -> None:
    """Phase 3 Continue-Pfad: exportiert Captions/Bilder aus _caption_stage.json.
    Kein Audit, kein Dedup, kein ArcFace-Neulauf.

    Wichtige Bucket-Regel fuer Subject-Profile-Captioning:
    - train_ready und keep_unused werden immer mit Profil-Captions exportiert
    - caption_remove und review sind explizit als Caption-Buckets vorbereitet
      und erhalten beim Export ebenfalls Caption-Dateien
    - caption_remove/review bleiben weiterhin aus der Subject-Profile-Auswertung
      ausgeschlossen; PROFILE_INPUT_BUCKETS steuert die Profilbildung separat
    """
    safe_print("🧬 Continuing from confirmed subject profile...")
    stage = load_caption_stage()
    subject_profile = load_confirmed_subject_profile(stage)
    if not subject_profile:
        raise RuntimeError("No subject profile available. Load/edit _subject_profile.json first.")

    all_rows = stage.get("all_rows", []) or []
    selected_sorted = stage.get("selected_sorted", []) or []
    review_items = stage.get("review_items", []) or []
    unselected_keep = stage.get("unselected_keep", []) or []
    reject_items = stage.get("reject_items", []) or []
    global_rules = stage.get("global_rules", {}) or {}
    identity_summary = stage.get("identity_summary", {}) or {}
    warnings = stage.get("warnings", []) or []
    valid_candidate_count = int(stage.get("valid_candidate_count", 0) or 0)

    clean_caption_output_dirs()
    row_index = {r.get("original_filename"): r for r in all_rows if r.get("original_filename")}

    # Auch die Non-Training-Buckets werden hier bewusst mit Captions aus dem
    # bestaetigten Subject Profile exportiert. So sind 03_caption_remove und
    # 04_review fuer spaetere manuelle Bearbeitung bereits captioned.
    counters = {"train_ready": 1, "keep_unused": 1, "caption_remove": 1, "review": 1}

    for row in selected_sorted:
        needs_text_cleanup = needs_caption_remove(row)
        if needs_text_cleanup and SEND_TEXT_IMAGES_TO_CAPTION_REMOVE:
            bucket = "caption_remove"
            out_dir = CAPTION_REMOVE_DIR
            new_basename = f"{SAFE_TRIGGER}-caption_remove_{counters[bucket]:03d}"
        else:
            bucket = "train_ready"
            out_dir = TRAIN_READY_DIR
            new_basename = f"{SAFE_TRIGGER}_{counters[bucket]:03d}"
        counters[bucket] += 1
        row["output_bucket"] = bucket
        row["selected"] = True
        _write_captioned_image(row, out_dir, new_basename, global_rules, subject_profile)
        _sync_row_update(row_index, row)

    if EXPORT_REVIEW_IMAGES:
        review_export = sorted(review_items, key=lambda r: -int(r.get("quality_total", 0)))
        for row in review_export:
            needs_text_cleanup = needs_caption_remove(row)
            if needs_text_cleanup and SEND_TEXT_IMAGES_TO_CAPTION_REMOVE:
                bucket = "caption_remove"
                out_dir = CAPTION_REMOVE_DIR
                new_basename = f"{SAFE_TRIGGER}-caption_remove_{counters['caption_remove']:03d}"
            else:
                bucket = "review"
                out_dir = REVIEW_DIR
                new_basename = f"{SAFE_TRIGGER}_review_{counters['review']:03d}"
            counters[bucket] += 1
            row["output_bucket"] = bucket
            try:
                _write_captioned_image(row, out_dir, new_basename, global_rules, subject_profile)
                _sync_row_update(row_index, row)
            except Exception as e:
                safe_print(f"   ⚠️ Review export failed for {row.get('original_filename','')}: {e}")

    keep_unused_sorted = sorted(unselected_keep, key=lambda r: -int(r.get("quality_total", 0)))
    for row in keep_unused_sorted:
        new_basename = f"{SAFE_TRIGGER}_unused_{counters['keep_unused']:03d}"
        counters["keep_unused"] += 1
        row["output_bucket"] = "keep_unused"
        try:
            _write_captioned_image(row, KEEP_UNUSED_DIR, new_basename, global_rules, subject_profile)
            _sync_row_update(row_index, row)
        except Exception as e:
            safe_print(f"   ⚠️ Keep-unused export failed for {row.get('original_filename','')}: {e}")

    if EXPORT_REJECT_IMAGES:
        reject_export = sorted(reject_items, key=lambda r: -int(r.get("quality_total", 0)))
        for idx, row in enumerate(reject_export, start=1):
            new_basename = f"{SAFE_TRIGGER}_reject_{idx:03d}"
            img_out = os.path.join(REJECT_DIR, f"{new_basename}.jpg")
            txt_out = os.path.join(REJECT_DIR, f"{new_basename}.txt")
            try:
                shutil.copy2(row["original_path"], img_out)
                reasons_str = build_reject_reason_string(row)
                with open(txt_out, "w", encoding="utf-8") as ft:
                    ft.write(f"REJECTED REASON: {reasons_str}\n")
                    ft.write(f"score={row.get('quality_total', 0)} | type={row.get('shot_type', '')} | file={row.get('original_filename', '')}\n")
            except Exception as e:
                safe_print(f"   ⚠️ Reject export failed for {row.get('original_filename','')}: {e}")

    write_caption_stage_reports(
        all_rows=all_rows,
        selected_sorted=selected_sorted,
        review_items=review_items,
        unselected_keep=unselected_keep,
        reject_items=reject_items,
        global_rules=global_rules,
        subject_profile=subject_profile,
        identity_summary=identity_summary,
        warnings=warnings,
        valid_candidate_count=valid_candidate_count,
    )

def local_status_override(item: Dict[str, Any]) -> Tuple[str, List[str]]:
    reasons = []

    shot = item.get("shot_type", "headshot")
    score = int(item.get("quality_total", 0))
    face_ratio = float(item.get("main_face_ratio", 0.0))
    multiple_people = bool(item.get("multiple_people", False))
    face_visible = bool(item.get("face_visible", False))
    face_occlusion = item.get("face_occlusion", "none")
    face_count_local = int(item.get("face_count_local", 0))
    main_subject_clear = bool(item.get("main_subject_clear", True))
    issues = set(item.get("issues", []))

    # ── Image-Medium-Check (hard reject, hoechste Prioritaet) ──
    # Nicht-photographische Bilder (Anime, Illustrationen, 3D-Renders,
    # Screenshots, AI-Generiert) vergiften das LoRA-Training. Selbst wenn
    # die Person erkannt wird, bringt das Modell falsche Visualstatistiken
    # bei (anatomische Vereinfachungen, Anime-Augen-Proportionen, etc.).
    # Strenger Filter, hoechste Prioritaet vor allen anderen Checks.
    image_medium = str(item.get("image_medium", "")).strip().lower()
    if image_medium and image_medium != "photograph":
        reasons.append(f"non_photographic_medium({image_medium})")
        item.setdefault("status_notes", []).append(
            f"image_medium_{image_medium}_hard_reject"
        )
        return "reject", reasons

    if multiple_people:
        sec_ratio = float(item.get("secondary_face_area_ratio", 0.0))

        # Strategie 1 (Default, empfohlen): immer auf review degradieren -
        # ABER mit Hard-Reject-Pfad fuer eindeutige Mehrpersonen-Bilder.
        # Wenn lokal ein zweites Gesicht erkannt wurde, das gross genug ist
        # (>= MULTIPLE_PEOPLE_HARD_REJECT_SECONDARY_FACE_RATIO), ist die
        # API-Aussage durch lokale Detection bestaetigt und das Bild ist
        # objektiv unbrauchbar (kein Review-Aufwand noetig).
        if MULTIPLE_PEOPLE_ALWAYS_DOWNGRADE_TO_REVIEW:
            hard_threshold = float(MULTIPLE_PEOPLE_HARD_REJECT_SECONDARY_FACE_RATIO)
            if (hard_threshold > 0.0
                    and face_count_local >= 2
                    and sec_ratio >= hard_threshold):
                reasons.append(
                    f"multiple_people_confirmed_local(sec_ratio={sec_ratio:.2f})"
                )
                item.setdefault("status_notes", []).append(
                    f"multiple_people_hard_reject_sec_ratio_{sec_ratio:.2f}"
                )
                return "reject", reasons
            reasons.append("multiple_people_downgraded_to_review")
            item.setdefault("status_notes", []).append(
                "multiple_people_always_downgrade_to_review"
            )
            return "review", reasons

        # Strategie 2 (Legacy): Dominance-Check. Wenn das Hauptgesicht klar
        # dominiert, ist die API-Meldung wahrscheinlich ein Mismatch (Reflexion
        # in Brille, Hintergrund-Statist, Spiegelbild). Dann statt hard reject
        # -> review. Sonst -> reject.
        if (ENABLE_MULTIPLE_PEOPLE_DOMINANCE_OVERRIDE
                and face_count_local >= 2
                and 0.0 < sec_ratio < CO_FACE_AREA_RATIO_THRESHOLD
                and score >= MULTIPLE_PEOPLE_SOFT_SCORE_MIN):
            reasons.append(
                f"multiple_people_dominant_main_face(sec_ratio={sec_ratio:.2f},score={score})"
            )
            item.setdefault("status_notes", []).append(
                f"multiple_people_downgraded_to_review_sec_ratio_{sec_ratio:.2f}"
            )
            return "review", reasons
        reasons.append("multiple_people")
        return "reject", reasons

    # ── SUBJECT-SANITY-CHECK (nach Gesichts-Erkennung) ─────────────────────
    # Bilder ohne sichtbares Gesicht UND ohne erkennbaren Torso sind
    # isolierte Gliedmassen (Fuesse, Haende) und fuer Person-LoRAs wertlos.
    # Greift NICHT bei sichtbaren Gesichtern und NICHT bei Rueckenansichten
    # mit klarem Torso (mind. SUBJECT_MIN_TORSO_LANDMARKS von 4 Landmarks).
    #
    # Robustheit: Wir vertrauen nicht nur der API-Angabe face_visible, sondern
    # kombinieren sie mit der lokalen MediaPipe-Face-Detection. Nur wenn BEIDE
    # kein Gesicht sehen, greift der Torso-Check. Verhindert False-Rejects,
    # wenn die API ein kleines, aber valides Gesicht uebersehen hat.
    # Der torso_landmark_count wurde bereits in local_subject_metrics
    # gesetzt (vermeidet zweiten MediaPipe-Call).
    if ENABLE_SUBJECT_SANITY_CHECK and not face_visible and face_count_local == 0:
        torso_count = int(item.get("torso_landmark_count", -1))
        # torso_count == -1 bedeutet MediaPipe nicht verfuegbar -> Check skippen
        if torso_count >= 0 and torso_count < SUBJECT_MIN_TORSO_LANDMARKS:
            item.setdefault("status_notes", []).append(
                f"subject_sanity_fail_torso_{torso_count}_of_4"
            )
            reasons.append("no_torso_no_face")
            return "reject", reasons

    # ── FACE-BBOX-BLUR-CHECK (Stufe 2) ─────────────────────────────────────
    # Fuer LoRA-Training ist Gesichtsschaerfe kritisch. Ein unscharfer
    # Hintergrund bei scharfem Gesicht ist ok, umgekehrt nicht.
    # Greift nur wenn ein Gesicht sichtbar ist und eine Face-Bbox vorliegt.
    # Die Stufe-1-Messung im Quick-Reject ist auf Totalausfall kalibriert;
    # hier prangern wir gezielt unscharfe Gesichter an.
    #
    # Konsistenz-Hinweis: Nach einem IG-Frame-Crop zeigt `original_path` auf
    # das gecropte Bild; die Face-Bbox (sowohl aus local_subject_metrics als
    # auch aus der AI) ist dann ebenfalls relativ zum gecropten Bild. Damit
    # passen Bbox und Pfad zusammen. Falls du den IG-Crop-Schritt aus der
    # Pipeline entfernst, muss diese Annahme neu geprueft werden.
    if USE_BLUR_FILTER and face_visible:
        face_bbox = item.get("main_face_bbox")
        orig_path = item.get("original_path")
        if face_bbox and orig_path and os.path.exists(orig_path):
            # Plausibilitaet: Bbox muss innerhalb der Bilddimensionen liegen.
            img_w = int(item.get("width", 0))
            img_h = int(item.get("height", 0))
            fx, fy, fw, fh = [int(v) for v in face_bbox]
            bbox_ok = (
                img_w > 0 and img_h > 0
                and fx >= 0 and fy >= 0 and fw > 0 and fh > 0
                and (fx + fw) <= img_w + 2 and (fy + fh) <= img_h + 2
            )
            if not bbox_ok:
                item.setdefault("status_notes", []).append("face_blur_skipped_bbox_inconsistent")
            else:
                face_var = local_blur_variance_in_face(orig_path, face_bbox)
                # Immer loggen (auch bei Keep), damit nachher die Verteilung
                # analysiert und der Threshold empirisch kalibriert werden kann.
                if face_var >= 0:
                    item["face_blur_variance"] = round(face_var, 1)
                # Shot-type-spezifische Schwelle: bei Headshots ist die
                # Face-Bbox so gross, dass glatte Hautflaechen die Variance
                # statistisch druecken, selbst wenn das Bild scharf ist.
                # Daher hat headshot eine niedrigere Schwelle als full_body.
                shot = str(item.get("shot_type", "")).strip().lower()
                if shot == "headshot" and FACE_MIN_BLUR_VARIANCE_HEADSHOT > 0:
                    threshold = float(FACE_MIN_BLUR_VARIANCE_HEADSHOT)
                elif shot == "medium" and FACE_MIN_BLUR_VARIANCE_MEDIUM > 0:
                    threshold = float(FACE_MIN_BLUR_VARIANCE_MEDIUM)
                elif shot == "full_body" and FACE_MIN_BLUR_VARIANCE_FULL_BODY > 0:
                    threshold = float(FACE_MIN_BLUR_VARIANCE_FULL_BODY)
                else:
                    threshold = float(FACE_MIN_BLUR_VARIANCE)
                if face_var >= 0 and face_var < threshold:
                    item.setdefault("status_notes", []).append(
                        f"face_blur_variance_{face_var:.1f}_below_{threshold}_shot_{shot or 'unknown'}"
                    )
                    reasons.append("face_blur_too_high")
                    return "reject", reasons

    if score < HARD_REJECT_SCORE:
        reasons.append(f"score_below_hard_reject_floor ({score}<{HARD_REJECT_SCORE})")
        return "reject", reasons

    if score < REVIEW_SCORE_MIN:
        reasons.append("score_below_review_threshold")
        return "reject", reasons

    if score < KEEP_SCORE_MIN:
        reasons.append(f"score_below_keep_threshold ({score}<{KEEP_SCORE_MIN})")

    if not main_subject_clear:
        reasons.append("main_subject_not_clear")

    # Dynamic Smart-Crop: Wenn das Bild riesig ist und das Gesicht winzig,
    # machen wir daraus automatisch einen Headshot (Smart Zoom), anstatt es wegzuwerfen!
    # AUSNAHME: Wenn das Gesicht bewusst verdeckt/nicht sichtbar ist (z.B. Rueckenansicht),
    # bleibt der Shot-Typ unveraendert – solche Bilder sollen als Full-Body gewertet werden.
    face_intentionally_hidden = (not face_visible) or (face_occlusion == "major")
    if face_ratio > 0.0001 and shot in MIN_FACE_RATIO and face_ratio < MIN_FACE_RATIO[shot]:
        if face_intentionally_hidden:
            # Gesicht ist absichtlich nicht sichtbar -> nicht zu Headshot umklassifizieren
            item.setdefault("status_notes", []).append("kept_as_fullbody_face_intentionally_hidden")
        elif item.get("width", 0) >= 1024 and item.get("height", 0) >= 1024:
            item["shot_type"] = "headshot"
            shot = "headshot"
            item.setdefault("status_notes", []).append("reclassified_to_headshot_for_smart_zoom")
        else:
            reasons.append(f"face_too_small_for_{shot}")

    if shot == "headshot" and not face_visible:
        reasons.append("headshot_without_clear_face")

    if shot in {"headshot", "medium"} and face_occlusion == "major":
        reasons.append("major_face_occlusion")

    # Local multi-face detection is too buggy (sees faces in trees). Trust API mostly.
 #   if face_count_local >= 3 and not multiple_people:
 #      reasons.append("local_multiple_faces_detected")

    if "sunglasses" in issues:
        reasons.append("sunglasses")

    if "strong_filter" in issues:
        reasons.append("strong_filter")

    if "motion_blur" in issues or "soft_focus" in issues:
        reasons.append("blur_soft_focus")

    # ── Extreme Winkel / isolierte Gliedmassen ──
    # extreme_angle = Bird's-Eye / Worm's-Eye / verzerrte Perspektive:
    #   fuer Person-LoRA-Training ungeeignet (Modell lernt Winkel statt Person).
    # cropped_limbs + kein Gesicht = isolierte Gliedmasse (z.B. nur Fuesse, nur
    #   Haende) ohne Torso-Kontext: wertlos. Bei sichtbarem Gesicht darf der
    #   Koerper gecropt sein (Headshot ist ja gerade das).
    if "extreme_angle" in issues:
        reasons.append("extreme_angle_unusable")
    if "cropped_limbs" in issues and not face_visible:
        reasons.append("isolated_limbs_no_face")

    # ── Face-Orientation-im-Frame: Anti-LoRA-Toxin ──
    # Bilder mit auf-dem-Kopf oder seitlich liegendem Gesicht im 2D-Frame
    # vergiften das LoRA-Training (Modell lernt verkehrte Anatomie).
    # 'inverted'  -> immer Downgrade keep -> review
    # 'sideways'  -> Downgrade nur bei niedriger Komposition (< Threshold)
    # 'tilted'    -> Downgrade nur bei deutlich schwacher Komposition,
    #                strikterer Threshold als sideways
    # Bei allen drei wird zusaetzlich der Pick-Score in adjusted_pick_score
    # bestraft. Hier nur die Status-Logik.
    if ENABLE_FACE_ORIENTATION_PENALTY:
        face_orient = str(item.get("face_orientation_in_frame", "")).strip().lower()
        if face_orient == "inverted" and FACE_ORIENTATION_DOWNGRADE_INVERTED_TO_REVIEW:
            reasons.append("face_inverted_in_frame")
            item.setdefault("status_notes", []).append("face_orientation_inverted_downgrade_to_review")
        elif face_orient == "sideways" and FACE_ORIENTATION_DOWNGRADE_SIDEWAYS_TO_REVIEW:
            comp_val = float(item.get("quality_composition", 0))
            if comp_val < float(FACE_ORIENTATION_SIDEWAYS_DOWNGRADE_COMPOSITION_MAX):
                reasons.append(
                    f"face_sideways_in_frame(composition={comp_val:.0f})"
                )
                item.setdefault("status_notes", []).append(
                    f"face_orientation_sideways_downgrade_to_review_composition_{comp_val:.0f}"
                )
        elif face_orient == "tilted" and FACE_ORIENTATION_DOWNGRADE_TILTED_TO_REVIEW:
            comp_val = float(item.get("quality_composition", 0))
            if comp_val < float(FACE_ORIENTATION_TILTED_DOWNGRADE_COMPOSITION_MAX):
                reasons.append(
                    f"face_tilted_in_frame(composition={comp_val:.0f})"
                )
                item.setdefault("status_notes", []).append(
                    f"face_orientation_tilted_downgrade_to_review_composition_{comp_val:.0f}"
                )

    if not reasons:
        return "keep", reasons

    hard_reject_reasons = {
        "multiple_people",
        "headshot_without_clear_face",
        "major_face_occlusion",
        "extreme_angle_unusable",
        "isolated_limbs_no_face",
    }
    # Bei Full-Body-Shots ist ein verdecktes/fehlendes Gesicht kein Hard-Reject
    # (z.B. Rueckenansichten sind wertvolle Trainingsdaten fuer Koerperhaltung/Kleidung)
    active_hard_rejects = hard_reject_reasons.copy()
    if item.get("shot_type") == "full_body":
        active_hard_rejects.discard("major_face_occlusion")
        active_hard_rejects.discard("headshot_without_clear_face")

    # Hard-Fail ohne Score-Bypass: diese Gruende machen das Bild intrinsisch
    # untrainierbar, unabhaengig vom Qualitaetsscore.
    unconditional_rejects = {"extreme_angle_unusable", "isolated_limbs_no_face"}
    if any(r in unconditional_rejects for r in reasons):
        return "reject", reasons

    if any(r in active_hard_rejects for r in reasons) and score < KEEP_SCORE_MIN:
        return "reject", reasons

    return "review", reasons


# ============================================================
# 8) CLUSTER / DIVERSITY / DUBLETTEN
# ============================================================

def build_outfit_cluster_key(item: Dict[str, Any]) -> str:
    clothing = coarse_key(item.get("clothing_description"), 4)
    shot_type = coarse_key(item.get("shot_type"), 1)
    return f"{clothing}|{shot_type}"


def build_session_cluster_key(item: Dict[str, Any]) -> str:
    bg = coarse_key(item.get("background_description"), 3)
    light = coarse_key(item.get("lighting_description"), 2)
    mirror = "mirror" if item.get("mirror_selfie", False) else "normal"
    mtime_bucket = item.get("mtime_bucket", "unknown")
    return f"{bg}|{light}|{mirror}|{mtime_bucket}"


def mark_duplicates(items: List[Dict[str, Any]]) -> None:
    """
    Near-Duplicate-Filter:
    1) pHash für pixelnahe Dubletten
    2) CLIP für semantisch sehr ähnliche Bilder

    Smart-Crop-Rows werden NICHT gegen ihr eigenes Original verglichen –
    die Auswahl zwischen Original und Crop übernimmt crop_dedup_selected().
    """
    candidates = [i for i in items if i.get("base_status") in {"keep", "review"}]
    candidates.sort(key=lambda x: x.get("quality_total", 0), reverse=True)

    representatives: List[Dict[str, Any]] = []

    for item in candidates:
        is_dup = False

        item_phash = item.get("phash")
        item_clip = item.get("clip_embedding")
        item_clothing = coarse_key(item.get("clothing_description"), 4)
        item_bg = coarse_key(item.get("background_description"), 3)
        item_shot = item.get("shot_type", "")
        item_session = build_session_cluster_key(item)
        item_is_crop = bool(item.get("is_smart_crop", False))
        item_crop_of = item.get("crop_of", "")

        for rep in representatives:
            rep_is_crop = bool(rep.get("is_smart_crop", False))
            rep_filename = rep.get("original_filename", "")
            rep_crop_of = rep.get("crop_of", "")

            # Original und sein eigener Crop sind KEIN Duplikat-Paar –
            # die werden durch crop_dedup_selected() geregelt.
            is_crop_original_pair = (
                (item_is_crop and item_crop_of == rep_filename) or
                (rep_is_crop and rep_crop_of == item.get("original_filename", ""))
            )
            if is_crop_original_pair:
                continue

            if USE_PHASH_DUPLICATE_SCORING:
                rep_phash = rep.get("phash")
                if item_phash is not None and rep_phash is not None:
                    dist = hamming_distance(item_phash, rep_phash)
                    if dist <= PHASH_HAMMING_THRESHOLD:
                        item["duplicate_of"] = rep["original_filename"]
                        item["duplicate_method"] = "phash"
                        item["duplicate_distance"] = dist
                        item["base_status"] = "reject"
                        item.setdefault("status_notes", []).append("near_duplicate_phash")
                        is_dup = True
                        break

            if USE_CLIP_DUPLICATE_SCORING and item_clip is not None and rep.get("clip_embedding") is not None:
                sim = clip_cosine(item_clip, rep["clip_embedding"])
                same_shot = item_shot == rep.get("shot_type", "")
                same_clothing = item_clothing == coarse_key(rep.get("clothing_description"), 4)
                same_bg = item_bg == coarse_key(rep.get("background_description"), 3)
                same_session = item_session == build_session_cluster_key(rep)

                if sim >= CLIP_COSINE_THRESHOLD and same_shot and (same_clothing or same_bg or same_session):
                    item["duplicate_of"] = rep["original_filename"]
                    item["duplicate_method"] = "clip"
                    item["duplicate_distance"] = round(sim, 6)
                    item["base_status"] = "reject"
                    item.setdefault("status_notes", []).append("near_duplicate_clip")
                    is_dup = True
                    break

        if not is_dup:
            item["duplicate_of"] = ""
            item["duplicate_method"] = ""
            item["duplicate_distance"] = ""
            representatives.append(item)


def crop_dedup_selected(selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Stellt sicher, dass Original und sein Smart-Crop NICHT beide im finalen
    Dataset landen. Wenn beide ausgewählt wurden, gewinnt der mit dem höheren
    quality_total. Bei Gleichstand gewinnt der Crop (er ist identity-optimierter).
    """
    originals_by_name = {
        r["original_filename"]: r
        for r in selected
        if not r.get("is_smart_crop", False)
    }
    crops = [r for r in selected if r.get("is_smart_crop", False)]

    to_remove: set = set()
    for crop in crops:
        crop_of = crop.get("crop_of", "")
        if crop_of and crop_of in originals_by_name:
            orig = originals_by_name[crop_of]
            crop_score = float(crop.get("quality_total", 0))
            orig_score = float(orig.get("quality_total", 0))
            if crop_score >= orig_score:
                # Crop ist besser oder gleich → Original raus
                to_remove.add(orig["original_filename"])
                safe_print(
                    f"   🔀 Crop wins: {crop.get('original_filename')} "
                    f"({crop_score:.1f}) > original ({orig_score:.1f})"
                )
            else:
                # Original ist besser → Crop raus
                to_remove.add(crop["original_filename"])
                safe_print(
                    f"   🔀 Original wins: {crop_of} "
                    f"({orig_score:.1f}) > crop ({crop_score:.1f})"
                )

    if SMART_PRECROP_ALLOW_DATASET_DUPLICATES:
        return selected  # Beide erlaubt – kein Dedup
    return [r for r in selected if r["original_filename"] not in to_remove]


def quotas_for_target(target_size: int, available_counts: Dict[str, int]) -> Dict[str, int]:
    raw = {
        "headshot": int(round(target_size * RATIO_HEADSHOT)),
        "medium": int(round(target_size * RATIO_MEDIUM)),
        "full_body": int(round(target_size * RATIO_FULL_BODY)),
            }
    diff = target_size - sum(raw.values())
    if diff != 0:
        raw["headshot"] += diff

    quotas = {k: min(raw[k], available_counts.get(k, 0)) for k in raw}
    return quotas


def diversity_penalty(item: Dict[str, Any], selected: List[Dict[str, Any]]) -> float:
    if not ENABLE_DIVERSITY_PENALTIES or not selected:
        return 0.0

    clothing_key = coarse_key(item.get("clothing_description"))
    bg_key = coarse_key(item.get("background_description"))
    light_key = coarse_key(item.get("lighting_description"))
    expr_key = coarse_key(item.get("expression"))
    mirror = bool(item.get("mirror_selfie", False))
    outfit_cluster = build_outfit_cluster_key(item)
    session_cluster = build_session_cluster_key(item)

    clothing_count = sum(1 for s in selected if coarse_key(s.get("clothing_description")) == clothing_key)
    bg_count = sum(1 for s in selected if coarse_key(s.get("background_description")) == bg_key)
    light_count = sum(1 for s in selected if coarse_key(s.get("lighting_description")) == light_key)
    expr_count = sum(1 for s in selected if coarse_key(s.get("expression")) == expr_key)
    mirror_count = sum(1 for s in selected if bool(s.get("mirror_selfie", False)) == mirror)

    outfit_count = sum(1 for s in selected if build_outfit_cluster_key(s) == outfit_cluster)
    session_count = sum(1 for s in selected if build_session_cluster_key(s) == session_cluster)

    penalty = 0.0
    penalty += max(0, clothing_count - 1) * 6.0
    penalty += max(0, bg_count - 1) * 4.0
    penalty += max(0, light_count - 2) * 2.5
    penalty += max(0, expr_count - 2) * 2.0
    penalty += max(0, mirror_count - 3) * 1.5

    if USE_SESSION_OUTFIT_CLUSTERING:
        penalty += max(0, outfit_count - 1) * 5.0
        penalty += max(0, session_count - 1) * 4.0

    # ── Pose-Bucket-Diversity ──
    # Wenn die KI head_pose_bucket geliefert hat, bestrafe Wiederholungen
    # innerhalb des bereits gewaehlten Sets. "unknown" und leere Werte
    # werden vom Penalty ausgenommen, damit nicht alle Bilder ohne klare
    # Pose-Klassifikation gegeneinander abgewertet werden.
    if ENABLE_POSE_DIVERSITY:
        pose_key = normalize_text(item.get("head_pose_bucket")) or "unknown"
        if pose_key not in {"unknown", ""}:
            pose_count = sum(
                1 for s in selected
                if (normalize_text(s.get("head_pose_bucket")) or "unknown") == pose_key
            )
            penalty += max(0, pose_count - POSE_DIVERSITY_SOFT_LIMIT) * POSE_DIVERSITY_PENALTY_WEIGHT

    return penalty


def body_visibility_bonus(item: Dict[str, Any]) -> float:
    """
    Bonus auf den Pick-Score zugunsten von Bildern mit gut sichtbarem Koerper
    (LoRA-Body-Learning). Wirkt nur auf die Final-Auswahl, nie auf
    keep/review/reject.

    Geltungsbereich nach shot_type:
      - full_body: voller Bonus (FULLBODY_HIGH / FULLBODY_MEDIUM)
      - medium:    halber Bonus (MEDIUM_SHOT_HIGH / MEDIUM_SHOT_MEDIUM)
      - headshot:  0 (Koerper nicht im Frame)

    body_skin_visibility-Werte 'low' und 'n_a' liefern 0 - kein Penalty,
    nur weniger Bonus.
    """
    if not ENABLE_BODY_VISIBILITY_BONUS:
        return 0.0
    visibility = str(item.get("body_skin_visibility", "")).strip().lower()
    shot = str(item.get("shot_type", "")).strip().lower()
    if visibility in ("", "low", "n_a") or shot == "headshot":
        return 0.0
    if shot == "full_body":
        if visibility == "high":
            return float(BODY_VISIBILITY_BONUS_FULLBODY_HIGH)
        if visibility == "medium":
            return float(BODY_VISIBILITY_BONUS_FULLBODY_MEDIUM)
    elif shot == "medium":
        if visibility == "high":
            return float(BODY_VISIBILITY_BONUS_MEDIUM_SHOT_HIGH)
        if visibility == "medium":
            return float(BODY_VISIBILITY_BONUS_MEDIUM_SHOT_MEDIUM)
    return 0.0


def face_orientation_penalty(item: Dict[str, Any]) -> float:
    """
    Pick-Score-Malus fuer Bilder, in denen das Gesicht im Frame nicht
    aufrecht orientiert ist. Bewertet ausschliesslich die 2D-Frame-
    Orientierung (siehe Audit-Prompt fuer face_orientation_in_frame),
    nicht die Pose der Person im Raum.

    Begruendung: Nicht-aufrechte Gesichter sind fuer's LoRA-Training
    toxisch, weil das Modell die Anatomie umgekehrt lernt. Inverted
    ist am schlimmsten (Augen unter Mund), sideways ebenfalls schwer,
    tilted noch tolerierbar.

    Status-Downgrade fuer 'inverted' und 'sideways' wird separat in
    local_status_override gehandhabt - hier nur der Pick-Score-Anteil.
    """
    if not ENABLE_FACE_ORIENTATION_PENALTY:
        return 0.0
    orient = str(item.get("face_orientation_in_frame", "")).strip().lower()
    if orient == "tilted":
        return float(FACE_ORIENTATION_PENALTY_TILTED)
    if orient == "sideways":
        return float(FACE_ORIENTATION_PENALTY_SIDEWAYS)
    if orient == "inverted":
        return float(FACE_ORIENTATION_PENALTY_INVERTED)
    return 0.0


def adjusted_pick_score(item: Dict[str, Any], selected: List[Dict[str, Any]]) -> float:
    # Identity ist das primäre Ziel – 3× stärker gewichtet als bisher
    base = float(item.get("quality_identity_usefulness", 0)) * 3.0
    base += float(item.get("quality_sharpness", 0)) * 1.5
    base += float(item.get("quality_lighting", 0)) * 1.0

    # Komposition als Veto: schlechte Komposition zieht ab, sehr gute gibt Bonus
    comp = float(item.get("quality_composition", 0))
    if comp < 30:
        base -= (30 - comp) * 2.0   # Starkes Malus bei wirklich schlechter Komposition
    elif comp >= 70:
        base += (comp - 70) * 0.3   # Kleiner Bonus für sehr gute Komposition

    face_ratio = float(item.get("main_face_ratio", 0.0))
    base += (face_ratio * 100.0) * 0.5
    base += min(5.0, float(item.get("file_size_mb", 0.0)))

    # Body-Visibility-Bonus: bevorzugt Body-Shots mit mehr sichtbarem Koerper
    # bei gleicher Bildqualitaet. Nur fuer full_body und medium relevant.
    base += body_visibility_bonus(item)

    # Face-Orientation-Penalty: bestraft Bilder mit gekippten/seitlichen/
    # umgekehrten Gesichtern im Frame (LoRA-Anti-Toxin).
    base -= face_orientation_penalty(item)

    if item.get("base_status") == "review":
        base -= 3.0
        if "main_subject_clear" in str(item.get("local_override_reasons", "")):
            base -= 1.5

    return base - diversity_penalty(item, selected)


def cluster_caps_allow(item: Dict[str, Any], selected: List[Dict[str, Any]]) -> bool:
    if not USE_SESSION_OUTFIT_CLUSTERING:
        return True

    outfit_cluster = build_outfit_cluster_key(item)
    session_cluster = build_session_cluster_key(item)

    outfit_count = sum(1 for s in selected if build_outfit_cluster_key(s) == outfit_cluster)
    session_count = sum(1 for s in selected if build_session_cluster_key(s) == session_cluster)

    if outfit_count >= MAX_PER_OUTFIT_CLUSTER:
        return False
    if session_count >= MAX_PER_SESSION_CLUSTER:
        return False
    return True


def choose_final_dataset(clean_keep_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    available_counts = Counter(i["shot_type"] for i in clean_keep_items)
    quotas = quotas_for_target(TARGET_DATASET_SIZE, available_counts)

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in clean_keep_items:
        by_type[item["shot_type"]].append(item)

    for shot_type in by_type:
        by_type[shot_type].sort(key=lambda x: x.get("quality_total", 0), reverse=True)

    selected: List[Dict[str, Any]] = []
    selected_ids = set()

    def greedy_pick(pool: List[Dict[str, Any]], count: int) -> None:
        nonlocal selected, selected_ids
        picked = 0
        while picked < count:
            remaining = [p for p in pool if p["original_filename"] not in selected_ids and cluster_caps_allow(p, selected)]
            if not remaining:
                fallback = [p for p in pool if p["original_filename"] not in selected_ids]
                if not fallback:
                    break
                best = max(fallback, key=lambda x: adjusted_pick_score(x, selected))
            else:
                best = max(remaining, key=lambda x: adjusted_pick_score(x, selected))
            selected.append(best)
            selected_ids.add(best["original_filename"])
            picked += 1

    for shot_type in ["headshot", "medium", "full_body"]:
        greedy_pick(by_type.get(shot_type, []), quotas.get(shot_type, 0))

    remaining_slots = TARGET_DATASET_SIZE - len(selected)
    if remaining_slots > 0:
        leftovers = [i for i in clean_keep_items if i["original_filename"] not in selected_ids]
        leftovers.sort(key=lambda x: x.get("quality_total", 0), reverse=True)
        greedy_pick(leftovers, remaining_slots)

    return selected


# ============================================================
# 9) CAPTIONING
# ============================================================

def photo_type_phrase(shot_type: str, mirror_selfie: bool) -> str:
    if mirror_selfie:
        return {
            "headshot": "mirror selfie photo",
            "medium": "mirror selfie photo",
            "full_body": "full-body mirror selfie photo",
        }.get(shot_type, "mirror selfie photo")

    return {
        "headshot": "close-up photo",
        "medium": "portrait photo",
        "full_body": "full-body photo",
    }.get(shot_type, "photo")


def compact_trait(text: str) -> str:
    """Kompakte Form eines Trait-Strings fuer Caption-Einbau.
    Entfernt das Wort 'visible' (das im Caption-Kontext redundant ist)
    und nutzt clean_audit_string fuer die Vorreinigung.
    """
    t = normalize_feature_value(text)
    if not t:
        return ""
    # 'visible tattoos on the left arm' -> 'tattoos on the left arm'
    t = re.sub(r"\bvisible\s+", "", t).strip()
    if is_invalid_trait_value(t):
        return ""
    return t


def normalize_hair_tag(raw: str) -> dict:
    """
    Normalisiert eine rohe KI-Haar-Beschreibung auf zwei saubere Tags:
      color: "blonde" | "red" | "auburn" | "blue" | "brown" | "dark" | "other" | None
      style: "loose" | "braid" | "updo" | "ponytail" | "pulled back" | "short" | None

    Gibt {"color": ..., "style": ..., "visible": bool} zurück.
    """
    d = raw.strip().lower() if raw else ""

    not_visible_markers = [
        "not visible", "not clearly", "covered by helmet", "covered by hat",
        "covered by beanie", "mostly covered", "hair not", "not applicable",
    ]
    if not d or d in {"none", "n/a"} or any(m in d for m in not_visible_markers):
        return {"color": None, "style": None, "visible": False}

    # Haarfarbe
    if any(x in d for x in ["red hair", "auburn", "reddish"]):
        color = "red"
    elif "blue" in d:
        color = "blue"
    elif any(x in d for x in ["dark brown", "dark hair", "brunette", "black"]):
        color = "dark"
    elif any(x in d for x in ["brown", "light brown"]):
        color = "brown"
    elif any(x in d for x in ["blonde", "blond", "light blonde", "light-blonde",
                               "light colored", "light-colored"]):
        color = "blonde"
    else:
        color = "other"

    # Frisur-Stil
    if any(x in d for x in ["bun", "updo", "top knot"]):
        style = "updo"
    elif "ponytail" in d:
        style = "ponytail"
    elif "braid" in d:
        style = "braid"
    elif any(x in d for x in ["pulled back", "tied back", "pulled away"]):
        style = "pulled back"
    elif any(x in d for x in ["short hair", "short blonde", "short red", "short brown"]):
        style = "short"
    else:
        style = "loose"  # loose/down/flowing/worn down etc.

    return {"color": color, "style": style, "visible": True}


def normalize_beard_tag(raw: str) -> dict:
    """
    Normalisiert eine rohe KI-Bart-Beschreibung auf saubere Tags:

    pattern: einer der 15 gaengigen Bart-Varianten (oder None bei nicht sichtbar)
      clean_shaven        - komplett glatt rasiert
      stubble             - kurze Stoppeln, 5-o-clock-shadow
      designer_stubble    - leicht laenger als stubble, gestylt
      short_beard         - gepflegter kurzer Vollbart, ~1cm
      full_beard          - voller Vollbart, mittellang bis lang
      long_beard          - sehr langer Bart (deutlich unter Kinn hinaus)
      goatee              - Spitzbart / Kinnbart
      mustache_only       - nur Schnurrbart, sonst rasiert
      mustache_goatee     - Schnurrbart + Kinnbart kombiniert (van dyke)
      chin_strap          - schmaler Bart entlang Kieferlinie
      mutton_chops        - voller Backenbart ohne Kinnbart
      soul_patch          - kleiner Fleck unter der Unterlippe
      circle_beard        - Schnurrbart + runder Kinnbart geschlossen
      handlebar_mustache  - gezwirbelter Schnurrbart
      neckbeard           - Bart nur am Hals, nicht im Gesicht
      other               - erkennbarer Bart, aber kein Pattern matcht

    color: Bartfarbe (oder None bei clean_shaven / nicht sichtbar):
      black/dark | brown | blonde | red | gray | white | salt_pepper | other

    visible: bool - True wenn der Bart sichtbar bewertbar war.
    """
    d = raw.strip().lower() if raw else ""

    not_visible_markers = [
        "not visible", "not clearly", "n/a", "not applicable",
        "covered by mask", "obscured", "cannot be determined",
    ]
    if not d or d in {"none"} or any(m in d for m in not_visible_markers):
        return {"pattern": None, "color": None, "visible": False}

    # ─── Clean-shaven Marker ───
    clean_shaven_markers = [
        "no beard", "clean shaven", "clean-shaven", "shaved", "no facial hair",
        "beard absent", "without beard",
    ]
    if any(m in d for m in clean_shaven_markers):
        return {"pattern": "clean_shaven", "color": None, "visible": True}

    # ─── Pattern-Erkennung (von spezifisch zu generisch) ───
    pattern = "other"

    if "handlebar" in d:
        pattern = "handlebar_mustache"
    elif "neckbeard" in d or "neck beard" in d:
        pattern = "neckbeard"
    elif "mutton chop" in d or "muttonchop" in d:
        pattern = "mutton_chops"
    elif "soul patch" in d and not any(x in d for x in ["beard", "goatee", "mustache", "moustache"]):
        pattern = "soul_patch"
    elif "chin strap" in d or "chinstrap" in d:
        pattern = "chin_strap"
    elif "circle beard" in d or "van dyke" in d or "vandyke" in d:
        pattern = "circle_beard"
    elif ("mustache" in d or "moustache" in d) and ("goatee" in d or "chin beard" in d):
        pattern = "mustache_goatee"
    elif "goatee" in d:
        pattern = "goatee"
    elif ("mustache" in d or "moustache" in d) and not any(x in d for x in ["beard", "stubble", "shadow"]):
        pattern = "mustache_only"
    elif "designer stubble" in d or "stylized stubble" in d:
        pattern = "designer_stubble"
    elif any(x in d for x in ["stubble", "5 o'clock shadow", "5-o-clock", "five o'clock",
                                "scruff", "scruffy", "facial shadow"]):
        pattern = "stubble"
    elif any(x in d for x in ["long beard", "very long beard", "lengthy beard"]):
        pattern = "long_beard"
    elif any(x in d for x in ["full beard", "thick beard", "bushy beard", "dense beard",
                                "heavy beard"]):
        pattern = "full_beard"
    elif any(x in d for x in ["short beard", "trimmed beard", "groomed beard",
                                "well-groomed beard", "neat beard", "tidy beard",
                                "short trimmed", "short groomed"]):
        pattern = "short_beard"
    elif "beard" in d:
        pattern = "short_beard"  # konservativer Fallback

    # ─── Farbe-Erkennung (Token-basiert) ───
    # Erfordert Bart-Kontext, damit "white shirt" oder "dark room" nicht
    # als Bartfarbe missdeutet werden. Reihenfolge: spezifisch -> generisch.
    if pattern == "clean_shaven":
        color = None
    else:
        beard_words = ["beard", "stubble", "mustache", "moustache", "goatee",
                        "facial hair", "shadow", "scruff", "patch", "chops"]
        has_beard_context = any(w in d for w in beard_words)

        if not has_beard_context:
            color = "other"
        elif "salt and pepper" in d or "salt-and-pepper" in d:
            color = "salt_pepper"
        elif re.search(r"\b(graying|greying|gray|grey)\b", d):
            color = "gray"
        elif "white" in d and "beard" in d and re.search(r"\bwhite\b", d):
            color = "white"
        elif re.search(r"\b(red|reddish|ginger|auburn)\b", d):
            color = "red"
        elif re.search(r"\b(blonde|blond)\b", d):
            color = "blonde"
        elif re.search(r"\bblack\b", d):
            color = "dark"
        elif re.search(r"\bdark\b", d):
            color = "dark"
        elif re.search(r"\b(brown|light brown)\b", d):
            color = "brown"
        else:
            color = "other"

    return {"pattern": pattern, "color": color, "visible": True}


def build_beard_caption_tag(item: Dict[str, Any], global_rules: Dict[str, Any]) -> Optional[str]:
    """
    Entscheidet ob und wie der Bart in die Caption kommt:
    - Bart-Pattern wird in eine kurze Caption-Phrase uebersetzt
    - Bartfarbe nur bei Abweichung vom Datensatz-Modus (analog zu Hair)
    - Bei clean_shaven wird kein Tag erzeugt (Default-Annahme, nicht erwaehnt)
    - Bei not visible wird kein Tag erzeugt
    """
    raw_beard = item.get("beard_description", "")
    parsed = normalize_beard_tag(raw_beard)

    if not parsed["visible"]:
        return None
    if parsed["pattern"] == "clean_shaven":
        return None

    beard_rule = global_rules.get("beard_description", {})
    stable_mode_raw = beard_rule.get("mode", "")
    stable_color = normalize_beard_tag(stable_mode_raw).get("color") if stable_mode_raw else None

    item_pattern = parsed["pattern"]
    item_color = parsed["color"]

    pattern_phrases = {
        "stubble": "stubble",
        "designer_stubble": "designer stubble",
        "short_beard": "short beard",
        "full_beard": "full beard",
        "long_beard": "long beard",
        "goatee": "goatee",
        "mustache_only": "mustache",
        "mustache_goatee": "mustache and goatee",
        "chin_strap": "chin strap beard",
        "mutton_chops": "mutton chops",
        "soul_patch": "soul patch",
        "circle_beard": "circle beard",
        "handlebar_mustache": "handlebar mustache",
        "neckbeard": "neckbeard",
        "other": "beard",
    }
    pattern_tag = pattern_phrases.get(item_pattern, "")
    if not pattern_tag:
        return None

    # Farbe: nur bei Abweichung vom Modus oder wenn kein Modus bekannt
    color_tag = ""
    if stable_color and item_color and item_color not in {"other"} and item_color != stable_color:
        color_tag = item_color.replace("_", " ")
    elif not stable_color and item_color and item_color not in {"other"}:
        color_tag = item_color.replace("_", " ")

    if color_tag:
        return f"{color_tag} {pattern_tag}"
    return pattern_tag


def build_hair_caption_tag(item: Dict[str, Any], global_rules: Dict[str, Any]) -> Optional[str]:
    """
    Entscheidet ob und wie Haare in die Caption kommen:
    - Haarfarbe nur wenn sie vom Datensatz-Modus abweicht (z.B. rot statt blond)
    - Frisur-Stil immer, wenn sichtbar (ausser wenn include_hair_always=False und kein Varianz-Flag)
    - Wenn include_hair_always=True: vollständiger Tag immer
    """
    raw_hair = item.get("hair_description", "")
    parsed = normalize_hair_tag(raw_hair)

    if not parsed["visible"]:
        return None

    hair_rule = global_rules.get("hair_description", {})
    stable_mode_raw = hair_rule.get("mode", "")
    stable_color = normalize_hair_tag(stable_mode_raw).get("color") if stable_mode_raw else None

    item_color = parsed["color"]
    item_style = parsed["style"]

    # Haarfarbe: nur bei Abweichung vom Modus
    color_tag = ""
    if stable_color and item_color and item_color != stable_color:
        color_tag = item_color  # z.B. "red", "blue"
    elif not stable_color:
        # Kein stabiler Modus bekannt -> Farbe immer erwähnen
        color_tag = item_color or ""

    # Frisur-Stil: immer, wenn sichtbar (ausser "loose" bei stabilem blond = Norm)
    style_tag = ""
    if item_style and item_style != "loose":
        style_tag = item_style  # braid / ponytail / updo / pulled back / short
    elif item_style == "loose" and color_tag:
        # Wenn Farbe abweicht, Stil mitnennen für Vollständigkeit
        style_tag = "loose"

    parts = [p for p in [color_tag, style_tag] if p]
    if not parts:
        return None
    return " ".join(parts) + " hair"


def build_caption(
    item: Dict[str, Any],
    global_rules: Dict[str, Any],
    subject_profile: Optional[Dict[str, Any]] = None,
) -> str:
    shot_type = item.get("shot_type", "headshot")
    mirror_selfie = bool(item.get("mirror_selfie", False))
    photo_type = photo_type_phrase(shot_type, mirror_selfie)
    caption_profile = normalize_caption_profile(globals().get("CAPTION_PROFILE", "ernie"))

    profile = subject_profile or {}
    stable_identity = profile.get("stable_identity", {}) if isinstance(profile, dict) else {}
    per_image_traits = profile.get("per_image_traits", {}) if isinstance(profile, dict) else {}
    image_id = item.get("profile_image_id") or profile_image_id(item)
    image_traits = per_image_traits.get(image_id, {}) if isinstance(per_image_traits, dict) else {}

    gender_class = normalize_feature_value(stable_identity.get("gender")) or normalize_feature_value(item.get("gender_class")) or "person"
    beard_desc = compact_trait(item.get("beard_description"))

    skin_tone = compact_trait(stable_identity.get("skin_tone")) or compact_trait(item.get("skin_tone"))
    eye_color = compact_trait(stable_identity.get("eye_color")) or compact_trait(item.get("eye_color"))
    # Body-Build-Sticky-Empty: Wenn das Profile bewusst body_build = "" gesetzt hat
    # (z.B. wegen Headshot-Dominanz oder User-Override im UI), darf NICHT auf den
    # per-image Audit-Wert zurueckgefallen werden. Sonst wuerde die Demotion sinnlos.
    if "body_build" in stable_identity:
        body_build = compact_trait(stable_identity.get("body_build"))
    else:
        body_build = compact_trait(item.get("body_build"))

    hair_color = image_traits.get("hair_color_base", "")
    hair_form = image_traits.get("hair_form", "")
    profile_hair_tag = profile_hair_caption(hair_color, hair_form)

    hair_desc = compact_trait(item.get("hair_description"))
    if profile_hair_tag:
        hair_tag = profile_hair_tag
    elif CAPTION_POLICY["include_hair_always"] and hair_desc:
        hair_tag = hair_desc
    elif CAPTION_POLICY["include_hair_when_variable"]:
        hair_tag = build_hair_caption_tag(item, global_rules)
    else:
        hair_tag = None

    makeup_token = image_traits.get("makeup_intensity", "")
    makeup_desc = ""
    if makeup_token and makeup_token not in {"none", "no"}:
        makeup_desc = f"{_phrase_from_token(makeup_token)} makeup"
    else:
        makeup_desc = compact_trait(item.get("makeup_description"))

    markers = profile.get("identity_markers", {}) if isinstance(profile, dict) else {}
    glasses_profile = markers.get("glasses", {}) if isinstance(markers, dict) else {}
    glasses_visible = bool(image_traits.get("glasses_visible")) or _profile_bool(item.get("has_glasses_now"))
    glasses_desc = ""
    if glasses_visible:
        glasses_desc = compact_trait(glasses_profile.get("canonical_description")) or compact_trait(item.get("glasses_description"))

    tattoo_map = _inventory_map(profile, "tattoos")
    piercing_map = _inventory_map(profile, "piercings")

    tattoo_bits: List[str] = []
    visible_tattoo_locations = image_traits.get("tattoo_locations_visible", [])
    if visible_tattoo_locations:
        for loc in visible_tattoo_locations:
            desc = tattoo_map.get(loc, "")
            if not desc:
                for t in item.get("tattoo_inventory_now") or []:
                    if normalize_text(t.get("location")) == loc:
                        desc = compact_trait(t.get("description")) or "tattoo"
                        break
            if desc:
                tattoo_bits.append(desc)
    elif bool(item.get("tattoos_visible", False)):
        tattoo_bits.append(compact_trait(item.get("tattoos_description")) or "visible tattoos")

    piercing_bits: List[str] = []
    visible_piercing_locations = image_traits.get("piercing_locations_visible", [])
    if visible_piercing_locations:
        for loc in visible_piercing_locations:
            desc = piercing_map.get(loc, "")
            if not desc:
                for p in item.get("piercing_inventory_now") or []:
                    if normalize_text(p.get("location")) == loc:
                        desc = compact_trait(p.get("description")) or "piercing"
                        break
            if desc:
                piercing_bits.append(desc)
    else:
        fallback_piercing = compact_trait(item.get("piercings_description"))
        if fallback_piercing:
            piercing_bits.append(fallback_piercing)

    # Earring-Doubletten dedupen: 'small hoop earring' und 'small hoop' sind
    # die gleiche Information - die KI liefert manchmal beide, weil sie sich
    # nicht entscheiden kann. Wir behalten den spezifischeren Eintrag.
    # Gleiche Logik fuer Tattoos: 'small floral/script tattoo' und
    # 'small script tattoo' sind dieselbe Beobachtung mit anderem Detail.
    piercing_bits = _dedupe_phrase_list(piercing_bits)
    tattoo_bits = _dedupe_phrase_list(tattoo_bits)

    # KI-Unentschiedenheit aufloesen: 'small hoop or stud nose piercing'
    # -> 'small nose piercing'. Zwei konkurrierende Adjektive werden zugunsten
    # des klaren Substantivs entfernt. Wirkt auf Piercings und Tattoos.
    # Auch Slash-Form: 'small floral/script tattoo' -> 'small tattoo'.
    piercing_bits = [_simplify_or_phrase(p) for p in piercing_bits]
    tattoo_bits = [_simplify_or_phrase(t) for t in tattoo_bits]

    # Nach Simplify nochmal dedupen, weil 'small floral/script tattoo'
    # und 'small script tattoo' nach Simplify beide zu 'small tattoo'
    # werden und dann substring-Doubletten sind.
    piercing_bits = _dedupe_phrase_list(piercing_bits)
    tattoo_bits = _dedupe_phrase_list(tattoo_bits)

    clothing = normalize_feature_value(item.get("clothing_description"))
    pose = normalize_feature_value(item.get("pose_description"))
    expression = normalize_feature_value(item.get("expression"))
    gaze = normalize_feature_value(item.get("gaze_direction"))
    background = normalize_feature_value(item.get("background_description"))
    lighting = normalize_feature_value(item.get("lighting_description"))

    anchor_parts: List[str] = []
    if caption_profile in {"ernie", "shared_compact"}:
        if hair_tag:
            anchor_parts.append(hair_tag)
        if CAPTION_POLICY.get("include_eye_color") and eye_color:
            anchor_parts.append(f"{eye_color} eyes")
        if CAPTION_POLICY["include_skin_tone"] and skin_tone:
            anchor_parts.append(f"{skin_tone} skin")

    first = f"A {photo_type} of {TRIGGER_WORD}"
    if CAPTION_POLICY["include_gender_class"] and gender_class:
        first += f", a {gender_class}"

    if anchor_parts:
        first += " with " + ", ".join(dict.fromkeys([p for p in anchor_parts if p]))

    trait_bits: List[str] = []

    if shot_type in {"medium", "full_body"} and CAPTION_POLICY["include_body_build"] and body_build:
        # Grammatical compact tag: "slim build" instead of a dangling "slim".
        trait_bits.append(body_build if "build" in body_build else f"{body_build} build")

    if caption_profile not in {"ernie", "shared_compact"} and hair_tag:
        trait_bits.append(hair_tag)

    beard_rule = global_rules.get("beard_description", {})
    beard_variable = beard_rule.get("variable", False)
    beard_mode = normalize_compact_text(beard_rule.get("mode", ""))

    # Normalisierter Beard-Tag (15 Patterns + Farbe). Konsistenter ueber den
    # Datensatz hinweg als der KI-Rohtext: "light stubble", "5 o'clock shadow"
    # und "scruff" werden alle zum gleichen Tag "stubble".
    beard_caption_tag = build_beard_caption_tag(item, global_rules)

    if CAPTION_POLICY["include_beard_always"]:
        if beard_caption_tag:
            trait_bits.append(beard_caption_tag)
        elif beard_desc:
            # Fallback wenn der Tag-Builder None liefert (clean_shaven oder
            # nicht sichtbar) aber User explizit immer captionen will.
            trait_bits.append(beard_desc)
    elif CAPTION_POLICY["include_beard_when_variable"]:
        if beard_variable:
            if beard_caption_tag:
                trait_bits.append(beard_caption_tag)
            elif beard_desc:
                trait_bits.append(beard_desc)
        elif not beard_variable and beard_caption_tag and beard_mode:
            # Stable Mode + Abweichung vom Modus -> Tag rein
            item_beard_mode = normalize_compact_text(item.get("beard_description", ""))
            if item_beard_mode and item_beard_mode != beard_mode:
                trait_bits.append(beard_caption_tag)

    if CAPTION_POLICY["include_glasses"] and glasses_desc:
        trait_bits.append(glasses_desc)

    if CAPTION_POLICY["include_piercings"]:
        trait_bits.extend(piercing_bits)

    if CAPTION_POLICY["include_makeup"] and makeup_desc:
        trait_bits.append(makeup_desc)

    if CAPTION_POLICY["include_tattoos"]:
        trait_bits.extend(tattoo_bits)

    if trait_bits:
        first += ", " + ", ".join(dict.fromkeys([t for t in trait_bits if t]))
    first += "."

    sentences = [first]
    pronoun = "They"
    if gender_class in ["woman", "girl"]:
        pronoun = "She"
    elif gender_class in ["man", "boy"]:
        pronoun = "He"

    if clothing:
        # Bug-Fix: KI laesst manchmal den Artikel weg ('wearing dark
        # sleeveless top'). Wir fuegen 'a' bzw 'an' ein wenn fehlt.
        clothing_with_article = _ensure_article(clothing)
        sentences.append(f"{pronoun} {'is' if pronoun in ['He', 'She'] else 'are'} wearing {clothing_with_article}.")

    pose_bits = []
    if pose and pose not in {"none", "unknown"}:
        # Bug-Fix: gelegentlich liefert die KI doppelt verschmolzene
        # Compound-Phrasen wie 'front-facing selfie seated in a car'.
        # Wir saeubern den Compound-Modifier-Praefix wenn er mit dem
        # nachfolgenden Verb kollidiert.
        pose_bits.append(_clean_pose_phrase(pose))

    # Eyes-closed-Sonderfall (Bug A + E):
    # Die KI markiert manchmal 'eyes closed' als Expression UND/ODER als Gaze.
    # 'eyes closed expression' ist grammatikalischer Unsinn (Expression
    # beschreibt Mund/Lippen/Augenbrauen, nicht die Augen). Wir behandeln
    # 'eyes closed' als eigenstaendigen Pose-Bit und vermeiden dabei
    # Mehrfach-Erwaehnung wenn beide Felder es liefern.
    eyes_closed_in_expr = bool(expression and re.search(r"\beyes closed\b", expression, re.IGNORECASE))
    eyes_closed_in_gaze = bool(gaze and re.search(r"\beyes closed\b", gaze, re.IGNORECASE))

    if CAPTION_POLICY["include_expression"] and expression and expression not in {"none", "unknown"}:
        # Bug-Fix: gelegentlich liefert die KI nur ein Adjektiv ohne
        # Substantiv ('neutral', 'pensive'), was zu kaputten Saetzen wie
        # 'with a neutral, looking at camera' fuehrt. Bei Mehrfach-Adjektiven
        # ('neutral, confident') wird mit 'and' verknuepft. 'eyes closed'
        # in Expression wird verworfen (s.o.).
        cleaned_expr = _clean_expression(expression)
        if cleaned_expr:
            pose_bits.append(f"with a {cleaned_expr}")
    if CAPTION_POLICY["include_gaze"] and gaze and gaze not in {"none", "unknown"}:
        # Wenn gaze=='eyes closed' und Expression auch eyes closed enthielt,
        # haengen wir 'with eyes closed' nur einmal an.
        if eyes_closed_in_gaze and eyes_closed_in_expr:
            # Beide Felder reden ueber geschlossene Augen -> einmal sauber anhaengen
            pose_bits.append("with eyes closed")
        elif eyes_closed_in_gaze:
            # Nur gaze ist eyes closed -> normal anhaengen
            pose_bits.append("with eyes closed")
        else:
            pose_bits.append(gaze)
    elif eyes_closed_in_expr:
        # Nur Expression hatte eyes closed (und wurde dort verworfen) -> hier anhaengen
        pose_bits.append("with eyes closed")

    if pose_bits:
        sentences.append(f"{pronoun} {'is' if pronoun in ['He', 'She'] else 'are'} " + ", ".join(pose_bits) + ".")

    if CAPTION_POLICY["include_lighting"] and lighting:
        sentences.append(f"{lighting.capitalize()}.")

    if CAPTION_POLICY["include_background"] and background:
        sentences.append(f"{background.capitalize()}.")

    caption = " ".join(sentences)
    caption = re.sub(r"\s+", " ", caption).strip()
    # Bug-Fix: 'eyeglasses' konsistent zu 'glasses' normalisieren. 'sunglasses'
    # bleibt unveraendert. Greift auch auf eingebaute Trait-Phrases der KI.
    caption = _normalize_glasses_token(caption)
    return caption


# ============================================================
# 10) CROP
# ============================================================

def body_aware_crop(image_path: str, item: Dict[str, Any]) -> Image.Image:
    pil_img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    img = np.array(pil_img)

    h, w = img.shape[:2]

    # Smart-Crop-Rows: Den Pre-Crop-Bereich (Face + Padding) direkt als
    # quadratische Crop-Region verwenden, NICHT nochmal ueber die hohen
    # Multiplikatoren (4.5/5.0) des normalen Headshot-Branches gehen.
    # Das sorgt fuer einen tatsaechlich engeren Zoom als das Original.
    #
    # Geometrie-Konvention:
    #   target_size_px = max(fw, fh) * (1 + 2 * SMART_PRECROP_PADDING_FACTOR_HALF)
    # also bei PADDING_FACTOR_HALF=0.6: ~2.2x die Gesichtsgroesse.
    # Damit kommt der Crop einem echten Headshot deutlich naeher als
    # die alte Logik, bei der das Padding faelschlich auf jede Seite
    # einzeln aufgeschlagen wurde (effektiv ~4-5x).
    #
    # Kompatibilitaet: SMART_PRECROP_PADDING_FACTOR bleibt der UI-Knopf,
    # wird aber jetzt als HALBES Padding interpretiert (pro Seite).
    # Default 0.6 -> Gesamtbreite = fw + 2*0.6*fw = 2.2 * fw.
    # Alte UI-Werte ueber ~1.0 fuehren also nicht mehr zu absurd grossen
    # Crops, sondern zu eng-bis-mittel-engen Headshots.
    if item.get("is_smart_crop") and item.get("smart_crop_bbox"):
        target_w, target_h = 1024, 1024
        fx, fy, fw, fh = item["smart_crop_bbox"]
        face_size = max(int(fw), int(fh))

        # Crop-Groesse: face + padding pro Seite. Ein Faktor von 0.6 pro
        # Seite gibt ~2.2x face_size als Gesamtgroesse, was Gesicht +
        # Haare + obere Schultern erfasst (klassischer Headshot-Bildaufbau).
        size = int(round(face_size * (1.0 + 2.0 * SMART_PRECROP_PADDING_FACTOR)))

        # Untere Schranke: mindestens 1.5x face_size, sonst wird selbst
        # bei winzigen PADDING_FACTOR-Werten kein Headroom mehr fuer Haare
        # gelassen.
        min_size = int(round(face_size * 1.5))
        # Obere Schranke: 80% der kleineren Bilddimension, damit der "Crop"
        # nicht zur Kopie des Originals degeneriert. Das war der Hauptgrund
        # warum vorher Crops fast wie Originale aussahen.
        max_size = int(round(min(w, h) * 0.80))

        size = clamp_int(size, min(min_size, max_size), max_size)

        # Zentrieren auf Face-Mitte, leicht nach oben versetzt damit der
        # Schwerpunkt des Bildes auf den Augen liegt (klassisch: Augen
        # bei ~38% der Bildhoehe von oben). Bei size weit ueber face_size
        # reicht 0.45; bei size knapp ueber face_size brauchen wir mehr
        # Headroom, damit die Haare nicht abgeschnitten werden.
        cx = fx + fw // 2
        cy = fy + fh // 2
        # Vertikale Versetzung: zwischen 0.35 (eng, Headroom-betont) und
        # 0.50 (locker, mittig) je nach Crop-Groesse relativ zum Gesicht.
        # Bei size = 1.5*face -> 0.35 (Stirn ggf. knapp, aber Haare drin)
        # Bei size = 3.0*face -> 0.50 (mittig, wie alte Logik)
        zoom_ratio = size / max(1, face_size)
        # Linear interpolieren von 0.35 (zoom=1.5) bis 0.50 (zoom=3.0+)
        v_offset_factor = max(0.35, min(0.50, 0.35 + (zoom_ratio - 1.5) * 0.10))

        sq_x1 = max(0, min(cx - size // 2, w - size))
        sq_y1 = max(0, min(cy - int(size * v_offset_factor), h - size))
        x1, y1, x2, y2 = sq_x1, sq_y1, sq_x1 + size, sq_y1 + size
        return pil_img.crop((x1, y1, x2, y2)).resize((target_w, target_h), Image.Resampling.LANCZOS)

    face_bbox = item.get("main_face_bbox")
    pose_bbox = item.get("pose_bbox")
    shot_type = item.get("shot_type", "headshot")

    def crop_box(x: int, y: int, cw: int, ch: int) -> Tuple[int, int, int, int]:
        x = max(0, min(x, w - cw))
        y = max(0, min(y, h - ch))
        return x, y, x + cw, y + ch

    if not USE_AI_TOOLKIT_CROP_PROFILES:
        target_w, target_h = 1024, 1024
        size = min(w, h)
        x1, y1, x2, y2 = crop_box((w - size) // 2, (h - size) // 2, size, size)
        return pil_img.crop((x1, y1, x2, y2)).resize((target_w, target_h), Image.Resampling.LANCZOS)

    if shot_type == "headshot":
        target_w, target_h = 1024, 1024
        if face_bbox:
            fx, fy, fw, fh = face_bbox
            cx = fx + fw // 2
            cy = fy + fh // 2
            size = int(max(fw * 4.5, fh * 5.0))
            # // 5 statt // 3 erlaubt einen viel tieferen Zoom für kleine Gesichter auf 4K Bildern!
            # Multiplikatoren 4.5/5.0 sorgen für ~38% Gesichtsfläche bei 1024px Output (statt ~18%).
            size = clamp_int(size, min(w, h) // 5, min(w, h))
            x1, y1, x2, y2 = crop_box(cx - size // 2, cy - int(size * 0.45), size, size)
        else:
            size = min(w, h)
            x1, y1, x2, y2 = crop_box((w - size) // 2, (h - size) // 3, size, size)

    elif shot_type == "medium":
        target_w, target_h = 832, 1216
        aspect = target_w / target_h
        if pose_bbox:
            px, py, pw, ph = pose_bbox
            crop_h = int(ph * 0.78)
            crop_w = int(crop_h * aspect)
            if crop_w > w:
                crop_w = w
                crop_h = int(crop_w / aspect)
            cx = px + pw // 2
            cy = py + int(ph * 0.32)
            x1, y1, x2, y2 = crop_box(cx - crop_w // 2, cy - int(crop_h * 0.22), crop_w, crop_h)
        elif face_bbox:
            fx, fy, fw, fh = face_bbox
            crop_h = int(max(fh * 7.5, h * 0.70))
            crop_h = min(crop_h, h)
            crop_w = int(crop_h * aspect)
            if crop_w > w:
                crop_w = w
                crop_h = int(crop_w / aspect)
            cx = fx + fw // 2
            cy = fy + fh // 2
            x1, y1, x2, y2 = crop_box(cx - crop_w // 2, cy - int(crop_h * 0.18), crop_w, crop_h)
        else:
            # Kein Pose- oder Face-Bbox: Breite bestimmt das Format.
            # X-Zentrierung ist ok; Y-Position: Gesicht sollte oben im Crop sein.
            crop_h = min(h, int(w / aspect))
            crop_w = int(crop_h * aspect)
            if crop_w > w:
                crop_w = w
                crop_h = int(crop_w / aspect)
            # Fallback-Y: oberes Viertel (Gesicht tipischerweise oben)
            y_fallback = (h - crop_h) // 4
            x1, y1, x2, y2 = crop_box((w - crop_w) // 2, y_fallback, crop_w, crop_h)

    elif shot_type == "full_body":
        target_w, target_h = 832, 1216
        aspect = target_w / target_h

        # Phase 2.1: Full-body exports must be conservative. The previous
        # pose_bbox branch used crop_h = ph * 1.12, which can zoom into an
        # incomplete MediaPipe torso/leg bbox and cut away context that was
        # still present in the original image. For LoRA curation it is safer
        # to preserve almost the entire original frame and only crop as much
        # as required by the target aspect ratio.
        crop_h = h
        crop_w = int(round(crop_h * aspect))
        if crop_w > w:
            crop_w = w
            crop_h = int(round(crop_w / aspect))

        # X: center on the detected body if available, otherwise on face or image.
        if pose_bbox:
            px, py, pw, ph = pose_bbox
            cx = px + pw // 2
        elif face_bbox:
            fx, fy, fw, fh = face_bbox
            cx = fx + fw // 2
        else:
            cx = w // 2
        x_start = clamp_int(cx - crop_w // 2, 0, w - crop_w)

        # Y: preserve full frame whenever possible. If the aspect-ratio crop
        # forces vertical trimming, bias toward keeping the top because raised
        # arms, phones, hair, and heads are more often lost there.
        if crop_h >= h - 2:
            y_start = 0
        elif face_bbox:
            fy_top = face_bbox[1]
            fh_val = face_bbox[3]
            y_start = clamp_int(fy_top - int(fh_val * 1.0), 0, h - crop_h)
        elif pose_bbox:
            px, py, pw, ph = pose_bbox
            # Keep a generous headroom above the detected pose box.
            y_start = clamp_int(py - int(ph * 0.12), 0, h - crop_h)
        else:
            y_start = 0
        x1, y1, x2, y2 = crop_box(x_start, y_start, crop_w, crop_h)

    else:
        # Fallback: API-Schema erlaubt nur headshot/medium/full_body.
        # Sollte nie eintreten. Sicherheitshalber wie full_body behandeln.
        target_w, target_h = 832, 1216
        aspect = target_w / target_h
        crop_h = h
        crop_w = int(h * aspect)
        if crop_w > w:
            crop_w = w
            crop_h = int(crop_w / aspect)
        x1, y1, x2, y2 = crop_box((w - crop_w) // 2, (h - crop_h) // 2, crop_w, crop_h)

    crop = pil_img.crop((x1, y1, x2, y2)).resize((target_w, target_h), Image.Resampling.LANCZOS)
    return crop


# ============================================================
# 11) REPORTS
# ============================================================

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_report_md(path: str, report: Dict[str, Any]) -> None:
    lines = []
    lines.append(f"# Dataset report for {TRIGGER_WORD}")
    lines.append("")
    lines.append(f"- Input folder: `{INPUT_FOLDER}`")
    lines.append(f"- Model used for audit: `{AI_MODEL}`")
    lines.append(f"- Target dataset size: `{TARGET_DATASET_SIZE}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for k, v in report["summary"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    if report.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for w in report["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    if report.get("global_rules"):
        lines.append("## Global rules")
        lines.append("")
        for field, info in report["global_rules"].items():
            lines.append(f"- {field}: mode=`{info.get('mode','')}`, variable={info.get('variable', False)}")
        lines.append("")

    # Identity-Check-Sektion
    ic = report.get("identity_check", {})
    if ic and ic.get("enabled"):
        lines.append("## Identity consistency check (ArcFace)")
        lines.append("")
        if not ic.get("centroid_present"):
            lines.append(f"- Skipped: {ic.get('skipped_reason', 'unknown')}")
        else:
            lines.append(f"- Faces detected: {ic.get('n_with_face', 0)}")
            lines.append(f"- No face detected: {ic.get('n_no_face', 0)}")
            lines.append(f"- OK (sim >= {ARCFACE_SOFT_THRESHOLD}): {ic.get('n_ok', 0)}")
            lines.append(f"- Soft-flagged ({ARCFACE_HARD_THRESHOLD} <= sim < {ARCFACE_SOFT_THRESHOLD}): {ic.get('n_soft', 0)}")
            lines.append(f"- Hard-flagged (sim < {ARCFACE_HARD_THRESHOLD}, moved to 06_needs_manual_review): {ic.get('n_hard', 0)}")
            if ic.get("hard_flagged"):
                lines.append("")
                lines.append("### Hard-flagged (removed from train_ready)")
                lines.append("")
                for fn in ic["hard_flagged"]:
                    lines.append(f"- `{fn}`")
            if ic.get("soft_flagged"):
                lines.append("")
                lines.append("### Soft-flagged (kept in train_ready, verify visually)")
                lines.append("")
                for fn in ic["soft_flagged"]:
                    lines.append(f"- `{fn}`")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 12) MAIN
# ============================================================


def generate_dashboard(all_rows: List[Dict[str, Any]], selected: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("============================================================")
    lines.append("📊 DATASET DASHBOARD & ANALYSE")
    lines.append("============================================================")

    scores = [float(r.get("quality_total", 0)) for r in all_rows if float(r.get("quality_total", 0)) > 0]
    bins = {"90-100":0, "80-89":0, "70-79":0, "60-69":0, "<60":0}
    for s in scores:
        if s >= 90: bins["90-100"] += 1
        elif s >= 80: bins["80-89"] += 1
        elif s >= 70: bins["70-79"] += 1
        elif s >= 60: bins["60-69"] += 1
        else: bins["<60"] += 1

    lines.append("\n📈 QUALITÄTS-HISTOGRAMM (Alle bewerteten Bilder)")
    max_count = max(bins.values()) if scores else 1
    for k, v in bins.items():
        bar = "█" * int((v / max(1, max_count)) * 20)
        lines.append(f" {k:>7} | {bar} ({v})")

    lines.append("\n🏆 TOP 10 BILDER IM DATENSATZ")
    top10 = sorted(all_rows, key=lambda x: float(x.get("quality_total", 0)), reverse=True)[:10]
    for i, r in enumerate(top10, 1):
        status = r.get('base_status', '')
        score = float(r.get("quality_total", 0))
        lines.append(f" {i:>2}. [{score:>4.1f}] {r['original_filename'][:35]:<35} ({r.get('shot_type','')}, {status})")

    lines.append("\n📉 HÄUFIGSTE LOKALE REVIEW/REJECT GRÜNDE")
    reasons = []
    for r in all_rows:
        rs = r.get("local_override_reasons", [])
        if isinstance(rs, str):
            rs = [x.strip() for x in rs.split(",") if x.strip()]
        reasons.extend(rs)
    from collections import Counter
    rc = Counter(reasons)
    if not rc:
        lines.append(" - Keine (Alle Bilder makellos oder keine lokalen Filter getriggert)")
    for k, v in rc.most_common(5):
        lines.append(f" - {v}x {k}")

    lines.append("\n🎯 BUCKET VERTEILUNG DER ENDAUSWAHL (Top 30)")
    sc = Counter([s.get("shot_type", "unknown") for s in selected])
    for k, v in sc.items():
        lines.append(f" - {k.capitalize()}: {v} Bilder")

    # Pose-Bucket-Verteilung im Final-Set
    pose_counter = Counter([
        (normalize_text(s.get("head_pose_bucket")) or "unknown")
        for s in selected
    ])
    if pose_counter:
        lines.append("\n🧭 KOPFPOSE-VERTEILUNG (Final-Set)")
        for k, v in pose_counter.most_common():
            lines.append(f" - {k}: {v}")

    # Identity-Konsistenz-Verteilung im Final-Set
    flag_counter = Counter([
        s.get("arcface_flag", "skipped")
        for s in selected
    ])
    if flag_counter and any(k in flag_counter for k in ("ok", "soft", "hard", "no_face")):
        lines.append("\n🪪 IDENTITY-CHECK (Final-Set)")
        for k in ("ok", "soft", "hard", "no_face", "skipped"):
            if k in flag_counter:
                lines.append(f" - {k}: {flag_counter[k]}")

    lines.append("============================================================\n")
    return "\n".join(lines)

def main() -> None:
    warnings: List[str] = []

    # Konfig-Banner: zeigt aktiv geladenes Modell + Cache-Schema-Version.
    # Dient zum schnellen Debug-Check, ob UI-Config-Overrides oder
    # Schema-Bumps wirklich gegriffen haben (alte Caches bei v6 vs v7
    # haben in der Vergangenheit zu Verwirrung gefuehrt).
    safe_print("=" * 60)
    safe_print(f"  Audit model:        {AI_MODEL}")
    safe_print(f"  Trigger model:      {TRIGGER_CHECK_MODEL}")
    safe_print(f"  Escalation:         {'ON (' + REVIEW_ESCALATION_MODEL + ')' if USE_REVIEW_ESCALATION and REVIEW_ESCALATION_MODEL else 'OFF'}")
    safe_print(f"  Audit cache schema: {AUDIT_CACHE_SCHEMA_VERSION}")
    safe_print(f"  Pipeline mode:      {PIPELINE_MODE}")
    safe_print("=" * 60)

    if CONTINUE_FROM_PROFILE:
        continue_caption_from_profile()
        return

    if USE_AI_TRIGGERWORD_CHECK:
        try:
            trigger_check = check_trigger_word_via_ai(TRIGGER_WORD)
            if trigger_check.get("is_potentially_problematic", False):
                warnings.append(
                    f"Trigger word '{TRIGGER_WORD}' may be problematic ({trigger_check.get('risk_level', 'unknown')}). "
                    f"{trigger_check.get('reason', '')}"
                )
                suggestion = trigger_check.get("suggested_trigger", "").strip()
                if suggestion and suggestion.lower() != TRIGGER_WORD.lower():
                    warnings.append(f"Suggested more robust trigger word: {suggestion}")
        except Exception as e:
            warnings.append(f"Trigger-word check failed: {e}")

    for w in warnings:
        safe_print(f"⚠️ {w}")

    image_paths = iter_input_images(INPUT_FOLDER)
    if not image_paths:
        safe_print("No images found.")
        return

    safe_print(f"Images found: {len(image_paths)}")

    # ── Early pHash-Dedup VOR der API ──────────────────────────────────────
    early_dup_paths: List[str] = []
    phash_cache: Dict[str, int] = {}
    if USE_EARLY_PHASH_DEDUP and USE_PHASH_DUPLICATE_SCORING:
        image_paths, early_dup_paths, phash_cache = early_phash_dedup(image_paths)

    safe_print(f"Starting audit for trigger word: {TRIGGER_WORD}")
    safe_print("")

    all_rows: List[Dict[str, Any]] = []
    # Sammelt alle Smart-Crop-Paare fuer den Vergleichs-Export
    crop_pairs: List[Dict[str, Any]] = []

    # Early-Duplikate als Reject-Rows eintragen (fuer vollständigen CSV-Export)
    for dup_path in early_dup_paths:
        all_rows.append({
            "original_filename": os.path.basename(dup_path),
            "original_path": dup_path,
            "base_status": "reject",
            "final_status": "reject",
            "quality_total": 0,
            "short_reason": "early_phash_duplicate",
            "status_notes": ["early_phash_dedup"],
            "selected": False,
            "output_bucket": "",
            "new_basename": "",
        })

    # PASS 1: Audit pro Bild
    for idx, image_path in enumerate(image_paths, start=1):
        original_filename = os.path.basename(image_path)
        safe_print(f"[{idx}/{len(image_paths)}] {original_filename}")

        try:
            width, height = image_dimensions(image_path)
            row: Dict[str, Any] = {
                "original_filename": original_filename,
                "original_path": image_path,
                "status_notes": [],
                "selected": False,
                "output_bucket": "",
                "new_basename": "",
            }

            if min(width, height) < HARD_MIN_SIDE_PX:
                row.update({
                    "width": width,
                    "height": height,
                    "quality_total": 0,
                    "base_status": "reject",
                    "final_status": "reject",
                    "short_reason": f"hard_pass_too_small_{width}x{height}",
                })
                all_rows.append(row)
                safe_print(f"   ❌ Reject: hard pass {width}x{height}")
                continue

            # ── Vorfilter STUFE 1: Dateigroesse (gratis, vor IG-Crop) ──────
            # Diese Pruefung kommt vor dem IG-Crop, weil sie quasi kostenlos
            # ist und das Laden kleiner Daten filtern kann, bevor wir
            # potenziell teure Frame-Detection starten.
            if USE_MIN_FILESIZE_FILTER:
                kb = local_filesize_kb(image_path)
                if kb < HARD_MIN_FILESIZE_KB:
                    reason = f"filesize_too_small_{kb:.0f}kb"
                    row.update({
                        "width": width,
                        "height": height,
                        "quality_total": 0,
                        "base_status": "reject",
                        "final_status": "reject",
                        "short_reason": reason,
                        "local_override_reasons": [reason],
                    })
                    all_rows.append(row)
                    safe_print(f"   ❌ Reject: {reason}")
                    continue

            file_hash = file_sha1(image_path)

            # ── Instagram-Frame Auto-Crop ──────────────────────────────────
            # Erkennt IG-Story-Rahmen und ersetzt image_path durch das
            # gecropte Bild, damit alle folgenden Schritte (Blur, Exposure,
            # API, Metriken, Hashing) auf dem bereinigten Bild arbeiten.
            if ENABLE_IG_FRAME_CROP:
                ig_cropped_path = run_with_heartbeat(
                    f"[{idx}/{len(image_paths)}] ig_frame_detect {original_filename}",
                    detect_and_crop_ig_frame,
                    image_path,
                )
                if ig_cropped_path:
                    # Dimensionen und Hash des bereinigten Bildes übernehmen
                    width, height = image_dimensions(ig_cropped_path)
                    file_hash = file_sha1(ig_cropped_path)
                    row["ig_frame_cropped"] = True
                    row.setdefault("status_notes", []).append("ig_frame_auto_cropped")
                    safe_print(f"   🖼️  IG frame detected → cropped to {width}x{height}")
                    # Für die weitere Pipeline das gecropte Bild verwenden
                    image_path = ig_cropped_path
                    row["original_path"] = ig_cropped_path

                    # Nach dem Crop Groesse erneut pruefen: Wenn der Crop zu
                    # klein geworden ist, jetzt erst verwerfen.
                    if min(width, height) < HARD_MIN_SIDE_PX:
                        reason = f"hard_pass_too_small_after_ig_crop_{width}x{height}"
                        row.update({
                            "width": width,
                            "height": height,
                            "quality_total": 0,
                            "base_status": "reject",
                            "final_status": "reject",
                            "short_reason": reason,
                            "local_override_reasons": [reason],
                        })
                        all_rows.append(row)
                        safe_print(f"   ❌ Reject: {reason}")
                        continue

            # ── Vorfilter STUFE 2: Blur/Exposure auf gecroptem Bild ────────
            # Diese Checks laufen NACH dem IG-Crop, damit z.B. ein schwarzer
            # Android-Nav-Bar die Helligkeits-Mediane nicht verfaelscht und
            # die Laplacian-Varianz nur den echten Bildinhalt bewertet.
            quick_reject_reason = local_quick_reject_post_crop(image_path, width, height)
            if quick_reject_reason:
                row.update({
                    "width": width,
                    "height": height,
                    "quality_total": 0,
                    "base_status": "reject",
                    "final_status": "reject",
                    "short_reason": quick_reject_reason,
                    "local_override_reasons": [quick_reject_reason],
                })
                all_rows.append(row)
                safe_print(f"   ❌ Reject: {quick_reject_reason}")
                continue

            primary_audit_cache_key = audit_cache_key(file_hash, AI_MODEL, "primary_audit")
            cached = load_cached_audit(primary_audit_cache_key)
            local_meta = run_with_heartbeat(
                f"[{idx}/{len(image_paths)}] local_subject_metrics {original_filename}",
                local_subject_metrics,
                image_path,
                phash_cache=phash_cache,
            )
            row.update(local_meta)
            row["file_hash"] = file_hash

            clip_embedding = None
            if USE_CLIP_DUPLICATE_SCORING:
                clip_embedding = run_with_heartbeat(
                    f"[{idx}/{len(image_paths)}] clip_embedding {original_filename}",
                    compute_clip_embedding,
                    image_path,
                    file_hash,
                )
            row["clip_embedding"] = clip_embedding

            if cached:
                audit = cached["audit"] if "audit" in cached else cached
                safe_print(f"   ↳ Primary audit cache used ({AI_MODEL})")
            else:
                audit = openai_audit_image(
                    image_path,
                    local_meta,
                    model=AI_MODEL,
                    phase_label=f"[{idx}/{len(image_paths)}] primary_audit {original_filename}",
                )

            if audit.get("NSFW_BLOCKED"):
                safe_print(f"      🔞 NSFW BLOCKED: {original_filename} -> needs manual review.")
                review_path = os.path.join(MANUAL_REVIEW_DIR, f"NSFW_{original_filename}")
                shutil.copy2(image_path, review_path)
                all_rows.append({
                    "original_filename": original_filename,
                    "original_path": image_path,
                    "base_status": "reject",
                    "final_status": "reject",
                    "quality_total": 0,
                    "short_reason": "NSFW_BLOCKED_NEEDS_MANUAL_REVIEW"
                })
                continue

            # FIX SCORES LOKAL (Nur wenn das Bild nicht aus dem Cache kommt!)

            # ---------------------------------------------------------
            # OVERWRITE LOCAL BBOX WITH AI BBOX
            # ---------------------------------------------------------
            if "face_bbox_ai" in audit:
                ai_bbox = audit.get("face_bbox_ai")
                face_visible = audit.get("face_visible", False)
                if not face_visible or not ai_bbox or not isinstance(ai_bbox, list) or len(ai_bbox) != 4:
                    # AI says no face -> clear local hallucinations (like necklaces mistaken for faces)
                    row["main_face_bbox"] = None
                    row["main_face_ratio"] = 0.0
                    row.setdefault("status_notes", []).append("cleared_local_face_by_ai")
                else:
                    try:
                        # AI returns relative coords [xmin, ymin, width, height] in 0.0 to 1.0
                        x_rel, y_rel, w_rel, h_rel = [float(v) for v in ai_bbox]

                        # Validierung: Werte muessen im Bereich 0.0-1.0 liegen
                        # und Breite/Hoehe mindestens 1% des Bildes sein
                        coords_valid = all(0.0 <= v <= 1.0 for v in [x_rel, y_rel, w_rel, h_rel])
                        size_valid = w_rel >= 0.01 and h_rel >= 0.01
                        bounds_valid = (x_rel + w_rel) <= 1.05 and (y_rel + h_rel) <= 1.05  # 5% Toleranz

                        if not coords_valid or not size_valid or not bounds_valid:
                            safe_print(
                                f"   ⚠️ Implausible AI face bbox: [{x_rel:.3f}, {y_rel:.3f}, "
                                f"{w_rel:.3f}, {h_rel:.3f}] – using local detection"
                            )
                            row.setdefault("status_notes", []).append("ai_face_bbox_invalid_fallback_local")
                        else:
                            # Auf Bildbereiche clampen (fuer minimal ueberhaengende BBoxen)
                            x_rel = min(x_rel, 1.0)
                            y_rel = min(y_rel, 1.0)
                            w_rel = min(w_rel, 1.0 - x_rel)
                            h_rel = min(h_rel, 1.0 - y_rel)

                            img_w = row.get("width", 1024)
                            img_h = row.get("height", 1024)
                            x_abs = clamp_int(int(x_rel * img_w), 0, img_w - 1)
                            y_abs = clamp_int(int(y_rel * img_h), 0, img_h - 1)
                            w_abs = clamp_int(int(w_rel * img_w), 1, img_w - x_abs)
                            h_abs = clamp_int(int(h_rel * img_h), 1, img_h - y_abs)

                            row["main_face_bbox"] = [x_abs, y_abs, w_abs, h_abs]
                            row["main_face_ratio"] = bbox_area_ratio(row["main_face_bbox"], img_w, img_h)
                            row.setdefault("status_notes", []).append("used_ai_face_bbox")
                    except Exception as e:
                        safe_print(f"   ⚠️ Error while parsing AI face bbox: {e}")
            # ---------------------------------------------------------
            if not cached:
                audit = normalize_audit_scores(audit)
                save_cached_audit(
                    primary_audit_cache_key,
                    audit_cache_payload(audit, AI_MODEL, "primary_audit"),
                )

            row.update(audit)
            # CSV-Audit: primaeren Score separat behalten. Falls spaeter eine
            # Review-Eskalation greift, wird quality_total durch das staerkere
            # Modell ueberschrieben; grundscore bleibt der Score der ersten
            # Bewertung.
            row["grundscore"] = row.get("quality_total", "")
            row["score_nach_eskalation"] = ""

            local_status, local_reasons = local_status_override(row)
            api_status = row.get("suggested_status", "review")

            if should_escalate_audit(api_status, local_status, float(row.get("quality_total", 0))):
                escalation_cache_key = audit_cache_key(file_hash, REVIEW_ESCALATION_MODEL, "escalation_audit")
                cached_escalation = load_cached_audit(escalation_cache_key)
                if cached_escalation:
                    escalated_audit = cached_escalation.get("audit", cached_escalation)
                    escalated_audit = normalize_audit_scores(escalated_audit)
                    safe_print(f"   ↳ Escalation cache used ({REVIEW_ESCALATION_MODEL})")
                else:
                    safe_print(f"   ↳ Escalating with {REVIEW_ESCALATION_MODEL}...")
                    escalated_audit = openai_audit_image(
                        image_path,
                        local_meta,
                        model=REVIEW_ESCALATION_MODEL,
                        phase_label=f"[{idx}/{len(image_paths)}] escalation_audit {original_filename}",
                    )
                    if not escalated_audit.get("NSFW_BLOCKED"):
                        escalated_audit = normalize_audit_scores(escalated_audit)
                        save_cached_audit(
                            escalation_cache_key,
                            audit_cache_payload(escalated_audit, REVIEW_ESCALATION_MODEL, "escalation_audit"),
                        )

                if not escalated_audit.get("NSFW_BLOCKED"):
                    row.update(escalated_audit)
                    row["score_nach_eskalation"] = row.get("quality_total", "")
                    row.setdefault("status_notes", []).append("review_escalation_applied")
                    row["audit_model_used"] = REVIEW_ESCALATION_MODEL
                    local_status, local_reasons = local_status_override(row)
                    api_status = row.get("suggested_status", "review")
                else:
                    row["audit_model_used"] = AI_MODEL
            else:
                row["audit_model_used"] = AI_MODEL

            if api_status == "reject" or local_status == "reject":
                base_status = "reject"
            elif api_status == "review" or local_status == "review":
                base_status = "review"
            else:
                base_status = "keep"

            row["base_status"] = base_status
            row["local_override_reasons"] = local_reasons

            safe_print(
                f"   score={row.get('quality_total', 0):>5.1f} | "
                f"type={row.get('shot_type', 'unknown'):<10} | "
                f"api={api_status:<6} | local={local_status:<6} | final={base_status}"
            )
            if row.get("short_reason"):
                safe_print(f"   ↳ {row['short_reason']}")

            all_rows.append(row)
            time.sleep(SLEEP_BETWEEN_CALLS)

            # ─────────────────────────────────────────────────────────────
            # SMART PRE-CROP: Post-API, basierend auf AI-BBox
            # Trigger: kein Headshot, Gesicht sichtbar, Bild gross genug,
            #          Gesichtsanteil unter Schwellwert, nicht bereits rejected
            # ─────────────────────────────────────────────────────────────
            if (
                ENABLE_SMART_PRECROP
                and base_status != "reject"
                and row.get("shot_type") in {"full_body", "medium"}
                and row.get("face_visible", False)
                and row.get("main_face_bbox") is not None
                and row.get("main_face_ratio", 0.0) < SMART_PRECROP_TRIGGER_RATIO
                and (row.get("width", 0) * row.get("height", 0)) >= 2_000_000
            ):
                ai_bbox = row["main_face_bbox"]   # bereits in Absolut-Pixel (Original)
                fw_check = ai_bbox[2]
                fh_check = ai_bbox[3]
                if min(fw_check, fh_check) >= SMART_PRECROP_MIN_FACE_PX:
                    crop_path = generate_headshot_crop(
                        image_path, ai_bbox, row["width"], row["height"]
                    )
                    if crop_path:
                        try:
                            safe_print("   ✂️  Smart pre-crop: evaluating headshot variant...")

                            # Eigener Cache-Key: Original-Hash + BBox-Koordinaten
                            bbox_str = "_".join(str(v) for v in ai_bbox)
                            crop_cache_key = f"{file_hash}_crop_{bbox_str}"
                            crop_hash = hashlib.sha1(crop_cache_key.encode()).hexdigest()

                            crop_primary_cache_key = audit_cache_key(crop_hash, AI_MODEL, "primary_crop_audit")
                            cached_crop = load_cached_audit(crop_primary_cache_key)
                            # Lokale Metriken (pHash, Pose etc.) IMMER berechnen,
                            # auch bei Cache-Hit, damit Duplikaterkennung funktioniert.
                            crop_local_meta = run_with_heartbeat(
                                f"[{idx}/{len(image_paths)}] crop_local_subject_metrics {original_filename}",
                                local_subject_metrics,
                                crop_path,
                            )
                            if cached_crop:
                                crop_audit = cached_crop["audit"] if "audit" in cached_crop else cached_crop
                                safe_print(f"   ↳ Crop audit cache used ({AI_MODEL})")
                            else:
                                crop_audit = openai_audit_image(
                                    crop_path,
                                    crop_local_meta,
                                    model=AI_MODEL,
                                    phase_label=f"[{idx}/{len(image_paths)}] primary_crop_audit {original_filename}",
                                )

                            if not crop_audit.get("NSFW_BLOCKED"):
                                crop_audit = normalize_audit_scores(crop_audit)

                                if not cached_crop:
                                    save_cached_audit(
                                        crop_primary_cache_key,
                                        audit_cache_payload(crop_audit, AI_MODEL, "primary_crop_audit"),
                                    )

                                crop_score = float(crop_audit.get("quality_total", 0))
                                crop_grundscore = crop_score
                                crop_score_nach_eskalation: Any = ""
                                orig_score = float(row.get("quality_total", 0))

                                if (
                                    should_use_review_escalation()
                                    and ESCALATE_SMART_CROP_CLOSE_CALLS
                                    and abs(crop_score - orig_score) <= SMART_CROP_ESCALATION_MAX_DELTA
                                ):
                                    crop_escalation_cache_key = audit_cache_key(crop_hash, REVIEW_ESCALATION_MODEL, "escalation_crop_audit")
                                    cached_crop_escalation = load_cached_audit(crop_escalation_cache_key)
                                    if cached_crop_escalation:
                                        crop_audit = cached_crop_escalation.get("audit", cached_crop_escalation)
                                        crop_audit = normalize_audit_scores(crop_audit)
                                        safe_print(f"   ↳ Crop escalation cache used ({REVIEW_ESCALATION_MODEL})")
                                    else:
                                        safe_print(f"   ↳ Escalating crop with {REVIEW_ESCALATION_MODEL}...")
                                        escalated_crop_audit = openai_audit_image(
                                            crop_path,
                                            crop_local_meta,
                                            model=REVIEW_ESCALATION_MODEL,
                                            phase_label=f"[{idx}/{len(image_paths)}] escalation_crop_audit {original_filename}",
                                        )
                                        if not escalated_crop_audit.get("NSFW_BLOCKED"):
                                            crop_audit = normalize_audit_scores(escalated_crop_audit)
                                            save_cached_audit(
                                                crop_escalation_cache_key,
                                                audit_cache_payload(crop_audit, REVIEW_ESCALATION_MODEL, "escalation_crop_audit"),
                                            )
                                    crop_score = float(crop_audit.get("quality_total", 0))
                                    crop_score_nach_eskalation = crop_score

                                safe_print(
                                        f"   ↳ Crop {crop_score:.1f} vs. original {orig_score:.1f} "
                                        f"(min gain: {SMART_PRECROP_MIN_GAIN})"
                                )

                                if crop_score >= orig_score + SMART_PRECROP_MIN_GAIN:
                                    # Crop als eigenstaendiger Row anlegen
                                    crop_row: Dict[str, Any] = {
                                        # Dateiname mit Suffix damit er eindeutig ist
                                        "original_filename": original_filename + "__headshot_crop",
                                        # Speichern erfolgt IMMER aus dem Original-Bild!
                                        "original_path": image_path,
                                        "is_smart_crop": True,
                                        "crop_of": original_filename,
                                        "smart_crop_bbox": ai_bbox,
                                        "status_notes": ["smart_precrop_headshot"],
                                        "selected": False,
                                        "output_bucket": "",
                                        "new_basename": "",
                                        "file_hash": crop_hash,
                                        "mtime_bucket": row.get("mtime_bucket"),
                                        "width": row["width"],
                                        "height": row["height"],
                                        "file_size_mb": row.get("file_size_mb", 0),
                                        # pHash/CLIP des Crops (immer berechnet)
                                        "phash": crop_local_meta.get("phash"),
                                        "clip_embedding": (
                                            run_with_heartbeat(
                                                f"[{idx}/{len(image_paths)}] crop_clip_embedding {original_filename}",
                                                compute_clip_embedding,
                                                crop_path,
                                                crop_hash,
                                            )
                                            if USE_CLIP_DUPLICATE_SCORING
                                            else None
                                        ),
                                    }
                                    crop_row.update(crop_audit)
                                    crop_row["grundscore"] = crop_grundscore
                                    crop_row["score_nach_eskalation"] = crop_score_nach_eskalation
                                    # Shot-Type immer Headshot, BBox auf Original-Koordinaten zuruecksetzen
                                    crop_row["shot_type"] = "headshot"
                                    crop_row["main_face_bbox"] = ai_bbox
                                    crop_row["main_face_ratio"] = row.get("main_face_ratio", 0.0)

                                    c_local_status, c_local_reasons = local_status_override(crop_row)
                                    c_api_status = crop_row.get("suggested_status", "review")
                                    if c_api_status == "reject" or c_local_status == "reject":
                                        c_base = "reject"
                                    elif c_api_status == "review" or c_local_status == "review":
                                        c_base = "review"
                                    else:
                                        c_base = "keep"
                                    crop_row["base_status"] = c_base
                                    crop_row["local_override_reasons"] = c_local_reasons

                                    safe_print(
                                        f"   ✅ Crop accepted: score={crop_score:.1f} | status={c_base}"
                                    )
                                    all_rows.append(crop_row)
                                    time.sleep(SLEEP_BETWEEN_CALLS)
                                    # Pair fuer spaetere Vergleichs-Export registrieren
                                    if EXPORT_SMART_CROP_COMPARISON:
                                        crop_pairs.append({
                                            "original_filename": original_filename,
                                            "original_path": image_path,
                                            "original_score": orig_score,
                                            "original_row": row,
                                            "crop_score": crop_score,
                                            "crop_row": crop_row,
                                            "ai_bbox": ai_bbox,
                                            "winner": None,  # wird nach crop_dedup_selected befuellt
                                        })
                                else:
                                    safe_print(
                                        f"   ❌ Crop rejected: gain too small "
                                        f"({crop_score:.1f} - {orig_score:.1f} < {SMART_PRECROP_MIN_GAIN})"
                                    )
                                    # Auch verworfene Crops protokollieren (fuer vollstaendigen Export)
                                    if EXPORT_SMART_CROP_COMPARISON:
                                        crop_pairs.append({
                                            "original_filename": original_filename,
                                            "original_path": image_path,
                                            "original_score": orig_score,
                                            "original_row": row,
                                            "crop_score": crop_score,
                                            "crop_row": None,  # nicht akzeptiert
                                            "ai_bbox": ai_bbox,
                                            "winner": "original",  # Original gewinnt automatisch
                                        })
                        except Exception as crop_e:
                            safe_print(f"   ⚠️ Smart pre-crop failed: {crop_e}")
                        finally:
                            if crop_path and os.path.exists(crop_path):
                                try:
                                    os.remove(crop_path)
                                except Exception:
                                    pass

        except Exception as e:
            tb = traceback.format_exc()
            safe_print(f"   ❌ Error: {e}")
            all_rows.append({
                "original_filename": original_filename,
                "original_path": image_path,
                "base_status": "reject",
                "final_status": "reject",
                "quality_total": 0,
                "short_reason": f"script_error: {e}",
                "traceback": tb,
            })

    # PASS 2: Duplicate-Filter
    mark_duplicates(all_rows)

    # PASS 3: Globale Regeln
    clean_candidates_for_rules = [
        r for r in all_rows
        if r.get("base_status") == "keep"
    ]
    global_rules = compute_global_rules(clean_candidates_for_rules)

    # PASS 4: Finale Auswahl
    valid_candidates = [r for r in all_rows if r.get("base_status") in {"keep", "review"}]
    review_items = [r for r in all_rows if r.get("base_status") == "review"]
    reject_items = [r for r in all_rows if r.get("base_status") == "reject"]

    selected = choose_final_dataset(valid_candidates)
    # Wenn sowohl Original als auch sein Smart-Crop ausgewählt wurden,
    # behalte nur den besseren von beiden.
    selected = crop_dedup_selected(selected)

    # ── Identity-Konsistenz-Check (ArcFace) ──────────────────────────────
    # Berechnet pro Bild die Aehnlichkeit zur "Set-Identitaet" (outlier-
    # getrimmter Centroid). Hard-Flags werden aus dem Train-Set entfernt
    # und gehen in 06_needs_manual_review; Soft-Flags bleiben drin und
    # werden im Markdown-Report markiert. Captions werden NIE veraendert.
    identity_summary = run_identity_consistency_check(selected)

    # Hard-Flags physisch entfernen und in MANUAL_REVIEW_DIR kopieren.
    # Das passiert VOR dem Train-Ready-Export, sodass die rausgefilterten
    # Bilder gar nicht erst in 01_train_ready landen.
    hard_flagged_rows: List[Dict[str, Any]] = []
    if identity_summary.get("centroid_present"):
        hard_flagged_rows = [r for r in selected if r.get("arcface_flag") == "hard"]
    if hard_flagged_rows:
        hard_flag_counter = 1
        for hf_row in hard_flagged_rows:
            try:
                src_path = hf_row.get("original_path", "")
                if not src_path or not os.path.exists(src_path):
                    continue
                src_name = hf_row.get("original_filename", os.path.basename(src_path))
                # Naming-Schema analog zu NSFW_<filename>: IDCHECK_<filename>
                # Praefix-Counter zusaetzlich, damit auch mehrere Hard-Flags
                # eindeutig sortiert sind.
                idcheck_name = f"IDCHECK_{hard_flag_counter:03d}_{src_name}"
                review_path = os.path.join(MANUAL_REVIEW_DIR, idcheck_name)
                shutil.copy2(src_path, review_path)

                # Begleitende .txt mit Distanzwert + Kontext.
                # Die ORIGINAL-Caption bleibt unangetastet (haengt am Bild
                # in 01_train_ready, wenn ueberhaupt - hier wird das Bild
                # aber rausgenommen). Diese .txt ist eine Diagnose-Datei,
                # KEINE Trainings-Caption.
                idcheck_txt = os.path.join(
                    MANUAL_REVIEW_DIR,
                    f"IDCHECK_{hard_flag_counter:03d}_{os.path.splitext(src_name)[0]}.txt"
                )
                sim_val = float(hf_row.get("arcface_distance_to_centroid", -1.0))
                with open(idcheck_txt, "w", encoding="utf-8") as fh:
                    fh.write(
                        "ArcFace identity mismatch detected.\n"
                        f"original_filename: {src_name}\n"
                        f"cosine_similarity_to_set_centroid: {sim_val:.4f}\n"
                        f"hard_threshold: {ARCFACE_HARD_THRESHOLD}\n"
                        f"soft_threshold: {ARCFACE_SOFT_THRESHOLD}\n"
                        f"shot_type: {hf_row.get('shot_type', '')}\n"
                        f"quality_total: {hf_row.get('quality_total', 0)}\n"
                        "\n"
                        "This image was selected for the training set but the face "
                        "embedding is unusually far from the rest of the dataset's "
                        "identity centroid. Possible causes:\n"
                        " - it's actually a different person (e.g. a sibling or "
                        "look-alike that got mixed in)\n"
                        " - it's the same person under heavy filter / make-up / "
                        "occlusion that breaks ArcFace\n"
                        " - it's a much older or younger photo of the same person\n"
                        "\n"
                        "Please verify visually. If it's the right person, you can "
                        "manually move it back into 01_train_ready.\n"
                    )
                hard_flag_counter += 1
            except Exception as e:
                safe_print(f"   ⚠️ Failed to move hard-flagged image {hf_row.get('original_filename','')}: {e}")

        # Aus dem selected-Set entfernen, damit der Train-Ready-Export
        # diese Bilder nicht mehr exportiert.
        hard_names = {r.get("original_filename") for r in hard_flagged_rows}
        selected = [r for r in selected if r.get("original_filename") not in hard_names]
        safe_print(
            f"   🛂 Removed {len(hard_flagged_rows)} hard-flagged image(s) from train_ready; "
            f"copies in 06_needs_manual_review."
        )

    selected_names = {r["original_filename"] for r in selected}
    for row in all_rows:
        if row["original_filename"] in selected_names:
            row["selected"] = True

    # Keep-Bilder, die qualitativ ok sind, aber durch Cluster-/Diversity-Selection
    # nicht ins finale Dataset gekommen sind, landen in einem eigenen Ordner
    # (02_keep_unused). Sie sind weder Review-Kandidaten (wo der Curator unsicher
    # war) noch Rejects, sondern "Overflow" – falls du manuell Bilder nachziehen
    # willst, weil dir das Training noch etwas Daten fehlt.
    unselected_keep = [
        r for r in all_rows
        if r.get("base_status") == "keep"
        and r["original_filename"] not in selected_names
        and r.get("arcface_flag") != "hard"
    ]
    for r in unselected_keep:
        r.setdefault("status_notes", []).append("keep_not_selected_by_diversity")

    # PASS 4b: Subject Profile (Phase 2)
    # Nur verwertbare Keep-Bilder: train_ready + keep_unused. Reject/Review
    # beeinflussen das Profil nicht, weil sie oft genau die fehlerhaften
    # Audit-Werte enthalten, die wir herausfiltern wollen.
    profile_source_rows = list(selected) + list(unselected_keep)
    subject_profile = build_subject_profile(profile_source_rows)

    # PASS 5: Speichern
    shot_order = {"headshot": 0, "medium": 1, "full_body": 2}
    selected_sorted = sorted(
        selected,
        key=lambda r: (shot_order.get(r.get("shot_type"), 9), -int(r.get("quality_total", 0)))
    )

    if PIPELINE_MODE == "profile_then_caption":
        save_caption_stage(
            all_rows=all_rows,
            selected_sorted=selected_sorted,
            review_items=review_items,
            unselected_keep=unselected_keep,
            reject_items=reject_items,
            global_rules=global_rules,
            subject_profile=subject_profile,
            identity_summary=identity_summary,
            warnings=warnings,
            valid_candidate_count=len(valid_candidates),
        )
        safe_print("")
        safe_print("=" * 70)
        safe_print("PROFILE READY - CAPTION EXPORT PAUSED")
        safe_print("=" * 70)
        safe_print(f"Subject profile: {output_subject_profile_path()}")
        safe_print(f"Caption stage:   {output_caption_stage_path()}")
        safe_print("Review or edit the profile in the UI, then click 'Start captioning from profile'.")
        safe_print("No train-ready captions were exported yet.")
        safe_print("=" * 70)
        return

    counters = {
        "train_ready": 1,
        "keep_unused": 1,
        "caption_remove": 1,
        "review": 1,
    }

    # Bug 1 fix: Der frueher hier befindliche Console-Override-Block (input()-basiert)
    # wurde entfernt, weil er im UI-Subprocess-Modus nie greifen konnte. Overrides
    # laufen jetzt ausschliesslich ueber den Subject-Profile-Tab in der UI bzw.
    # ueber _profile_override.json (deep-merged in build_subject_profile).

    for row in selected_sorted:
        needs_text_cleanup = needs_caption_remove(row)

        if needs_text_cleanup and SEND_TEXT_IMAGES_TO_CAPTION_REMOVE:
            bucket = "caption_remove"
            out_dir = CAPTION_REMOVE_DIR
        else:
            bucket = "train_ready"
            out_dir = TRAIN_READY_DIR

        if bucket == "caption_remove":
            new_basename = f"{SAFE_TRIGGER}-caption_remove_{counters[bucket]:03d}"
        else:
            new_basename = f"{SAFE_TRIGGER}_{counters[bucket]:03d}"
        counters[bucket] += 1

        row["output_bucket"] = bucket
        row["new_basename"] = new_basename
        caption = build_caption(row, global_rules, subject_profile)
        row["final_caption"] = caption

        cropped = body_aware_crop(row["original_path"], row)
        img_out = os.path.join(out_dir, f"{new_basename}.jpg")
        txt_out = os.path.join(out_dir, f"{new_basename}.txt")

        cropped.save(img_out, "JPEG", quality=100)
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(caption)

    if EXPORT_REVIEW_IMAGES:
        review_export = sorted(review_items, key=lambda r: -int(r.get("quality_total", 0)))
        for row in review_export:
            needs_text_cleanup = needs_caption_remove(row)

            if needs_text_cleanup and SEND_TEXT_IMAGES_TO_CAPTION_REMOVE:
                bucket = "caption_remove"
                out_dir = CAPTION_REMOVE_DIR
                new_basename = f"{SAFE_TRIGGER}-caption_remove_{counters['caption_remove']:03d}"
            else:
                bucket = "review"
                out_dir = REVIEW_DIR
                new_basename = f"{SAFE_TRIGGER}_review_{counters['review']:03d}"

            counters[bucket] += 1
            row["output_bucket"] = bucket
            row["new_basename"] = new_basename
            row["final_caption"] = build_caption(row, global_rules, subject_profile)

            try:
                cropped = body_aware_crop(row["original_path"], row)
                img_out = os.path.join(out_dir, f"{new_basename}.jpg")
                txt_out = os.path.join(out_dir, f"{new_basename}.txt")
                cropped.save(img_out, "JPEG", quality=100)
                with open(txt_out, "w", encoding="utf-8") as f:
                    f.write(row["final_caption"])
            except Exception:
                pass

    # Keep-Unused-Export: qualitativ als keep eingestufte Bilder, die wegen
    # Cluster-/Diversity-Selection nicht im finalen Dataset gelandet sind.
    # Werden inklusive Caption exportiert, sodass sie bei Bedarf direkt ins
    # Training-Set gezogen werden koennen.
    keep_unused_sorted = sorted(unselected_keep, key=lambda r: -int(r.get("quality_total", 0)))
    for row in keep_unused_sorted:
        new_basename = f"{SAFE_TRIGGER}_unused_{counters['keep_unused']:03d}"
        counters["keep_unused"] += 1
        row["output_bucket"] = "keep_unused"
        row["new_basename"] = new_basename
        row["final_caption"] = build_caption(row, global_rules, subject_profile)

        try:
            cropped = body_aware_crop(row["original_path"], row)
            img_out = os.path.join(KEEP_UNUSED_DIR, f"{new_basename}.jpg")
            txt_out = os.path.join(KEEP_UNUSED_DIR, f"{new_basename}.txt")
            cropped.save(img_out, "JPEG", quality=100)
            with open(txt_out, "w", encoding="utf-8") as f:
                f.write(row["final_caption"])
        except Exception:
            pass

    if EXPORT_REJECT_IMAGES:
        reject_export = sorted(reject_items, key=lambda r: -int(r.get("quality_total", 0)))
        for idx, row in enumerate(reject_export, start=1):
            new_basename = f"{SAFE_TRIGGER}_reject_{idx:03d}"
            img_out = os.path.join(REJECT_DIR, f"{new_basename}.jpg")
            txt_out = os.path.join(REJECT_DIR, f"{new_basename}.txt")

            # Bild kopieren
            try:
                shutil.copy2(row["original_path"], img_out)
            except Exception as copy_err:
                safe_print(f"   ⚠️ Failed to copy reject image: {row.get('original_filename','')} – {copy_err}")

            # Reason-Datei: gemeinsamer Helper baut den vollstaendigen String
            try:
                reasons_str = build_reject_reason_string(row)
                with open(txt_out, "w", encoding="utf-8") as ft:
                    ft.write(f"REJECTED REASON: {reasons_str}\n")
                    ft.write(f"score={row.get('quality_total', 0)} | "
                             f"type={row.get('shot_type', '')} | "
                             f"file={row.get('original_filename', '')}\n")
            except Exception as txt_err:
                safe_print(f"   ⚠️ Failed to write reject text file: {row.get('original_filename','')} – {txt_err}")

    if len(review_items) > 100:
        SECOND_CHOICE_DIR = os.path.join(OUTPUT_ROOT, "08_train_ready_2nd_choice")
        os.makedirs(SECOND_CHOICE_DIR, exist_ok=True)

        review_by_type = defaultdict(list)
        for r in review_items:
            if not r.get("selected", False):
                review_by_type[r.get("shot_type", "medium")].append(r)

        for st in ["headshot", "medium", "full_body"]:
            pool = review_by_type.get(st, [])
            pool.sort(key=lambda x: adjusted_pick_score(x, []), reverse=True)
            second_choice = pool[30:60]
            if not second_choice:
                safe_print(f"   ⚠️  Second choice {st}: pool too small (<30 review images), nothing exported.")
                continue

            for idx2, row in enumerate(second_choice, start=1):
                try:
                    new_basename = f"{SAFE_TRIGGER}_second_{st}_{idx2:03d}"
                    img_out = os.path.join(SECOND_CHOICE_DIR, f"{new_basename}.jpg")
                    txt_out = os.path.join(SECOND_CHOICE_DIR, f"{new_basename}.txt")

                    cropped = body_aware_crop(row["original_path"], row)
                    cropped.save(img_out, "JPEG", quality=100)

                    caption_text = row.get("final_caption") or ""
                    with open(txt_out, "w", encoding="utf-8") as f2:
                        f2.write(caption_text)
                except Exception as e:
                    pass


    # PASS 5b: Smart-Crop Vergleichs-Export
    if EXPORT_SMART_CROP_COMPARISON and crop_pairs:
        safe_print(f"\n📸 Exporting {len(crop_pairs)} smart-crop comparison pairs...")

        # Gewinner aus crop_dedup_selected rueckwirkend eintragen
        for pair in crop_pairs:
            if pair["winner"] is not None:
                continue  # bereits gesetzt (verworfene Crops)
            orig_fn = pair["original_filename"]
            crop_fn = orig_fn + "__headshot_crop"
            orig_selected = orig_fn in selected_names
            crop_selected = crop_fn in selected_names
            if crop_selected and not orig_selected:
                pair["winner"] = "crop"
            elif orig_selected and not crop_selected:
                pair["winner"] = "original"
            else:
                # Keiner wurde ins finale Dataset gewaehlt (z.B. durch Diversity-Penalty)
                orig_score = pair["original_score"]
                crop_score = pair["crop_score"]
                pair["winner"] = "crop" if crop_score >= orig_score else "original"
                pair["winner"] += "_not_selected"

        for pair_idx, pair in enumerate(crop_pairs, start=1):
            try:
                orig_fn   = pair["original_filename"]
                orig_path = pair["original_path"]
                orig_row  = pair["original_row"]
                orig_score = pair["original_score"]
                crop_row_data = pair.get("crop_row")
                crop_score = pair["crop_score"]
                winner     = pair.get("winner", "unknown")
                ai_bbox    = pair["ai_bbox"]

                safe_name = re.sub(r"[^\w\-]", "_", os.path.splitext(orig_fn)[0])[:40]
                prefix = f"pair_{pair_idx:03d}_{safe_name}"

                # --- Original-Crop exportieren (body_aware_crop des Originals) ---
                # Wer hat tatsaechlich den hoeheren Score? ("_not_selected" ignorieren)
                actual_winner = winner.replace("_not_selected", "")
                orig_label = "WINNER" if actual_winner == "original" else "loser"
                orig_out = os.path.join(
                    SMART_CROP_COMPARISON_DIR,
                    f"{prefix}__A_original_{orig_label}_s{orig_score:.0f}.jpg"
                )
                orig_cropped = body_aware_crop(orig_path, orig_row)
                orig_cropped.save(orig_out, "JPEG", quality=100)

                # Caption-Datei fuer Original
                orig_caption = orig_row.get("final_caption") or build_caption(orig_row, {})
                with open(orig_out.replace(".jpg", ".txt"), "w", encoding="utf-8") as fc:
                    fc.write(
                        f"ORIGINAL | score={orig_score:.1f} | type={orig_row.get('shot_type','')} | "
                        f"winner={orig_label}\n\n{orig_caption}"
                    )

                # --- Headshot-Crop exportieren ---
                crop_label = "WINNER" if actual_winner == "crop" else "loser"
                crop_out = os.path.join(
                    SMART_CROP_COMPARISON_DIR,
                    f"{prefix}__B_headshot_crop_{crop_label}_s{crop_score:.0f}.jpg"
                )
                # Crop aus Original-Bild mit AI-BBox neu erzeugen
                img_w = orig_row.get("width", 1024)
                img_h = orig_row.get("height", 1024)
                crop_tmp = generate_headshot_crop(orig_path, ai_bbox, img_w, img_h)
                if crop_tmp:
                    try:
                        # body_aware_crop auf den Headshot anwenden
                        fake_row = dict(orig_row)
                        fake_row["shot_type"] = "headshot"
                        fake_row["main_face_bbox"] = ai_bbox
                        fake_row["is_smart_crop"] = True
                        fake_row["smart_crop_bbox"] = ai_bbox
                        headshot_cropped = body_aware_crop(orig_path, fake_row)
                        headshot_cropped.save(crop_out, "JPEG", quality=100)
                    finally:
                        try:
                            os.remove(crop_tmp)
                        except Exception:
                            pass
                elif crop_row_data:
                    # Fallback: direkt aus Original mit Headshot-Logik
                    fake_row = dict(crop_row_data)
                    fake_row["original_path"] = orig_path
                    headshot_cropped = body_aware_crop(orig_path, fake_row)
                    headshot_cropped.save(crop_out, "JPEG", quality=100)

                # Caption-Datei fuer Crop
                crop_caption = (crop_row_data or {}).get("final_caption") or ""
                with open(crop_out.replace(".jpg", ".txt"), "w", encoding="utf-8") as fc:
                    fc.write(
                        f"HEADSHOT CROP | score={crop_score:.1f} | winner={crop_label}\n\n{crop_caption}"
                    )

                safe_print(f"   Pair {pair_idx:03d}: {orig_fn[:35]} | winner: {winner}")

            except Exception as ep:
                safe_print(f"   ⚠️ Comparison export for pair {pair_idx} failed: {ep}")

        safe_print(f"✅ Smart-crop comparisons: {SMART_CROP_COMPARISON_DIR}")


    # PASS 6: Reports
    csv_fields = [
        "original_filename",
        "base_status",
        "selected",
        "output_bucket",
        "new_basename",
        "quality_total",
        "grundscore",
        "score_nach_eskalation",
        "quality_sharpness",
        "quality_lighting",
        "quality_composition",
        "quality_identity_usefulness",
        "shot_type",
        "gender_class",
        "face_visible",
        "face_occlusion",
        "multiple_people",
        "main_subject_clear",
        "watermark_or_overlay",
        "prominent_readable_text",
        "image_medium",
        "mirror_selfie",
        "hair_description",
        "beard_description",
        "glasses_description",
        "piercings_description",
        "makeup_description",
        "skin_tone",
        "body_build",
        "body_skin_visibility",
        "face_orientation_in_frame",
        "tattoos_visible",
        "tattoos_description",
        "clothing_description",
        "pose_description",
        "expression",
        "gaze_direction",
        "head_pose_bucket",
        "background_description",
        "lighting_description",
        "issues",
        "short_reason",
        "local_override_reasons",
        "duplicate_of",
        "duplicate_method",
        "duplicate_distance",
        "main_face_ratio",
        "secondary_face_area_ratio",
        "face_count_local",
        "width",
        "height",
        "file_size_mb",
        "arcface_distance_to_centroid",
        "arcface_flag",
        "final_caption",
    ]

    csv_path = os.path.join(OUTPUT_ROOT, f"dataset_audit_{SAFE_TRIGGER}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            row_copy = dict(row)
            row_copy["issues"] = ", ".join(row_copy.get("issues", [])) if isinstance(row_copy.get("issues"), list) else row_copy.get("issues", "")
            row_copy["local_override_reasons"] = ", ".join(row_copy.get("local_override_reasons", []))
            if isinstance(row_copy.get("clip_embedding"), np.ndarray):
                row_copy["clip_embedding"] = ""
            writer.writerow(row_copy)

    jsonl_path = os.path.join(OUTPUT_ROOT, f"dataset_audit_{SAFE_TRIGGER}.jsonl")
    json_rows = []
    for row in all_rows:
        row_copy = dict(row)
        if isinstance(row_copy.get("clip_embedding"), np.ndarray):
            row_copy["clip_embedding"] = None
        json_rows.append(row_copy)
    write_jsonl(jsonl_path, json_rows)

    summary = {
        "input_images": len(all_rows),
        "kept_clean_candidates_before_selection": len(valid_candidates),
        "review_candidates": len(review_items),
        "keep_unused_overflow": len(unselected_keep),
        "rejected": len(reject_items),
        "selected_total": len(selected_sorted),
        "selected_train_ready": sum(1 for r in selected_sorted if r.get("output_bucket") == "train_ready"),
        "selected_caption_remove": sum(1 for r in selected_sorted if r.get("output_bucket") == "caption_remove"),
        "selected_headshots": sum(1 for r in selected_sorted if r.get("shot_type") == "headshot"),
        "selected_medium": sum(1 for r in selected_sorted if r.get("shot_type") == "medium"),
        "selected_full_body": sum(1 for r in selected_sorted if r.get("shot_type") == "full_body"),
        "smart_crop_pairs_evaluated": len(crop_pairs),
        "smart_crop_pairs_accepted": sum(1 for p in crop_pairs if p.get("crop_row") is not None),
        "smart_crop_pairs_won": sum(1 for p in crop_pairs if p.get("winner","").startswith("crop") and "not" not in p.get("winner","")),
        "identity_check_enabled": identity_summary.get("enabled", False),
        "identity_check_centroid_present": identity_summary.get("centroid_present", False),
        "identity_check_n_with_face": identity_summary.get("n_with_face", 0),
        "identity_check_n_no_face": identity_summary.get("n_no_face", 0),
        "identity_check_n_ok": identity_summary.get("n_ok", 0),
        "identity_check_n_soft_flagged": identity_summary.get("n_soft", 0),
        "identity_check_n_hard_flagged_removed": identity_summary.get("n_hard", 0),
        "subject_profile_enabled": bool(subject_profile),
        "subject_profile_normalizer_model": (subject_profile or {}).get("normalizer_model", ""),
        "subject_profile_sample_size": (subject_profile or {}).get("sample_size", 0),
        "subject_profile_total_usable_images": (subject_profile or {}).get("total_usable_images", 0),
    }

    if len(selected_sorted) < TARGET_DATASET_SIZE:
        warnings.append(
            f"Intentionally selected only {len(selected_sorted)} instead of {TARGET_DATASET_SIZE} images, "
            f"because quality and/or balance matter more than filler content."
        )

    if summary["selected_full_body"] == 0:
        warnings.append("No final full-body images were selected. Full-body generation will likely be weaker.")
    if summary["selected_headshots"] < max(5, int(TARGET_DATASET_SIZE * 0.25)):
        warnings.append("Relatively few headshots were selected. Identity/face quality may suffer.")

    # Identity-Check-Warnings
    if identity_summary.get("enabled") and identity_summary.get("centroid_present"):
        if identity_summary.get("n_hard", 0) > 0:
            hard_list = ", ".join(identity_summary.get("hard_flagged", []))
            warnings.append(
                f"Identity check: {identity_summary['n_hard']} image(s) hard-flagged "
                f"and moved to 06_needs_manual_review (likely different person): {hard_list}"
            )
        if identity_summary.get("n_soft", 0) > 0:
            soft_list = ", ".join(identity_summary.get("soft_flagged", []))
            warnings.append(
                f"Identity check: {identity_summary['n_soft']} image(s) soft-flagged "
                f"(borderline identity match, kept in train_ready - please verify visually): {soft_list}"
            )
    elif identity_summary.get("skipped_reason"):
        warnings.append(
            f"Identity check skipped: {identity_summary['skipped_reason']}"
        )

    report = {
        "summary": summary,
        "warnings": warnings,
        "global_rules": global_rules,
        "identity_check": identity_summary,
        "subject_profile": subject_profile_report_summary(subject_profile),
    }

    md_path = os.path.join(OUTPUT_ROOT, f"dataset_report_{SAFE_TRIGGER}.md")
    save_report_md(md_path, report)

    safe_print("")
    safe_print("=" * 70)
    safe_print(f"DONE: {TRIGGER_WORD}")
    safe_print("=" * 70)
    for k, v in summary.items():
        safe_print(f"{k}: {v}")
    safe_print("-" * 70)
    if warnings:
        safe_print("WARNINGS:")
        for w in warnings:
            safe_print(f" - {w}")
        safe_print("-" * 70)
    safe_print(f"CSV:   {csv_path}")
    safe_print(f"JSONL: {jsonl_path}")
    safe_print(f"MD:    {md_path}")
    safe_print(f"Train-ready:     {TRAIN_READY_DIR}")
    if unselected_keep:
        safe_print(f"Keep-unused:     {KEEP_UNUSED_DIR} ({len(unselected_keep)} overflow)")
    if EXPORT_SMART_CROP_COMPARISON and crop_pairs:
        safe_print(f"Crop comparisons: {SMART_CROP_COMPARISON_DIR} ({len(crop_pairs)} pairs)")
    safe_print(f"Caption-remove:  {CAPTION_REMOVE_DIR}")
    if EXPORT_REVIEW_IMAGES:
        safe_print(f"Review:          {REVIEW_DIR}")
    safe_print("=" * 70)


if __name__ == "__main__":
    main()
