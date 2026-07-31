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
from PIL import Image, ImageOps, UnidentifiedImageError

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
AI_MODEL = "gpt-5.6-luna"
TRIGGER_CHECK_MODEL = "gpt-5.6-luna"
OPENAI_TOKEN_LIMIT_TOTAL = 0  # 0 = disabled

# GPT-5.6 stage-specific reasoning. Image audits and per-image captions are
# extraction tasks and use no reasoning. The single dataset-wide subject
# profile call gets a small reasoning budget because it must reconcile
# recurring and variable traits across many images.
AUDIT_REASONING_EFFORT = "none"
TRIGGER_CHECK_REASONING_EFFORT = "none"
REVIEW_ESCALATION_REASONING_EFFORT = "low"
PROFILE_REASONING_EFFORT = "low"
KREA_CAPTION_REASONING_EFFORT = "none"
KREA_CAPTION_REPAIR_REASONING_EFFORT = "low"

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

# Kleiner Soft-Penalty für Bilder, die praktisch farblos sind
# (echte Schwarz-Weiß-/Graustufenfotos oder starke Monochrom-Filter).
# Kein Hard-Reject: der Abzug soll nur knappe Fälle etwas nach unten ziehen.
USE_GRAYSCALE_PENALTY = True
GRAYSCALE_SCORE_PENALTY = 5.0
# Konservative Schwellen: beide Bedingungen müssen erfüllt sein, damit normale
# entsättigte/stimmungsvolle Farbfotos nicht versehentlich bestraft werden.
GRAYSCALE_SATURATION_THRESHOLD = 0.08
GRAYSCALE_CHANNEL_DELTA_THRESHOLD = 3.5

# Relaxed fallback for JPEG/social-media B/W images:
# Some visually black-and-white images have tiny RGB channel differences
# from compression, filters, tiles/walls, or app processing.
# The strict path stays unchanged; this fallback only catches near-monochrome
# images with low saturation and mostly channel-even pixels.
GRAYSCALE_RELAXED_SATURATION_THRESHOLD = 0.06
GRAYSCALE_RELAXED_CHANNEL_DELTA_THRESHOLD = 6.0
GRAYSCALE_PIXEL_DELTA_THRESHOLD = 8.0
GRAYSCALE_PIXEL_SHARE_THRESHOLD = 0.92
# Color-Tint-Detection: erkennt Bilder mit dominantem Farbstich (Blau-/Sepia-/
# Grün-Filter etc.). Dient ausschliesslich der Caption-Markierung, kein
# Quality-Penalty - getoente Bilder sind grundsaetzlich brauchbar, der LoRA
# muss nur wissen dass der Tint ein Bild-Attribut ist und nicht zur Person
# gehoert.
USE_COLOR_TINT_CAPTION = True
# Channel-Asymmetrie ab der ein Tint ueberhaupt in Betracht kommt. Werte unter
# 0.15 sind im Bereich von natuerlich warmer/kalter Beleuchtung, die wir nicht
# als Filter behandeln wollen.
TINT_MIN_ASYMMETRY = 0.15
# Asymmetrie ab der Strength = 1.0. Bilder mit asym >= 0.45 sind eindeutige
# Filter (Sepia, Heavy-Blue-Filter, etc.).
TINT_STRONG_ASYMMETRY = 0.45
# Strength-Schwelle ab der das Tint-Label tatsaechlich in die Caption geht.
# Werte zwischen MIN_ASYMMETRY und dieser Schwelle gelten als "leichter
# Farbstich" und werden NICHT captionen, um false positives bei Sonnenuntergang
# / kaltem Tageslicht zu vermeiden.
TINT_MIN_STRENGTH_FOR_CAPTION = 0.30
# Sepia-Spezialfall: braeunlicher Vintage-Look (R > G > B mit deutlicher
# R-B-Differenz im 0..255 Mittel). Wird nur dann als sepia getaggt, wenn die
# RGB-Reihenfolge stimmt UND mean_r - mean_b ueber dieser Schwelle liegt.
TINT_SEPIA_MIN_R_B_DELTA = 25.0

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
USE_CLIP_DUPLICATE_SCORING = True
USE_PHASH_DUPLICATE_SCORING = True

PHASH_HAMMING_THRESHOLD = 8

# Soft-Threshold: CLIP-Aehnlichkeit ab der ein Bild als near-duplicate gilt,
# WENN zusaetzlich Metadaten-Match (gleicher Shot + gleiche Clothing/BG/Session)
# zustimmt. Frueher 0.985 - praktisch nie ausgeloest. 0.96 fängt semantische
# Naehe ohne dass unterschiedliche Outfits faelschlich verschmolzen werden.
CLIP_COSINE_THRESHOLD = 0.96

# Hard-Threshold: Bei sehr hoher CLIP-Aehnlichkeit (>= 0.92) wird ohne
# Metadaten-Bedingung als Duplicate gewertet. Schuetzt vor Style-Cluster-
# Dominanz (z.B. 5 B/W-Closeups derselben Person mit minimal anderer Pose und
# nominell unterschiedlicher Sweater-Beschreibung). Hard-Threshold MUSS unter
# Soft-Threshold liegen.
CLIP_HARD_DUPLICATE_THRESHOLD = 0.92

# Visual-Style-Diversity: Erkennt Konzentration von Bildern mit gleichem
# Bild-Stil (B/W oder dominanter Farbstich) im Final-Set. Soft-Penalty,
# kein Hard-Reject.
ENABLE_VISUAL_STYLE_DIVERSITY = True
VISUAL_STYLE_SOFT_LIMIT = 2          # ab dem 3. Bild gleichen Stils -> Penalty
VISUAL_STYLE_PENALTY_WEIGHT = 8.0    # pro ueberzaehligem Bild gleichen Stils

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
# Export normalization is deliberately separate from content crops.
# False (default): preserve the selected image/crop composition and let the
# trainer bucket mixed aspect ratios. True: normalize only at final export to
# 1024x1024 (headshots) or 832x1216 (medium/full body).
USE_CONTROLLED_BUCKETS = False
# Backward-compatible alias for older config files.
USE_AI_TOOLKIT_CROP_PROFILES = USE_CONTROLLED_BUCKETS

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

# Trainingsziel und Caption-Regeln
# TRAINING_TARGET waehlt die Pipeline/Prompt-Familie. CAPTION_PROFILE bleibt als
# Rueckwaertskompatibilitaets-Alias fuer alte Configs, darf aber nicht mehr aus
# einzelnen Caption-Checkboxen abgeleitet werden.
TRAINING_TARGET = "ernie"  # "ernie" | "z_image_base" | "krea2"
CAPTION_PROFILE = "ernie"  # legacy alias, wird beim Config-Load aus TRAINING_TARGET gesetzt
# How variable visual traits are captioned:
# - canonical_deviations: canonical baseline belongs to the trigger; only deviations are captioned.
# - all_visible_when_variable: once genuine variation is detected, caption every visible state.
VARIABLE_FEATURE_CAPTION_MODE = "canonical_deviations"

# Krea 2 quality-caption mode. Only final selected images are sent for a
# dedicated natural-language caption after the subject profile is known.
USE_KREA_AI_CAPTIONING = True
KREA_CAPTION_MODEL = "gpt-5.6-luna"
USE_KREA_CAPTION_REPAIR = True
KREA_CAPTION_REPAIR_MODEL = "gpt-5.6-terra"
KREA_CAPTION_IMAGE_DETAIL = "high"
KREA_CAPTION_PROMPT_VERSION = "krea2-natural-v5-canon-selection-tattoo-policy"
CAPTION_POLICY = {
    "include_gender_class": True,
    "include_skin_tone": True, 
    "include_body_build": True,
    "include_freckles": True,
    "include_tattoos": True,
    "include_glasses": True,
    "include_glasses_when_variable": False,
    "include_piercings": True,
    "include_makeup": True,
    "include_background": True,
    "include_lighting": True,
    "include_gaze": True,
    "include_expression": True,
    "include_hair_always": True,   
    "include_hair_when_variable": True,
    "include_eye_color_when_variable": True,
    "include_costume_accessories": True,
    "include_beard_always": False,
    "include_beard_when_variable": True,
    "include_mirror_selfie_marker": True,
    "include_eye_color": True,        # ← NEU: Augenfarbe (siehe unten)
    "include_visual_style": True,    # B/W- und Tint-Marker als Caption-Praefix
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

# Identity-/Appearance-Clusterung fuer den Subject-Profile-UI-Bereich.
# Wichtig: diese Rollen sind keine Caption-Tokens. Sie steuern nur, welche
# Bilder bei Phase-3-Export in 01_train_ready kommen und wie stark sie im
# Ranking bevorzugt werden.
ENABLE_IDENTITY_APPEARANCE_CLUSTERING = True
IDENTITY_CLUSTER_SCHEMA_VERSION = "v2"
IDENTITY_CLUSTER_CORE_SCORE_BOOST = 6.0
IDENTITY_CLUSTER_VARIATION_SCORE_BOOST = 1.5
IDENTITY_CLUSTER_BODY_SCORE_BOOST = 2.5
IDENTITY_CLUSTER_MAX_CORE_SHARE = 0.60
IDENTITY_CLUSTER_CORE_OVERFLOW_PENALTY = 18.0
IDENTITY_CLUSTER_TRAIN_ROLES = {"core", "variation", "body_reference"}
IDENTITY_CLUSTER_NONTRAIN_ROLES = {"review", "exclude"}

# Weiche Canon-Repräsentation bei der finalen Auswahl. Diese Logik greift erst
# nach dem bestätigten Subject Profile, weil die kanonische Erscheinung eine
# bewusste Nutzerentscheidung sein kann und nicht zwingend der Statistik folgt.
# Sie verändert niemals die Shot-Type-Quoten und zieht weder Review- noch
# Reject-Bilder automatisch in den Trainingssatz.
ENABLE_CANON_REPRESENTATION_BONUS = True
CANON_REPRESENTATION_TARGET = 3
CANON_REPRESENTATION_MAX_QUALITY_GAP = 5.0
CANON_REPRESENTATION_BONUS_SCHEDULE = [6.0, 4.0, 2.0, 1.0, 0.5]

# Body build soll nicht verworfen werden, nur weil das Dataset headshot-lastig
# ist, wenn wenigstens einige brauchbare Medium-/Fullbody-Bilder vorhanden sind.
PROFILE_BODY_BUILD_MIN_ABSOLUTE = 3
PROFILE_BODY_BUILD_MIN_FRACTION = 0.30
PROFILE_BODY_PRIORITY_SAMPLE_MAX = 24

# Normalizer-Modell: Terra balances cross-image consistency and cost.
PROFILE_NORMALIZER_MODEL = "gpt-5.6-terra"

# Cache-Version fuer zentrale Subject-Profile. Bei Aenderungen am
# Profile-Schema oder an der Normalizer-Logik inkrementieren.
#   v1: initial (Phase 2)
#   v2: confidence ist jetzt ein Objekt {level, reasoning, outliers}
#       statt nur ein String. Alte Cache-Eintraege werden invalidiert.
#   v3: robustere Brillenlogik. Sonnenbrillen duerfen nicht mehr durch die
#       kanonische Profil-Brille ueberschrieben werden.
#   v7: Erweiterte Profile-Vokabulare inkl. hair_length und
#       body_height_impression; Body-Build ersetzt stocky durch broad_build.
PROFILE_CACHE_SCHEMA_VERSION = "v13"

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

# Medium rescue crop: a separate content-recovery mechanism for weak full-body
# images. It tries to retain head, shoulders, torso and hips instead of turning
# every distant image into a square face crop.
ENABLE_MEDIUM_RESCUE_CROP = True
MEDIUM_RESCUE_MIN_GAIN = 4.0
MEDIUM_RESCUE_TRIGGER_COMPOSITION_MAX = 65.0
MEDIUM_RESCUE_MIN_FACE_PX = 90
MEDIUM_RESCUE_TARGET_ASPECT = 0.80  # 4:5 content crop; bucket normalization is later/optional

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


def _normalize_training_target_bootstrap(value: Optional[str]) -> str:
    v = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if v in {"z_image_base", "zimage", "z_image"}:
        return "z_image_base"
    if v in {"krea2", "krea_2", "krea2_character", "krea_2_character"}:
        return "krea2"
    return "ernie"


def _caption_profile_for_target_bootstrap(value: Optional[str]) -> str:
    target = _normalize_training_target_bootstrap(value)
    return "krea2_character" if target == "krea2" else target


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

# New setting wins when explicitly present in the UI config. Older configs can
# still provide the legacy name. A fresh installation keeps controlled buckets
# disabled and therefore preserves the natural composition by default.
if isinstance(globals().get("_ui_cfg"), dict) and "USE_CONTROLLED_BUCKETS" in _ui_cfg:
    USE_CONTROLLED_BUCKETS = bool(_ui_cfg.get("USE_CONTROLLED_BUCKETS"))
elif isinstance(globals().get("_ui_cfg"), dict) and "USE_AI_TOOLKIT_CROP_PROFILES" in _ui_cfg:
    USE_CONTROLLED_BUCKETS = bool(_ui_cfg.get("USE_AI_TOOLKIT_CROP_PROFILES"))
USE_AI_TOOLKIT_CROP_PROFILES = bool(USE_CONTROLLED_BUCKETS)

# TRAINING_TARGET is the single source of truth for prompt family and caption
# engine. Legacy CAPTION_PROFILE values are migrated, but individual caption
# checkboxes never change the target.
if isinstance(globals().get("_ui_cfg"), dict) and "TRAINING_TARGET" in _ui_cfg:
    TRAINING_TARGET = _normalize_training_target_bootstrap(_ui_cfg.get("TRAINING_TARGET"))
elif isinstance(globals().get("_ui_cfg"), dict) and "CAPTION_PROFILE" in _ui_cfg:
    TRAINING_TARGET = _normalize_training_target_bootstrap(_ui_cfg.get("CAPTION_PROFILE"))
else:
    TRAINING_TARGET = _normalize_training_target_bootstrap(TRAINING_TARGET)
CAPTION_PROFILE = _caption_profile_for_target_bootstrap(TRAINING_TARGET)
USE_KREA_AI_CAPTIONING = TRAINING_TARGET == "krea2"

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
# Wichtig: shot_type bleibt fuer Auswahl/Quoten unveraendert bei
# headshot | medium | full_body. frame_subtype ist nur ein Zusatzfeld fuer
# Caption/Analyse und beeinflusst die Zielverteilung nicht.
HAIR_FORM_VOCAB: List[str] = [
    "loose_straight", "loose_wavy", "loose_curly", "loose_coily",
    "afro_natural",
    "ponytail", "low_ponytail", "high_ponytail",
    "pigtails", "two_braids", "single_braid",
    "box_braids", "knotless_braids", "cornrows", "dreadlocks",
    "bun", "low_bun", "high_bun", "messy_bun",
    "updo", "half_up", "pulled_back",
    "pixie_cut", "bob_cut", "lob_cut", "short_cut",
    "buzz_cut", "shaved_head", "undercut", "side_shaved",
    "bangs", "curtain_bangs", "covered_hair", "other",
]

HAIR_LENGTH_VOCAB: List[str] = [
    "shaved", "very_short", "short", "chin_length", "shoulder_length",
    "medium_length", "long", "very_long", "not_visible", "unclear",
]

HAIR_COLOR_VOCAB: List[str] = [
    "black", "dark_brown", "brown", "light_brown",
    "dark_blonde", "blonde", "platinum", "strawberry_blonde",
    "red", "copper", "auburn", "burgundy",
    "gray", "silver", "white",
    "blue", "pink", "purple", "green",
    "dyed_other", "multicolor", "ombre", "highlights",
    "not_visible", "unclear",
]

EYE_COLOR_VOCAB: List[str] = [
    "blue", "blue_green", "green", "hazel", "brown", "dark_brown",
    "gray", "gray_blue", "amber", "not_visible", "unclear",
]

SKIN_TONE_VOCAB: List[str] = [
    "very_fair", "fair", "light", "medium", "tan", "olive",
    "brown", "dark", "deep", "unclear",
]

BODY_BUILD_VOCAB: List[str] = [
    "petite", "slim", "average", "athletic", "curvy",
    "plus_size", "muscular", "broad_build", "unclear",
]

BODY_HEIGHT_IMPRESSION_VOCAB: List[str] = [
    "short", "average_height", "tall", "unclear",
]

MAKEUP_INTENSITY_VOCAB: List[str] = [
    "none", "minimal", "natural", "defined", "full", "dramatic",
    "stage_makeup", "costume_makeup", "face_paint", "unclear",
]

LIGHTING_TYPE_VOCAB: List[str] = [
    "studio_softbox", "studio_ringlight", "studio_other",
    "natural_outdoor_sun", "natural_outdoor_overcast", "harsh_direct_sun",
    "golden_hour", "natural_indoor_window",
    "indoor_artificial", "camera_flash", "mixed", "low_light",
    "backlit", "neon_colored", "colored_stage_light", "other",
]

BACKGROUND_TYPE_VOCAB: List[str] = [
    "studio_plain", "studio_textured",
    "indoor_room", "indoor_bathroom", "indoor_kitchen",
    "indoor_bedroom", "indoor_office", "indoor_gym",
    "outdoor_urban", "outdoor_nature", "outdoor_forest",
    "outdoor_beach", "outdoor_snow", "outdoor_mountain",
    "outdoor_event", "outdoor_other",
    "vehicle_interior", "public_transport",
    "mirror_selfie", "transparent_or_isolated", "other",
]

GLASSES_FRAME_SHAPE_VOCAB: List[str] = [
    "round", "square", "rectangular", "oval", "aviator", "cat_eye",
    "oversized", "rimless", "semi_rimless", "browline",
    "geometric", "wayfarer", "shield", "other",
]

GLASSES_FRAME_MATERIAL_VOCAB: List[str] = [
    "wire_frame", "metal_frame", "plastic_frame", "acetate_frame",
    "rimless", "semi_rimless", "mixed_material", "unclear",
]

GLASSES_LENS_TYPE_VOCAB: List[str] = [
    "clear_lenses", "tinted_lenses", "sunglasses",
    "reflective_lenses", "blue_light_lenses", "unclear",
]

FRAME_SUBTYPE_VOCAB: List[str] = [
    "close_up", "portrait", "selfie", "mirror_selfie",
    "three_quarter_body", "full_body", "faceless_body",
    "detail_only", "unclear",
]

GAZE_VOCAB: List[str] = [
    "looking_at_camera", "looking_left", "looking_right",
    "looking_up", "looking_down", "looking_away",
    "eyes_closed", "partly_closed", "unclear",
]

EXPRESSION_VOCAB: List[str] = [
    "neutral", "slight_smile", "smile", "wide_smile",
    "serious", "pensive", "playful", "laughing",
    "surprised", "sad", "angry", "duckface",
    "winking", "eyes_closed", "other",
]

OCCLUSION_TYPE_VOCAB: List[str] = [
    "none", "hair_covering_face", "hand_covering_face",
    "object_covering_face", "sunglasses_occluding_eyes",
    "mask", "hat_shadow", "motion_blur", "crop_cutoff",
    "face_partly_out_of_frame", "other",
]

VISUAL_STYLE_VOCAB: List[str] = [
    "normal_color", "black_and_white", "sepia",
    "warm_tinted", "cool_tinted", "green_tinted", "blue_tinted",
    "high_contrast", "low_contrast", "beauty_filter",
    "heavy_smoothing", "vintage_filter", "screenshot", "other",
]

EYE_APPEARANCE_VOCAB: List[str] = [
    "natural_eyes", "colored_contact_lenses", "circle_lenses",
    "cosmetic_lenses", "unnatural_eye_color", "unclear",
]

LOOK_CONTEXT_VOCAB: List[str] = [
    "regular_photo", "fashion", "glamour", "gyaru_style",
    "cosplay", "character_costume", "fantasy_costume",
    "stage_costume", "swimwear_costume", "lingerie_costume",
    "unclear",
]

MAKEUP_STYLE_VOCAB: List[str] = [
    "natural_makeup", "gyaru_makeup", "cosplay_makeup",
    "anime_inspired_makeup", "dramatic_eyeliner",
    "smoky_eye_makeup", "false_eyelashes", "glossy_lips",
    "face_paint", "fantasy_makeup", "unclear",
]

COSTUME_ACCESSORY_VOCAB: List[str] = [
    "animal_ears", "cat_ears", "fox_ears", "bunny_ears",
    "elf_ears", "pointed_ears", "horns", "antlers",
    "wings", "feather_headpiece", "headband", "hair_bow",
    "hair_ribbon", "forehead_jewel", "tiara", "crown",
    "halo", "veil", "hood", "hat", "cap", "helmet",
    "mask", "choker", "collar", "necklace",
    "gloves", "arm_guards", "wrist_cuffs",
    "fantasy_armor", "shoulder_armor",
    "prop_weapon", "prop_sword", "prop_gun", "prop_staff",
    "prop_bottle", "prop_book", "other_prop",
    "none_visible", "unclear",
]

PROFILE_APPEARANCE_MODE_VOCAB: List[str] = [
    "natural_identity", "fashion_identity",
    "cosplay_identity", "high_variation_model_identity",
]

# --- Tattoo-Locations als kontrolliertes ENUM ----------------------------
# Wird im Audit-Schema als Strict-ENUM verwendet, weil Tattoo-Lokationen
# fuer die Inventar-Deduplizierung in Phase 2 deterministisch gleich
# benannt sein muessen.
TATTOO_LOCATION_ENUM: List[str] = [
    "forearm_left", "forearm_right",
    "inner_forearm_left", "inner_forearm_right",
    "upper_arm_left", "upper_arm_right",
    "inner_upper_arm_left", "inner_upper_arm_right",
    "elbow_left", "elbow_right",
    "hand_left", "hand_right",
    "wrist_left", "wrist_right",
    "shoulder_left", "shoulder_right",
    "neck_left", "neck_right", "neck_back",
    "chest_upper", "chest_left", "chest_right",
    "chest_sternum", "sternum_underboob",
    "collarbone_left", "collarbone_right",
    "ribcage_left", "ribcage_right",
    "abdomen", "hip_left", "hip_right",
    "back_upper", "upper_back_left", "upper_back_right",
    "back_lower", "lower_back_left", "lower_back_right",
    "buttock_left", "buttock_right",
    "thigh_left", "thigh_right",
    "knee_left", "knee_right",
    "shin_left", "shin_right",
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
    "ear_conch_left", "ear_conch_right",
    "ear_daith_left", "ear_daith_right",
    "ear_rook_left", "ear_rook_right",
    "ear_industrial_left", "ear_industrial_right",
    "ear_snug_left", "ear_snug_right",
    "ear_gauge_left", "ear_gauge_right",
    "nose_left", "nose_right", "nose_septum",
    "nose_high_nostril_left", "nose_high_nostril_right",
    "nose_bridge", "eyebrow_left", "eyebrow_right",
    "lip_upper", "lip_lower", "lip_corner_left", "lip_corner_right",
    "lip_labret", "lip_medusa", "lip_monroe_left", "lip_monroe_right",
    "cheek_left", "cheek_right",
    "tongue", "navel", "nipple_left", "nipple_right", "other",
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


def normalize_training_target(value: Optional[str]) -> str:
    v = normalize_text(value)
    if v in {"ernie", "shared_compact"}:
        return "ernie"
    if v in {"z_image_base", "z-image_base", "zimage", "z_image"}:
        return "z_image_base"
    if v in {"krea2", "krea_2", "krea2_character", "krea_2_character"}:
        return "krea2"
    return "ernie"


def caption_profile_for_training_target(value: Optional[str]) -> str:
    target = normalize_training_target(value)
    return "krea2_character" if target == "krea2" else target


def normalize_caption_profile(value: Optional[str]) -> str:
    # Legacy helper: old configs/reports may still contain caption-profile names.
    return caption_profile_for_training_target(value)


def training_target_audit_guidance(target: Optional[str] = None) -> str:
    t = normalize_training_target(target or globals().get("TRAINING_TARGET", "ernie"))
    if t == "krea2":
        return (
            "TRAINING TARGET: Krea 2 character LoRA. Preserve natural-language-relevant "
            "scene facts, full-person usefulness, body orientation, camera framing, temporary "
            "appearance changes and exact visible accessories. Keep stable identity facts raw; "
            "the Subject Profile decides later whether they belong to the trigger or caption."
        )
    if t == "z_image_base":
        return (
            "TRAINING TARGET: Z-Image Base character LoRA. Prioritize compact, controllable "
            "visual attributes and reliable separation of stable identity from variable traits."
        )
    return (
        "TRAINING TARGET: ERNIE Image character LoRA. Capture visible identity anchors as well "
        "as scene-specific details because the downstream caption policy is intentionally explicit."
    )


def training_target_profile_guidance(target: Optional[str] = None) -> str:
    t = normalize_training_target(target or globals().get("TRAINING_TARGET", "ernie"))
    if t == "krea2":
        return (
            "For Krea 2, treat the Subject Profile as the canonical identity and terminology layer. "
            "Record all recurring piercings/accessories in inventories even when they are not canonical."
        )
    if t == "z_image_base":
        return "For Z-Image Base, favor a clean canonical identity and explicit variable-feature policies."
    return "For ERNIE, retain reliable visible identity anchors for explicit downstream captions."


def enforce_caption_policy_profile(profile: Optional[str], policy: Dict[str, Any]) -> Dict[str, Any]:
    """Return the explicit caption policy without changing the training target.

    Target-specific recommendations are applied by the UI preset. Once the user
    changes a checkbox, that choice remains authoritative; it may not switch the
    caption engine and is not silently re-enabled/disabled here. Missing keys are
    filled conservatively for backward compatibility only.
    """
    result = dict(policy or {})
    for key in (
        "include_gender_class", "include_skin_tone", "include_body_build",
        "include_freckles", "include_tattoos", "include_glasses",
        "include_glasses_when_variable", "include_piercings", "include_makeup",
        "include_background", "include_lighting", "include_gaze",
        "include_expression", "include_hair_always", "include_hair_when_variable",
        "include_eye_color_when_variable", "include_costume_accessories",
        "include_beard_always", "include_beard_when_variable",
        "include_mirror_selfie_marker", "include_eye_color", "include_visual_style",
    ):
        result.setdefault(key, False)
    return result


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


OPENAI_PRICING_PER_1M_TOKENS: Dict[str, Dict[str, float]] = {
    # Keep this table small and explicit. Unknown/custom models simply suppress
    # the estimated-cost display instead of producing a misleading number.
    "gpt-5.6-luna": {"input": 1.00, "output": 6.00},
    "gpt-5.6-terra": {"input": 2.50, "output": 15.00},
    "gpt-5.6-sol": {"input": 5.00, "output": 30.00},
    "gpt-5.6": {"input": 5.00, "output": 30.00},
}

_OPENAI_USAGE_LOCK = threading.Lock()
OPENAI_USAGE_STATS: Dict[str, Any] = {
    "requests": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "by_model": {},
    "by_phase": {},
}


class OpenAITokenBudgetExceeded(RuntimeError):
    """Raised when the configured OpenAI token budget for this run is exhausted."""


def openai_token_limit_enabled() -> bool:
    try:
        return int(OPENAI_TOKEN_LIMIT_TOTAL or 0) > 0
    except Exception:
        return False


def current_openai_total_tokens() -> int:
    with _OPENAI_USAGE_LOCK:
        return int(OPENAI_USAGE_STATS.get("total_tokens", 0))


def remaining_openai_token_budget() -> Optional[int]:
    if not openai_token_limit_enabled():
        return None
    limit = int(OPENAI_TOKEN_LIMIT_TOTAL or 0)
    return max(0, limit - current_openai_total_tokens())


def assert_openai_token_budget_available(context: str = "") -> None:
    if not openai_token_limit_enabled():
        return
    limit = int(OPENAI_TOKEN_LIMIT_TOTAL or 0)
    used = current_openai_total_tokens()
    if used >= limit:
        detail = f" during {context}" if context else ""
        raise OpenAITokenBudgetExceeded(
            f"OpenAI token limit reached{detail}: {used:,} / {limit:,} tokens used."
        )


def _usage_bucket_key(value: Optional[str], fallback: str) -> str:
    text = normalize_text(value)
    return text or fallback


def _extract_usage_int(usage: Dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return 0


def record_openai_usage(model: str, phase_label: str, data: Dict[str, Any]) -> Dict[str, int]:
    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    if not isinstance(usage, dict):
        usage = {}

    input_tokens = _extract_usage_int(usage, "input_tokens", "prompt_tokens")
    output_tokens = _extract_usage_int(usage, "output_tokens", "completion_tokens")
    total_tokens = _extract_usage_int(usage, "total_tokens")
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens

    request_usage = {
        "requests": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

    model_key = _usage_bucket_key(model, "unknown_model")
    phase_key = _usage_bucket_key(phase_label, "responses_api")

    with _OPENAI_USAGE_LOCK:
        OPENAI_USAGE_STATS["requests"] += 1
        OPENAI_USAGE_STATS["input_tokens"] += input_tokens
        OPENAI_USAGE_STATS["output_tokens"] += output_tokens
        OPENAI_USAGE_STATS["total_tokens"] += total_tokens

        model_bucket = OPENAI_USAGE_STATS["by_model"].setdefault(
            model_key,
            {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        phase_bucket = OPENAI_USAGE_STATS["by_phase"].setdefault(
            phase_key,
            {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

        for key, value in request_usage.items():
            model_bucket[key] += value
            phase_bucket[key] += value

    return request_usage


def estimate_openai_cost_usd() -> Optional[float]:
    total_cost = 0.0
    with _OPENAI_USAGE_LOCK:
        by_model = dict(OPENAI_USAGE_STATS.get("by_model", {}))

    for model_name, usage in by_model.items():
        pricing = OPENAI_PRICING_PER_1M_TOKENS.get(model_name)
        if not pricing:
            return None
        input_price = float(pricing.get("input", 0.0))
        output_price = float(pricing.get("output", 0.0))
        total_cost += (float(usage.get("input_tokens", 0)) / 1_000_000.0) * input_price
        total_cost += (float(usage.get("output_tokens", 0)) / 1_000_000.0) * output_price

    return round(total_cost, 6)


def build_openai_usage_summary() -> Dict[str, Any]:
    with _OPENAI_USAGE_LOCK:
        summary = {
            "requests": int(OPENAI_USAGE_STATS.get("requests", 0)),
            "input_tokens": int(OPENAI_USAGE_STATS.get("input_tokens", 0)),
            "output_tokens": int(OPENAI_USAGE_STATS.get("output_tokens", 0)),
            "total_tokens": int(OPENAI_USAGE_STATS.get("total_tokens", 0)),
            "by_model": json.loads(json.dumps(OPENAI_USAGE_STATS.get("by_model", {}))),
            "by_phase": json.loads(json.dumps(OPENAI_USAGE_STATS.get("by_phase", {}))),
        }

    estimated_cost = estimate_openai_cost_usd()
    summary["estimated_cost_usd"] = estimated_cost
    token_limit_total = int(OPENAI_TOKEN_LIMIT_TOTAL or 0) if openai_token_limit_enabled() else 0
    summary["token_limit_total"] = token_limit_total
    summary["token_limit_enabled"] = token_limit_total > 0
    summary["token_limit_reached"] = token_limit_total > 0 and summary["total_tokens"] >= token_limit_total
    summary["token_limit_remaining"] = (
        max(0, token_limit_total - summary["total_tokens"])
        if token_limit_total > 0
        else None
    )
    return summary


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


def is_crop_variant(item: Dict[str, Any]) -> bool:
    return bool(item.get("is_smart_crop") or item.get("is_rescue_crop"))


def generate_medium_rescue_crop(
    image_path: str,
    face_bbox: Optional[List[int]],
    pose_bbox: Optional[List[int]],
    img_w: int,
    img_h: int,
) -> Tuple[Optional[str], Optional[List[int]]]:
    """Create a 4:5-ish medium-shot candidate from a weak full-body image."""
    if not ENABLE_MEDIUM_RESCUE_CROP or not face_bbox:
        return None, None
    try:
        import tempfile
        fx, fy, fw, fh = [int(v) for v in face_bbox]
        if min(fw, fh) < MEDIUM_RESCUE_MIN_FACE_PX:
            return None, None

        aspect = float(MEDIUM_RESCUE_TARGET_ASPECT or 0.8)
        if pose_bbox:
            px, py, pw, ph = [int(v) for v in pose_bbox]
            top = max(0, min(fy - int(fh * 0.8), py - int(ph * 0.05)))
            bottom = min(img_h, max(fy + int(fh * 5.5), py + int(ph * 0.72)))
            center_x = px + pw // 2
        else:
            top = max(0, fy - int(fh * 0.9))
            bottom = min(img_h, top + int(max(fh * 7.0, img_h * 0.55)))
            center_x = fx + fw // 2

        crop_h = max(1, bottom - top)
        crop_w = int(round(crop_h * aspect))
        if crop_w > img_w:
            crop_w = img_w
            crop_h = int(round(crop_w / aspect))
        if crop_h > img_h:
            crop_h = img_h
            crop_w = int(round(crop_h * aspect))

        x1 = max(0, min(center_x - crop_w // 2, img_w - crop_w))
        y1 = max(0, min(top, img_h - crop_h))
        bbox = [x1, y1, crop_w, crop_h]

        with Image.open(image_path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
            cropped = pil_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="medium_rescue_")
            os.close(tmp_fd)
            cropped.save(tmp_path, "JPEG", quality=100)
            return tmp_path, bbox
    except Exception:
        return None, None


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

def _classify_color_tint(
    arr: np.ndarray,
    saturation_mean: float,
) -> Tuple[str, float]:
    """Erkennt einen dominanten Farbstich im Bild.

    Args:
        arr: HxWx3 RGB-Array (float, 0..255).
        saturation_mean: Bereits berechnete mittlere HSV-Saettigung.

    Returns:
        (label, strength) - label ist eines von:
            "" (kein Tint), "blue", "warm", "green", "purple", "sepia"
        strength ist 0..1 (geclamped).

    Hinweise:
        - Naturwarme/-kalte Beleuchtung (asym < TINT_MIN_ASYMMETRY) gibt "".
        - Sehr niedrige Saettigung (< 0.05) gibt "" - das ist Grayscale-Territorium
          und wird separat ueber is_grayscale_filter geflagged.
        - "warm" ist der Sammelbegriff fuer R-dominante Tints, die NICHT die
          Sepia-Kriterien erfuellen (z.B. Instagram-Warm-Filter).
    """
    if arr.size == 0:
        return ("", 0.0)
    if saturation_mean < 0.05:
        # Grayscale-Territorium - separater Pfad
        return ("", 0.0)

    mean_r = float(np.mean(arr[:, :, 0]))
    mean_g = float(np.mean(arr[:, :, 1]))
    mean_b = float(np.mean(arr[:, :, 2]))
    avg = (mean_r + mean_g + mean_b) / 3.0
    if avg < 1e-6:
        return ("", 0.0)

    rel_r = mean_r / avg
    rel_g = mean_g / avg
    rel_b = mean_b / avg
    asymmetry = max(rel_r, rel_g, rel_b) - min(rel_r, rel_g, rel_b)

    if asymmetry < float(TINT_MIN_ASYMMETRY):
        return ("", 0.0)

    # Strength normalisieren
    span = max(1e-6, float(TINT_STRONG_ASYMMETRY) - float(TINT_MIN_ASYMMETRY))
    strength = min(1.0, max(0.0, (asymmetry - float(TINT_MIN_ASYMMETRY)) / span))

    # Sepia: R > G > B mit deutlicher R-B-Differenz im absoluten Mittel.
    # Reihenfolge der Pruefungen wichtig: Sepia VOR Warm, weil Sepia ein
    # Spezialfall von "R-dominant" ist.
    sepia_r_b_delta = mean_r - mean_b
    if (
        rel_r > rel_g
        and rel_g > rel_b
        and sepia_r_b_delta >= float(TINT_SEPIA_MIN_R_B_DELTA)
    ):
        return ("sepia", strength)

    # Reine Channel-Dominanz
    if rel_b > rel_r and rel_b > rel_g:
        return ("blue", strength)
    if rel_g > rel_r and rel_g > rel_b:
        return ("green", strength)
    if rel_r > rel_g and rel_r > rel_b:
        # R-dominant ohne Sepia-Kriterium = warmer Filter
        return ("warm", strength)
    # R+B hoch, G niedrig = magenta/purple (z.B. Disco/Konzert-Lighting)
    if rel_r > rel_g and rel_b > rel_g:
        return ("purple", strength)

    return ("", 0.0)

def local_colorfulness_metrics(image_path: str) -> Dict[str, Any]:
    """Erkennt nahezu farblose Bilder UND Bilder mit dominantem Farbstich
    lokal und guenstig.

    Drei Output-Signale:
    - is_grayscale_filter (bool): praktisch farblos (B/W oder reines Greyscale)
    - color_tint_label (str): "" | "blue" | "warm" | "green" | "purple" | "sepia"
    - color_tint_strength (float, 0..1): wie stark der Tint ausgeprägt ist

    is_grayscale_filter ist konservativ - es muessen sowohl die mittlere
    HSV-Saettigung als auch die mittlere RGB-Kanalabweichung sehr niedrig sein.

    color_tint_label wird nur gesetzt, wenn der Tint stark genug ist
    (strength >= TINT_MIN_STRENGTH_FOR_CAPTION). Naturwarme/-kalte Beleuchtung
    bleibt unmarkiert.
    """
    result: Dict[str, Any] = {
        "color_saturation_mean": 0.0,
        "color_channel_delta_mean": 0.0,
        "is_grayscale_filter": False,
        "color_tint_label": "",
        "color_tint_strength": 0.0,
    }
    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        # Fuer die Statistik klein rechnen: schneller, aber stabil genug.
        thumb = img.copy()
        thumb.thumbnail((256, 256), Image.Resampling.BILINEAR)
        arr = np.asarray(thumb).astype(np.float32)
        if arr.size == 0:
            return result

        rgb = arr / 255.0
        maxc = rgb.max(axis=2)
        minc = rgb.min(axis=2)
        sat = np.where(maxc > 1e-6, (maxc - minc) / np.maximum(maxc, 1e-6), 0.0)

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        channel_delta = (np.abs(r - g) + np.abs(r - b) + np.abs(g - b)) / 3.0

        saturation_mean = float(np.mean(sat))
        channel_delta_mean = float(np.mean(channel_delta))

        strict_grayscale = (
            saturation_mean <= float(GRAYSCALE_SATURATION_THRESHOLD)
            and channel_delta_mean <= float(GRAYSCALE_CHANNEL_DELTA_THRESHOLD)
        )

        # Pixel-share fallback:
        # A visually B/W image can fail the mean channel-delta threshold by a tiny margin
        # because of JPEG compression, social-media processing, or slightly tinted wall/tile areas.
        # We only accept the relaxed path if most pixels are still near channel-equal.
        near_mono_pixel_share = float(
            np.mean(channel_delta <= float(GRAYSCALE_PIXEL_DELTA_THRESHOLD))
        )

        relaxed_grayscale = (
            saturation_mean <= float(GRAYSCALE_RELAXED_SATURATION_THRESHOLD)
            and channel_delta_mean <= float(GRAYSCALE_RELAXED_CHANNEL_DELTA_THRESHOLD)
            and near_mono_pixel_share >= float(GRAYSCALE_PIXEL_SHARE_THRESHOLD)
        )

        is_grayscale = strict_grayscale or relaxed_grayscale

        result["color_saturation_mean"] = round(saturation_mean, 4)
        result["color_channel_delta_mean"] = round(channel_delta_mean, 3)
        result["is_grayscale_filter"] = bool(is_grayscale)

        # Tint nur wenn nicht schon Grayscale
        if not is_grayscale:
            tint_label, tint_strength = _classify_color_tint(arr, saturation_mean)
            # Erst ab MIN_STRENGTH_FOR_CAPTION wirklich als Tint flaggen,
            # darunter bleibt das Bild als "neutral" gewertet.
            if tint_label and tint_strength >= float(TINT_MIN_STRENGTH_FOR_CAPTION):
                result["color_tint_label"] = tint_label
                result["color_tint_strength"] = round(tint_strength, 3)
        return result
    except Exception:
        return result

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
        "color_saturation_mean": 0.0,
        "color_channel_delta_mean": 0.0,
        "is_grayscale_filter": False,
        "color_tint_label": "",
        "color_tint_strength": 0.0,
    }
    metrics.update(local_colorfulness_metrics(image_path))

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
#   v10: Audit/Profile/Caption-Pipeline um freckles_description erweitert.
#        Freckles werden als flexibler, sichtbarkeitsabhaengiger Marker
#        behandelt und muessen in alten Caches neu erhoben werden.
#   v11: Erweiterte Profil-Vokabulare und Aux-Felder: hair_length,
#        body_height_impression, frame_subtype, gaze/expression categories,
#        occlusion_type, visual_style_type sowie Brillenmaterial/-linsentyp.
#   v12: Cosplay-/High-Variation-Felder ohne Herkunftserkennung:
#        eye_appearance, look_context, makeup_style, costume_accessories
#        plus datasetweite Hair-/Eye-Variability-Policies fuer Stable Profile.
AUDIT_CACHE_SCHEMA_VERSION = "v15"
EARLY_RESULT_CACHE_SCHEMA_VERSION = "v1"


def audit_cache_key(
    base_hash: str,
    model: str,
    variant: str = "audit",
    reasoning_effort: Optional[str] = None,
) -> str:
    if reasoning_effort is None:
        reasoning_effort = (
            REVIEW_ESCALATION_REASONING_EFFORT
            if "escalation" in str(variant).lower()
            else AUDIT_REASONING_EFFORT
        )
    raw = "|".join([
        AUDIT_CACHE_SCHEMA_VERSION,
        str(variant),
        str(base_hash),
        (model or "").strip().lower(),
        str(reasoning_effort or "none"),
        normalize_training_target(globals().get("TRAINING_TARGET", "ernie")),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def early_result_cache_path() -> str:
    return os.path.join(CACHE_DIR, "early_results.json")


def early_result_cache_settings() -> Dict[str, Any]:
    return {
        "HARD_MIN_SIDE_PX": int(HARD_MIN_SIDE_PX),
        "USE_MIN_FILESIZE_FILTER": bool(USE_MIN_FILESIZE_FILTER),
        "HARD_MIN_FILESIZE_KB": float(HARD_MIN_FILESIZE_KB),
        "USE_EARLY_PHASH_DEDUP": bool(USE_EARLY_PHASH_DEDUP),
        "USE_PHASH_DUPLICATE_SCORING": bool(USE_PHASH_DUPLICATE_SCORING),
        "USE_EARLY_PHASH_LOOP1": bool(globals().get("USE_EARLY_PHASH_LOOP1", True)),
        "EARLY_PHASH_HAMMING_THRESHOLD_1": int(globals().get("EARLY_PHASH_HAMMING_THRESHOLD_1", 1)),
        "EARLY_PHASH_KEEP_PER_GROUP_1": int(globals().get("EARLY_PHASH_KEEP_PER_GROUP_1", 1)),
        "EARLY_PHASH_LOOP1_PREFER_RESOLUTION": bool(globals().get("EARLY_PHASH_LOOP1_PREFER_RESOLUTION", True)),
        "USE_EARLY_PHASH_LOOP2": bool(globals().get("USE_EARLY_PHASH_LOOP2", True)),
        "EARLY_PHASH_HAMMING_THRESHOLD_2": int(globals().get("EARLY_PHASH_HAMMING_THRESHOLD_2", EARLY_PHASH_HAMMING_THRESHOLD)),
        "EARLY_PHASH_KEEP_PER_GROUP_2": int(globals().get("EARLY_PHASH_KEEP_PER_GROUP_2", EARLY_PHASH_KEEP_PER_GROUP)),
        "EARLY_PHASH_HAMMING_THRESHOLD": int(EARLY_PHASH_HAMMING_THRESHOLD),
        "EARLY_PHASH_KEEP_PER_GROUP": int(EARLY_PHASH_KEEP_PER_GROUP),
    }


def early_result_settings_fingerprint() -> str:
    payload = early_result_cache_settings()
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def dataset_fingerprint(image_paths: List[str]) -> str:
    rows: List[List[Any]] = []
    for path in sorted(image_paths, key=lambda p: p.lower()):
        try:
            rel_path = os.path.relpath(path, INPUT_FOLDER)
        except Exception:
            rel_path = path
        try:
            st = os.stat(path)
            size = int(st.st_size)
            mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
        except Exception:
            size = -1
            mtime_ns = -1
        rows.append([rel_path.replace("\\", "/"), size, mtime_ns])
    raw = json.dumps(rows, ensure_ascii=False, sort_keys=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_cached_early_results(dataset_fp: str, settings_fp: str) -> Optional[Dict[str, Any]]:
    path = early_result_cache_path()
    if not ENABLE_CACHE or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if data.get("schema_version") != EARLY_RESULT_CACHE_SCHEMA_VERSION:
            return None
        if data.get("dataset_fingerprint") != dataset_fp:
            return None
        if data.get("settings_fingerprint") != settings_fp:
            return None
        return data
    except Exception:
        return None


def save_cached_early_results(payload: Dict[str, Any]) -> None:
    if not ENABLE_CACHE:
        return
    path = early_result_cache_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def apply_early_static_rejects(image_paths: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    survivors: List[str] = []
    reject_rows: List[Dict[str, Any]] = []
    for image_path in image_paths:
        original_filename = os.path.basename(image_path)
        try:
            width, height = image_dimensions(image_path)
        except (OSError, UnidentifiedImageError, ValueError) as e:
            reason = "unreadable_or_corrupt_image"
            reject_rows.append({
                "original_filename": original_filename,
                "original_path": image_path,
                "width": 0,
                "height": 0,
                "quality_total": 0,
                "base_status": "reject",
                "final_status": "reject",
                "short_reason": reason,
                "local_override_reasons": [reason],
                "status_notes": [f"image_open_error: {type(e).__name__}: {e}"],
                "selected": False,
                "output_bucket": "",
                "new_basename": "",
            })
            continue

        if min(width, height) < HARD_MIN_SIDE_PX:
            reject_rows.append({
                "original_filename": original_filename,
                "original_path": image_path,
                "width": width,
                "height": height,
                "quality_total": 0,
                "base_status": "reject",
                "final_status": "reject",
                "short_reason": f"hard_pass_too_small_{width}x{height}",
                "status_notes": [],
                "selected": False,
                "output_bucket": "",
                "new_basename": "",
            })
            continue

        if USE_MIN_FILESIZE_FILTER:
            kb = local_filesize_kb(image_path)
            if kb < HARD_MIN_FILESIZE_KB:
                reason = f"filesize_too_small_{kb:.0f}kb"
                reject_rows.append({
                    "original_filename": original_filename,
                    "original_path": image_path,
                    "width": width,
                    "height": height,
                    "quality_total": 0,
                    "base_status": "reject",
                    "final_status": "reject",
                    "short_reason": reason,
                    "local_override_reasons": [reason],
                    "status_notes": [],
                    "selected": False,
                    "output_bucket": "",
                    "new_basename": "",
                })
                continue

        survivors.append(image_path)

    return survivors, reject_rows


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
    raw = "|".join([
        str(trigger_word).strip().lower(),
        str(TRIGGER_CHECK_MODEL).strip().lower(),
        str(TRIGGER_CHECK_REASONING_EFFORT or "none"),
    ])
    suffix = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    key = slugify_filename(trigger_word.lower())
    return os.path.join(TRIGGER_CACHE_DIR, f"{key}_{suffix}.json")


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
    """Collect the complete visible text from a Responses API result.

    A response may contain more than one ``output_text`` content part.  The old
    implementation returned only the first part, which could turn otherwise
    valid structured JSON into an apparently truncated string.
    """
    if response_json.get("NSFW_BLOCKED"):
        return '{"NSFW_BLOCKED": true}'

    parts: List[str] = []
    for item in response_json.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if part.get("type") == "output_text" and part.get("text"):
                parts.append(str(part["text"]))
    if parts:
        return "".join(parts).strip()
    raise ValueError("Kein output_text in Responses-Antwort gefunden.")


def _parse_json_object_text(text: str) -> Dict[str, Any]:
    """Parse one JSON object and tolerate harmless wrappers/trailing text.

    Structured-output responses should already be JSON.  This helper only
    repairs transport/presentation artefacts such as Markdown fences or a
    short trailing explanation.  Incomplete JSON is *not* guessed locally; it
    triggers a real model retry instead.
    """
    raw = str(text or "").strip().lstrip("\ufeff")
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as first_error:
        start = raw.find("{")
        if start < 0:
            raise first_error
        decoder = json.JSONDecoder()
        try:
            value, _end = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            raise first_error
    if not isinstance(value, dict):
        raise ValueError("Structured response is not a JSON object.")
    return value


def _responses_incomplete_reason(response_json: Dict[str, Any]) -> str:
    status = str((response_json or {}).get("status", "") or "").strip().lower()
    if status != "incomplete":
        return ""
    details = (response_json or {}).get("incomplete_details") or {}
    if isinstance(details, dict):
        reason = str(details.get("reason", "") or "").strip()
        return reason or "response_status_incomplete"
    return str(details or "response_status_incomplete")


def _validate_subject_profile_core(profile: Dict[str, Any]) -> None:
    required = ("subject_id", "stable_identity", "confidence", "identity_markers", "normalizer_notes")
    missing = [key for key in required if key not in profile]
    if missing:
        raise ValueError("Subject profile missing required keys: " + ", ".join(missing))
    if not isinstance(profile.get("stable_identity"), dict):
        raise ValueError("Subject profile stable_identity must be an object.")
    if not isinstance(profile.get("confidence"), dict):
        raise ValueError("Subject profile confidence must be an object.")
    if not isinstance(profile.get("identity_markers"), dict):
        raise ValueError("Subject profile identity_markers must be an object.")


def normalize_reasoning_effort_for_model(model: str, effort: Optional[str]) -> Optional[str]:
    """Validate UI reasoning values and keep them compatible with model families."""
    if effort is None:
        return None
    normalized = str(effort).strip().lower()
    if normalized in {"", "auto", "default"}:
        return None
    allowed = {"none", "low", "medium", "high", "xhigh", "max"}
    if normalized not in allowed:
        safe_print(f"   ⚠️ Unknown reasoning effort '{effort}', using 'none'.")
        return "none"

    model_name = str(model or "").strip().lower()
    if normalized == "max" and not model_name.startswith("gpt-5.6"):
        safe_print(
            f"   ⚠️ Reasoning effort 'max' is only used for GPT-5.6 in this curator; "
            f"using 'xhigh' for {model}."
        )
        return "xhigh"
    return normalized


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
            assert_openai_token_budget_available(phase_label)
            request_payload = {"model": model, **payload}
            reasoning_effort = normalize_reasoning_effort_for_model(
                model, request_payload.pop("_reasoning_effort", None)
            )
            if reasoning_effort:
                request_payload["reasoning"] = {"effort": reasoning_effort}

            # Current GPT-5.4/5.5/5.6 reasoning configurations do not need
            # sampling parameters. Remove temperature whenever effort is set.
            model_name = str(model).strip().lower()
            if reasoning_effort and model_name.startswith(("gpt-5.4", "gpt-5.5", "gpt-5.6")):
                request_payload.pop("temperature", None)

            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                json=request_payload,
                timeout=180,
            )
            if response.status_code >= 400:
                try:
                    err = response.json()
                except Exception:
                    err = {"error": {"message": response.text}}
                message = err.get("error", {}).get("message", f"HTTP {response.status_code}")
                raise RuntimeError(message)
            data = response.json()
            request_usage = record_openai_usage(model, phase_label, data)
            stop_phase_heartbeat(attempt_label, started_at, stop_event, thread, success=True)
            safe_print(
                f"   ↳ API response ok: status={response.status_code} | phase={phase_label}"
            )
            safe_print(
                "   💰 OpenAI usage: "
                f"req+={request_usage['requests']} | "
                f"in+={request_usage['input_tokens']:,} | "
                f"out+={request_usage['output_tokens']:,} | "
                f"total+={request_usage['total_tokens']:,}"
            )
            if openai_token_limit_enabled():
                limit = int(OPENAI_TOKEN_LIMIT_TOTAL or 0)
                used = current_openai_total_tokens()
                remaining = max(0, limit - used)
                safe_print(
                    f"   🧮 OpenAI token budget: used={used:,} / {limit:,} | remaining={remaining:,}"
                )
                if used >= limit:
                    raise OpenAITokenBudgetExceeded(
                        f"OpenAI token limit reached after {phase_label}: {used:,} / {limit:,} tokens used."
                    )
            return data
        except OpenAITokenBudgetExceeded:
            stop_phase_heartbeat(attempt_label, started_at, stop_event, thread, success=False)
            raise
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
        "_reasoning_effort": TRIGGER_CHECK_REASONING_EFFORT,
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
            "freckles_description": {
                "type": "string",
                "description": "Visible freckles of the main subject, if present and discernible. Use short factual phrases like 'light freckles across the nose and cheeks'. Empty string if no freckles are visible or they cannot be determined reliably."
            },
            "skin_tone": {"type": "string"},
            "eye_color": {
                "type": "string",
                "description": "Eye color of the main subject. Use one of the controlled values where possible: blue, blue_green, green, hazel, brown, dark_brown, gray, gray_blue, amber. Empty string only if eyes are not visible."
            },
            "eye_appearance": {
                "type": "string",
                "description": "Visible eye appearance marker. Use: natural_eyes, colored_contact_lenses, circle_lenses, cosmetic_lenses, unnatural_eye_color, unclear. Do not speculate; use unclear if not confident."
            },
            "body_build": {
                "type": "string",
                "description": "Body build if the body is actually readable. Use one of: petite, slim, average, athletic, curvy, plus_size, muscular, broad_build. Empty string for headshots or unclear views. Do not use 'stocky'."
            },
            "body_height_impression": {
                "type": "string",
                "description": "Only if enough body context is visible. Use one of: short, average_height, tall. Empty string if not readable."
            },
            "hair_length": {
                "type": "string",
                "description": "Hair length using controlled values: shaved, very_short, short, chin_length, shoulder_length, medium_length, long, very_long, not_visible, unclear."
            },
            "frame_subtype": {
                "type": "string",
                "description": "Fine frame subtype for caption/report only. Does NOT replace shot_type. Use: close_up, portrait, selfie, mirror_selfie, three_quarter_body, full_body, faceless_body, detail_only, unclear."
            },
            "gaze_category": {
                "type": "string",
                "description": "Controlled gaze category: looking_at_camera, looking_left, looking_right, looking_up, looking_down, looking_away, eyes_closed, partly_closed, unclear."
            },
            "expression_category": {
                "type": "string",
                "description": "Controlled expression category: neutral, slight_smile, smile, wide_smile, serious, pensive, playful, laughing, surprised, sad, angry, duckface, winking, eyes_closed, other."
            },
            "occlusion_type": {
                "type": "string",
                "description": "Main face/body occlusion category: none, hair_covering_face, hand_covering_face, object_covering_face, sunglasses_occluding_eyes, mask, hat_shadow, motion_blur, crop_cutoff, face_partly_out_of_frame, other."
            },
            "visual_style_type": {
                "type": "string",
                "description": "Image style marker, not image origin. Use: normal_color, black_and_white, sepia, warm_tinted, cool_tinted, green_tinted, blue_tinted, high_contrast, low_contrast, beauty_filter, heavy_smoothing, vintage_filter, screenshot, other."
            },
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
            "body_orientation": {
                "type": "string",
                "enum": ["front", "three_quarter", "side", "back", "mixed", "unclear"],
                "description": "Orientation of the subject's torso/body relative to the camera."
            },
            "camera_angle": {
                "type": "string",
                "enum": ["eye_level", "slightly_high", "high_angle", "slightly_low", "low_angle", "overhead", "dutch_angle", "unclear"],
                "description": "Visible camera angle, using the least extreme accurate category."
            },
            "depth_of_field": {
                "type": "string",
                "enum": ["shallow", "moderate", "deep", "unclear"],
                "description": "How strongly the background is separated by focus blur."
            },
            "action_description": {"type": "string"},
            "prominent_objects": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short names of visually important objects interacting with the subject. Use [] if none."
            },
            "composition_description": {"type": "string"},
            "silhouette_clarity": {
                "type": "string",
                "enum": ["clear", "partly_obscured", "poor", "n_a"]
            },
            "limb_completeness": {
                "type": "string",
                "enum": ["complete", "minor_crop", "major_crop", "not_visible", "n_a"]
            },
            "body_reference_usefulness": {
                "type": "number", "minimum": 0, "maximum": 10,
                "description": "0-10 usefulness specifically for learning body proportions, posture and face-to-body connection. Headshots should score 0-2."
            },
            "perspective_distortion": {
                "type": "string",
                "enum": ["none", "mild", "strong", "unclear"]
            },

            # --- NEU (Phase 1): kategoriale Aux-Felder fuer Profile-Stage ---
            "lighting_type": {
                "type": "string",
                "description": (
                    "Categorical lighting label. Allowed values: studio_softbox, "
                    "studio_ringlight, studio_other, natural_outdoor_sun, "
                    "natural_outdoor_overcast, harsh_direct_sun, golden_hour, "
                    "natural_indoor_window, indoor_artificial, camera_flash, mixed, "
                    "low_light, backlit, neon_colored, colored_stage_light, other. "
                    "Use empty string only if truly indeterminable. "
                    "This is critical for studio-bias correction in skin-tone profiling."
                )
            },
            "background_type": {
                "type": "string",
                "description": (
                    "Categorical background label. Allowed values: studio_plain, "
                    "studio_textured, indoor_room, indoor_bathroom, indoor_kitchen, "
                    "indoor_bedroom, indoor_office, indoor_gym, outdoor_urban, "
                    "outdoor_nature, outdoor_forest, outdoor_beach, outdoor_snow, "
                    "outdoor_mountain, outdoor_event, outdoor_other, vehicle_interior, "
                    "public_transport, mirror_selfie, transparent_or_isolated, other. "
                    "Empty string only if no background visible."
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
                    "minimal, natural, defined, full, dramatic, stage_makeup, "
                    "costume_makeup, face_paint, unclear. "
                    "NEVER use 'or'-phrases like 'minimal or no'. If unclear, pick the "
                    "closest single value."
                )
            },
            "makeup_style": {
                "type": "string",
                "description": "Makeup style if visually clear. Use: natural_makeup, gyaru_makeup, cosplay_makeup, anime_inspired_makeup, dramatic_eyeliner, smoky_eye_makeup, false_eyelashes, glossy_lips, face_paint, fantasy_makeup, unclear."
            },
            "look_context": {
                "type": "string",
                "description": "Overall visible styling/context of this image. Use: regular_photo, fashion, glamour, gyaru_style, cosplay, character_costume, fantasy_costume, stage_costume, swimwear_costume, lingerie_costume, unclear."
            },
            "costume_accessories": {
                "type": "array",
                "description": "Visible costume/headpiece/prop accessories. Use controlled tokens only. Use [] if none are visible or if uncertain.",
                "items": {"type": "string", "enum": COSTUME_ACCESSORY_VOCAB}
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
                    "semi_rimless, browline, geometric, wayfarer, shield, other. "
                    "Empty string if no glasses."
                )
            },
            "glasses_frame_material": {
                "type": "string",
                "description": "If has_glasses_now is true: material/type. One of: wire_frame, metal_frame, plastic_frame, acetate_frame, rimless, semi_rimless, mixed_material, unclear. Empty string if no glasses."
            },
            "glasses_lens_type": {
                "type": "string",
                "description": "If glasses/sunglasses are visible: clear_lenses, tinted_lenses, sunglasses, reflective_lenses, blue_light_lenses, unclear. Empty string if no glasses."
            },
            "glasses_position": {
                "type": "string",
                "enum": ["on_face", "on_head", "held", "hanging_from_clothing", "other", "not_visible"],
                "description": "Where the glasses are in this image. Use on_face only when worn over the eyes. Use not_visible when no glasses are visible."
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
            "freckles_description",
            "skin_tone",
            "eye_color",
            "eye_appearance",
            "body_build",
            "body_height_impression",
            "hair_length",
            "frame_subtype",
            "gaze_category",
            "expression_category",
            "occlusion_type",
            "visual_style_type",
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
            "body_orientation",
            "camera_angle",
            "depth_of_field",
            "action_description",
            "prominent_objects",
            "composition_description",
            "silhouette_clarity",
            "limb_completeness",
            "body_reference_usefulness",
            "perspective_distortion",
            "lighting_type",
            "background_type",
            "hair_texture",
            "makeup_intensity",
            "makeup_style",
            "look_context",
            "costume_accessories",
            "has_glasses_now",
            "glasses_frame_shape",
            "glasses_frame_material",
            "glasses_lens_type",
            "glasses_position",
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
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    schema = build_api_schema()
    image_b64 = resize_and_encode_for_api(image_path)
    chosen_model = (model or AI_MODEL).strip() or AI_MODEL

    instructions = f"""
You are auditing a single image for a person LoRA training dataset for a realistic image model.
Trigger word: "{TRIGGER_WORD}".

{training_target_audit_guidance()}

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
- Describe eye color PRECISELY if visible. Prefer controlled values such as blue, blue_green, green, gray_blue, hazel, brown, dark_brown. Return empty string only if eyes are not visible.
- Describe skin_tone as a neutral factual value (e.g. very_fair, fair, light, medium, tan, olive, brown, dark, deep). Never return empty.
- Describe beard/glasses/piercings/makeup only as visible raw facts.
- Describe freckles only when actually visible. Use short factual phrases like
  'light freckles across the nose and cheeks' or 'prominent facial freckles'.
  If freckles are not visible or cannot be determined reliably because of
  distance, makeup, filter smoothing or lighting, return an empty string.
- body_build: ONLY judge body build when the body is actually visible.
    * On HEADSHOTS (only head and shoulders visible): body_build MUST be empty string "". Do not guess.
    * On medium shots: only fill body_build if torso shape is clearly readable.
    * On full_body shots: judge accurately.
    * Resist the tendency to default to "slim" or "average". Use "curvy", "plus_size",
      "athletic", "muscular", "petite" or "broad_build" when the body actually shows those traits. Do not soften.
    * Allowed values: petite | slim | average | athletic | curvy | plus_size | muscular | broad_build | "" (empty for headshots/unclear).
    * Do NOT use the term "stocky". Use broad_build if that is the intended neutral meaning.
- body_height_impression: only fill when enough full/three-quarter body context is visible. Use short | average_height | tall | "". Do not infer height from headshots.
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
- KREA/BODY-REFERENCE FIELDS:
  * body_orientation describes the torso, not the face.
  * camera_angle and perspective_distortion must distinguish a normal slight selfie angle from anatomy-distorting wide-angle views.
  * body_reference_usefulness is high only when body proportions, posture and the connection between face and body are readable. A beautiful distant photo with hidden/cropped limbs can still be a poor body reference.
  * limb_completeness describes visible cropping; do not invent hidden limbs.
  * action_description and prominent_objects should contain only visually important, reproducible details.
  * composition_description should be one concise natural-language phrase useful for later captioning.

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
For the categorical aux fields, use ONLY these values. Do not invent new tokens.
shot_type remains ONLY headshot | medium | full_body and is used for dataset quotas;
frame_subtype is an extra descriptive field for captions/reports only.

lighting_type:
  studio_softbox | studio_ringlight | studio_other | natural_outdoor_sun |
  natural_outdoor_overcast | harsh_direct_sun | golden_hour |
  natural_indoor_window | indoor_artificial | camera_flash | mixed |
  low_light | backlit | neon_colored | colored_stage_light | other

background_type:
  studio_plain | studio_textured | indoor_room | indoor_bathroom |
  indoor_kitchen | indoor_bedroom | indoor_office | indoor_gym |
  outdoor_urban | outdoor_nature | outdoor_forest | outdoor_beach |
  outdoor_snow | outdoor_mountain | outdoor_event | outdoor_other |
  vehicle_interior | public_transport | mirror_selfie |
  transparent_or_isolated | other

hair_texture (natural texture, separate from style):
  straight | wavy | curly | coily | afro_textured

hair_length:
  shaved | very_short | short | chin_length | shoulder_length |
  medium_length | long | very_long | not_visible | unclear

eye_appearance:
  natural_eyes | colored_contact_lenses | circle_lenses | cosmetic_lenses |
  unnatural_eye_color | unclear

makeup_intensity (pick exactly ONE):
  none | minimal | natural | defined | full | dramatic |
  stage_makeup | costume_makeup | face_paint | unclear

makeup_style:
  natural_makeup | gyaru_makeup | cosplay_makeup | anime_inspired_makeup |
  dramatic_eyeliner | smoky_eye_makeup | false_eyelashes | glossy_lips |
  face_paint | fantasy_makeup | unclear

look_context:
  regular_photo | fashion | glamour | gyaru_style | cosplay |
  character_costume | fantasy_costume | stage_costume | swimwear_costume |
  lingerie_costume | unclear

costume_accessories (array, use [] if none visible):
  animal_ears | cat_ears | fox_ears | bunny_ears | elf_ears | pointed_ears |
  horns | antlers | wings | feather_headpiece | headband | hair_bow |
  hair_ribbon | forehead_jewel | tiara | crown | halo | veil | hood |
  hat | cap | helmet | mask | choker | collar | necklace | gloves |
  arm_guards | wrist_cuffs | fantasy_armor | shoulder_armor |
  prop_weapon | prop_sword | prop_gun | prop_staff | prop_bottle |
  prop_book | other_prop | none_visible | unclear

glasses_frame_shape (only if has_glasses_now is true):
  round | square | rectangular | oval | aviator | cat_eye | oversized |
  rimless | semi_rimless | browline | geometric | wayfarer | shield | other

glasses_frame_material:
  wire_frame | metal_frame | plastic_frame | acetate_frame | rimless |
  semi_rimless | mixed_material | unclear

glasses_lens_type:
  clear_lenses | tinted_lenses | sunglasses | reflective_lenses |
  blue_light_lenses | unclear

glasses_position:
  on_face | on_head | held | hanging_from_clothing | other | not_visible

Hair color rule:
  hair_description may describe highlights, ombre or streaks, but keep the underlying
  base color explicit whenever visible, e.g. "brown hair with blonde highlights".
  Do not use "highlights" or "ombre" as if it were a complete base hair color.

frame_subtype:
  close_up | portrait | selfie | mirror_selfie | three_quarter_body |
  full_body | faceless_body | detail_only | unclear

gaze_category:
  looking_at_camera | looking_left | looking_right | looking_up |
  looking_down | looking_away | eyes_closed | partly_closed | unclear

expression_category:
  neutral | slight_smile | smile | wide_smile | serious | pensive |
  playful | laughing | surprised | sad | angry | duckface | winking |
  eyes_closed | other

occlusion_type:
  none | hair_covering_face | hand_covering_face | object_covering_face |
  sunglasses_occluding_eyes | mask | hat_shadow | motion_blur |
  crop_cutoff | face_partly_out_of_frame | other

visual_style_type:
  normal_color | black_and_white | sepia | warm_tinted | cool_tinted |
  green_tinted | blue_tinted | high_contrast | low_contrast |
  beauty_filter | heavy_smoothing | vintage_filter | screenshot | other

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
        "max_output_tokens": 2600,  # Krea/body-reference fields included
        "store": False,
        "temperature": 0.1,
        "_reasoning_effort": reasoning_effort or AUDIT_REASONING_EFFORT,
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
    audit["body_reference_usefulness"] = round(_to_unit(audit.get("body_reference_usefulness", 0)) * 10.0, 1)

    # Gewichtete Summe direkt auf den 0-10-Werten (einmal *10 indirekt
    # ueber Gewichte). Ergebnis liegt garantiert in [0.0, 100.0].
    weighted = (qs10 * 4.0) + (ql10 * 2.5) + (qc10 * 2.0) + (qi10 * 1.5)
    audit["quality_total"] = round(min(100.0, max(0.0, weighted)), 1)

    return audit


def apply_local_score_adjustments(row: Dict[str, Any]) -> Dict[str, Any]:
    """Wendet deterministische lokale Soft-Penalties auf den finalen Score an."""
    try:
        current_score = float(row.get("quality_total", 0) or 0)
    except Exception:
        current_score = 0.0

    row["quality_total_before_local_penalties"] = round(current_score, 1)
    penalties_applied: List[str] = []
    total_penalty = 0.0

    if USE_GRAYSCALE_PENALTY and bool(row.get("is_grayscale_filter")):
        total_penalty += float(GRAYSCALE_SCORE_PENALTY)
        penalties_applied.append(f"grayscale_filter_penalty_{GRAYSCALE_SCORE_PENALTY:.1f}")

    final_score = max(0.0, min(100.0, current_score - total_penalty))
    row["grayscale_penalty"] = round(float(GRAYSCALE_SCORE_PENALTY), 1) if bool(row.get("is_grayscale_filter")) else 0.0
    row["local_score_penalty_total"] = round(total_penalty, 1)
    row["quality_total"] = round(final_score, 1)

    if penalties_applied:
        notes = row.setdefault("status_notes", [])
        if isinstance(notes, list):
            for note in penalties_applied:
                if note not in notes:
                    notes.append(note)

    return row


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
    # Splitte nur die erste Phrase (bis zum ersten Konnektor).
    # Komma ist Sonderfall: braucht keinen Whitespace davor (typisch "top,").
    # Andere Konnektoren (over, with, and, ...) brauchen Whitespace beidseits.
    first_segment = re.split(
        r"(?:\s+(?:over|with|and|under|above|on|in)\s+|,\s*)",
        p, maxsplit=1
    )[0]
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
    Legacy-Hook fuer Brillen-Wording.

    Wichtig: Profil-Canonical-Wording muss in Captions erhalten bleiben.
    Wenn das Subject Profile z.B. "round wire-frame eyeglasses" als
    canonical_description setzt, darf dieser Begriff nicht am Ende wieder
    zu "round wire-frame glasses" normalisiert werden.

    Die eigentliche Sicherheitslogik (Sonnenbrillen nicht durch Profilbrillen
    ueberschreiben, normale Brillen mit dem Profil vereinheitlichen) passiert
    in resolve_visible_glasses_description(...).
    """
    return text or ""


def _is_sunglasses_description(text: Optional[str]) -> bool:
    """Erkennt Beschreibungen, die explizit Sonnenbrillen/Shades meinen.

    Wichtig: Diese Bilder duerfen NICHT durch die kanonische Profil-Brille
    (z.B. "thin rectangular glasses") ueberschrieben werden.
    """
    t = normalize_compact_text(text)
    if not t:
        return False
    keywords = [
        "sunglasses", "sun glasses", "shades",
        "dark sunglasses", "aviator sunglasses",
        "tinted lenses", "dark lenses", "black lenses",
        "mirrored lenses", "mirror lenses", "reflective lenses",
        "tinted shades", "dark shades",
    ]
    return any(k in t for k in keywords)


def _is_regular_glasses_description(text: Optional[str]) -> bool:
    """Erkennt normale optische Brillenbeschreibungen grob heuristisch."""
    t = normalize_compact_text(text)
    if not t or _is_sunglasses_description(t):
        return False
    keywords = [
        "glasses", "eyeglasses", "spectacles", "frames", "rimless",
        "browline", "rectangular", "round", "square", "oval", "cat eye",
        "cat-eye", "geometric", "clear lenses", "clear lens",
    ]
    return any(k in t for k in keywords)


def resolve_visible_glasses_description(item: Dict[str, Any], profile: Dict[str, Any], image_traits: Dict[str, Any]) -> str:
    """Waehlt eine sichere Brillenbeschreibung fuer die Caption.

    Regeln:
    - Wenn im Einzelbild Sonnenbrille sichtbar ist, IMMER die Einzelbild-
      Beschreibung behalten.
    - Wenn das Profil Sonnenbrille sagt, das Einzelbild aber normale Brille,
      ebenfalls die Einzelbild-Beschreibung bevorzugen.
    - Nur kompatible normale Brillen duerfen durch die kanonische Profil-
      Beschreibung vereinheitlicht werden.
    """
    glasses_visible = bool(image_traits.get("glasses_visible")) or _profile_bool(item.get("has_glasses_now"))
    if not glasses_visible:
        return ""

    markers = profile.get("identity_markers", {}) if isinstance(profile, dict) else {}
    glasses_profile = markers.get("glasses", {}) if isinstance(markers, dict) else {}

    item_desc = compact_trait(item.get("glasses_description"))
    profile_desc = compact_trait(glasses_profile.get("canonical_description"))

    if _is_sunglasses_description(item_desc):
        return item_desc
    if _is_sunglasses_description(profile_desc) and item_desc:
        return item_desc
    if item_desc and _is_regular_glasses_description(item_desc) and _is_regular_glasses_description(profile_desc):
        return profile_desc or item_desc
    if item_desc:
        return item_desc
    return profile_desc




def _eye_color_family(token: str) -> str:
    t = normalize_text(token)
    if t in {"blue", "gray_blue", "blue_green"}:
        return "blue_green_family"
    if t in {"brown", "dark_brown"}:
        return "brown_family"
    # Green, hazel, amber and blue-green remain separate. For identity LoRAs
    # it is safer to caption a possible contact-lens deviation than to merge
    # genuinely different eye colors into one broad family.
    return t


def _stats_mode_token(stats: Any) -> str:
    if isinstance(stats, dict):
        return normalize_text(stats.get("mode", ""))
    return ""


def _feature_deviation(current: str, baseline: str, family_fn=None) -> bool:
    cur = normalize_text(current)
    base = normalize_text(baseline)
    if not cur:
        return False
    if not base:
        return True
    if family_fn is not None:
        cur_f = normalize_text(family_fn(cur))
        base_f = normalize_text(family_fn(base))
        if cur_f and base_f:
            return cur_f != base_f
    return cur != base


def get_hair_feature_state(item: Dict[str, Any], profile: Dict[str, Any], image_traits: Dict[str, Any], global_rules: Dict[str, Any], active_policy: Dict[str, Any], caption_profile: str) -> Dict[str, Any]:
    canonical = profile.get("canonical_features", {}) if isinstance(profile, dict) else {}
    variability = profile.get("profile_variability_stats", {}) if isinstance(profile, dict) else {}
    color_stats = variability.get("hair_color", {}) if isinstance(variability, dict) else {}
    form_stats = variability.get("hair_form", {}) if isinstance(variability, dict) else {}

    current_color = normalize_text(image_traits.get("hair_color_base", ""))
    current_modifier = normalize_text(image_traits.get("hair_color_modifier", ""))
    current_form = normalize_text(image_traits.get("hair_form", ""))
    baseline_color = normalize_text(canonical.get("hair_color", "")) or _stats_mode_token(color_stats)
    baseline_form = normalize_text(canonical.get("hair_form", "")) or _stats_mode_token(form_stats)

    color_variable = bool(color_stats.get("variation_detected", color_stats.get("unique", 0) >= 2))
    form_variable = bool(form_stats.get("variation_detected", form_stats.get("unique", 0) >= 2))
    mode = normalize_text(globals().get("VARIABLE_FEATURE_CAPTION_MODE", "canonical_deviations"))

    include_all = bool(active_policy.get("include_hair_always"))
    include_variable = bool(active_policy.get("include_hair_when_variable"))
    color_deviation = _feature_deviation(current_color, baseline_color, _appearance_hair_family)
    form_deviation = _feature_deviation(current_form, baseline_form)

    if include_all:
        include_color = bool(current_color)
        include_form = bool(current_form)
    elif include_variable and mode == "all_visible_when_variable":
        include_color = bool(current_color and color_variable)
        include_form = bool(current_form and form_variable)
    elif include_variable:
        include_color = bool(current_color and color_deviation)
        include_form = bool(current_form and form_deviation)
    else:
        include_color = include_form = False

    phrase = profile_hair_caption(current_color if include_color else "", current_form if include_form else "")
    if current_modifier and (include_color or include_variable or include_all):
        modifier_phrase = {
            "blonde_highlights": "blonde highlights",
            "red_highlights": "red highlights",
            "highlights": "highlights",
            "ombre": "ombre coloring",
            "balayage": "balayage",
        }.get(current_modifier, _phrase_from_token(current_modifier))
        if phrase:
            phrase = f"{phrase} with {modifier_phrase}"
        elif current_color:
            phrase = f"{_phrase_from_token(current_color)} hair with {modifier_phrase}"
        else:
            phrase = modifier_phrase
    return {
        "phrase": phrase,
        "must_caption": bool(phrase),
        "current": current_color,
        "baseline": baseline_color,
        "current_form": current_form,
        "current_modifier": current_modifier,
        "baseline_form": baseline_form,
        "color_variable": color_variable,
        "form_variable": form_variable,
        "mode": mode,
    }

def get_eye_feature_state(item: Dict[str, Any], profile: Dict[str, Any], image_traits: Dict[str, Any], active_policy: Dict[str, Any]) -> Dict[str, Any]:
    canonical = profile.get("canonical_features", {}) if isinstance(profile, dict) else {}
    stable_identity = profile.get("stable_identity", {}) if isinstance(profile, dict) else {}
    stats = (profile.get("profile_variability_stats", {}) or {}).get("eye_color", {}) if isinstance(profile, dict) else {}
    reliable = bool(image_traits.get("eye_color_reliable"))
    current = normalize_text(image_traits.get("eye_color_base", "")) if reliable else ""
    baseline = normalize_text(canonical.get("eye_color", "")) or normalize_text(stable_identity.get("eye_color", "")) or _stats_mode_token(stats)
    variable = bool(stats.get("variation_detected", stats.get("unique", 0) >= 2))
    mode = normalize_text(globals().get("VARIABLE_FEATURE_CAPTION_MODE", "canonical_deviations"))
    enabled = bool(active_policy.get("include_eye_color_when_variable"))
    if enabled and mode == "all_visible_when_variable":
        must_caption = bool(current and variable)
    else:
        # Eye color is especially error-prone. A single contrary audit is not
        # enough to establish a real identity variation; require at least two
        # reliable minority observations before captioning deviations.
        must_caption = bool(enabled and variable and current and _feature_deviation(current, baseline, _eye_color_family))
    phrase = f"{_phrase_from_token(current)} eyes" if must_caption and current else ""
    return {
        "phrase": phrase,
        "must_caption": bool(phrase),
        "current": current,
        "baseline": baseline,
        "variable": variable,
        "reliable": reliable,
        "mode": mode,
    }

def get_beard_feature_state(item: Dict[str, Any], global_rules: Dict[str, Any], active_policy: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    profile = profile or {}
    canonical = profile.get("canonical_features", {}) if isinstance(profile, dict) else {}
    beard_rule = global_rules.get("beard_description", {}) if isinstance(global_rules, dict) else {}
    parsed = normalize_beard_tag(item.get("beard_description", ""))
    current_pattern = normalize_text(parsed.get("pattern")) if parsed.get("visible") else ""
    current_color = normalize_text(parsed.get("color")) if parsed.get("visible") else ""
    baseline_pattern = normalize_text(canonical.get("beard_pattern", "")) or normalize_text(beard_rule.get("mode_pattern", ""))
    baseline_color = normalize_text(canonical.get("beard_color", "")) or normalize_text(beard_rule.get("mode_color", ""))
    variable = bool(beard_rule.get("variation_detected", beard_rule.get("variable", False)))
    mode = normalize_text(globals().get("VARIABLE_FEATURE_CAPTION_MODE", "canonical_deviations"))

    deviation = False
    if current_pattern:
        deviation = current_pattern != baseline_pattern
        if not deviation and current_pattern != "clean_shaven" and current_color and baseline_color:
            deviation = current_color not in {"other"} and current_color != baseline_color

    enabled_all = bool(active_policy.get("include_beard_always"))
    enabled_variable = bool(active_policy.get("include_beard_when_variable"))
    if enabled_all:
        must_caption = bool(current_pattern)
    elif enabled_variable and mode == "all_visible_when_variable":
        must_caption = bool(current_pattern and variable)
    else:
        must_caption = bool(enabled_variable and current_pattern and deviation)

    phrase = ""
    if must_caption:
        if current_pattern == "clean_shaven":
            phrase = "clean-shaven"
        else:
            phrase = build_beard_caption_tag(item, global_rules) or compact_trait(item.get("beard_description"))

    return {
        "phrase": phrase,
        "must_caption": bool(phrase),
        "current_pattern": current_pattern,
        "baseline_pattern": baseline_pattern,
        "current_color": current_color,
        "baseline_color": baseline_color,
        "variable": variable,
        "mode": mode,
    }


def _glasses_shape_family(token: str) -> str:
    t = normalize_text(token)
    if t in {"round", "oval"}:
        return "rounded"
    if t in {"rectangular", "square"}:
        return "angular"
    return t


def _glasses_material_family(token: str) -> str:
    t = normalize_text(token)
    if t in {"wire_frame", "metal_frame"}:
        return "metal_frame"
    if t in {"plastic_frame", "acetate_frame"}:
        return "plastic_frame"
    return t


def _glasses_lens_family(token: str) -> str:
    t = normalize_text(token)
    if t in {"clear_lenses", "blue_light_lenses"}:
        return "clear_lenses"
    if t in {"tinted_lenses", "sunglasses", "reflective_lenses"}:
        return "sunglasses"
    return t


def _canonical_glasses_phrase(shape: str, material: str, lens: str, fallback: str = "") -> str:
    lens_f = _glasses_lens_family(lens)
    if lens_f == "sunglasses":
        shape_txt = _phrase_from_token(_glasses_shape_family(shape))
        return f"{shape_txt} sunglasses".strip() if shape_txt else "sunglasses"
    shape_txt = _phrase_from_token(_glasses_shape_family(shape))
    material_f = _glasses_material_family(material)
    material_txt = {
        "metal_frame": "metal-frame",
        "plastic_frame": "plastic-frame",
        "rimless": "rimless",
        "semi_rimless": "semi-rimless",
        "mixed_material": "mixed-frame",
    }.get(material_f, "")
    bits = [b for b in [shape_txt, material_txt, "glasses"] if b]
    phrase = " ".join(bits).strip()
    return phrase or compact_trait(fallback) or "eyeglasses"


def _glasses_fingerprint(shape: str, material: str, lens: str) -> str:
    return "|".join([
        _glasses_shape_family(shape) or "unclear_shape",
        _glasses_material_family(material) or "unclear_material",
        _glasses_lens_family(lens) or "clear_lenses",
    ])


def get_glasses_feature_state(item: Dict[str, Any], profile: Dict[str, Any], image_traits: Dict[str, Any], active_policy: Dict[str, Any]) -> Dict[str, Any]:
    markers = profile.get("identity_markers", {}) if isinstance(profile, dict) else {}
    canonical = profile.get("canonical_features", {}) if isinstance(profile, dict) else {}
    glasses_profile = markers.get("glasses", {}) if isinstance(markers, dict) else {}
    variability = (profile.get("profile_variability_stats", {}) or {}).get("glasses", {}) if isinstance(profile, dict) else {}
    frame_variability = (profile.get("profile_variability_stats", {}) or {}).get("glasses_frame", {}) if isinstance(profile, dict) else {}

    visible = bool(image_traits.get("glasses_visible")) or _profile_bool(item.get("has_glasses_now"))
    current_position = normalize_text(image_traits.get("glasses_position", "")) or infer_glasses_position(item)
    baseline_desc = compact_trait(glasses_profile.get("canonical_description"))
    baseline_shape = normalize_text(canonical.get("glasses_frame_shape", ""))
    baseline_material = normalize_text(canonical.get("glasses_frame_material", ""))
    baseline_lens = normalize_text(canonical.get("glasses_lens_type", ""))
    baseline_fingerprint = _glasses_fingerprint(baseline_shape, baseline_material, baseline_lens)
    baseline_family = "regular_glasses" if glasses_profile.get("wears_regularly") else "no_glasses"
    if baseline_desc and _is_sunglasses_description(baseline_desc):
        baseline_family = "sunglasses"

    current_shape = normalize_text(image_traits.get("glasses_frame_shape", "")) or normalize_text(item.get("glasses_frame_shape", ""))
    current_material = normalize_text(image_traits.get("glasses_frame_material", "")) or normalize_text(item.get("glasses_frame_material", ""))
    current_lens = normalize_text(image_traits.get("glasses_lens_type", "")) or normalize_text(item.get("glasses_lens_type", ""))
    item_desc = compact_trait(image_traits.get("glasses_description")) or compact_trait(item.get("glasses_description"))

    if visible:
        current_family = "sunglasses" if _is_sunglasses_description(item_desc) or _glasses_lens_family(current_lens) == "sunglasses" else "regular_glasses"
        current_fingerprint = _glasses_fingerprint(current_shape, current_material, current_lens)
        same_regular_frame = (
            current_family == baseline_family == "regular_glasses"
            and (
                current_fingerprint == baseline_fingerprint
                or (not current_shape and not current_material)
                or (not baseline_shape and not baseline_material)
            )
        )
        if same_regular_frame:
            phrase = baseline_desc or _canonical_glasses_phrase(current_shape, current_material, current_lens, item_desc)
        else:
            phrase = _canonical_glasses_phrase(current_shape, current_material, current_lens, item_desc)
    else:
        current_family = "no_glasses"
        current_fingerprint = "no_glasses"
        same_regular_frame = False
        phrase = ""

    if current_family != "no_glasses" and current_position != "on_face":
        current_fingerprint = f"{current_fingerprint}|{current_position}"
        same_regular_frame = False

    variable = bool(
        variability.get("variation_detected", variability.get("unique", 0) >= 2)
        or frame_variability.get("variation_detected", frame_variability.get("unique", 0) >= 2)
    )
    mode = normalize_text(globals().get("VARIABLE_FEATURE_CAPTION_MODE", "canonical_deviations"))
    always = bool(active_policy.get("include_glasses"))
    when_variable = bool(active_policy.get("include_glasses_when_variable"))

    if always:
        must_caption = visible and bool(phrase)
    elif when_variable and mode == "all_visible_when_variable" and variable:
        must_caption = True
    elif when_variable:
        must_caption = current_family != baseline_family
        if current_family == baseline_family == "regular_glasses":
            must_caption = not same_regular_frame
    else:
        must_caption = False

    if must_caption and current_family == "no_glasses":
        phrase = "without glasses"
    elif must_caption and current_family == "regular_glasses":
        phrase = phrase or baseline_desc or "eyeglasses"
        if current_position == "on_head":
            phrase = f"{phrase} resting on the head"
        elif current_position == "held":
            phrase = f"holding {phrase}"
        elif current_position == "hanging_from_clothing":
            phrase = f"{phrase} hanging from the clothing"
    elif must_caption and current_family == "sunglasses":
        phrase = phrase or "sunglasses"
    else:
        phrase = "" if not must_caption else phrase

    return {
        "phrase": phrase,
        "must_caption": bool(must_caption and phrase),
        "current_family": current_family,
        "baseline_family": baseline_family,
        "current_desc": compact_trait(phrase),
        "baseline_desc": baseline_desc,
        "current_fingerprint": current_fingerprint,
        "baseline_fingerprint": baseline_fingerprint,
        "variable": variable,
        "position": current_position,
        "mode": mode,
    }

def _caption_contains_term(caption: str, term: str) -> bool:
    c = normalize_compact_text(caption)
    t = normalize_compact_text(term)
    return bool(c and t and t in c)


def get_visible_tattoo_state(
    item: Dict[str, Any],
    profile: Optional[Dict[str, Any]],
    active_policy: Dict[str, Any],
) -> Dict[str, Any]:
    visible: List[str] = []
    for entry in item.get("tattoo_inventory_now") or []:
        if not isinstance(entry, dict):
            continue
        desc = compact_trait(entry.get("description"))
        loc = _phrase_from_token(entry.get("location", ""))
        phrase = desc or (f"tattoo on the {loc}" if loc else "tattoo")
        if phrase:
            visible.append(phrase)
    if not visible and bool(item.get("tattoos_visible", False)):
        fallback = compact_trait(item.get("tattoos_description")) or "visible tattoo"
        visible.append(fallback)
    visible = _dedupe_phrase_list(visible)
    must_caption = bool(active_policy.get("include_tattoos", False) and visible)
    return {"phrases": visible, "visible": bool(visible), "must_caption": must_caption}


def _validate_krea_caption_features(caption: str, feature_states: Dict[str, Dict[str, Any]]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    text = normalize_compact_text(caption)

    def contains_any(tokens: List[str]) -> bool:
        return any(normalize_compact_text(t) in text for t in tokens if normalize_compact_text(t))

    for name in ("hair", "eye", "beard", "glasses"):
        state = feature_states.get(name, {}) or {}
        phrase = compact_trait(state.get("phrase"))
        if state.get("must_caption") and phrase and not _caption_contains_term(text, phrase):
            # Accept common wording variants for a few canonical feature phrases.
            alternatives: List[str] = []
            if name == "glasses" and state.get("current_family") == "regular_glasses":
                position = normalize_text(state.get("position", ""))
                if position == "on_head":
                    alternatives = ["glasses on the head", "glasses on her head", "glasses on his head", "glasses resting on", "glasses pushed up"]
                elif position == "held":
                    alternatives = ["holding glasses", "held glasses", "glasses in hand"]
                elif position == "hanging_from_clothing":
                    alternatives = ["glasses hanging", "glasses on the collar", "glasses on the shirt", "glasses at the neckline"]
                else:
                    alternatives = ["glasses", "eyeglasses", "spectacles"]
            elif name == "glasses" and state.get("current_family") == "no_glasses":
                alternatives = ["without glasses", "no glasses"]
            elif name == "beard" and state.get("current_pattern") == "clean_shaven":
                alternatives = ["clean-shaven", "clean shaven"]
            if not contains_any(alternatives):
                reasons.append(f"missing required {name}: {phrase}")

    eye = feature_states.get("eye", {}) or {}
    if not eye.get("must_caption") and eye.get("current"):
        eye_tokens = [f"{_phrase_from_token(eye.get('current'))} eyes"]
        if contains_any(eye_tokens):
            reasons.append("canonical eye color should be omitted")

    beard = feature_states.get("beard", {}) or {}
    if not beard.get("must_caption"):
        if contains_any(["beard", "stubble", "mustache", "moustache", "goatee", "soul patch", "mutton chops", "chin strap", "clean-shaven", "clean shaven"]):
            reasons.append("canonical beard state should be omitted")

    glasses = feature_states.get("glasses", {}) or {}
    if not glasses.get("must_caption"):
        if contains_any(["glasses", "eyeglasses", "spectacles", "without glasses", "no glasses", "sunglasses"]):
            reasons.append("canonical glasses state should be omitted")

    hair = feature_states.get("hair", {}) or {}
    if not hair.get("must_caption"):
        if contains_any([" hair", "hair ", "braids", "ponytail", "pigtails", "bun", "updo", "dreadlocks", "cornrows", "pixie cut", "bob cut"]):
            reasons.append("canonical hair state should be omitted")

    piercings = feature_states.get("piercings", {}) or {}
    if piercings.get("must_caption"):
        for phrase in piercings.get("phrases", []) or []:
            anchor = _piercing_caption_anchor(str(phrase))
            alternatives = [anchor]
            if anchor in {"lower lip", "labret"}:
                alternatives.extend(["lower-lip", "labret"])
            if anchor == "earring":
                alternatives.extend(["earrings", "ear jewelry"])
            if not contains_any(alternatives):
                reasons.append(f"missing required piercing/accessory: {phrase}")

    tattoos = feature_states.get("tattoos", {}) or {}
    tattoo_tokens = ["tattoo", "tattoos", "tattooed", "inked"]
    if tattoos.get("must_caption"):
        if tattoos.get("visible") and not contains_any(tattoo_tokens):
            reasons.append("missing required visible tattoo")
    elif contains_any(tattoo_tokens):
        reasons.append("tattoos should be omitted by caption policy")

    return (len(reasons) == 0), reasons

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

    # Pattern 3: 'X or Y' am Phrasen-Ende (kein nachfolgendes Substantiv).
    # Hier ist X selbst das Substantiv, Y die Alternative dazu.
    # Beispiel: 'left ear lobe earring or stud' -> 'left ear lobe earring'
    # Wir behalten das erste Substantiv (X) als das spezifischere/uebliche.
    #
    # WICHTIG: Wir matchen NUR bei einer Whitelist von Schmuck-/Piercing-
    # Substantiven, weil ein freies 'X or Y'-Pattern legitime 'or'-Phrasen
    # in Background/Lighting-Saetzen kaputtmacht ('curtain or wall',
    # 'daylight or flash', 'standing or sitting').
    JEWELRY_NOUNS = (
        "earring|earrings|stud|studs|hoop|hoops|piercing|piercings|"
        "ring|rings|necklace|necklaces|bracelet|bracelets|pendant|pendants"
    )
    pattern_or_terminal = re.compile(
        rf"\b({JEWELRY_NOUNS}) or ({JEWELRY_NOUNS})(?=\s*(?:[,.;]|$))",
        re.IGNORECASE,
    )

    def replace_terminal(m: re.Match) -> str:
        return m.group(1)

    text = pattern_or_terminal.sub(replace_terminal, text)

    return text


def _ensure_gaze_verb(gaze: str) -> str:
    """
    Setzt 'looking' vor reine Direction-Adverbien, damit der Caption-Satz
    grammatikalisch sauber ist:
    - 'downward' -> 'looking downward'
    - 'toward camera' -> 'toward camera' (hat schon eine Praeposition,
       in der naechsten Pose-Bit-Aufzaehlung lesbar)
    - 'looking at the camera' -> unveraendert (Verb schon vorhanden)

    Behebt den Bug 'holding cards, downward.' der entsteht wenn die KI
    nur das Adverb liefert ohne Verb.
    """
    g = gaze.strip()
    if not g:
        return g
    # Verb schon vorhanden? Pruefe nach gaengigen Blick-Verben am Anfang
    GAZE_VERBS = {"looking", "gazing", "staring", "glancing", "facing",
                  "watching", "peering", "looks", "gazes", "stares"}
    first_word = g.split()[0].lower().rstrip(",.")
    if first_word in GAZE_VERBS:
        return g
    # Praeposition vorhanden? ('toward', 'at', 'into', etc.) - dann ist es
    # eine vollstaendige Phrase die in der Caption-Aufzaehlung sauber liest
    GAZE_PREPS = {"toward", "towards", "at", "into", "upon", "across", "past",
                  "through", "above", "below", "behind", "ahead"}
    if first_word in GAZE_PREPS:
        return g
    # Reine Direction-Adverbien -> 'looking' davorsetzen
    DIRECTION_ADVERBS = {"downward", "upward", "sideways", "leftward",
                         "rightward", "forward", "backward", "away", "down",
                         "up", "left", "right", "outward", "inward"}
    if first_word in DIRECTION_ADVERBS:
        return f"looking {g}"
    # Fallback: ungewoehnliche gaze-Phrase, unveraendert lassen
    return g


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

    def beard_mode_info(min_fraction_for_stable: float = 0.85) -> Dict[str, Any]:
        values: List[Tuple[str, str]] = []
        for i in items:
            parsed = normalize_beard_tag(str(i.get("beard_description", "")))
            if not parsed.get("visible"):
                continue
            pattern = normalize_text(parsed.get("pattern")) or ""
            color = normalize_text(parsed.get("color")) or ""
            if not pattern:
                continue
            values.append((pattern, color))
        if not values:
            return {"mode": "", "stable": False, "variable": True, "override_candidates": [], "counts": {}, "mode_pattern": "", "mode_color": ""}

        counts = Counter(values)
        (mode_pattern, mode_color), mode_count = counts.most_common(1)[0]
        total = max(1, len(values))
        frac = mode_count / total
        stable = frac >= min_fraction_for_stable

        def _fmt(pattern: str, color: str) -> str:
            if pattern == "clean_shaven":
                return "clean shaven"
            tag = {
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
            }.get(pattern, "beard")
            color_txt = color.replace("_", " ").strip()
            return f"{color_txt} {tag}".strip() if color_txt and color_txt != "other" else tag

        override_candidates = [] if stable else [_fmt(p, c) for (p, c), _n in counts.most_common(5)]
        pretty_counts = {_fmt(p, c): n for (p, c), n in counts.most_common(10)}
        return {
            "mode": _fmt(mode_pattern, mode_color),
            "stable": stable,
            "variable": not stable,
            "override_candidates": override_candidates,
            "counts": pretty_counts,
            "mode_pattern": mode_pattern,
            "mode_color": mode_color,
        }

    # Wir berechnen globale Regeln NUR für Features, bei denen es die "when_variable" Logik gibt!
    # Brillen, Tattoos etc. sind fest durch CAPTION_POLICY geregelt und brauchen keine Mehrheitsentscheidung.
    rules["hair_description"] = mode_info("hair_description", 0.85)
    rules["beard_description"] = beard_mode_info(0.85)

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
            "freckles_description": row.get("freckles_description", ""),
            "hair_texture": row.get("hair_texture", ""),
            "beard_description": row.get("beard_description", ""),
            "glasses_description": row.get("glasses_description", ""),
            "has_glasses_now": row.get("has_glasses_now", False),
            "glasses_frame_shape": row.get("glasses_frame_shape", ""),
            "glasses_frame_material": row.get("glasses_frame_material", ""),
            "glasses_lens_type": row.get("glasses_lens_type", ""),
            "glasses_position": row.get("glasses_position", ""),
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
        "normalizer_model": str(PROFILE_NORMALIZER_MODEL).strip().lower(),
        "reasoning_effort": str(PROFILE_REASONING_EFFORT),
        "training_target": normalize_training_target(globals().get("TRAINING_TARGET", "ernie")),
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

    Profilbildung braucht mehr Koerperinformation als die finale Train-Auswahl:
    Keep- und Keep-Unused-Bilder gehen upstream ohnehin in `rows`. Wenn das
    Dataset > PROFILE_SAMPLE_THRESHOLD ist, werden Medium-/Fullbody-Bilder
    bevorzugt in das Sample gehoben, damit body_build nicht von Headshots
    ueberstimmt wird. Danach wird der Rest wie bisher stratifiziert nach
    lighting × shot_type × quality-tier aufgefuellt.
    """
    if len(rows) <= int(PROFILE_SAMPLE_THRESHOLD):
        return list(rows)

    target = max(1, int(PROFILE_SAMPLE_SIZE))
    selected: List[Dict[str, Any]] = []
    selected_ids: set = set()

    def _sid(row: Dict[str, Any]) -> str:
        return profile_image_id(row)

    # 1) Body-relevante Bilder zuerst, aber begrenzt, damit Face-Identity nicht
    #    ueberrollt wird. Score-Sortierung sorgt fuer gute Body-Referenzen.
    body_rows = [
        r for r in rows
        if normalize_text(r.get("shot_type")) in {"medium", "full_body"}
    ]
    body_rows.sort(key=lambda r: (normalize_text(r.get("shot_type")) != "full_body", -float(r.get("quality_total", 0)), _sid(r)))
    for row in body_rows[: min(len(body_rows), int(PROFILE_BODY_PRIORITY_SAMPLE_MAX), target)]:
        sid = _sid(row)
        if sid not in selected_ids:
            selected.append(row)
            selected_ids.add(sid)

    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sid = _sid(row)
        if sid in selected_ids:
            continue
        key = (
            normalize_text(row.get("lighting_type")) or "unknown_lighting",
            normalize_text(row.get("shot_type")) or "unknown_shot",
            _quality_tier(row),
        )
        groups[key].append(row)

    for key in groups:
        groups[key].sort(key=lambda r: _sid(r))

    # 2) Mindestens einen Vertreter je Stratum.
    for key in sorted(groups.keys()):
        if len(selected) >= target:
            break
        if groups[key]:
            row = groups[key].pop(0)
            selected.append(row)
            selected_ids.add(_sid(row))

    # 3) Round-robin auffuellen.
    while len(selected) < target:
        progressed = False
        for key in sorted(groups.keys()):
            if len(selected) >= target:
                break
            if groups[key]:
                row = groups[key].pop(0)
                selected.append(row)
                selected_ids.add(_sid(row))
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
                    "body_height_impression": {"type": "string"},
                },
                "required": ["gender", "skin_tone", "eye_color", "hair_texture", "body_build", "body_height_impression"],
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
                    "body_height_impression": _confidence_field_schema(),
                },
                "required": ["gender", "skin_tone", "eye_color", "hair_texture", "body_build", "body_height_impression"],
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
                    "piercing_inventory": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "canonical_description": {"type": "string"},
                                "frequency": {"type": "string"},
                                "category": {"type": "string", "enum": ["body_piercing", "ear_jewelry"]},
                                "role": {"type": "string", "enum": ["canonical", "variable", "accessory", "ignore"]},
                            },
                            "required": ["location", "canonical_description", "frequency", "category", "role"],
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
                    "freckles": {
                        "type": "object",
                        "properties": {
                            "has_freckles": {"type": "boolean"},
                            "canonical_description": {"type": "string"},
                            "frequency": {"type": "string"},
                        },
                        "required": ["has_freckles", "canonical_description", "frequency"],
                        "additionalProperties": False,
                    },
                },
                "required": ["glasses", "tattoo_inventory", "piercing_inventory", "piercing_baseline", "freckles"],
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
                "eye_appearance": row.get("eye_appearance", ""),
                "body_build": row.get("body_build", ""),
                "body_height_impression": row.get("body_height_impression", ""),
                "hair_description": row.get("hair_description", ""),
                "hair_length": row.get("hair_length", ""),
                "hair_texture": row.get("hair_texture", ""),
                "glasses_description": row.get("glasses_description", ""),
                "has_glasses_now": row.get("has_glasses_now", False),
                "glasses_frame_shape": row.get("glasses_frame_shape", ""),
                "glasses_frame_material": row.get("glasses_frame_material", ""),
                "glasses_lens_type": row.get("glasses_lens_type", ""),
            "glasses_position": row.get("glasses_position", ""),
                "makeup_description": row.get("makeup_description", ""),
                "makeup_intensity": row.get("makeup_intensity", ""),
                "makeup_style": row.get("makeup_style", ""),
                "look_context": row.get("look_context", ""),
                "costume_accessories": row.get("costume_accessories", []),
                "freckles_description": row.get("freckles_description", ""),
                "tattoo_inventory_now": row.get("tattoo_inventory_now", []),
                "piercing_inventory_now": row.get("piercing_inventory_now", []),
                "lighting_description": row.get("lighting_description", ""),
                "background_description": row.get("background_description", ""),
                "frame_subtype": row.get("frame_subtype", ""),
                "gaze_category": row.get("gaze_category", ""),
                "expression_category": row.get("expression_category", ""),
                "occlusion_type": row.get("occlusion_type", ""),
                "visual_style_type": row.get("visual_style_type", ""),
            },
        })
    return payload_rows


def call_subject_profile_normalizer(rows: List[Dict[str, Any]], input_hash: str, total_count: int) -> Dict[str, Any]:
    instructions = """
You consolidate raw per-image audits into one Subject Identity Profile for a person LoRA dataset.
All input images are intended to show the same subject. Some outliers may exist.

{training_target_profile_guidance()}

Important:
- Stable identity traits must be canonical and consistent across captions.
- Use single, clean tokens or short phrases. No hedge words, no 'or'-phrases, no 'none visible'.
- For skin tone, account for studio-lighting bias: studio or ring-light images can make darker skin read lighter.
- For eye color, treat mirror selfies, filters, and extreme lighting as possible outliers.
- Body build and body height impression are unreliable on headshots. If less than ~30% of input images
  are medium/full_body and fewer than the configured minimum body-reference shots exist, set body_build
  and body_height_impression to "" (empty string) and set their confidence levels to "low" with reasoning
  "few full-body observations". Vision models tend to over-label women as 'slim'/'average' on headshots
  due to RLHF politeness bias - resist this tendency.
- Do not use the word "stocky". If a compact/wide build is intended, use body_build="broad_build".
- Do not try to decide whether hair is a wig. Instead, use dataset-wide variance: if hair color or hair form varies strongly across images, this will be handled later by profile policies and should not be treated as a stable identity cue.
- If eye colors vary strongly or cosmetic/circle-lens looks recur, do not over-stabilize eye color. This will be handled later by profile policies and captioned as a variable attribute when needed.
- Cosplay, character-costume, fantasy-costume, gyaru or high-variation styling should be preserved in per-image fields; do not merge those costume traits into stable identity.
- Glasses are regular only if visible in at least about 60% of sampled usable images.
- Do not let occasional sunglasses overwrite a regular prescription-glasses baseline.
- If a subject regularly wears normal eyeglasses, per-image sunglasses must remain
  sunglasses in downstream captions and must not be normalized to the profile glasses description.
- Prefer a non-sunglasses canonical_description unless sunglasses are genuinely the regular baseline.
- Freckles are a flexible visibility-dependent identity marker: if they recur across the face in a substantial subset of images, preserve them as a canonical marker, but they must still only be captioned when visible in the current image.
- Piercing inventory must list every repeatedly observed piercing or ear-jewelry location, even below the canonical threshold. Classify ear locations as ear_jewelry/accessory by default and other locations as body_piercing/variable unless clearly canonical.
- Piercing baseline includes only inventory entries that are canonical and visible in at least about 40% of sampled usable images.
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
    instructions = instructions.replace("{training_target_profile_guidance()}", training_target_profile_guidance())

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
            "body_height_impression": BODY_HEIGHT_IMPRESSION_VOCAB,
            "makeup_intensity": MAKEUP_INTENSITY_VOCAB,
            "makeup_style": MAKEUP_STYLE_VOCAB,
            "look_context": LOOK_CONTEXT_VOCAB,
            "costume_accessories": COSTUME_ACCESSORY_VOCAB,
            "profile_appearance_mode": PROFILE_APPEARANCE_MODE_VOCAB,
            "hair_form": HAIR_FORM_VOCAB,
            "hair_length": HAIR_LENGTH_VOCAB,
            "hair_color": HAIR_COLOR_VOCAB,
            "lighting_type": LIGHTING_TYPE_VOCAB,
            "background_type": BACKGROUND_TYPE_VOCAB,
            "glasses_frame_shape": GLASSES_FRAME_SHAPE_VOCAB,
            "glasses_frame_material": GLASSES_FRAME_MATERIAL_VOCAB,
            "glasses_lens_type": GLASSES_LENS_TYPE_VOCAB,
            "glasses_position": ["on_face", "on_head", "held", "hanging_from_clothing", "other", "not_visible"],
            "frame_subtype": FRAME_SUBTYPE_VOCAB,
            "gaze_category": GAZE_VOCAB,
            "expression_category": EXPRESSION_VOCAB,
            "occlusion_type": OCCLUSION_TYPE_VOCAB,
            "visual_style_type": VISUAL_STYLE_VOCAB,
            "eye_appearance": EYE_APPEARANCE_VOCAB,
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
        # Reasoning tokens count against this budget. 2600 was too small for
        # larger profiles and could truncate otherwise valid strict JSON.
        "max_output_tokens": 7000,
        "store": False,
        "temperature": 0.1,
        "_reasoning_effort": PROFILE_REASONING_EFFORT,
    }

    primary_error = ""
    parsed: Optional[Dict[str, Any]] = None

    try:
        data = responses_api_call(
            PROFILE_NORMALIZER_MODEL, payload, phase_label="subject_profile_normalizer_primary"
        )
        incomplete_reason = _responses_incomplete_reason(data)
        if incomplete_reason:
            raise ValueError(f"Responses API returned incomplete output: {incomplete_reason}")
        parsed = _parse_json_object_text(extract_response_text(data))
        _validate_subject_profile_core(parsed)
    except Exception as exc:
        primary_error = str(exc)
        safe_print(
            "   ⚠️ Subject profile response was incomplete or invalid; "
            f"retrying once with {PROFILE_NORMALIZER_MODEL}: {primary_error}"
        )

        repair_instructions = instructions + """

RETRY REQUIREMENT:
The previous structured response was incomplete or invalid. Rebuild the complete profile from the
original input below. Do not quote or discuss the previous error. Return exactly one complete JSON
object matching the supplied strict schema. Keep notes concise so the response cannot be truncated.
"""
        repair_payload = {
            **payload,
            "instructions": repair_instructions,
            "max_output_tokens": 9000,
        }
        repair_data = responses_api_call(
            PROFILE_NORMALIZER_MODEL, repair_payload, phase_label="subject_profile_normalizer_retry"
        )
        incomplete_reason = _responses_incomplete_reason(repair_data)
        if incomplete_reason:
            raise ValueError(
                "Subject profile retry returned incomplete output: " + incomplete_reason
            )
        parsed = _parse_json_object_text(extract_response_text(repair_data))
        _validate_subject_profile_core(parsed)
        parsed.setdefault("normalizer_notes", []).append(
            "Automatic profile-normalizer retry succeeded after an invalid primary response."
        )

    assert parsed is not None
    parsed["profile_schema_version"] = PROFILE_CACHE_SCHEMA_VERSION
    parsed["input_hash"] = input_hash
    parsed["normalizer_model"] = PROFILE_NORMALIZER_MODEL
    parsed["normalizer_source"] = "gpt_retry" if primary_error else "gpt_primary"
    parsed["normalizer_retry_count"] = 1 if primary_error else 0
    parsed["normalizer_primary_error"] = primary_error
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
    regular_glasses_descs = [d for d in glasses_descs if not _is_sunglasses_description(d)]
    if regular_glasses_descs:
        glasses_mode = Counter(regular_glasses_descs).most_common(1)[0][0]
    else:
        glasses_mode = Counter(glasses_descs).most_common(1)[0][0] if glasses_descs else ""
    freckles_rows = [r for r in rows if compact_trait(r.get("freckles_description"))]
    freckles_descs = [compact_trait(r.get("freckles_description")) for r in freckles_rows]
    freckles_descs = [d for d in freckles_descs if d]
    freckles_mode = Counter(freckles_descs).most_common(1)[0][0] if freckles_descs else ""

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
        "normalizer_source": "local_fallback",
        "normalizer_retry_count": 1,
        "normalizer_primary_error": str(reason or ""),
        "sample_size": len(rows),
        "total_usable_images": len(rows),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stable_identity": {
            "gender": _mode_clean(rows, "gender_class") or "person",
            "skin_tone": _mode_clean(rows, "skin_tone"),
            "eye_color": _mode_clean(rows, "eye_color"),
            "hair_texture": _mode_clean(rows, "hair_texture"),
            "body_build": body_build_value,
            "body_height_impression": _mode_clean(rows, "body_height_impression"),
        },
        "canonical_features": {
            "hair_color": Counter([canonical_hair_color(r) for r in rows if canonical_hair_color(r)]).most_common(1)[0][0] if any(canonical_hair_color(r) for r in rows) else "",
            "hair_form": Counter([canonical_hair_form(r) for r in rows if canonical_hair_form(r)]).most_common(1)[0][0] if any(canonical_hair_form(r) for r in rows) else "",
            "eye_color": canonical_eye_color({"eye_color": _mode_clean(rows, "eye_color")}),
            "beard_pattern": "",
            "beard_color": "",
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
            "body_height_impression": {
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
            "freckles": {
                "has_freckles": (len(freckles_rows) / n) >= 0.25,
                "canonical_description": freckles_mode,
                "frequency": f"{len(freckles_rows)}/{n}",
            },
            "tattoo_inventory": inv_items(tattoos_by_loc, min_fraction=0.0),
            "piercing_inventory": [],
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
    # Highlights/ombre/balayage are modifiers, not the base color. Remove the
    # modifier phrase before detecting the underlying canonical color, e.g.
    # "brown hair with blonde highlights" -> base=brown, modifier=blonde_highlights.
    base_text = re.sub(
        r"\b(?:blonde|blond|red|copper|auburn|light|dark|colored|coloured)?\s*(?:highlights?|streaks?)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    base_text = re.sub(r"\b(?:ombre|ombré|balayage)(?:\s+coloring)?\b", " ", base_text, flags=re.IGNORECASE)
    base_text = re.sub(r"\s+", " ", base_text).strip()
    text = base_text or text
    # Continue scanning the cleaned description for the underlying color.
    if _contains_any(text, ["multi-colored", "multicolored", "multi color", "multicolor", "rainbow"]):
        return "multicolor"
    if _contains_any(text, ["platinum", "white-blonde", "white blonde", "very light blonde", "very light ash", "silver blonde"]):
        return "platinum"
    if _contains_any(text, ["silver hair", "silvery", "silver-gray", "silver grey"]):
        return "silver"
    if _contains_any(text, ["burgundy", "wine-red", "wine red", "deep red", "dark red"]):
        return "burgundy"
    if _contains_any(text, ["strawberry blonde", "strawberry-blonde"]):
        return "strawberry_blonde"
    if _contains_any(text, ["copper"]):
        return "copper"
    if _contains_any(text, ["auburn"]):
        return "auburn"
    if _contains_any(text, ["red hair", "red-haired", "reddish"]):
        return "red"
    if _contains_any(text, ["blue hair", "blue-dyed"]):
        return "blue"
    if _contains_any(text, ["pink hair", "pink-dyed"]):
        return "pink"
    if _contains_any(text, ["purple hair", "violet hair"]):
        return "purple"
    if _contains_any(text, ["green hair", "green-dyed"]):
        return "green"
    if _contains_any(text, ["black hair", "jet black", "raven", "black braided", "black braids", "dark black"]):
        return "black"
    if _contains_any(text, ["dark brown", "deep brown", "brunette"]):
        return "dark_brown"
    if _contains_any(text, ["light brown", "dirty blonde"]):
        return "light_brown"
    if _contains_any(text, ["dark blonde", "dark-blonde"]):
        return "dark_blonde"
    if _contains_any(text, ["blonde", "blond", "ash-blonde", "ash blonde"]):
        return "blonde"
    if _contains_any(text, ["brown hair", "brown wavy", "brown straight"]):
        return "brown"
    if _contains_any(text, ["gray", "grey"]):
        return "gray"
    if "white" in text:
        return "white"
    return ""




def canonical_hair_color_modifier(row: Dict[str, Any]) -> str:
    text = normalize_text(" ".join([
        str(row.get("hair_description", "")),
        str(row.get("hair_texture", "")),
    ]))
    if not text:
        return ""
    if _contains_any(text, ["balayage"]):
        return "balayage"
    if _contains_any(text, ["ombre", "ombré"]):
        return "ombre"
    if _contains_any(text, ["highlight", "highlights", "streaks"]):
        if _contains_any(text, ["blonde highlight", "blond highlight", "light highlight"]):
            return "blonde_highlights"
        if _contains_any(text, ["red highlight", "copper highlight", "auburn highlight"]):
            return "red_highlights"
        return "highlights"
    return ""


def eye_color_is_reliable(row: Dict[str, Any]) -> bool:
    color = normalize_text(row.get("eye_color", ""))
    if not color or not _profile_bool(row.get("face_visible")):
        return False
    if bool(row.get("is_grayscale_filter")):
        return False
    if normalize_text(row.get("visual_style_type")) in {"black_and_white", "heavy_smoothing", "beauty_filter"}:
        return False
    if normalize_text(row.get("occlusion_type")) in {"sunglasses_occluding_eyes", "mask", "hat_shadow", "motion_blur", "face_partly_out_of_frame"}:
        return False
    if normalize_text(row.get("gaze_category")) in {"eyes_closed", "partly_closed"}:
        return False
    if normalize_text(row.get("eye_appearance")) in {"colored_contact_lenses", "circle_lenses", "cosmetic_lenses", "unnatural_eye_color", "unclear"}:
        return False
    if normalize_text(row.get("glasses_lens_type")) in {"sunglasses", "tinted_lenses", "reflective_lenses"}:
        return False
    if float(row.get("main_face_ratio", 0.0) or 0.0) < 0.035:
        return False
    if float(row.get("color_tint_strength", 0.0) or 0.0) >= 0.55:
        return False
    return True


def infer_glasses_position(row: Dict[str, Any]) -> str:
    explicit = normalize_text(row.get("glasses_position", ""))
    allowed = {"on_face", "on_head", "held", "hanging_from_clothing", "other", "not_visible"}
    if explicit in allowed:
        return explicit
    desc = normalize_text(row.get("glasses_description", ""))
    if not (_profile_bool(row.get("has_glasses_now")) or desc):
        return "not_visible"
    if any(k in desc for k in ["on head", "on top of head", "on top of her head", "on top of his head", "atop head", "resting on head", "resting on top", "pushed up", "in hair"]):
        return "on_head"
    if any(k in desc for k in ["holding", "held", "in hand"]):
        return "held"
    if any(k in desc for k in ["hanging", "on shirt", "from collar", "on neckline"]):
        return "hanging_from_clothing"
    return "on_face"

def canonical_eye_color(row: Dict[str, Any]) -> str:
    text = normalize_text(str(row.get("eye_color", "")))
    if not text:
        return ""
    text = text.replace("grey", "gray")
    if "blue_green" in text or "blue green" in text or "green blue" in text:
        return "blue_green"
    if "gray_blue" in text or "gray blue" in text or "blue gray" in text:
        return "gray_blue"
    for token in ["dark_brown", "blue", "green", "hazel", "brown", "gray", "amber"]:
        if token in text:
            return token
    return ""


def canonical_eye_appearance(row: Dict[str, Any]) -> str:
    text = normalize_text(" ".join([
        str(row.get("eye_appearance", "")),
        str(row.get("eye_color", "")),
        str(row.get("makeup_description", "")),
        str(row.get("makeup_style", "")),
    ]))
    if not text:
        return ""
    if any(k in text for k in ["circle_lens", "circle lenses", "enlarging lens", "doll-like eyes"]):
        return "circle_lenses"
    if any(k in text for k in ["colored contact", "coloured contact", "contact lenses", "colored lenses", "colour contact"]):
        return "colored_contact_lenses"
    if any(k in text for k in ["cosmetic lens", "cosmetic lenses"]):
        return "cosmetic_lenses"
    if any(k in text for k in ["unnatural eye", "bright blue", "vivid blue", "vivid green", "anime-like eyes"]):
        return "unnatural_eye_color"
    if "natural_eyes" in text or "natural eyes" in text:
        return "natural_eyes"
    return normalize_text(row.get("eye_appearance")) if normalize_text(row.get("eye_appearance")) in EYE_APPEARANCE_VOCAB else ""


def canonical_makeup_style(row: Dict[str, Any]) -> str:
    text = normalize_text(" ".join([
        str(row.get("makeup_style", "")),
        str(row.get("makeup_description", "")),
    ]))
    if not text:
        return ""
    if any(k in text for k in ["gyaru", "gal makeup"]):
        return "gyaru_makeup"
    if any(k in text for k in ["cosplay makeup", "character makeup"]):
        return "cosplay_makeup"
    if any(k in text for k in ["anime inspired", "anime-style", "anime style"]):
        return "anime_inspired_makeup"
    if any(k in text for k in ["dramatic eyeliner", "winged eyeliner", "heavy eyeliner", "cat eye liner", "cat-eye liner"]):
        return "dramatic_eyeliner"
    if any(k in text for k in ["smoky eye", "smokey eye"]):
        return "smoky_eye_makeup"
    if any(k in text for k in ["false eyelashes", "fake eyelashes", "long false lashes", "heavy lashes"]):
        return "false_eyelashes"
    if any(k in text for k in ["glossy lips", "lip gloss"]):
        return "glossy_lips"
    if "face_paint" in text or "face paint" in text:
        return "face_paint"
    if any(k in text for k in ["fantasy makeup", "elf makeup", "demon makeup"]):
        return "fantasy_makeup"
    if "natural_makeup" in text or "natural makeup" in text:
        return "natural_makeup"
    token = normalize_text(row.get("makeup_style"))
    return token if token in MAKEUP_STYLE_VOCAB else ""


def canonical_look_context(row: Dict[str, Any]) -> str:
    text = normalize_text(" ".join([
        str(row.get("look_context", "")),
        str(row.get("clothing_description", "")),
        str(row.get("background_description", "")),
        str(row.get("makeup_description", "")),
    ]))
    if not text:
        return ""
    if any(k in text for k in ["gyaru", "gal style"]):
        return "gyaru_style"
    if any(k in text for k in ["cosplay", "cosplayer"]):
        return "cosplay"
    if any(k in text for k in ["character costume", "anime costume", "game character", "character outfit"]):
        return "character_costume"
    if any(k in text for k in ["fantasy costume", "elf", "demon", "armor", "armour", "horns"]):
        return "fantasy_costume"
    if any(k in text for k in ["stage costume", "performance costume", "theatrical"]):
        return "stage_costume"
    if any(k in text for k in ["swimwear", "bikini", "swimsuit"]):
        return "swimwear_costume"
    if any(k in text for k in ["lingerie", "underwear", "bra and", "lace bra"]):
        return "lingerie_costume"
    if "glamour" in text:
        return "glamour"
    if "fashion" in text:
        return "fashion"
    token = normalize_text(row.get("look_context"))
    return token if token in LOOK_CONTEXT_VOCAB else "regular_photo"


def canonical_costume_accessories(row: Dict[str, Any]) -> List[str]:
    raw = row.get("costume_accessories", [])
    tokens: List[str] = []
    if isinstance(raw, list):
        tokens.extend(normalize_text(x) for x in raw if normalize_text(str(x)))
    elif isinstance(raw, str) and raw.strip():
        tokens.extend(normalize_text(x) for x in re.split(r"[,;/|]+", raw) if normalize_text(x))

    text = normalize_text(" ".join([
        str(row.get("clothing_description", "")),
        str(row.get("pose_description", "")),
        str(row.get("background_description", "")),
    ]))
    patterns = [
        ("cat_ears", ["cat ears"]),
        ("fox_ears", ["fox ears"]),
        ("bunny_ears", ["bunny ears", "rabbit ears"]),
        ("animal_ears", ["animal ears"]),
        ("elf_ears", ["elf ears", "elven ears"]),
        ("pointed_ears", ["pointed ears"]),
        ("horns", ["horns", "demon horns"]),
        ("antlers", ["antlers"]),
        ("wings", ["wings"]),
        ("feather_headpiece", ["feather headpiece", "feathered headpiece", "black feathers"]),
        ("headband", ["headband"]),
        ("hair_bow", ["hair bow", "bow in her hair"]),
        ("hair_ribbon", ["hair ribbon", "ribbon in her hair"]),
        ("forehead_jewel", ["forehead jewel", "forehead gem", "jewel on forehead"]),
        ("tiara", ["tiara"]),
        ("crown", ["crown"]),
        ("halo", ["halo"]),
        ("veil", ["veil"]),
        ("hood", ["hood"]),
        ("hat", ["hat"]),
        ("cap", ["cap"]),
        ("helmet", ["helmet"]),
        ("mask", ["mask"]),
        ("choker", ["choker"]),
        ("collar", ["collar"]),
        ("necklace", ["necklace"]),
        ("gloves", ["gloves"]),
        ("arm_guards", ["arm guards", "bracers"]),
        ("wrist_cuffs", ["wrist cuffs", "cuffs"]),
        ("fantasy_armor", ["fantasy armor", "fantasy armour", "armor", "armour"]),
        ("shoulder_armor", ["shoulder armor", "shoulder armour", "pauldron"]),
        ("prop_sword", ["sword"]),
        ("prop_gun", ["gun", "pistol"]),
        ("prop_staff", ["staff", "wand"]),
        ("prop_bottle", ["bottle"]),
        ("prop_book", ["book"]),
    ]
    for token, needles in patterns:
        if any(n in text for n in needles):
            tokens.append(token)
    valid = [t for t in tokens if t in COSTUME_ACCESSORY_VOCAB and t not in {"none_visible", "unclear"}]
    return sorted(dict.fromkeys(valid))

def canonical_hair_form(row: Dict[str, Any]) -> str:
    text = normalize_text(" ".join([
        str(row.get("hair_description", "")),
        str(row.get("hair_texture", "")),
    ]))
    if not text:
        return ""
    if _contains_any(text, ["covered hair", "hair covered", "headscarf", "hijab", "hood covers hair", "beanie covers hair"]):
        return "covered_hair"
    if _contains_any(text, ["side shaved", "shaved side"]):
        return "side_shaved"
    if "undercut" in text:
        return "undercut"
    if _contains_any(text, ["curtain bangs"]):
        return "curtain_bangs"
    if "bangs" in text or "fringe" in text:
        return "bangs"
    if _contains_any(text, ["dreadlock", "locs", "dreads"]):
        return "dreadlocks"
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
    if "high ponytail" in text:
        return "high_ponytail"
    if "low ponytail" in text:
        return "low_ponytail"
    if "ponytail" in text:
        return "ponytail"
    if _contains_any(text, ["messy bun"]):
        return "messy_bun"
    if _contains_any(text, ["high bun", "top knot"]):
        return "high_bun"
    if "low bun" in text:
        return "low_bun"
    if _contains_any(text, ["bun", "chignon"]):
        return "bun"
    if _contains_any(text, ["updo", "up-do"]):
        return "updo"
    if _contains_any(text, ["half up", "half-up"]):
        return "half_up"
    if _contains_any(text, ["pulled back", "tied back", "slicked back"]):
        return "pulled_back"
    if _contains_any(text, ["shaved head", "completely shaved"]):
        return "shaved_head"
    if _contains_any(text, ["buzz cut", "buzzcut"]):
        return "buzz_cut"
    if _contains_any(text, ["pixie"]):
        return "pixie_cut"
    if _contains_any(text, ["long bob", "lob cut", "lob"]):
        return "lob_cut"
    if _contains_any(text, ["bob cut", "bob haircut", "bob"]):
        return "bob_cut"
    if _contains_any(text, ["short hair", "short cut"]):
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


def canonical_hair_length(row: Dict[str, Any]) -> str:
    explicit = normalize_text(row.get("hair_length"))
    if explicit in HAIR_LENGTH_VOCAB:
        return explicit
    text = normalize_text(row.get("hair_description"))
    if not text:
        return ""
    if _contains_any(text, ["not visible", "covered", "hidden"]):
        return "not_visible"
    if _contains_any(text, ["shaved head", "shaved"]):
        return "shaved"
    if _contains_any(text, ["very short", "buzz cut", "buzzcut", "pixie"]):
        return "very_short"
    if _contains_any(text, ["short hair", "short cut"]):
        return "short"
    if _contains_any(text, ["chin length", "chin-length", "bob"]):
        return "chin_length"
    if _contains_any(text, ["shoulder length", "shoulder-length", "to the shoulders"]):
        return "shoulder_length"
    if _contains_any(text, ["medium length", "medium-length"]):
        return "medium_length"
    if _contains_any(text, ["very long", "waist length", "waist-length"]):
        return "very_long"
    if _contains_any(text, ["long hair", "long "]):
        return "long"
    return ""


def canonical_body_height_impression(row: Dict[str, Any]) -> str:
    explicit = normalize_text(row.get("body_height_impression"))
    if explicit in BODY_HEIGHT_IMPRESSION_VOCAB:
        return explicit
    text = normalize_text(" ".join([
        str(row.get("body_build", "")),
        str(row.get("pose_description", "")),
        str(row.get("short_reason", "")),
    ]))
    if not text:
        return ""
    if _contains_any(text, ["short stature", "appears short", "shorter"]):
        return "short"
    if _contains_any(text, ["tall", "long-limbed", "long limbed"]):
        return "tall"
    if _contains_any(text, ["average height", "average-height"]):
        return "average_height"
    return ""

def canonical_makeup_intensity(row: Dict[str, Any]) -> str:
    explicit = normalize_text(row.get("makeup_intensity"))
    if explicit in MAKEUP_INTENSITY_VOCAB:
        return explicit
    text = normalize_text(row.get("makeup_description"))
    if not text:
        return ""
    if _contains_any(text, ["face paint", "face-paint", "painted face"]):
        return "face_paint"
    if _contains_any(text, ["costume makeup", "cosplay makeup", "special effects makeup", "sfx makeup"]):
        return "costume_makeup"
    if _contains_any(text, ["stage makeup", "theatrical makeup"]):
        return "stage_makeup"
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
        "low_ponytail": f"{color_prefix()}hair in a low ponytail",
        "high_ponytail": f"{color_prefix()}hair in a high ponytail",
        "pigtails": f"{color_prefix()}hair in pigtails",
        "bun": f"{color_prefix()}hair in a bun",
        "low_bun": f"{color_prefix()}hair in a low bun",
        "high_bun": f"{color_prefix()}hair in a high bun",
        "messy_bun": f"{color_prefix()}hair in a messy bun",
        "updo": f"{color_prefix()}hair in an updo",
        "two_braids": f"{color_prefix()}hair in two braids",
        "single_braid": f"{color_prefix()}hair in a single braid",
        "box_braids": f"{color_prefix()}box braids",
        "knotless_braids": f"{color_prefix()}knotless braids",
        "cornrows": f"{color_prefix()}cornrows",
        "dreadlocks": f"{color_prefix()}dreadlocks",
        "pixie_cut": f"{color_prefix()}pixie cut",
        "bob_cut": f"{color_prefix()}bob cut",
        "lob_cut": f"{color_prefix()}long bob cut",
        "short_cut": f"short {color_prefix()}hair".strip(),
        "buzz_cut": f"{color_prefix()}buzz cut",
        "shaved_head": "shaved head",
        "undercut": f"{color_prefix()}hair with an undercut",
        "side_shaved": f"{color_prefix()}hair with one side shaved",
        "bangs": f"{color_prefix()}hair with bangs",
        "curtain_bangs": f"{color_prefix()}hair with curtain bangs",
        "covered_hair": "covered hair",
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
        items = markers.get("piercing_inventory", []) or markers.get("piercing_baseline", [])
    else:
        items = []
    out: Dict[str, str] = {}
    for item in items or []:
        if marker_key == "piercings" and normalize_text(item.get("role", "")) == "ignore":
            continue
        loc = normalize_text(item.get("location"))
        desc = compact_trait(item.get("canonical_description"))
        if loc and desc:
            out[loc] = desc
    return out


def _piercing_inventory_by_location(profile: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    markers = (profile or {}).get("identity_markers", {}) if isinstance(profile, dict) else {}
    items = markers.get("piercing_inventory", []) or markers.get("piercing_baseline", []) or []
    out: Dict[str, Dict[str, Any]] = {}
    for entry in items:
        if not isinstance(entry, dict):
            continue
        loc = _canonical_piercing_location(entry.get("location"))
        if not loc:
            continue
        role = normalize_text(entry.get("role", "")) or (
            "accessory" if _piercing_category(loc) == "ear_jewelry" else "canonical"
        )
        out[loc] = {
            "location": loc,
            "canonical_description": compact_trait(entry.get("canonical_description")) or "piercing",
            "frequency": str(entry.get("frequency", "") or ""),
            "category": normalize_text(entry.get("category", "")) or _piercing_category(loc),
            "role": role,
        }
    return out


def get_visible_piercing_state(
    item: Dict[str, Any],
    profile: Dict[str, Any],
    image_traits: Dict[str, Any],
    active_policy: Dict[str, Any],
    caption_profile: str,
) -> Dict[str, Any]:
    """Resolve visible piercings/ear jewelry against editable profile roles.

    ERNIE/all-fields profiles may describe every visible non-ignored marker.
    Z-Image/Krea profiles omit canonical body markers and describe only visible
    `variable` or `accessory` entries. This keeps the trigger responsible for
    the canon while still separating temporary jewelry from identity.
    """
    if not active_policy.get("include_piercings"):
        return {"phrases": [], "must_caption": False, "entries": []}

    inventory = _piercing_inventory_by_location(profile)
    locations = image_traits.get("piercing_locations_visible", [])
    if not isinstance(locations, list):
        locations = []

    raw_by_location: Dict[str, str] = {}
    for raw in item.get("piercing_inventory_now") or []:
        if not isinstance(raw, dict):
            continue
        loc = _canonical_piercing_location(raw.get("location"))
        raw_by_location[loc] = _canonicalize_piercing_description(loc, str(raw.get("description", "")))
        if loc not in locations:
            locations.append(loc)

    resolved: List[Dict[str, Any]] = []
    for loc_raw in locations:
        loc = _canonical_piercing_location(loc_raw)
        if not loc:
            continue
        meta = dict(inventory.get(loc, {}))
        if not meta:
            meta = {
                "location": loc,
                "canonical_description": raw_by_location.get(loc) or "piercing",
                "frequency": "",
                "category": _piercing_category(loc),
                "role": "accessory" if _piercing_category(loc) == "ear_jewelry" else "variable",
            }
        role = normalize_text(meta.get("role", "")) or "variable"
        if role == "ignore":
            continue
        # Stable body markers belong to the trigger in identity-focused targets.
        if caption_profile not in {"ernie", "shared_compact"} and role == "canonical":
            continue
        if normalize_text(meta.get("category")) == "ear_jewelry":
            phrase = raw_by_location.get(loc) or compact_trait(meta.get("canonical_description")) or "earring"
        else:
            phrase = compact_trait(meta.get("canonical_description")) or raw_by_location.get(loc) or "piercing"
        if phrase:
            resolved.append({**meta, "canonical_description": phrase})

    phrases = _dedupe_phrase_list([str(x.get("canonical_description", "")) for x in resolved])
    return {
        "phrases": phrases,
        "must_caption": bool(phrases),
        "entries": resolved,
    }


def _piercing_caption_anchor(phrase: str) -> str:
    t = normalize_compact_text(phrase)
    for anchor in (
        "septum", "lower lip", "lower-lip", "labret", "upper lip", "upper-lip",
        "nose", "eyebrow", "navel", "ear gauge", "earring", "lip", "piercing",
    ):
        if anchor in t:
            return anchor.replace("-", " ")
    words = t.split()
    return " ".join(words[-2:]) if len(words) >= 2 else t


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
        "hair_color_modifier": canonical_hair_color_modifier(row),
        "hair_form": canonical_hair_form(row),
        "hair_length": canonical_hair_length(row),
        "eye_color_base": canonical_eye_color(row) if eye_color_is_reliable(row) else "",
        "eye_color_reliable": eye_color_is_reliable(row),
        "eye_appearance": canonical_eye_appearance(row),
        "body_height_impression": canonical_body_height_impression(row),
        "makeup_intensity": canonical_makeup_intensity(row),
        "makeup_style": canonical_makeup_style(row),
        "look_context": canonical_look_context(row),
        "costume_accessories": canonical_costume_accessories(row),
        "freckles_visible": bool(compact_trait(row.get("freckles_description"))),
        "freckles_description": compact_trait(row.get("freckles_description")),
        "glasses_visible": _profile_bool(row.get("has_glasses_now")) or bool(compact_trait(row.get("glasses_description"))),
        "glasses_description": compact_trait(row.get("glasses_description")),
        "glasses_frame_shape": normalize_text(row.get("glasses_frame_shape")),
        "beard_pattern": normalize_text(normalize_beard_tag(row.get("beard_description", "")).get("pattern", "")),
        "beard_color": normalize_text(normalize_beard_tag(row.get("beard_description", "")).get("color", "")),
        "beard_visible": bool(normalize_beard_tag(row.get("beard_description", "")).get("visible", False)),
        "tattoo_locations_visible": sorted(set(tattoos_visible)),
        "piercing_locations_visible": sorted(set(piercings_visible)),
        "frame_subtype": normalize_text(row.get("frame_subtype")),
        "gaze_category": normalize_text(row.get("gaze_category")),
        "expression_category": normalize_text(row.get("expression_category")),
        "occlusion_type": normalize_text(row.get("occlusion_type")),
        "visual_style_type": normalize_text(row.get("visual_style_type")),
        "glasses_frame_material": normalize_text(row.get("glasses_frame_material")),
        "glasses_lens_type": normalize_text(row.get("glasses_lens_type")),
        "glasses_position": infer_glasses_position(row),
    }



def _appearance_hair_family(token: str) -> str:
    t = normalize_text(token)
    if t in {"blonde", "dark_blonde", "platinum"}:
        return "blonde_family"
    if t in {"light_brown", "brown", "dark_brown", "black"}:
        return "brown_dark_family"
    if t in {"strawberry_blonde", "auburn", "red", "copper", "burgundy"}:
        return "red_auburn_family"
    if t in {"gray", "silver", "white"}:
        return "gray_white_family"
    if not t:
        return "unknown_hair"
    return f"{t}_family"


def _appearance_glasses_family(row: Dict[str, Any], traits: Dict[str, Any]) -> str:
    desc = normalize_text(row.get("glasses_description"))
    if any(k in desc for k in ["sunglass", "dark_lens", "tinted_lens", "shades"]):
        return "sunglasses"
    if _profile_bool(row.get("has_glasses_now")) or bool(traits.get("glasses_visible")):
        return "regular_glasses"
    if desc:
        return "glasses_unclear"
    return "no_glasses"


def _appearance_visual_group(row: Dict[str, Any]) -> str:
    if bool(row.get("is_grayscale_filter")):
        return "filtered_bw"
    label = normalize_text(row.get("color_tint_label"))
    strength = float(row.get("color_tint_strength", 0.0) or 0.0)
    if label and strength >= float(TINT_MIN_STRENGTH_FOR_CAPTION):
        return "filtered_tinted"
    return "clean"


def _appearance_frame_group(row: Dict[str, Any]) -> str:
    shot = normalize_text(row.get("shot_type"))
    if shot == "full_body":
        return "body"
    if shot == "medium":
        return "medium"
    return "face"


def _safe_cluster_id(label: str, existing: set) -> str:
    base = re.sub(r"[^a-zA-Z0-9_\-]+", "_", label).strip("_").lower() or "look"
    cid = base
    i = 2
    while cid in existing:
        cid = f"{base}_{i}"
        i += 1
    existing.add(cid)
    return cid


def build_identity_appearance_clusters(rows: List[Dict[str, Any]], profile: Dict[str, Any]) -> Dict[str, Any]:
    """Baut grobe, UI-taugliche Appearance-Cluster fuer den Personality-/Profile-Bereich.

    Ziel ist NICHT perfekte automatische Phasenerkennung. Ziel ist ein Entscheidungsboard:
    wenige grobe Gruppen, die der User als core / variation / body_reference / review / exclude
    markieren kann. Tints, B/W, Sonnenbrillen und Body-Shots werden bewusst nicht als Core
    vorgeschlagen.
    """
    if not ENABLE_IDENTITY_APPEARANCE_CLUSTERING:
        return {"clusters": [], "member_roles": {}, "member_clusters": {}, "warnings": []}

    usable = [r for r in rows if r.get("base_status") in {"keep", "review"} and r.get("arcface_flag") != "hard"]
    if not usable:
        return {"clusters": [], "member_roles": {}, "member_clusters": {}, "warnings": []}

    per_img = profile.get("per_image_traits", {}) or {}

    # Haupt-Haarfamilie aus sauberen Face/Medium-Bildern bestimmen. Das ist nur
    # ein Priorisierungssignal, kein harter Reject.
    hair_counter = Counter()
    for row in usable:
        image_id = profile_image_id(row)
        traits = per_img.get(image_id) or per_image_profile_traits(row, profile)
        frame = _appearance_frame_group(row)
        visual = _appearance_visual_group(row)
        if frame in {"face", "medium"} and visual == "clean":
            hair_counter[_appearance_hair_family(str(traits.get("hair_color_base", "")))] += 1
    main_hair = hair_counter.most_common(1)[0][0] if hair_counter else "unknown_hair"

    raw: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    row_meta: Dict[str, Dict[str, str]] = {}
    for row in usable:
        image_id = profile_image_id(row)
        traits = per_img.get(image_id) or per_image_profile_traits(row, profile)
        frame = _appearance_frame_group(row)
        hair = _appearance_hair_family(str(traits.get("hair_color_base", "")))
        glasses = _appearance_glasses_family(row, traits)
        visual = _appearance_visual_group(row)
        # Seltene Einzelfaelle nicht als 10 Einzel-Looks aufblasen.
        key = (frame, hair, glasses, visual)
        raw[key].append(row)
        row_meta[image_id] = {"frame": frame, "hair": hair, "glasses": glasses, "visual": visual}

    # Zweite Zusammenfassung: seltene Outlier und Filterbilder grob bündeln.
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for key, members in raw.items():
        frame, hair, glasses, visual = key
        if frame == "body":
            new_key = ("body", hair if hair == main_hair else "other_hair", glasses, "filtered" if visual != "clean" else "clean")
        elif visual != "clean":
            # Tinted/BW trennt Core nicht weiter in zig Mini-Looks.
            new_key = (frame, hair if hair == main_hair else "other_hair", glasses, "filtered")
        elif len(members) == 1 and hair != main_hair:
            new_key = ("outlier", "other_hair", "mixed_glasses", "clean")
        else:
            new_key = key
        grouped[new_key].extend(members)

    clusters: List[Dict[str, Any]] = []
    member_roles: Dict[str, str] = {}
    member_clusters: Dict[str, str] = {}
    used_ids: set = set()
    warnings: List[str] = []

    for key, members in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        frame, hair, glasses, visual = key
        n = len(members)
        role = "variation"
        if frame == "body":
            role = "body_reference"
        elif visual != "clean" or visual == "filtered":
            role = "variation" if n >= 2 else "review"
        elif glasses == "sunglasses":
            role = "variation" if n >= 2 else "review"
        elif frame in {"face", "medium"} and hair == main_hair and n >= 2:
            role = "core"
        elif frame == "outlier":
            role = "review"

        label_parts = [frame, hair, glasses, visual]
        cid = _safe_cluster_id("look_" + "_".join(label_parts), used_ids)

        shot_counts = Counter(normalize_text(r.get("shot_type")) or "unknown" for r in members)
        style_counts = Counter(_appearance_visual_group(r) for r in members)
        quality_avg = sum(float(r.get("quality_total", 0) or 0) for r in members) / max(1, n)
        identity_avg = sum(float(r.get("quality_identity_usefulness", 0) or 0) for r in members) / max(1, n)
        member_ids = [profile_image_id(r) for r in members]
        filenames = [r.get("original_filename", "") for r in members]
        image_paths = [r.get("original_path", "") or r.get("source_path", "") or r.get("image_path", "") for r in members]
        summary = f"{frame} | {hair} | {glasses} | {visual}"
        cluster = {
            "cluster_id": cid,
            "role": role,
            "n": n,
            "summary": summary,
            "frame_group": frame,
            "hair_family": hair,
            "glasses_family": glasses,
            "visual_group": visual,
            "avg_quality_total": round(quality_avg, 1),
            "avg_identity_usefulness": round(identity_avg, 1),
            "shot_counts": dict(shot_counts),
            "style_counts": dict(style_counts),
            "members": member_ids,
            "filenames": filenames,
            "image_paths": image_paths,
        }
        clusters.append(cluster)
        for mid in member_ids:
            member_roles[mid] = role
            member_clusters[mid] = cid

    if len(clusters) > 10:
        warnings.append(
            f"Detected {len(clusters)} appearance clusters. If this is hard to review, merge/re-bucket variable traits first."
        )
    if any(c.get("visual_group") != "clean" and c.get("role") == "core" for c in clusters):
        warnings.append("Filtered / black-and-white / tinted clusters should not be core by default.")
    if len(hair_counter) > 2:
        warnings.append("High hair-family variance across clean portrait clusters: " + ", ".join(f"{k}={v}" for k, v in hair_counter.most_common()))

    return {
        "schema": IDENTITY_CLUSTER_SCHEMA_VERSION,
        "main_hair_family": main_hair,
        "clusters": clusters,
        "member_roles": member_roles,
        "member_clusters": member_clusters,
        "warnings": warnings,
    }


def attach_identity_clusters_to_profile(profile: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    data = build_identity_appearance_clusters(rows, profile)
    profile["identity_cluster_schema_version"] = IDENTITY_CLUSTER_SCHEMA_VERSION
    profile["identity_clusters"] = data.get("clusters", [])
    profile["identity_cluster_member_roles"] = data.get("member_roles", {})
    profile["identity_cluster_member_clusters"] = data.get("member_clusters", {})
    notes = profile.setdefault("normalizer_notes", [])
    if isinstance(notes, list):
        for w in data.get("warnings", []) or []:
            notes.append("Identity clustering: " + str(w))
        profile["normalizer_notes"] = notes[-30:]
    return profile


def identity_cluster_role_for_row(row: Dict[str, Any], profile: Optional[Dict[str, Any]]) -> str:
    if not profile:
        return ""
    roles = profile.get("identity_cluster_member_roles", {}) or {}
    image_id = row.get("profile_image_id") or profile_image_id(row)
    role = str(roles.get(image_id, "") or "").strip().lower()
    if role not in IDENTITY_CLUSTER_TRAIN_ROLES and role not in IDENTITY_CLUSTER_NONTRAIN_ROLES:
        return ""
    return role


def apply_identity_cluster_roles_to_rows(rows: List[Dict[str, Any]], profile: Optional[Dict[str, Any]]) -> None:
    for row in rows:
        role = identity_cluster_role_for_row(row, profile)
        if role:
            row["identity_cluster_role"] = role


def identity_cluster_role_bonus(item: Dict[str, Any], selected: List[Dict[str, Any]]) -> float:
    role = str(item.get("identity_cluster_role", "") or "").strip().lower()
    if not role:
        return 0.0
    if role == "core":
        core_count = sum(1 for s in selected if str(s.get("identity_cluster_role", "") or "").strip().lower() == "core")
        max_core = max(2, int(round(float(TARGET_DATASET_SIZE) * float(IDENTITY_CLUSTER_MAX_CORE_SHARE))))
        bonus = float(IDENTITY_CLUSTER_CORE_SCORE_BOOST)
        if core_count >= max_core:
            bonus -= float(IDENTITY_CLUSTER_CORE_OVERFLOW_PENALTY) * (core_count - max_core + 1)
        return bonus
    if role == "variation":
        return float(IDENTITY_CLUSTER_VARIATION_SCORE_BOOST)
    if role == "body_reference":
        return float(IDENTITY_CLUSTER_BODY_SCORE_BOOST)
    if role in {"review", "exclude"}:
        return -9999.0
    return 0.0


def rebuild_selection_from_identity_roles(
    all_rows: List[Dict[str, Any]],
    profile: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Phase-3 Re-Ranking nach UI-Clusterrollen.

    Fuer AI Toolkit gibt es weiterhin nur 'rein' oder 'raus'. Die Rollen steuern hier:
    - core / variation / body_reference -> Kandidaten fuer 01_train_ready
    - review -> 04_review
    - exclude -> 02_keep_unused
    Core bekommt einen kleinen Score-Boost, wird aber durch MAX_CORE_SHARE begrenzt,
    damit Variation nicht wegoptimiert wird.
    """
    apply_identity_cluster_roles_to_rows(all_rows, profile)
    has_roles = any(r.get("identity_cluster_role") for r in all_rows)
    has_canon_selection = bool(
        ENABLE_CANON_REPRESENTATION_BONUS
        and normalize_text((profile.get("canonical_features", {}) or {}).get("hair_color", ""))
    )
    if not has_roles and not has_canon_selection:
        return [], [], []

    train_candidates: List[Dict[str, Any]] = []
    review_items: List[Dict[str, Any]] = []
    overflow_items: List[Dict[str, Any]] = []
    for row in all_rows:
        if row.get("arcface_flag") == "hard" or row.get("base_status") == "reject":
            continue
        role = str(row.get("identity_cluster_role", "") or "").strip().lower()
        if not has_roles:
            if row.get("base_status") == "keep":
                train_candidates.append(row)
            elif row.get("base_status") == "review":
                review_items.append(row)
            continue
        if role in IDENTITY_CLUSTER_TRAIN_ROLES:
            train_candidates.append(row)
        elif role == "review" or row.get("base_status") == "review":
            review_items.append(row)
        elif role == "exclude":
            overflow_items.append(row)
        elif row.get("base_status") == "keep":
            # Falls kein Cluster-Mapping fuer ein Bild existiert: nicht verlieren,
            # aber nur als Overflow bereitstellen.
            overflow_items.append(row)

    if not train_candidates:
        return [], [], []

    selected = choose_final_dataset(train_candidates, profile)
    selected = crop_dedup_selected(selected)
    selected, final_duplicate_rows = dedup_final_selected_scene_variants(selected)
    selected_names = {r.get("original_filename") for r in selected}

    unselected = [
        r for r in train_candidates
        if r.get("original_filename") not in selected_names
        and r.get("base_status") != "reject"
    ] + overflow_items

    shot_order = {"headshot": 0, "medium": 1, "full_body": 2}
    selected_sorted = sorted(
        selected,
        key=lambda r: (
            shot_order.get(r.get("shot_type"), 9),
            {"core": 0, "variation": 1, "body_reference": 2}.get(str(r.get("identity_cluster_role", "")), 9),
            -float(r.get("quality_total", 0) or 0),
        ),
    )
    review_sorted = sorted(review_items, key=lambda r: -float(r.get("quality_total", 0) or 0))
    unselected_sorted = sorted(unselected, key=lambda r: -float(r.get("quality_total", 0) or 0))
    return selected_sorted, review_sorted, unselected_sorted


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
                "body_height_impression": "",
            },
            "canonical_features": {
                "hair_color": "",
                "hair_form": "",
                "eye_color": "",
                "beard_pattern": "",
                "beard_color": "",
            },
            "identity_markers": {
                "glasses": {
                    "wears_regularly": False,
                    "canonical_description": "",
                },
                "freckles": {
                    "has_freckles": False,
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
            and str(profile.get("normalizer_model", "")).strip().lower() != "fallback_local"
        ):
            return profile
    except Exception:
        return None
    return None



def _trait_variance(values: List[str], min_unique: int = 3, max_mode_fraction: float = 0.70) -> Tuple[bool, Dict[str, Any]]:
    clean = [normalize_text(v) for v in values if normalize_text(v) and normalize_text(v) not in {"none", "unknown", "unclear", "not_visible", "n_a"}]
    if not clean:
        return False, {"total": 0, "unique": 0, "mode": "", "mode_fraction": 0.0, "minority_count": 0, "variation_detected": False, "counts": {}}
    counts = Counter(clean)
    mode, count = counts.most_common(1)[0]
    total = len(clean)
    unique = len(counts)
    mode_fraction = count / max(1, total)
    minority_count = total - count
    # "variable" remains the stronger high-variation signal used for cosplay/
    # appearance-mode classification. "variation_detected" is intentionally
    # more permissive and drives the UI caption policy. Two agreeing minority
    # observations are enough; a single outlier does not force all baseline
    # images to be captioned in the all-visible mode.
    variation_detected = unique >= 2 and minority_count >= 2
    variable = unique >= min_unique and mode_fraction <= max_mode_fraction
    return variable, {
        "total": total,
        "unique": unique,
        "mode": mode,
        "mode_fraction": round(mode_fraction, 3),
        "minority_count": minority_count,
        "variation_detected": variation_detected,
        "counts": dict(counts.most_common(12)),
    }


def _state_stats(values: List[str]) -> Dict[str, Any]:
    clean = [normalize_text(v) for v in values if normalize_text(v)]
    if not clean:
        return {"total": 0, "unique": 0, "mode": "", "mode_fraction": 0.0, "minority_count": 0, "variation_detected": False, "counts": {}}
    counts = Counter(clean)
    mode, count = counts.most_common(1)[0]
    total = len(clean)
    minority_count = total - count
    return {
        "total": total,
        "unique": len(counts),
        "mode": mode,
        "mode_fraction": round(count / max(1, total), 3),
        "minority_count": minority_count,
        "variation_detected": len(counts) >= 2 and minority_count >= 2,
        "counts": dict(counts.most_common(12)),
    }

def attach_profile_variability_policies(profile: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Attach canonical baselines and measurable per-feature variation.

    The Subject Profile owns the canonical appearance. Caption policy is a
    separate UI decision: either only deviations from that canonical baseline
    are captioned, or every visible state is captioned once genuine variation
    is detected. Normalization therefore never decides omission by itself.
    """
    per_image = profile.get("per_image_traits", {}) or {}
    trait_rows = list(per_image.values()) if isinstance(per_image, dict) else []

    hair_color_variable, hair_color_stats = _trait_variance(
        [t.get("hair_color_base", "") for t in trait_rows], min_unique=4, max_mode_fraction=0.72
    )
    hair_form_variable, hair_form_stats = _trait_variance(
        [t.get("hair_form", "") for t in trait_rows], min_unique=5, max_mode_fraction=0.70
    )
    eye_color_variable, eye_color_stats = _trait_variance(
        [t.get("eye_color_base", "") for t in trait_rows if bool(t.get("eye_color_reliable"))], min_unique=3, max_mode_fraction=0.78
    )
    hair_modifier_stats = _state_stats([
        t.get("hair_color_modifier", "") for t in trait_rows
        if normalize_text(t.get("hair_color_modifier", ""))
    ])

    beard_patterns = [normalize_text(t.get("beard_pattern", "")) for t in trait_rows if t.get("beard_visible")]
    beard_colors = [normalize_text(t.get("beard_color", "")) for t in trait_rows if t.get("beard_visible") and normalize_text(t.get("beard_pattern", "")) != "clean_shaven"]
    beard_pattern_stats = _state_stats(beard_patterns)
    beard_color_stats = _state_stats(beard_colors)

    glasses_states: List[str] = []
    for t in trait_rows:
        desc = normalize_text(t.get("glasses_description", ""))
        lens = normalize_text(t.get("glasses_lens_type", ""))
        if any(k in desc for k in ["sunglass", "shades"]) or lens in {"sunglasses", "tinted_lenses", "reflective_lenses"}:
            glasses_states.append("sunglasses")
        elif bool(t.get("glasses_visible")):
            glasses_states.append("regular_glasses")
        else:
            glasses_states.append("no_glasses")
    glasses_stats = _state_stats(glasses_states)
    regular_glasses_traits = [t for t in trait_rows if bool(t.get("glasses_visible")) and _glasses_lens_family(t.get("glasses_lens_type", "")) != "sunglasses"]
    glasses_frame_stats = _state_stats([
        _glasses_fingerprint(t.get("glasses_frame_shape", ""), t.get("glasses_frame_material", ""), t.get("glasses_lens_type", ""))
        for t in regular_glasses_traits
    ])
    glasses_shape_stats = _state_stats([_glasses_shape_family(t.get("glasses_frame_shape", "")) for t in regular_glasses_traits])
    glasses_material_stats = _state_stats([_glasses_material_family(t.get("glasses_frame_material", "")) for t in regular_glasses_traits])
    glasses_lens_stats = _state_stats([_glasses_lens_family(t.get("glasses_lens_type", "")) for t in regular_glasses_traits])
    glasses_position_stats = _state_stats([
        normalize_text(t.get("glasses_position", "")) for t in trait_rows
        if normalize_text(t.get("glasses_position", ""))
    ])

    eye_appearance_counts = Counter(
        normalize_text(t.get("eye_appearance", "")) for t in trait_rows
        if normalize_text(t.get("eye_appearance", "")) and normalize_text(t.get("eye_appearance", "")) not in {"natural_eyes", "unclear"}
    )
    look_counts = Counter(
        normalize_text(t.get("look_context", "")) for t in trait_rows
        if normalize_text(t.get("look_context", "")) and normalize_text(t.get("look_context", "")) != "unclear"
    )
    cosplay_tokens = {"cosplay", "character_costume", "fantasy_costume", "stage_costume", "gyaru_style"}
    cosplay_count = sum(c for k, c in look_counts.items() if k in cosplay_tokens)
    total = max(1, len(trait_rows))
    cosplay_fraction = cosplay_count / total

    if cosplay_fraction >= 0.25:
        appearance_mode = "cosplay_identity"
    elif hair_color_variable or hair_form_variable or eye_color_variable:
        appearance_mode = "high_variation_model_identity"
    elif look_counts.get("fashion", 0) + look_counts.get("glamour", 0) >= max(3, int(total * 0.25)):
        appearance_mode = "fashion_identity"
    else:
        appearance_mode = "natural_identity"

    canonical = profile.setdefault("canonical_features", {})
    if not normalize_text(canonical.get("hair_color", "")):
        canonical["hair_color"] = hair_color_stats.get("mode", "")
    if not normalize_text(canonical.get("hair_form", "")):
        canonical["hair_form"] = hair_form_stats.get("mode", "")
    if not normalize_text(canonical.get("eye_color", "")):
        canonical["eye_color"] = normalize_text((profile.get("stable_identity", {}) or {}).get("eye_color", "")) or eye_color_stats.get("mode", "")
    if not normalize_text(canonical.get("beard_pattern", "")):
        canonical["beard_pattern"] = beard_pattern_stats.get("mode", "")
    if not normalize_text(canonical.get("beard_color", "")):
        canonical["beard_color"] = beard_color_stats.get("mode", "")
    if not normalize_text(canonical.get("glasses_frame_shape", "")):
        canonical["glasses_frame_shape"] = glasses_shape_stats.get("mode", "")
    if not normalize_text(canonical.get("glasses_frame_material", "")):
        canonical["glasses_frame_material"] = glasses_material_stats.get("mode", "")
    if not normalize_text(canonical.get("glasses_lens_type", "")):
        canonical["glasses_lens_type"] = glasses_lens_stats.get("mode", "")

    policies = profile.setdefault("profile_policies", {})
    policies.update({
        "appearance_mode": appearance_mode,
        "hair_color_policy": "variable" if hair_color_stats.get("variation_detected") else "stable",
        "hair_form_policy": "variable" if hair_form_stats.get("variation_detected") else "stable",
        "eye_color_policy": "variable" if eye_color_stats.get("variation_detected") or bool(eye_appearance_counts) else "stable",
        "beard_policy": "variable" if beard_pattern_stats.get("variation_detected") or beard_color_stats.get("variation_detected") else "stable",
        "glasses_policy": "variable" if glasses_stats.get("variation_detected") or glasses_frame_stats.get("variation_detected") else "stable",
        "makeup_policy": "always_caption_when_visible" if appearance_mode in {"cosplay_identity", "high_variation_model_identity"} else "caption_when_visible",
        "costume_policy": "always_caption_when_visible" if appearance_mode in {"cosplay_identity", "high_variation_model_identity"} else "caption_when_visible",
    })
    profile["profile_appearance_mode"] = appearance_mode
    profile["profile_variability_stats"] = {
        "hair_color": hair_color_stats,
        "hair_form": hair_form_stats,
        "hair_color_modifier": hair_modifier_stats,
        "eye_color": eye_color_stats,
        "beard_pattern": beard_pattern_stats,
        "beard_color": beard_color_stats,
        "glasses": glasses_stats,
        "glasses_frame": glasses_frame_stats,
        "glasses_shape": glasses_shape_stats,
        "glasses_material": glasses_material_stats,
        "glasses_lens": glasses_lens_stats,
        "glasses_position": glasses_position_stats,
        "eye_appearance_counts": dict(eye_appearance_counts.most_common(10)),
        "look_context_counts": dict(look_counts.most_common(10)),
        "cosplay_fraction": round(cosplay_fraction, 3),
    }

    notes = profile.setdefault("normalizer_notes", [])
    if hair_color_stats.get("variation_detected"):
        notes.append(
            f"Hair-color variation detected. Canonical baseline is '{canonical.get('hair_color','')}'. Caption behavior is controlled by the UI variable-feature mode."
        )
    if eye_color_stats.get("variation_detected") or eye_appearance_counts:
        notes.append(
            f"Eye-color/lens variation detected. Canonical baseline remains '{canonical.get('eye_color','')}' and is not discarded."
        )
    if beard_pattern_stats.get("variation_detected") or beard_color_stats.get("variation_detected"):
        notes.append(
            f"Beard variation detected. Canonical baseline is '{canonical.get('beard_pattern','')}'."
        )
    if glasses_stats.get("variation_detected") or glasses_frame_stats.get("variation_detected"):
        notes.append("Glasses-state/frame variation detected; the canonical glasses description and structured frame fingerprint remain the terminology anchor.")
    if appearance_mode == "cosplay_identity":
        notes.append("Cosplay mode enabled from dataset-wide look_context distribution; costume/headpiece/makeup traits remain per-image attributes.")
    profile["normalizer_notes"] = notes[-30:]
    return profile


def _canonical_piercing_location(location: str) -> str:
    loc = normalize_text(location)
    aliases = {
        "lip_labret": "lip_lower",
        "labret": "lip_lower",
        "lower_lip": "lip_lower",
        "septum": "nose_septum",
        "nose_ring_septum": "nose_septum",
        "left_ear_lobe": "ear_lobe_left",
        "right_ear_lobe": "ear_lobe_right",
        "left_ear_helix": "ear_helix_left",
        "right_ear_helix": "ear_helix_right",
    }
    return aliases.get(loc, loc or "other")


def _piercing_category(location: str) -> str:
    return "ear_jewelry" if _canonical_piercing_location(location).startswith("ear_") else "body_piercing"


def _canonicalize_piercing_description(location: str, description: str) -> str:
    loc = _canonical_piercing_location(location)
    desc = normalize_compact_text(description)
    color = ""
    for c in ("gold", "silver", "black", "rose gold"):
        if c in desc:
            color = c
            break
    prefix = (color + " ") if color else ""
    if loc.startswith("ear_"):
        if any(k in desc for k in ["gauge", "plug", "tunnel"]):
            return f"{prefix}ear gauge".strip()
        if any(k in desc for k in ["dangling", "drop earring", "dangle"]):
            return f"{prefix}dangling earring".strip()
        if any(k in desc for k in ["hoop", "ring"]):
            return f"{prefix}hoop earring".strip()
        return f"{prefix}stud earring".strip() if "stud" in desc else (f"{prefix}earring".strip())
    if loc == "nose_septum":
        return f"{prefix}septum ring".strip()
    if loc.startswith("nose_"):
        return f"{prefix}nose ring".strip() if any(k in desc for k in ["ring", "hoop"]) else f"{prefix}nose stud".strip()
    if loc in {"lip_lower", "lip_labret"}:
        return f"{prefix}lower-lip ring".strip() if any(k in desc for k in ["ring", "hoop"]) else f"{prefix}lower-lip stud".strip()
    if loc.startswith("lip_"):
        return f"{prefix}lip ring".strip() if any(k in desc for k in ["ring", "hoop"]) else f"{prefix}lip stud".strip()
    if loc.startswith("eyebrow_"):
        return f"{prefix}eyebrow piercing".strip()
    if loc == "navel":
        return f"{prefix}navel piercing".strip()
    return compact_trait(description) or "piercing"


def attach_piercing_inventory(profile: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep all observed piercings visible in the profile, separate from canon."""
    total = max(1, len(rows))
    observations: Dict[str, List[str]] = defaultdict(list)
    image_hits: Counter = Counter()
    for row in rows:
        seen_locations = set()
        for entry in row.get("piercing_inventory_now") or []:
            loc = _canonical_piercing_location(entry.get("location"))
            desc = _canonicalize_piercing_description(loc, str(entry.get("description", "")))
            observations[loc].append(desc)
            seen_locations.add(loc)
        for loc in seen_locations:
            image_hits[loc] += 1

    markers = profile.setdefault("identity_markers", {})
    existing = {
        _canonical_piercing_location(x.get("location")): x
        for x in (markers.get("piercing_inventory", []) or [])
        if isinstance(x, dict) and normalize_text(x.get("location"))
    }
    inventory = []
    for loc in sorted(observations):
        desc_counts = Counter(observations[loc])
        canonical_desc, canonical_count = desc_counts.most_common(1)[0]
        hits = int(image_hits.get(loc, 0))
        category = _piercing_category(loc)
        if category == "ear_jewelry" and len(desc_counts) >= 2 and (canonical_count / max(1, len(observations[loc]))) < 0.70:
            canonical_desc = "earring"
        old = existing.get(loc, {})
        role = normalize_text(old.get("role", ""))
        if role not in {"canonical", "variable", "accessory", "ignore"}:
            if category == "ear_jewelry":
                role = "accessory"
            elif (hits / total) >= 0.40:
                role = "canonical"
            else:
                role = "variable"
        inventory.append({
            "location": loc,
            "canonical_description": canonical_desc,
            "frequency": f"{hits}/{total}",
            "category": category,
            "role": role,
        })

    markers["piercing_inventory"] = inventory
    markers["piercing_baseline"] = [
        {
            "location": x["location"],
            "canonical_description": x["canonical_description"],
            "frequency": x["frequency"],
        }
        for x in inventory if x.get("role") == "canonical"
    ]
    return profile

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
    # Vorher wurde body_build bereits bei <30% Medium/Fullbody geleert.
    # Fuer Identity-LoRAs mit wenigen, aber wertvollen Body-Shots ist das zu hart:
    # 3-5 gute Body-Referenzen sollen den Koerperbau ins Profil einbringen duerfen.
    if body_eligible < int(PROFILE_BODY_BUILD_MIN_ABSOLUTE) and body_eligible_fraction < float(PROFILE_BODY_BUILD_MIN_FRACTION):
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
            stable["body_height_impression"] = ""
            height_conf = conf.get("body_height_impression", {})
            if not isinstance(height_conf, dict):
                height_conf = {"level": str(height_conf or ""), "reasoning": "", "outliers": []}
            height_conf["level"] = "low"
            height_conf["reasoning"] = existing.get("reasoning", "few full-body observations")
            height_conf.setdefault("outliers", [])
            conf["body_height_impression"] = height_conf
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
    profile = attach_piercing_inventory(profile, rows)
    profile = attach_profile_variability_policies(profile, rows)
    profile["input_hash"] = input_hash
    profile["profile_schema_version"] = PROFILE_CACHE_SCHEMA_VERSION
    profile["subject_id"] = profile.get("subject_id") or SAFE_TRIGGER
    profile["force_only_when_visible"] = True

    # Identity-/Appearance-Cluster fuer den UI-Personality-Bereich.
    # Wird nach per_image_traits gebaut, damit die Cluster dieselben normalisierten
    # Merkmale nutzen wie die Captions.
    profile = attach_identity_clusters_to_profile(profile, rows)

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
        f"body={stable.get('body_build','') or '-'} | "
        f"height={stable.get('body_height_impression','') or '-'} | "
        f"mode={profile.get('profile_appearance_mode','') or '-'}"
    )
    return profile


def subject_profile_report_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    if not profile:
        return {}
    return {
        "subject_id": profile.get("subject_id", ""),
        "profile_schema_version": profile.get("profile_schema_version", ""),
        "normalizer_model": profile.get("normalizer_model", ""),
        "normalizer_source": profile.get("normalizer_source", ""),
        "normalizer_retry_count": profile.get("normalizer_retry_count", 0),
        "normalizer_primary_error": profile.get("normalizer_primary_error", ""),
        "sample_size": profile.get("sample_size", 0),
        "total_usable_images": profile.get("total_usable_images", 0),
        "force_only_when_visible": profile.get("force_only_when_visible", True),
        "stable_identity": profile.get("stable_identity", {}),
        "profile_appearance_mode": profile.get("profile_appearance_mode", ""),
        "profile_policies": profile.get("profile_policies", {}),
        "profile_variability_stats": profile.get("profile_variability_stats", {}),
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


def build_reject_export_text(
    row: Dict[str, Any],
    global_rules: Optional[Dict[str, Any]] = None,
    subject_profile: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Baut den Textinhalt fuer exportierte Reject-.txt-Dateien.

    Behaelt immer den diagnostischen Reject-Header und haengt darunter eine
    normale Caption an, falls fuer die Row genug AI-Daten vorhanden sind und
    der Caption-Build erfolgreich ist.
    """
    reasons_str = build_reject_reason_string(row)
    header = (
        f"REJECTED REASON: {reasons_str}\n"
        f"score={row.get('quality_total', 0)} | "
        f"type={row.get('shot_type', '')} | "
        f"file={row.get('original_filename', '')}\n"
    )

    caption_text = ""
    try:
        caption_text = build_caption(row, global_rules or {}, subject_profile or {})
    except Exception:
        caption_text = ""

    if caption_text:
        row["final_caption"] = caption_text
        return f"{header}\nCaption:\n{caption_text}"
    return header


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



def backfill_train_ready_selection(
    selected: List[Dict[str, Any]],
    candidate_pool: List[Dict[str, Any]],
    target_size: Optional[int] = None,
    subject_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fill replacements after caption-remove/hard-review exclusions.

    TARGET_DATASET_SIZE refers to usable images in 01_train_ready, not to rows
    that later move to caption_remove. Existing selected rows are kept; clean
    high-quality keep candidates are appended until the actual train-ready
    count reaches the target or no safe candidate remains.
    """
    target = int(target_size or TARGET_DATASET_SIZE)
    out = list(selected)
    selected_names = {r.get("original_filename") for r in out}
    clean_count = sum(1 for r in out if not needs_caption_remove(r) and r.get("arcface_flag") != "hard")
    if clean_count >= target:
        return out, []

    desired = quotas_for_target(target, {
        "headshot": sum(1 for r in candidate_pool if r.get("shot_type") == "headshot"),
        "medium": sum(1 for r in candidate_pool if r.get("shot_type") == "medium"),
        "full_body": sum(1 for r in candidate_pool if r.get("shot_type") == "full_body"),
    })
    shot_counts = Counter(r.get("shot_type") for r in out if not needs_caption_remove(r))

    candidates = []
    for row in candidate_pool:
        if row.get("original_filename") in selected_names:
            continue
        if row.get("base_status") != "keep" or row.get("arcface_flag") == "hard":
            continue
        if needs_caption_remove(row):
            continue
        role = normalize_text(row.get("identity_cluster_role", ""))
        if role in {"review", "exclude"}:
            continue
        candidates.append(row)

    added: List[Dict[str, Any]] = []
    while clean_count < target and candidates:
        def score(row: Dict[str, Any]) -> float:
            shot = row.get("shot_type", "headshot")
            quota_gap = max(0, int(desired.get(shot, 0)) - int(shot_counts.get(shot, 0)))
            return (
                float(adjusted_pick_score(row, out))
                + (8.0 if quota_gap > 0 else 0.0)
                + canon_representation_bonus(row, out, subject_profile, candidates)
            )
        best = max(candidates, key=score)
        applied_canon_bonus = canon_representation_bonus(best, out, subject_profile, candidates)
        if applied_canon_bonus > 0:
            best["canon_representation_bonus_applied"] = round(applied_canon_bonus, 3)
            best["canonical_hair_match_strength"] = round(canonical_hair_match_strength(best, subject_profile), 3)
            best.setdefault("status_notes", []).append("backfilled_with_soft_canon_representation_bonus")
        candidates.remove(best)
        best.setdefault("status_notes", []).append("backfill_after_final_bucket_exclusion")
        best["selected"] = True
        out.append(best)
        added.append(best)
        selected_names.add(best.get("original_filename"))
        shot_counts[best.get("shot_type", "headshot")] += 1
        clean_count += 1

    if added:
        safe_print(f"   🔄 Backfill added {len(added)} replacement image(s); actual train-ready target={target}.")
    return out, added

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
        "quality_total", "quality_total_before_local_penalties", "grundscore", "score_nach_eskalation", "quality_sharpness",
        "quality_lighting", "quality_composition", "quality_identity_usefulness", "shot_type",
        "body_orientation", "camera_angle", "depth_of_field", "action_description",
        "prominent_objects", "composition_description", "silhouette_clarity",
        "limb_completeness", "body_reference_usefulness", "perspective_distortion",
        "is_grayscale_filter", "grayscale_penalty", "local_score_penalty_total",
        "color_saturation_mean", "color_channel_delta_mean",
        "color_tint_label", "color_tint_strength",
        "gender_class", "face_visible", "face_occlusion", "multiple_people",
        "main_subject_clear", "watermark_or_overlay", "prominent_readable_text",
        "image_medium",
        "mirror_selfie", "frame_subtype", "visual_style_type",
        "look_context",
        "hair_description", "hair_length", "beard_description", "glasses_description",
        "piercings_description", "makeup_description", "makeup_intensity", "makeup_style",
        "skin_tone", "eye_color", "eye_appearance", "body_build",
        "body_height_impression", "freckles_description", "costume_accessories",
        "body_skin_visibility",
        "face_orientation_in_frame",
        "tattoos_visible", "tattoos_description", "clothing_description", "pose_description",
        "expression", "expression_category", "gaze_direction", "gaze_category",
        "head_pose_bucket", "occlusion_type", "background_description",
        "lighting_description", "lighting_type", "background_type", "hair_texture",
        "has_glasses_now", "glasses_frame_shape",
        "glasses_frame_material", "glasses_lens_type", "glasses_position", "issues",
        "short_reason", "local_override_reasons", "duplicate_of", "duplicate_method",
        "duplicate_distance", "main_face_ratio", "secondary_face_area_ratio",
        "face_count_local", "width", "height",
        "file_size_mb", "arcface_distance_to_centroid", "arcface_flag",
        "canonical_hair_match_strength", "canon_representation_bonus_applied",
        "caption_source", "caption_model", "caption_retry_count", "caption_validation_error", "final_caption",
    ]

    csv_path = os.path.join(OUTPUT_ROOT, f"dataset_audit_{SAFE_TRIGGER}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            row_copy = dict(row)
            row_copy["issues"] = ", ".join(row_copy.get("issues", [])) if isinstance(row_copy.get("issues"), list) else row_copy.get("issues", "")
            row_copy["costume_accessories"] = ", ".join(row_copy.get("costume_accessories", [])) if isinstance(row_copy.get("costume_accessories"), list) else row_copy.get("costume_accessories", "")
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
        "subject_profile_normalizer_source": (subject_profile or {}).get("normalizer_source", ""),
        "subject_profile_normalizer_retry_count": (subject_profile or {}).get("normalizer_retry_count", 0),
        "subject_profile_normalizer_primary_error": (subject_profile or {}).get("normalizer_primary_error", ""),
        "subject_profile_sample_size": (subject_profile or {}).get("sample_size", 0),
        "subject_profile_total_usable_images": (subject_profile or {}).get("total_usable_images", 0),
        "variable_feature_mode": str(VARIABLE_FEATURE_CAPTION_MODE),
        "canon_representation_enabled": canon_representation_summary(all_rows, selected_sorted, subject_profile).get("enabled", False),
        "canonical_hair_color": canon_representation_summary(all_rows, selected_sorted, subject_profile).get("canonical_hair_color", ""),
        "canon_representation_target": canon_representation_summary(all_rows, selected_sorted, subject_profile).get("target", 0),
        "canon_representation_selected": canon_representation_summary(all_rows, selected_sorted, subject_profile).get("selected", 0),
        "canon_representation_eligible_keep": canon_representation_summary(all_rows, selected_sorted, subject_profile).get("eligible_keep_candidates", 0),
        "canon_representation_review_candidates": canon_representation_summary(all_rows, selected_sorted, subject_profile).get("review_candidates", 0),
        "canon_representation_reject_candidates": canon_representation_summary(all_rows, selected_sorted, subject_profile).get("reject_candidates", 0),
        "canon_representation_max_quality_gap": canon_representation_summary(all_rows, selected_sorted, subject_profile).get("max_quality_gap", 0),
        "training_target": normalize_training_target(TRAINING_TARGET),
        "caption_profile": caption_profile_for_training_target(TRAINING_TARGET),
        "audit_model": AI_MODEL,
        "krea_ai_captioning": bool(normalize_training_target(TRAINING_TARGET) == "krea2" and USE_KREA_AI_CAPTIONING),
        "krea_caption_model": KREA_CAPTION_MODEL if normalize_training_target(TRAINING_TARGET) == "krea2" else "",
        "krea_caption_repair_enabled": bool(normalize_training_target(TRAINING_TARGET) == "krea2" and USE_KREA_CAPTION_REPAIR),
        "krea_caption_repair_model": KREA_CAPTION_REPAIR_MODEL if normalize_training_target(TRAINING_TARGET) == "krea2" and USE_KREA_CAPTION_REPAIR else "",
        "caption_primary_count": sum(1 for r in selected_sorted if r.get("caption_source") == "gpt_primary"),
        "caption_repair_count": sum(1 for r in selected_sorted if r.get("caption_source") == "gpt_repair"),
        "caption_local_fallback_count": sum(1 for r in selected_sorted if r.get("caption_source") == "local_fallback"),
        "controlled_buckets": bool(USE_CONTROLLED_BUCKETS),
        "medium_rescue_crop_enabled": bool(ENABLE_MEDIUM_RESCUE_CROP),
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

    # Backward-compatible profile migration: older paused runs do not yet
    # contain canonical_features or the generalized hair/eye/beard/glasses
    # variability statistics. Rebuild only the normalized per-image profile
    # layer from the already audited rows; no OpenAI call or image audit runs.
    needs_feature_migration = (
        subject_profile.get("profile_schema_version") != PROFILE_CACHE_SCHEMA_VERSION
        or not isinstance(subject_profile.get("canonical_features"), dict)
        or not isinstance((subject_profile.get("profile_variability_stats") or {}).get("glasses"), dict)
        or not isinstance((subject_profile.get("profile_variability_stats") or {}).get("beard_pattern"), dict)
        or not isinstance(((subject_profile.get("identity_markers") or {}).get("piercing_inventory")), list)
    )
    if needs_feature_migration:
        migration_rows = [
            r for r in all_rows
            if r.get("base_status") == "keep" and r.get("arcface_flag") != "hard"
        ]
        if migration_rows:
            refreshed_traits: Dict[str, Any] = {}
            for row in migration_rows:
                image_id = profile_image_id(row)
                row["profile_image_id"] = image_id
                refreshed_traits[image_id] = per_image_profile_traits(row, subject_profile)
            subject_profile["per_image_traits"] = refreshed_traits
            subject_profile = attach_piercing_inventory(subject_profile, migration_rows)
            subject_profile = attach_profile_variability_policies(subject_profile, migration_rows)
            subject_profile["profile_schema_version"] = PROFILE_CACHE_SCHEMA_VERSION
            subject_profile["force_only_when_visible"] = True
            subject_profile.setdefault("normalizer_notes", []).append(
                "Profile feature policies migrated locally during continue-from-profile; no new audit/API call was used."
            )
            save_subject_profile(subject_profile)
            safe_print("   🧬 Migrated legacy profile to generalized feature-policy schema.")

    reranked_selected, reranked_review, reranked_unused = rebuild_selection_from_identity_roles(all_rows, subject_profile)
    if reranked_selected:
        safe_print(
            f"   🧩 Identity roles applied: train_ready={len(reranked_selected)} | "
            f"review={len(reranked_review)} | keep_unused={len(reranked_unused)}"
        )
        selected_sorted = reranked_selected
        review_items = reranked_review
        unselected_keep = reranked_unused

    selected_sorted, backfill_added = backfill_train_ready_selection(
        selected_sorted, list(unselected_keep) + list(all_rows), TARGET_DATASET_SIZE, subject_profile
    )
    if backfill_added:
        added_names = {r.get("original_filename") for r in backfill_added}
        unselected_keep = [r for r in unselected_keep if r.get("original_filename") not in added_names]
        warnings.append(f"Backfilled {len(backfill_added)} image(s) so 01_train_ready can reach the requested target after caption-remove/review exclusions.")

    canon_summary = canon_representation_summary(all_rows, selected_sorted, subject_profile)
    if canon_summary.get("enabled") and canon_summary.get("selected", 0) < canon_summary.get("target", 0):
        warnings.append(
            "Canonical hair representation below soft target: "
            f"{canon_summary.get('selected', 0)}/{canon_summary.get('target', 0)} "
            f"for '{canon_summary.get('canonical_hair_color', '')}'. "
            f"Eligible keep={canon_summary.get('eligible_keep_candidates', 0)}, "
            f"review={canon_summary.get('review_candidates', 0)}, "
            f"reject={canon_summary.get('reject_candidates', 0)}. "
            "Review/reject candidates are never promoted automatically."
        )

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
                if should_copy_reject_original(row):
                    shutil.copy2(row["original_path"], img_out)
                else:
                    cropped = body_aware_crop(row["original_path"], row)
                    cropped.save(img_out, "JPEG", quality=100)
                with open(txt_out, "w", encoding="utf-8") as ft:
                    ft.write(build_reject_export_text(row, global_rules, subject_profile))
                _sync_row_update(row_index, row)
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
  
    # ── POSE-FILTER ────────────────────────────────────────────────────
    # Filtere unvorteilhafte Posen (kniend nach vorn, liegend, auf allen Vieren)
    # direkt auf "reject", da das LoRA-Modell diese Haltungen ueberlernt.
    pose_desc = str(item.get("pose_description", "")).lower()
    bad_pose_keywords = [
        "crouched on hands and knees", "on hands and knees", "all fours",
        "lying on a bed", "lying on", "reclining", 
        "kneeling forward", "leaning forward", 
        "crouching forward", "crouched forward"
    ]
    for bad_kw in bad_pose_keywords:
        if bad_kw in pose_desc:
            reasons.append(f"bad_lora_pose: {bad_kw}")
            item.setdefault("status_notes", []).append(f"pose_override: {pose_desc}")
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

    # Ein kleines Gesicht veraendert den tatsaechlichen Shot-Typ des Originals
    # nicht. Stattdessen darf die nachgelagerte Crop-Pipeline einen separaten
    # Headshot- oder Medium-Rettungskandidaten erzeugen. So bleiben Analyse,
    # Rettungs-Crop und finaler Export fachlich getrennt.
    face_intentionally_hidden = (not face_visible) or (face_occlusion == "major")
    if face_ratio > 0.0001 and shot in MIN_FACE_RATIO and face_ratio < MIN_FACE_RATIO[shot]:
        if face_intentionally_hidden:
            item.setdefault("status_notes", []).append("small_face_intentionally_hidden")
        elif item.get("width", 0) >= 1024 and item.get("height", 0) >= 1024:
            item.setdefault("status_notes", []).append("small_face_crop_candidate")
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
        item_is_crop = is_crop_variant(item)
        item_crop_of = item.get("crop_of", "")

        for rep in representatives:
            rep_is_crop = is_crop_variant(rep)
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

                # Hard-Threshold: sehr hohe semantische Aehnlichkeit -> Duplicate
                # OHNE Metadaten-Bedingung. Faengt z.B. mehrere Closeups derselben
                # Person mit minimal anderer Pose / Beschreibung im selben Stil.
                if sim >= CLIP_HARD_DUPLICATE_THRESHOLD:
                    item["duplicate_of"] = rep["original_filename"]
                    item["duplicate_method"] = "clip_hard"
                    item["duplicate_distance"] = round(sim, 6)
                    item["base_status"] = "reject"
                    item.setdefault("status_notes", []).append("near_duplicate_clip_hard")
                    is_dup = True
                    break

                # Soft-Threshold: mittlere semantische Aehnlichkeit -> Duplicate
                # nur wenn Metadaten zustimmen (gleicher Shot + gleiche
                # Clothing/BG/Session). Schuetzt vor false positives bei
                # unterschiedlichen Outfits.
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
    Stellt sicher, dass Original und seine Crop-Variante NICHT beide im finalen
    Dataset landen. Wenn beide ausgewählt wurden, gewinnt der mit dem höheren
    quality_total. Bei Gleichstand gewinnt der Crop (er ist identity-optimierter).
    """
    originals_by_name = {
        r["original_filename"]: r
        for r in selected
        if not is_crop_variant(r)
    }
    crops = [r for r in selected if is_crop_variant(r)]

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


# --- Final selected-set duplicate guard ------------------------------------
# Early pHash/CLIP dedup runs before the final dataset is assembled. In real
# phone/Google/WhatsApp exports the same source moment can still survive as a
# crop/resized variant with a different filename. This final guard is deliberately
# later and stricter in scope: it only compares images that were already selected
# for 01_train_ready and suppresses one when the scene, outfit and pose are close
# enough that both would teach the same frame.
_FINAL_DUP_STOPWORDS = {
    "a", "an", "and", "are", "at", "both", "from", "he", "her", "his", "in",
    "is", "of", "on", "she", "the", "they", "to", "with", "wearing",
    "standing", "sitting", "looking", "photo", "portrait", "image",
}


def _duplicate_token_set(value: Any) -> set:
    text = normalize_compact_text(value)
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", text):
        if len(token) < 3 or token in _FINAL_DUP_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _token_jaccard(a: Any, b: Any) -> float:
    ta = _duplicate_token_set(a)
    tb = _duplicate_token_set(b)
    if not ta and not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def _final_duplicate_text_blob(item: Dict[str, Any]) -> str:
    fields = [
        "clothing_description",
        "pose_description",
        "expression",
        "gaze_direction",
        "background_description",
        "lighting_description",
    ]
    return " ".join(str(item.get(f, "") or "") for f in fields)


def _safe_float_value(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def export_crop_retention_fraction(item: Dict[str, Any]) -> float:
    """Approximate how much of the original frame survives the export crop.

    This is mainly used as a tie-breaker for near-duplicate scene variants. For
    full-body landscape shots a more conservative/wider source often preserves
    arms/legs better after conversion to the AI-Toolkit training aspect ratio.
    """
    w = int(_safe_float_value(item.get("width"), 0))
    h = int(_safe_float_value(item.get("height"), 0))
    if w <= 0 or h <= 0:
        return 0.0

    shot_type = str(item.get("shot_type", "") or "").strip().lower()
    if shot_type in {"medium", "full_body"}:
        aspect = 832 / 1216
    else:
        aspect = 1.0

    crop_h = h
    crop_w = int(round(crop_h * aspect))
    if crop_w > w:
        crop_w = w
        crop_h = int(round(crop_w / aspect))
    return max(0.0, min(1.0, (crop_w * crop_h) / max(1.0, float(w * h))))


def final_duplicate_keeper_score(item: Dict[str, Any]) -> float:
    score = _safe_float_value(item.get("quality_total"), 0.0)
    shot_type = str(item.get("shot_type", "") or "").strip().lower()
    retention = export_crop_retention_fraction(item)
    # Full-body duplicates should prefer the source that survives the export
    # crop better, even when its API quality score is marginally lower.
    if shot_type == "full_body":
        score += retention * 20.0
    elif shot_type == "medium":
        score += retention * 8.0
    else:
        score += retention * 3.0

    width = _safe_float_value(item.get("width"), 0.0)
    height = _safe_float_value(item.get("height"), 0.0)
    score += min((width * height) / 1_000_000.0, 16.0) * 0.15
    return score


def is_same_smart_crop_family(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    a_name = str(a.get("original_filename", "") or "")
    b_name = str(b.get("original_filename", "") or "")
    a_crop_of = str(a.get("crop_of", "") or "")
    b_crop_of = str(b.get("crop_of", "") or "")
    return bool(
        (a_crop_of and a_crop_of == b_name) or
        (b_crop_of and b_crop_of == a_name) or
        (a_crop_of and b_crop_of and a_crop_of == b_crop_of)
    )


def is_final_scene_duplicate(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, str, float]:
    if is_same_smart_crop_family(a, b):
        return True, "smart_crop_family", 1.0

    if str(a.get("shot_type", "") or "") != str(b.get("shot_type", "") or ""):
        return False, "", 0.0

    # CLIP, if present, is the strongest signal. Keep the threshold lower than
    # the global duplicate filter because this only runs inside the already
    # selected final set.
    if USE_CLIP_DUPLICATE_SCORING and a.get("clip_embedding") is not None and b.get("clip_embedding") is not None:
        sim = clip_cosine(a.get("clip_embedding"), b.get("clip_embedding"))
        if sim >= 0.90:
            return True, "final_clip_scene", float(sim)

    clothing_sim = _token_jaccard(a.get("clothing_description"), b.get("clothing_description"))
    background_sim = _token_jaccard(a.get("background_description"), b.get("background_description"))
    pose_sim = _token_jaccard(
        " ".join(str(a.get(f, "") or "") for f in ["pose_description", "expression", "gaze_direction"]),
        " ".join(str(b.get(f, "") or "") for f in ["pose_description", "expression", "gaze_direction"]),
    )
    combo_sim = _token_jaccard(_final_duplicate_text_blob(a), _final_duplicate_text_blob(b))

    # Very conservative text fallback: outfit + background + pose/expression
    # need to agree, or the combined caption-like scene text must be highly
    # overlapping. This catches same-moment crop/original exports without
    # removing ordinary same-outfit variation too aggressively.
    if combo_sim >= 0.52:
        return True, "final_scene_text", combo_sim
    if clothing_sim >= 0.38 and background_sim >= 0.45 and pose_sim >= 0.25:
        return True, "final_scene_text_parts", min(0.99, (clothing_sim + background_sim + pose_sim) / 3.0)

    return False, "", max(combo_sim, clothing_sim, background_sim, pose_sim)


def dedup_final_selected_scene_variants(selected: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Remove final-set crop/scene variants so they do not land in train_ready
    *or* keep_unused.

    Returns (kept_selected, suppressed_duplicates). Suppressed rows are marked as
    rejects/duplicates so the later keep_unused export will not preserve both
    versions.
    """
    if len(selected) < 2:
        return selected, []

    kept: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []

    # Strongest candidate first, but still allow a later item to replace an
    # earlier representative if it is clearly a better export source.
    for item in sorted(selected, key=final_duplicate_keeper_score, reverse=True):
        duplicate_rep = None
        duplicate_method = ""
        duplicate_distance = 0.0
        for rep in kept:
            is_dup, method, distance = is_final_scene_duplicate(item, rep)
            if is_dup:
                duplicate_rep = rep
                duplicate_method = method
                duplicate_distance = distance
                break

        if duplicate_rep is None:
            kept.append(item)
            continue

        item_score = final_duplicate_keeper_score(item)
        rep_score = final_duplicate_keeper_score(duplicate_rep)
        if item_score > rep_score + 0.25:
            # Replace previous representative with the better export source.
            kept.remove(duplicate_rep)
            suppressed.append(duplicate_rep)
            kept.append(item)
            loser = duplicate_rep
            winner = item
        else:
            suppressed.append(item)
            loser = item
            winner = duplicate_rep

        loser["base_status"] = "reject"
        loser["final_status"] = "reject"
        loser["selected"] = False
        loser["duplicate_of"] = winner.get("original_filename", "")
        loser["duplicate_method"] = duplicate_method or "final_scene_duplicate"
        loser["duplicate_distance"] = round(float(duplicate_distance), 6)
        loser.setdefault("status_notes", []).append("final_selected_scene_duplicate")
        loser["short_reason"] = (
            f"final_selected_scene_duplicate_of:{winner.get('original_filename','')} "
            f"method:{duplicate_method or 'final_scene_duplicate'}"
        )
        safe_print(
            f"   🔁 Final duplicate suppressed: {loser.get('original_filename','')} "
            f"→ duplicate of {winner.get('original_filename','')} "
            f"({duplicate_method}, score {final_duplicate_keeper_score(loser):.1f} vs "
            f"{final_duplicate_keeper_score(winner):.1f})"
        )

    return kept, suppressed


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

def visual_style_cluster_key(item: Dict[str, Any]) -> str:
    """Cluster-Key fuer visuellen Bildstil.

    - 'bw'    : Schwarz-Weiss-Bilder (is_grayscale_filter)
    - 'sepia' / 'blue' / 'warm' / 'green' / 'purple' : starker Farbstich
    - 'color' : neutral (kein dominanter Stich)

    Wird im diversity_penalty genutzt, um Konzentration eines Stil-Clusters
    im Final-Set zu bestrafen.
    """
    if bool(item.get("is_grayscale_filter")):
        return "bw"
    label = (item.get("color_tint_label") or "").strip().lower()
    if label:
        return label
    return "color"

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

    # ── Visual-Style-Diversity ──
    # Bestraft Konzentration eines Bildstils (B/W oder Tint) im Final-Set.
    # Wirkt zusaetzlich zur CLIP-Duplicate-Detection - dort werden visuell
    # nahezu identische Bilder hart rejected, hier wird verhindert dass die
    # Auswahl insgesamt von einem Stil-Cluster dominiert wird.
    if ENABLE_VISUAL_STYLE_DIVERSITY:
        style_key = visual_style_cluster_key(item)
        # 'color' (neutral) wird nicht bestraft - das ist der Default-Bucket.
        if style_key != "color":
            style_count = sum(
                1 for s in selected
                if visual_style_cluster_key(s) == style_key
            )
            penalty += max(0, style_count - VISUAL_STYLE_SOFT_LIMIT) * VISUAL_STYLE_PENALTY_WEIGHT

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


def _selection_profile_traits(row: Dict[str, Any], profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    per_image = profile.get("per_image_traits", {}) or {}
    image_id = row.get("profile_image_id") or profile_image_id(row)
    traits = per_image.get(image_id, {}) if isinstance(per_image, dict) else {}
    return traits if isinstance(traits, dict) else {}


def canonical_hair_match_strength(row: Dict[str, Any], profile: Optional[Dict[str, Any]]) -> float:
    """Return 0..1 match strength against the user-confirmed canonical hair color.

    Exact enum matches receive 1.0. Closely related blonde variants may count as
    near-canonical because the UI intentionally keeps them as separate visible
    values while still allowing a blonde canon to be represented by useful
    dark-/platinum-blonde photos. Black and dark brown remain distinct.
    """
    if not isinstance(profile, dict):
        return 0.0
    canonical = normalize_text((profile.get("canonical_features", {}) or {}).get("hair_color", ""))
    if not canonical:
        return 0.0
    traits = _selection_profile_traits(row, profile)
    current = normalize_text(traits.get("hair_color_base", "")) or canonical_hair_color(row)
    if not current:
        return 0.0
    if current == canonical:
        return 1.0

    near_map = {
        "blonde": {"dark_blonde": 0.90, "platinum": 0.85, "strawberry_blonde": 0.50},
        "dark_blonde": {"blonde": 0.90, "platinum": 0.70},
        "platinum": {"blonde": 0.85, "dark_blonde": 0.70},
        "strawberry_blonde": {"blonde": 0.50},
    }
    return float((near_map.get(canonical, {}) or {}).get(current, 0.0))


def canonical_hair_representation_count(selected: List[Dict[str, Any]], profile: Optional[Dict[str, Any]]) -> int:
    # Only full/near matches count toward the target. Reduced strawberry-blonde
    # matches can receive a small bonus but do not silently satisfy a blonde canon.
    return sum(1 for row in selected if canonical_hair_match_strength(row, profile) >= 0.70)


def canon_representation_bonus(
    item: Dict[str, Any],
    selected: List[Dict[str, Any]],
    profile: Optional[Dict[str, Any]],
    competing_pool: Optional[List[Dict[str, Any]]] = None,
) -> float:
    if not bool(ENABLE_CANON_REPRESENTATION_BONUS) or not isinstance(profile, dict):
        return 0.0
    target = max(0, int(CANON_REPRESENTATION_TARGET or 0))
    current_count = canonical_hair_representation_count(selected, profile)
    if target <= 0 or current_count >= target:
        return 0.0

    strength = canonical_hair_match_strength(item, profile)
    if strength <= 0.0:
        return 0.0

    pool = competing_pool or [item]
    best_quality = max((float(r.get("quality_total", 0) or 0) for r in pool), default=0.0)
    item_quality = float(item.get("quality_total", 0) or 0)
    if (best_quality - item_quality) > float(CANON_REPRESENTATION_MAX_QUALITY_GAP):
        return 0.0

    schedule = list(CANON_REPRESENTATION_BONUS_SCHEDULE or [])
    base_bonus = float(schedule[current_count]) if current_count < len(schedule) else 0.0
    return base_bonus * strength


def canon_representation_summary(
    all_rows: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    profile: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    canonical = ""
    if isinstance(profile, dict):
        canonical = normalize_text((profile.get("canonical_features", {}) or {}).get("hair_color", ""))
    def count(rows, predicate=lambda r: True):
        return sum(1 for r in rows if predicate(r) and canonical_hair_match_strength(r, profile) >= 0.70)
    return {
        "enabled": bool(ENABLE_CANON_REPRESENTATION_BONUS and canonical),
        "canonical_hair_color": canonical,
        "target": int(CANON_REPRESENTATION_TARGET or 0),
        "selected": count(selected),
        "eligible_keep_candidates": count(all_rows, lambda r: r.get("base_status") == "keep" and r.get("arcface_flag") != "hard"),
        "review_candidates": count(all_rows, lambda r: r.get("base_status") == "review" or normalize_text(r.get("identity_cluster_role")) == "review"),
        "reject_candidates": count(all_rows, lambda r: r.get("base_status") == "reject" or r.get("arcface_flag") == "hard"),
        "max_quality_gap": float(CANON_REPRESENTATION_MAX_QUALITY_GAP),
    }


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

    # Krea-2-Character soll nicht nur das Gesicht, sondern die Verbindung von
    # Gesicht, Haltung und Koerperproportionen lernen. Der Audit-Wert wird nur
    # in diesem Profil als moderater Auswahlbonus genutzt; starke Perspektiv-
    # verzerrungen oder unklare Silhouetten ziehen entsprechend ab.
    if normalize_caption_profile(globals().get("CAPTION_PROFILE", "ernie")) == "krea2_character":
        shot_type = str(item.get("shot_type", "")).strip().lower()
        if shot_type in {"medium", "full_body"}:
            body_ref = float(item.get("body_reference_usefulness", 0.0) or 0.0)
            base += min(8.0, body_ref * 0.08)
            distortion = str(item.get("perspective_distortion", "")).strip().lower()
            if distortion in {"strong", "severe", "extreme"}:
                base -= 8.0
            silhouette = str(item.get("silhouette_clarity", "")).strip().lower()
            if silhouette in {"poor", "unclear", "low"}:
                base -= 4.0

    # UI-Clusterrollen aus dem Subject-Profile: core darf im Ranking etwas
    # nach oben, aber mit Core-Share-Bremse, damit Variation nicht stirbt.
    base += identity_cluster_role_bonus(item, selected)

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


def choose_final_dataset(clean_keep_items: List[Dict[str, Any]], subject_profile: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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
                scoring_pool = fallback
            else:
                scoring_pool = remaining
            best = max(
                scoring_pool,
                key=lambda x: adjusted_pick_score(x, selected)
                + canon_representation_bonus(x, selected, subject_profile, scoring_pool),
            )
            applied_canon_bonus = canon_representation_bonus(best, selected, subject_profile, scoring_pool)
            if applied_canon_bonus > 0:
                best["canon_representation_bonus_applied"] = round(applied_canon_bonus, 3)
                best["canonical_hair_match_strength"] = round(canonical_hair_match_strength(best, subject_profile), 3)
                best.setdefault("status_notes", []).append("selected_with_soft_canon_representation_bonus")
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

def photo_type_phrase(shot_type: str, mirror_selfie: bool, frame_subtype: str = "") -> str:
    # shot_type bleibt die harte Auswahl-/Quotengruppe. frame_subtype ist nur
    # fuer eine etwas natuerlichere Caption relevant.
    subtype = normalize_text(frame_subtype)
    if mirror_selfie or subtype == "mirror_selfie":
        return {
            "headshot": "mirror selfie photo",
            "medium": "mirror selfie photo",
            "full_body": "full-body mirror selfie photo",
        }.get(shot_type, "mirror selfie photo")

    subtype_map = {
        "close_up": "close-up photo",
        "portrait": "portrait photo",
        "selfie": "selfie photo",
        "three_quarter_body": "three-quarter body portrait photo",
        "full_body": "full-body photo",
        "faceless_body": "faceless body-reference photo",
        "detail_only": "detail photo",
    }
    if subtype in subtype_map:
        return subtype_map[subtype]

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
    if any(x in d for x in ["strawberry blonde", "strawberry-blonde"]):
        color = "strawberry blonde"
    elif any(x in d for x in ["platinum", "white blonde", "white-blonde"]):
        color = "platinum"
    elif any(x in d for x in ["auburn"]):
        color = "auburn"
    elif any(x in d for x in ["red hair", "reddish"]):
        color = "red"
    elif "copper" in d:
        color = "copper"
    elif "blue" in d:
        color = "blue"
    elif "pink" in d:
        color = "pink"
    elif any(x in d for x in ["purple", "violet"]):
        color = "purple"
    elif "green" in d or "mint" in d:
        color = "green"
    elif any(x in d for x in ["dark brown", "dark hair", "brunette", "black"]):
        color = "dark"
    elif any(x in d for x in ["light brown", "dirty blonde"]):
        color = "light brown"
    elif "dark blonde" in d:
        color = "dark blonde"
    elif any(x in d for x in ["brown"]):
        color = "brown"
    elif any(x in d for x in ["blonde", "blond", "light blonde", "light-blonde",
                               "light colored", "light-colored"]):
        color = "blonde"
    elif any(x in d for x in ["white hair", "white wig", "silver hair", "gray hair", "grey hair"]):
        color = "white"
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
    stable_pattern = normalize_text(beard_rule.get("mode_pattern", ""))
    stable_color = normalize_text(beard_rule.get("mode_color", ""))
    if not stable_pattern and stable_mode_raw:
        parsed_mode = normalize_beard_tag(stable_mode_raw)
        stable_pattern = normalize_text(parsed_mode.get("pattern", ""))
        stable_color = normalize_text(parsed_mode.get("color", ""))

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

def build_visual_style_phrase(item: Dict[str, Any]) -> str:
    """Liefert einen optionalen Style-Praefix fuer die Caption.

    Wird vor das photo_type gesetzt. Beispiele:
        "black and white"
        "sepia-toned"
        "blue-tinted"
        "warm-tinted"
        "green-tinted"
        "purple-tinted"

    Leerstring bei Bildern ohne erkennbaren Filter / Stil.
    """
    if not CAPTION_POLICY.get("include_visual_style", True):
        return ""

    if bool(item.get("is_grayscale_filter")):
        return "black and white"

    if not USE_COLOR_TINT_CAPTION:
        return ""

    label = normalize_compact_text(item.get("color_tint_label", ""))
    if not label:
        return ""

    # Mapping label -> Caption-Phrase. Zentrale Stelle, leicht zu pflegen.
    return {
        "sepia": "sepia-toned",
        "blue": "blue-tinted",
        "warm": "warm-tinted",
        "green": "green-tinted",
        "purple": "purple-tinted",
    }.get(label, "")

def build_local_caption(
    item: Dict[str, Any],
    global_rules: Dict[str, Any],
    subject_profile: Optional[Dict[str, Any]] = None,
) -> str:
    shot_type = item.get("shot_type", "headshot")
    mirror_selfie = bool(item.get("mirror_selfie", False))
    photo_type = photo_type_phrase(shot_type, mirror_selfie, item.get("frame_subtype", ""))
    caption_profile = normalize_caption_profile(globals().get("CAPTION_PROFILE", "ernie"))
    active_policy = enforce_caption_policy_profile(caption_profile, CAPTION_POLICY)

    profile = subject_profile or {}
    stable_identity = profile.get("stable_identity", {}) if isinstance(profile, dict) else {}
    per_image_traits = profile.get("per_image_traits", {}) if isinstance(profile, dict) else {}
    image_id = item.get("profile_image_id") or profile_image_id(item)
    image_traits = per_image_traits.get(image_id, {}) if isinstance(per_image_traits, dict) else {}

    gender_class = normalize_feature_value(stable_identity.get("gender")) or normalize_feature_value(item.get("gender_class")) or "person"
    beard_desc = compact_trait(item.get("beard_description"))

    profile_policies = profile.get("profile_policies", {}) if isinstance(profile, dict) else {}
    skin_tone = compact_trait(stable_identity.get("skin_tone")) or compact_trait(item.get("skin_tone"))
    eye_policy = normalize_text(profile_policies.get("eye_color_policy"))
    if eye_policy == "caption_when_clear_or_variable":
        eye_color = compact_trait(image_traits.get("eye_color_base")) or compact_trait(item.get("eye_color"))
    else:
        eye_color = compact_trait(stable_identity.get("eye_color")) or compact_trait(item.get("eye_color"))
    # Body-Build-Sticky-Empty: Wenn das Profile bewusst body_build = "" gesetzt hat
    # (z.B. wegen Headshot-Dominanz oder User-Override im UI), darf NICHT auf den
    # per-image Audit-Wert zurueckgefallen werden. Sonst wuerde die Demotion sinnlos.
    if "body_build" in stable_identity:
        body_build = compact_trait(stable_identity.get("body_build"))
    else:
        body_build = compact_trait(item.get("body_build"))

    hair_state = get_hair_feature_state(item, profile, image_traits, global_rules, active_policy, caption_profile)
    hair_tag = hair_state.get("phrase", "") or None

    makeup_token = image_traits.get("makeup_intensity", "")
    makeup_style_token = image_traits.get("makeup_style", "")
    makeup_desc = ""
    if makeup_style_token and makeup_style_token not in {"none", "no", "unclear", "natural_makeup"}:
        makeup_desc = _phrase_from_token(makeup_style_token)
    elif makeup_token and makeup_token not in {"none", "no", "unclear"}:
        makeup_desc = f"{_phrase_from_token(makeup_token)} makeup"
    else:
        makeup_desc = compact_trait(item.get("makeup_description"))

    costume_accessories = image_traits.get("costume_accessories", [])
    if not isinstance(costume_accessories, list):
        costume_accessories = []
    costume_bits = [
        _phrase_from_token(t) for t in costume_accessories
        if normalize_text(t) not in {"", "none", "none_visible", "unclear"}
    ]

    markers = profile.get("identity_markers", {}) if isinstance(profile, dict) else {}
    glasses_profile = markers.get("glasses", {}) if isinstance(markers, dict) else {}
    freckles_profile = markers.get("freckles", {}) if isinstance(markers, dict) else {}
    glasses_visible = bool(image_traits.get("glasses_visible")) or _profile_bool(item.get("has_glasses_now"))
    glasses_desc = resolve_visible_glasses_description(item, profile, image_traits) if glasses_visible else ""
    freckles_visible = bool(image_traits.get("freckles_visible")) or bool(compact_trait(item.get("freckles_description")))
    freckles_desc = ""
    if freckles_visible:
        freckles_desc = compact_trait(image_traits.get("freckles_description")) or compact_trait(freckles_profile.get("canonical_description")) or compact_trait(item.get("freckles_description"))

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

    piercing_state = get_visible_piercing_state(item, profile, image_traits, active_policy, caption_profile)
    piercing_bits: List[str] = list(piercing_state.get("phrases", []) or [])

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
        if active_policy.get("include_eye_color") and eye_color:
            anchor_parts.append(f"{_phrase_from_token(eye_color)} eyes")
        if active_policy["include_skin_tone"] and skin_tone:
            anchor_parts.append(f"{skin_tone} skin")

    visual_style = build_visual_style_phrase(item)
    if visual_style:
        first = f"A {visual_style} {photo_type} of {TRIGGER_WORD}"
    else:
        first = f"A {photo_type} of {TRIGGER_WORD}"
    if active_policy["include_gender_class"] and gender_class:
        first += f", a {gender_class}"

    if anchor_parts:
        first += " with " + ", ".join(dict.fromkeys([p for p in anchor_parts if p]))

    trait_bits: List[str] = []

    if shot_type in {"medium", "full_body"} and active_policy["include_body_build"] and body_build:
        # Grammatical compact tag: "slim build" instead of a dangling "slim".
        body_build_phrase = _phrase_from_token(body_build)
        trait_bits.append(body_build_phrase if "build" in body_build_phrase else f"{body_build_phrase} build")

    if caption_profile not in {"ernie", "shared_compact"} and hair_tag and hair_state.get("must_caption"):
        trait_bits.append(hair_tag)

    eye_state = get_eye_feature_state(item, profile, image_traits, active_policy)
    if caption_profile not in {"ernie", "shared_compact"} and eye_state.get("must_caption") and eye_state.get("phrase"):
        trait_bits.append(eye_state.get("phrase"))

    beard_state = get_beard_feature_state(item, global_rules, active_policy, profile)
    if beard_state.get("must_caption") and beard_state.get("phrase"):
        trait_bits.append(beard_state.get("phrase"))
    elif active_policy["include_beard_always"] and beard_desc:
        trait_bits.append(beard_desc)

    glasses_state = get_glasses_feature_state(item, profile, image_traits, active_policy)
    if glasses_state.get("must_caption") and glasses_state.get("phrase"):
        trait_bits.append(glasses_state.get("phrase"))

    if active_policy.get("include_freckles") and freckles_desc:
        trait_bits.append(freckles_desc)

    if active_policy["include_piercings"]:
        trait_bits.extend(piercing_bits)

    if active_policy["include_makeup"] and makeup_desc:
        trait_bits.append(makeup_desc)

    if active_policy.get("include_costume_accessories") and costume_bits:
        trait_bits.extend(costume_bits)

    if active_policy["include_tattoos"]:
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

    if active_policy["include_expression"] and expression and expression not in {"none", "unknown"}:
        # Bug-Fix: gelegentlich liefert die KI nur ein Adjektiv ohne
        # Substantiv ('neutral', 'pensive'), was zu kaputten Saetzen wie
        # 'with a neutral, looking at camera' fuehrt. Bei Mehrfach-Adjektiven
        # ('neutral, confident') wird mit 'and' verknuepft. 'eyes closed'
        # in Expression wird verworfen (s.o.).
        cleaned_expr = _clean_expression(expression)
        if cleaned_expr:
            pose_bits.append(f"with a {cleaned_expr}")
    def _pose_bits_have_eyes_closed() -> bool:
        return any(
            re.search(r"\beyes closed\b", str(bit), re.IGNORECASE)
            for bit in pose_bits
        )

    if active_policy["include_gaze"] and gaze and gaze not in {"none", "unknown"}:
        # Wenn gaze == "eyes closed", nur ergänzen, wenn es nicht bereits
        # in pose_description / pose_bits vorkommt.
        if eyes_closed_in_gaze:
            if not _pose_bits_have_eyes_closed():
                pose_bits.append("with eyes closed")
        else:
            # Bug G: KI liefert manchmal gaze als reines Direction-Adverb
            # ('downward', 'upward', 'sideways'), was zu losgeloesten Saetzen
            # fuehrt: 'holding cards, downward.'. Wir setzen 'looking' davor
            # wenn gaze eine kurze Direction-Phrase ohne eigenes Verb ist.
            pose_bits.append(_ensure_gaze_verb(gaze))
    elif eyes_closed_in_expr:
        # Nur Expression hatte eyes closed und wurde dort verworfen.
        # Auch hier nicht doppeln, falls pose_description es bereits enthält.
        if not _pose_bits_have_eyes_closed():
            pose_bits.append("with eyes closed")

    if pose_bits:
        sentences.append(f"{pronoun} {'is' if pronoun in ['He', 'She'] else 'are'} " + ", ".join(pose_bits) + ".")

    if active_policy["include_lighting"] and lighting:
        sentences.append(f"{lighting.capitalize()}.")

    if active_policy["include_background"] and background:
        sentences.append(f"{background.capitalize()}.")

    caption = " ".join(sentences)
    caption = re.sub(r"\s+", " ", caption).strip()
    # Brillen-Wording absichtlich NICHT global vereinheitlichen:
    # der Canonical-Begriff aus dem Subject Profile soll wortgleich in die
    # Caption gehen (z.B. "eyeglasses" bleibt "eyeglasses").
    caption = _normalize_glasses_token(caption)
    if caption_profile == "krea2_character":
        # Keep the deterministic fallback compatible with the Krea sidecars:
        # exact trigger token first, natural sentence afterwards, no duplicate
        # trigger later in the sentence.
        pattern = rf"^A\s+(.+?)\s+of\s+{re.escape(TRIGGER_WORD)}\b"
        caption = re.sub(
            pattern,
            lambda m: f"{TRIGGER_WORD}, a {m.group(1)}",
            caption,
            count=1,
            flags=re.IGNORECASE,
        )
        caption = _clean_krea_caption(caption)
    return caption


def _encode_pil_for_api(pil_img: Image.Image, max_side: int = API_MAX_IMAGE_SIDE) -> str:
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        img = img.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            Image.Resampling.LANCZOS,
        )
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _krea_caption_cache_path(item: Dict[str, Any], subject_profile: Dict[str, Any]) -> str:
    source_key = str(item.get("file_hash") or "")
    if not source_key:
        path = str(item.get("original_path") or "")
        source_key = file_sha1(path) if path and os.path.exists(path) else profile_image_id(item)
    profile_key = hashlib.sha1(
        json.dumps(subject_profile or {}, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    raw = "|".join([
        KREA_CAPTION_PROMPT_VERSION,
        str(KREA_CAPTION_MODEL),
        str(KREA_CAPTION_REASONING_EFFORT),
        str(bool(USE_KREA_CAPTION_REPAIR)),
        str(KREA_CAPTION_REPAIR_MODEL),
        str(KREA_CAPTION_REPAIR_REASONING_EFFORT),
        str(VARIABLE_FEATURE_CAPTION_MODE),
        str(source_key),
        profile_key,
        str(item.get("crop_variant") or "original"),
        str(TRIGGER_WORD),
    ])
    key = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    folder = os.path.join(CACHE_DIR, "krea_captions")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{key}.json")


def _clean_krea_caption(text: str) -> str:
    caption = re.sub(r"\s+", " ", str(text or "")).strip().strip('"')
    caption = re.sub(r"^(caption\s*:\s*)", "", caption, flags=re.IGNORECASE)
    # Enforce the exact trigger token at the start without duplicating it.
    caption = re.sub(rf"^{re.escape(TRIGGER_WORD)}\s*[,;:\-]*\s*", "", caption, flags=re.IGNORECASE)
    caption = f"{TRIGGER_WORD}, {caption}" if caption else TRIGGER_WORD
    return caption.strip()


def _set_caption_metadata(
    item: Dict[str, Any],
    *,
    source: str,
    model: str,
    retry_count: int,
    validation_error: str = "",
) -> None:
    """Persist caption provenance on the row for CSV/JSONL/report diagnostics."""
    item["caption_source"] = str(source or "")
    item["caption_model"] = str(model or "")
    item["caption_retry_count"] = int(retry_count or 0)
    item["caption_validation_error"] = str(validation_error or "")


def _call_krea_caption_model(
    *,
    model: str,
    reasoning_effort: str,
    instructions: str,
    text_payload: Dict[str, Any],
    image_b64: str,
    phase_label: str,
) -> str:
    payload = {
        "instructions": instructions,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": json.dumps(text_payload, ensure_ascii=False)},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": KREA_CAPTION_IMAGE_DETAIL,
                },
            ],
        }],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "krea2_caption",
                "schema": {
                    "type": "object",
                    "properties": {"caption": {"type": "string"}},
                    "required": ["caption"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        },
        "max_output_tokens": 300,
        "store": False,
        "temperature": 0.1,
        "_reasoning_effort": reasoning_effort,
    }
    data = responses_api_call(model, payload, phase_label=phase_label)
    parsed = json.loads(extract_response_text(data))
    return _clean_krea_caption(parsed.get("caption", ""))


def _save_krea_caption_cache(
    cache_path: str,
    *,
    caption: str,
    source: str,
    model: str,
    retry_count: int,
    validation_error: str = "",
) -> None:
    if not ENABLE_CACHE:
        return
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "caption": caption,
                "caption_source": source,
                "model": model,
                "retry_count": int(retry_count or 0),
                "validation_error": validation_error,
                "primary_model": KREA_CAPTION_MODEL,
                "repair_enabled": bool(USE_KREA_CAPTION_REPAIR),
                "repair_model": KREA_CAPTION_REPAIR_MODEL,
                "prompt_version": KREA_CAPTION_PROMPT_VERSION,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def build_krea_ai_caption(
    item: Dict[str, Any],
    global_rules: Dict[str, Any],
    subject_profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a dataset-aware Krea 2 caption with one automatic repair attempt.

    Flow:
      1. Generate with the primary caption model.
      2. Validate against the confirmed Subject Profile and caption policy.
      3. If generation or validation fails, call the configured repair model once
         with the original attempt and exact validation errors.
      4. Use the deterministic local caption only if both AI attempts fail.
    """
    profile = subject_profile or {}
    cache_path = _krea_caption_cache_path(item, profile)

    profile_policies = profile.get("profile_policies", {}) if isinstance(profile, dict) else {}
    active_policy = enforce_caption_policy_profile(
        normalize_caption_profile(globals().get("CAPTION_PROFILE", "ernie")),
        CAPTION_POLICY,
    )
    image_id = profile_image_id(item)
    image_traits = (
        (profile.get("per_image_traits", {}) or {}).get(image_id, {})
        if isinstance(profile, dict)
        else {}
    )
    hair_state = get_hair_feature_state(item, profile, image_traits, global_rules, active_policy, "krea2_character")
    eye_state = get_eye_feature_state(item, profile, image_traits, active_policy)
    beard_state = get_beard_feature_state(item, global_rules, active_policy, profile)
    glasses_state = get_glasses_feature_state(item, profile, image_traits, active_policy)
    piercing_state = get_visible_piercing_state(item, profile, image_traits, active_policy, "krea2_character")
    tattoo_state = get_visible_tattoo_state(item, profile, active_policy)
    feature_states = {
        "hair": hair_state,
        "eye": eye_state,
        "beard": beard_state,
        "glasses": glasses_state,
        "piercings": piercing_state,
        "tattoos": tattoo_state,
    }

    # A valid cache entry is still revalidated against the current profile.
    if ENABLE_CACHE and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_caption = _clean_krea_caption(cached.get("caption", ""))
            valid_cached, cached_reasons = _validate_krea_caption_features(cached_caption, feature_states)
            if cached_caption and cached_caption != TRIGGER_WORD and valid_cached:
                _set_caption_metadata(
                    item,
                    source=str(cached.get("caption_source") or "cache"),
                    model=str(cached.get("model") or KREA_CAPTION_MODEL),
                    retry_count=int(cached.get("retry_count") or 0),
                    validation_error=str(cached.get("validation_error") or ""),
                )
                return cached_caption
            if cached_reasons:
                safe_print(
                    f"   ℹ️ Ignoring invalid Krea caption cache for {item.get('original_filename', '')}: "
                    + "; ".join(cached_reasons)
                )
        except Exception:
            pass

    fallback = build_local_caption(item, global_rules, profile)
    primary_caption = ""
    primary_errors: List[str] = []
    repair_errors: List[str] = []

    try:
        exported_view = body_aware_crop(str(item.get("original_path") or ""), item)
        image_b64 = _encode_pil_for_api(exported_view)

        visible_facts = {
            "shot_type": item.get("shot_type", ""),
            "frame_subtype": item.get("frame_subtype", ""),
            "body_orientation": item.get("body_orientation", ""),
            "camera_angle": item.get("camera_angle", ""),
            "depth_of_field": item.get("depth_of_field", ""),
            "pose": item.get("pose_description", ""),
            "action": item.get("action_description", ""),
            "expression": item.get("expression", ""),
            "gaze": item.get("gaze_direction", ""),
            "clothing": item.get("clothing_description", ""),
            "makeup": item.get("makeup_description", ""),
            "glasses": glasses_state.get("current_desc", ""),
            "glasses_position": glasses_state.get("position", ""),
            "hair": hair_state.get("phrase", "") or item.get("hair_description", ""),
            "hair_color_modifier": hair_state.get("current_modifier", ""),
            "eye_color": eye_state.get("current", "") if eye_state.get("reliable") else "",
            "eye_color_reliable": bool(eye_state.get("reliable")),
            "visible_piercings_and_ear_jewelry": piercing_state.get("phrases", []),
            "visible_tattoos": tattoo_state.get("phrases", []) if tattoo_state.get("must_caption") else [],
            "background": item.get("background_description", ""),
            "lighting": item.get("lighting_description", ""),
            "composition": item.get("composition_description", ""),
            "prominent_objects": item.get("prominent_objects", []),
            "visual_style": item.get("visual_style_type", ""),
            "mirror_selfie": bool(item.get("mirror_selfie", False)),
            "profile_hair_policy": profile_policies.get("hair_color_policy", ""),
            "profile_eye_policy": profile_policies.get("eye_color_policy", ""),
            "feature_policy": {
                "hair": {
                    "current": hair_state.get("current", ""),
                    "baseline": hair_state.get("baseline", ""),
                    "preferred_phrase": hair_state.get("phrase", ""),
                    "must_caption": bool(hair_state.get("must_caption")),
                },
                "eye_color": {
                    "current": eye_state.get("current", ""),
                    "baseline": eye_state.get("baseline", ""),
                    "preferred_phrase": eye_state.get("phrase", ""),
                    "must_caption": bool(eye_state.get("must_caption")),
                },
                "beard": {
                    "preferred_phrase": beard_state.get("phrase", ""),
                    "must_caption": bool(beard_state.get("must_caption")),
                    "baseline_pattern": beard_state.get("baseline_pattern", ""),
                    "current_pattern": beard_state.get("current_pattern", ""),
                    "baseline_color": beard_state.get("baseline_color", ""),
                    "current_color": beard_state.get("current_color", ""),
                },
                "glasses": {
                    "preferred_phrase": glasses_state.get("phrase", ""),
                    "must_caption": bool(glasses_state.get("must_caption")),
                    "baseline_desc": glasses_state.get("baseline_desc", ""),
                    "current_desc": glasses_state.get("current_desc", ""),
                    "baseline_family": glasses_state.get("baseline_family", ""),
                    "current_family": glasses_state.get("current_family", ""),
                    "position": glasses_state.get("position", ""),
                },
                "piercings_and_ear_jewelry": {
                    "preferred_phrases": piercing_state.get("phrases", []),
                    "must_caption": bool(piercing_state.get("must_caption")),
                    "entries": piercing_state.get("entries", []),
                },
                "tattoos": {
                    "preferred_phrases": tattoo_state.get("phrases", []),
                    "visible": bool(tattoo_state.get("visible")),
                    "must_caption": bool(tattoo_state.get("must_caption")),
                },
            },
        }

        instructions = f"""
You create final natural-language captions for a Krea 2 character LoRA dataset.
Return exactly one fluent English caption and nothing else.

The caption MUST begin with the exact trigger token: {TRIGGER_WORD},
Target length: 25-80 words, up to 100 only for genuinely complex images.
Describe only what is visible in the exported image: framing, body/head orientation,
pose or action, expression, gaze, clothing, temporary accessories, important objects,
environment, camera angle, depth of field, lighting, composition and visible medium/style.

Identity policy:
- The trigger word carries stable physical identity.
- Do NOT describe stable skin tone, body build, body proportions, facial structure,
  freckles, permanent piercings, scars or other fixed body markers.
- Tattoos: follow feature_policy.tattoos literally. If must_caption=false, omit all tattoos even when visible. If must_caption=true, mention only tattoos visible in the exported image.
- Do NOT describe stable hair or eye color unless the supplied feature_policy says must_caption=true.
- Treat beard and glasses the same way: if feature_policy.must_caption is true, you MUST include the supplied preferred_phrase. If must_caption is false, omit that stable feature.
- Glasses, makeup, costume elements and hairstyle changes may be described when visible, but follow feature_policy exactly for hair color, eye color, beard and glasses.
- For piercings and ear jewelry, include every preferred phrase when piercings_and_ear_jewelry.must_caption=true. Canonical fixed piercings are already omitted by the profile policy; never invent or relocate jewelry.
- Do not identify the person, guess a name, exact age, location, brand or relationship.
- No booru tags, keyword lists, filename, markdown, labels, 'This image shows', hedging,
  explanations, quality scores or training advice.
- Do not mention removed social-media frames or invisible/cropped-out details.
""".strip()

        # Primary attempt.
        try:
            primary_caption = _call_krea_caption_model(
                model=KREA_CAPTION_MODEL,
                reasoning_effort=KREA_CAPTION_REASONING_EFFORT,
                instructions=instructions,
                text_payload=visible_facts,
                image_b64=image_b64,
                phase_label=f"krea_caption:{item.get('original_filename', '')}",
            )
            if not primary_caption or primary_caption == TRIGGER_WORD:
                primary_errors = ["empty Krea caption"]
            else:
                valid_primary, primary_errors = _validate_krea_caption_features(primary_caption, feature_states)
                if valid_primary:
                    _set_caption_metadata(
                        item,
                        source="gpt_primary",
                        model=KREA_CAPTION_MODEL,
                        retry_count=0,
                    )
                    _save_krea_caption_cache(
                        cache_path,
                        caption=primary_caption,
                        source="gpt_primary",
                        model=KREA_CAPTION_MODEL,
                        retry_count=0,
                    )
                    return primary_caption
        except Exception as exc:
            primary_errors = [f"primary API error: {exc}"]

        primary_error_text = "; ".join(primary_errors) or "unknown primary caption failure"
        safe_print(
            f"   ↻ Krea caption validation failed for {item.get('original_filename', '')}: "
            f"{primary_error_text}"
        )

        # One automatic repair attempt. It receives the exact reason the first
        # attempt failed, so no full audit/profile rerun is necessary.
        if bool(USE_KREA_CAPTION_REPAIR):
            repair_instructions = f"""
You repair one invalid Krea 2 LoRA training caption.
Return exactly one corrected fluent English caption and nothing else.
The corrected caption MUST begin with the exact trigger token: {TRIGGER_WORD},
Follow the supplied feature_policy literally, and fix every listed validation error.
Do not add stable identity traits merely to make the sentence more descriptive.
Preserve accurate useful scene details from the previous attempt when possible.
""".strip()
            repair_payload = {
                "visible_facts": visible_facts,
                "previous_caption": primary_caption,
                "validation_errors": primary_errors,
                "required_action": "Return a fully rewritten valid caption, not an explanation.",
            }
            try:
                repaired_caption = _call_krea_caption_model(
                    model=KREA_CAPTION_REPAIR_MODEL,
                    reasoning_effort=KREA_CAPTION_REPAIR_REASONING_EFFORT,
                    instructions=repair_instructions,
                    text_payload=repair_payload,
                    image_b64=image_b64,
                    phase_label=f"krea_caption_repair:{item.get('original_filename', '')}",
                )
                if not repaired_caption or repaired_caption == TRIGGER_WORD:
                    repair_errors = ["empty repaired Krea caption"]
                else:
                    valid_repair, repair_errors = _validate_krea_caption_features(repaired_caption, feature_states)
                    if valid_repair:
                        _set_caption_metadata(
                            item,
                            source="gpt_repair",
                            model=KREA_CAPTION_REPAIR_MODEL,
                            retry_count=1,
                            validation_error=primary_error_text,
                        )
                        _save_krea_caption_cache(
                            cache_path,
                            caption=repaired_caption,
                            source="gpt_repair",
                            model=KREA_CAPTION_REPAIR_MODEL,
                            retry_count=1,
                            validation_error=primary_error_text,
                        )
                        safe_print(
                            f"   ✅ Caption repaired with {KREA_CAPTION_REPAIR_MODEL}: "
                            f"{item.get('original_filename', '')}"
                        )
                        return repaired_caption
            except Exception as exc:
                repair_errors = [f"repair API error: {exc}"]

        repair_error_text = "; ".join(repair_errors)
        combined_error = primary_error_text
        if repair_error_text:
            combined_error += " | repair: " + repair_error_text
        _set_caption_metadata(
            item,
            source="local_fallback",
            model="local_deterministic",
            retry_count=1 if bool(USE_KREA_CAPTION_REPAIR) else 0,
            validation_error=combined_error,
        )
        safe_print(
            f"   ⚠️ Krea caption attempts failed for {item.get('original_filename', '')}: "
            f"{combined_error}; using local caption."
        )
        return fallback

    except Exception as exc:
        # Preparation failures (for example an unreadable image) cannot be
        # repaired through a second API request because the image payload is unavailable.
        error_text = f"caption preparation error: {exc}"
        _set_caption_metadata(
            item,
            source="local_fallback",
            model="local_deterministic",
            retry_count=0,
            validation_error=error_text,
        )
        safe_print(
            f"   ⚠️ Krea caption preparation failed for {item.get('original_filename', '')}: "
            f"{exc}; using local caption."
        )
        return fallback

def build_caption(
    item: Dict[str, Any],
    global_rules: Dict[str, Any],
    subject_profile: Optional[Dict[str, Any]] = None,
) -> str:
    profile_name = caption_profile_for_training_target(globals().get("TRAINING_TARGET", globals().get("CAPTION_PROFILE", "ernie")))
    use_ai = (
        normalize_training_target(globals().get("TRAINING_TARGET", "ernie")) == "krea2"
        and bool(USE_KREA_AI_CAPTIONING)
        and bool(item.get("selected") or item.get("output_bucket") == "train_ready")
    )
    if use_ai:
        return build_krea_ai_caption(item, global_rules, subject_profile)
    return build_local_caption(item, global_rules, subject_profile)


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
        content_crop = pil_img.crop((x1, y1, x2, y2))
        if USE_CONTROLLED_BUCKETS:
            return content_crop.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return content_crop

    if item.get("is_rescue_crop") and item.get("rescue_crop_bbox"):
        rx, ry, rw, rh = [int(v) for v in item["rescue_crop_bbox"]]
        rx = max(0, min(rx, w - 1))
        ry = max(0, min(ry, h - 1))
        rw = max(1, min(rw, w - rx))
        rh = max(1, min(rh, h - ry))
        content_crop = pil_img.crop((rx, ry, rx + rw, ry + rh))
        if USE_CONTROLLED_BUCKETS:
            return ImageOps.fit(
                content_crop,
                (832, 1216),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.38),
            )
        return content_crop

    face_bbox = item.get("main_face_bbox")
    pose_bbox = item.get("pose_bbox")
    shot_type = item.get("shot_type", "headshot")

    def crop_box(x: int, y: int, cw: int, ch: int) -> Tuple[int, int, int, int]:
        x = max(0, min(x, w - cw))
        y = max(0, min(y, h - ch))
        return x, y, x + cw, y + ch

    if not USE_CONTROLLED_BUCKETS:
        # Content cleanup/crops have already happened. Preserve the selected
        # composition and let AI Toolkit bucket the natural aspect ratio.
        return pil_img

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


def should_copy_reject_original(row: Dict[str, Any]) -> bool:
    """Bestimmt, ob ein Reject unverändert statt gecroppt exportiert werden soll.

    Early-Hard-Rejects sollen als Diagnosematerial im Originalzustand erhalten
    bleiben, damit kleine/duplizierte/problematische Inputs nicht künstlich
    auf Trainingsgrößen hochskaliert oder neu beschnitten werden.
    """
    reason = str(row.get("short_reason", "") or "")
    return (
        reason.startswith("hard_pass_too_small")
        or reason.startswith("filesize_too_small")
        or reason == "early_phash_duplicate"
    )


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
    openai_usage = report.get("openai_usage", {}) or {}
    if openai_usage:
        lines.append("## OpenAI usage")
        lines.append("")
        lines.append(f"- requests: {openai_usage.get('requests', 0)}")
        lines.append(f"- input_tokens: {openai_usage.get('input_tokens', 0)}")
        lines.append(f"- output_tokens: {openai_usage.get('output_tokens', 0)}")
        lines.append(f"- total_tokens: {openai_usage.get('total_tokens', 0)}")
        if openai_usage.get("token_limit_enabled"):
            lines.append(f"- token_limit_total: {openai_usage.get('token_limit_total', 0)}")
            lines.append(f"- token_limit_remaining: {openai_usage.get('token_limit_remaining', 0)}")
            lines.append(f"- token_limit_reached: {openai_usage.get('token_limit_reached', False)}")
        estimated_cost = openai_usage.get("estimated_cost_usd")
        lines.append(
            f"- estimated_cost_usd: {estimated_cost}"
            if estimated_cost is not None
            else "- estimated_cost_usd: n/a (no local pricing table configured)"
        )
        if openai_usage.get("by_model"):
            lines.append("")
            lines.append("### By model")
            lines.append("")
            for model_name, usage in openai_usage["by_model"].items():
                lines.append(
                    f"- `{model_name}`: requests={usage.get('requests', 0)}, "
                    f"input_tokens={usage.get('input_tokens', 0)}, "
                    f"output_tokens={usage.get('output_tokens', 0)}, "
                    f"total_tokens={usage.get('total_tokens', 0)}"
                )
        if openai_usage.get("by_phase"):
            lines.append("")
            lines.append("### By phase")
            lines.append("")
            for phase_name, usage in openai_usage["by_phase"].items():
                lines.append(
                    f"- `{phase_name}`: requests={usage.get('requests', 0)}, "
                    f"input_tokens={usage.get('input_tokens', 0)}, "
                    f"output_tokens={usage.get('output_tokens', 0)}, "
                    f"total_tokens={usage.get('total_tokens', 0)}"
                )
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
    budget_limit_reached = False

    # Konfig-Banner: zeigt aktiv geladenes Modell + Cache-Schema-Version.
    # Dient zum schnellen Debug-Check, ob UI-Config-Overrides oder
    # Schema-Bumps wirklich gegriffen haben (alte Caches bei v6 vs v7
    # haben in der Vergangenheit zu Verwirrung gefuehrt).
    safe_print("=" * 60)
    safe_print(f"  Audit model:        {AI_MODEL}")
    safe_print(f"  Trigger model:      {TRIGGER_CHECK_MODEL}")
    safe_print(f"  Escalation:         {'ON (' + REVIEW_ESCALATION_MODEL + ')' if USE_REVIEW_ESCALATION and REVIEW_ESCALATION_MODEL else 'OFF'}")
    safe_print(
        f"  Token limit:        {f'{int(OPENAI_TOKEN_LIMIT_TOTAL):,}' if openai_token_limit_enabled() else 'OFF'}"
    )
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
        except OpenAITokenBudgetExceeded as e:
            budget_limit_reached = True
            warnings.append(str(e))
            safe_print(f"🛑 {e}")
        except Exception as e:
            warnings.append(f"Trigger-word check failed: {e}")

    for w in warnings:
        safe_print(f"⚠️ {w}")

    image_paths = iter_input_images(INPUT_FOLDER)
    if not image_paths:
        safe_print("No images found.")
        return

    safe_print(f"Images found: {len(image_paths)}")

    dataset_fp = dataset_fingerprint(image_paths)
    settings_fp = early_result_settings_fingerprint()
    cached_early = load_cached_early_results(dataset_fp, settings_fp)

    early_reject_rows: List[Dict[str, Any]] = []
    early_dup_paths: List[str] = []
    phash_cache: Dict[str, int] = {}

    if cached_early:
        image_paths = [
            p for p in cached_early.get("survivor_paths", [])
            if isinstance(p, str) and os.path.exists(p)
        ]
        early_dup_paths = [
            p for p in cached_early.get("early_duplicate_paths", [])
            if isinstance(p, str) and os.path.exists(p)
        ]
        raw_phash_cache = cached_early.get("phash_cache", {}) or {}
        if isinstance(raw_phash_cache, dict):
            phash_cache = {
                str(p): int(v)
                for p, v in raw_phash_cache.items()
                if isinstance(p, str) and os.path.exists(p)
            }
        raw_reject_rows = cached_early.get("early_reject_rows", []) or []
        early_reject_rows = [r for r in raw_reject_rows if isinstance(r, dict)]
        safe_print(
            f"   ↳ Early result cache used: {len(image_paths)} survivors, "
            f"{len(early_reject_rows)} early rejects, {len(early_dup_paths)} early duplicates"
        )
    else:
        image_paths, early_reject_rows = apply_early_static_rejects(image_paths)
        if USE_EARLY_PHASH_DEDUP and USE_PHASH_DUPLICATE_SCORING:
            image_paths, early_dup_paths, phash_cache = early_phash_dedup(image_paths)
        save_cached_early_results({
            "schema_version": EARLY_RESULT_CACHE_SCHEMA_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset_fingerprint": dataset_fp,
            "settings_fingerprint": settings_fp,
            "settings": early_result_cache_settings(),
            "survivor_paths": list(image_paths),
            "early_duplicate_paths": list(early_dup_paths),
            "phash_cache": phash_cache,
            "early_reject_rows": early_reject_rows,
        })
        safe_print(
            f"   ↳ Early result cache saved: {len(image_paths)} survivors, "
            f"{len(early_reject_rows)} early rejects, {len(early_dup_paths)} early duplicates"
        )

    safe_print(f"Starting audit for trigger word: {TRIGGER_WORD}")
    safe_print("")

    all_rows: List[Dict[str, Any]] = list(early_reject_rows)
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

            row = apply_local_score_adjustments(row)
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
                        reasoning_effort=REVIEW_ESCALATION_REASONING_EFFORT,
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
                    row = apply_local_score_adjustments(row)
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

                                crop_grundscore = float(crop_audit.get("quality_total", 0))
                                crop_score = crop_grundscore
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
                                            reasoning_effort=REVIEW_ESCALATION_REASONING_EFFORT,
                                        )
                                        if not escalated_crop_audit.get("NSFW_BLOCKED"):
                                            crop_audit = normalize_audit_scores(escalated_crop_audit)
                                            save_cached_audit(
                                                crop_escalation_cache_key,
                                                audit_cache_payload(crop_audit, REVIEW_ESCALATION_MODEL, "escalation_crop_audit"),
                                            )
                                    crop_score_nach_eskalation = float(crop_audit.get("quality_total", 0))
                                    crop_score = crop_score_nach_eskalation

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

                                    crop_row = apply_local_score_adjustments(crop_row)
                                    crop_score = float(crop_row.get("quality_total", 0))

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

            # ─────────────────────────────────────────────────────────────
            # MEDIUM RESCUE CROP: separate from headshot smart crop.
            # Tries to salvage a weak full-body composition as a medium shot.
            # ─────────────────────────────────────────────────────────────
            rescue_issues = set(row.get("issues") or [])
            rescue_triggered = (
                ENABLE_MEDIUM_RESCUE_CROP
                and base_status != "reject"
                and row.get("shot_type") == "full_body"
                and row.get("face_visible", False)
                and row.get("main_face_bbox") is not None
                and (row.get("width", 0) * row.get("height", 0)) >= 2_000_000
                and (
                    float(row.get("quality_composition", 0) or 0) <= MEDIUM_RESCUE_TRIGGER_COMPOSITION_MAX
                    or base_status == "review"
                    or bool(rescue_issues.intersection({"cropped_limbs", "busy_background", "small_face", "extreme_angle"}))
                )
            )
            if rescue_triggered:
                rescue_path, rescue_bbox = generate_medium_rescue_crop(
                    image_path,
                    row.get("main_face_bbox"),
                    row.get("pose_bbox"),
                    int(row.get("width", 0)),
                    int(row.get("height", 0)),
                )
                if rescue_path and rescue_bbox:
                    try:
                        safe_print("   ✂️  Medium rescue: evaluating torso/hip crop...")
                        bbox_str = "_".join(str(v) for v in rescue_bbox)
                        rescue_key = f"{file_hash}_medium_rescue_{bbox_str}"
                        rescue_hash = hashlib.sha1(rescue_key.encode()).hexdigest()
                        rescue_cache_key = audit_cache_key(rescue_hash, AI_MODEL, "primary_medium_rescue_audit")
                        cached_rescue = load_cached_audit(rescue_cache_key)
                        rescue_local_meta = run_with_heartbeat(
                            f"[{idx}/{len(image_paths)}] medium_rescue_local_metrics {original_filename}",
                            local_subject_metrics,
                            rescue_path,
                        )
                        if cached_rescue:
                            rescue_audit = cached_rescue.get("audit", cached_rescue)
                            safe_print(f"   ↳ Medium rescue audit cache used ({AI_MODEL})")
                        else:
                            rescue_audit = openai_audit_image(
                                rescue_path,
                                rescue_local_meta,
                                model=AI_MODEL,
                                phase_label=f"[{idx}/{len(image_paths)}] primary_medium_rescue_audit {original_filename}",
                            )
                        if not rescue_audit.get("NSFW_BLOCKED"):
                            rescue_audit = normalize_audit_scores(rescue_audit)
                            if not cached_rescue:
                                save_cached_audit(
                                    rescue_cache_key,
                                    audit_cache_payload(rescue_audit, AI_MODEL, "primary_medium_rescue_audit"),
                                )
                            rescue_score = float(rescue_audit.get("quality_total", 0) or 0)
                            orig_score = float(row.get("quality_total", 0) or 0)
                            safe_print(
                                f"   ↳ Medium rescue {rescue_score:.1f} vs. original {orig_score:.1f} "
                                f"(min gain: {MEDIUM_RESCUE_MIN_GAIN})"
                            )
                            if rescue_score >= orig_score + MEDIUM_RESCUE_MIN_GAIN:
                                rescue_row: Dict[str, Any] = {
                                    "original_filename": original_filename + "__medium_rescue",
                                    # During local validation this must point to the actual
                                    # rescue image, because its face/pose bboxes are relative
                                    # to that crop. It is restored to the source image before
                                    # the temporary file is removed; final export reapplies
                                    # rescue_crop_bbox to the source image.
                                    "original_path": rescue_path,
                                    "source_original_path": image_path,
                                    "is_rescue_crop": True,
                                    "crop_variant": "medium_rescue",
                                    "crop_of": original_filename,
                                    "rescue_crop_bbox": rescue_bbox,
                                    "status_notes": ["medium_rescue_crop"],
                                    "selected": False,
                                    "output_bucket": "",
                                    "new_basename": "",
                                    "file_hash": rescue_hash,
                                    "mtime_bucket": row.get("mtime_bucket"),
                                    "width": rescue_local_meta.get("width", 0),
                                    "height": rescue_local_meta.get("height", 0),
                                    "file_size_mb": rescue_local_meta.get("file_size_mb", 0),
                                    "phash": rescue_local_meta.get("phash"),
                                    "clip_embedding": (
                                        run_with_heartbeat(
                                            f"[{idx}/{len(image_paths)}] medium_rescue_clip_embedding {original_filename}",
                                            compute_clip_embedding,
                                            rescue_path,
                                            rescue_hash,
                                        )
                                        if USE_CLIP_DUPLICATE_SCORING
                                        else None
                                    ),
                                }
                                rescue_row.update(rescue_audit)
                                rescue_row["grundscore"] = rescue_score
                                rescue_row["score_nach_eskalation"] = ""
                                rescue_row["shot_type"] = "medium"
                                # Keep the rescue audit's own bboxes and face ratio. They
                                # are relative to rescue_path and therefore valid for local
                                # blur/sanity checks.
                                rescue_row = apply_local_score_adjustments(rescue_row)
                                r_local_status, r_local_reasons = local_status_override(rescue_row)
                                r_api_status = rescue_row.get("suggested_status", "review")
                                if r_api_status == "reject" or r_local_status == "reject":
                                    r_base = "reject"
                                elif r_api_status == "review" or r_local_status == "review":
                                    r_base = "review"
                                else:
                                    r_base = "keep"
                                rescue_row["base_status"] = r_base
                                rescue_row["local_override_reasons"] = r_local_reasons
                                rescue_row["original_path"] = image_path
                                safe_print(
                                    f"   ✅ Medium rescue accepted: score={rescue_row.get('quality_total', 0):.1f} | status={r_base}"
                                )
                                all_rows.append(rescue_row)
                                time.sleep(SLEEP_BETWEEN_CALLS)
                            else:
                                safe_print("   ❌ Medium rescue rejected: gain too small")
                    except Exception as rescue_e:
                        safe_print(f"   ⚠️ Medium rescue failed: {rescue_e}")
                    finally:
                        try:
                            if os.path.exists(rescue_path):
                                os.remove(rescue_path)
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
            if isinstance(e, OpenAITokenBudgetExceeded):
                budget_limit_reached = True
                warnings.append(str(e))
                safe_print(f"🛑 {e}")
                break

        if budget_limit_reached:
            break

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
    # behalte nur den besseren von beiden. Anschließend finaler Schutz gegen
    # Crop-/Resize-/Scene-Varianten, die mit anderem Dateinamen in derselben
    # Endauswahl gelandet sind. Unterdrückte Varianten werden als Duplicate
    # markiert und landen dadurch nicht zusätzlich in keep_unused.
    selected = crop_dedup_selected(selected)
    selected, final_duplicate_rows = dedup_final_selected_scene_variants(selected)
    if final_duplicate_rows:
        reject_items.extend(final_duplicate_rows)

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

    selected, backfill_added = backfill_train_ready_selection(selected, valid_candidates, TARGET_DATASET_SIZE)
    if backfill_added:
        warnings.append(f"Backfilled {len(backfill_added)} image(s) after hard/caption-remove exclusions to preserve the train-ready target.")

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

            # Early-Hard-Rejects im Originalzustand behalten; alle anderen wie
            # Keep/Review auf Exportgröße crop/resize.
            try:
                if should_copy_reject_original(row):
                    shutil.copy2(row["original_path"], img_out)
                else:
                    cropped = body_aware_crop(row["original_path"], row)
                    cropped.save(img_out, "JPEG", quality=100)
            except Exception as export_err:
                safe_print(f"   ⚠️ Failed to export reject image: {row.get('original_filename','')} – {export_err}")

            # Reason-Datei: gemeinsamer Helper baut den vollstaendigen String
            try:
                with open(txt_out, "w", encoding="utf-8") as ft:
                    ft.write(build_reject_export_text(row, global_rules, subject_profile))
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
        "quality_total_before_local_penalties",
        "quality_total",
        "grundscore",
        "score_nach_eskalation",
        "quality_sharpness",
        "quality_lighting",
        "quality_composition",
        "is_grayscale_filter",
        "grayscale_penalty",
        "local_score_penalty_total",
        "color_saturation_mean",
        "color_tint_label",
        "color_tint_strength",
        "color_channel_delta_mean",
        "quality_identity_usefulness",
        "shot_type",
        "body_orientation",
        "camera_angle",
        "depth_of_field",
        "action_description",
        "prominent_objects",
        "composition_description",
        "silhouette_clarity",
        "limb_completeness",
        "body_reference_usefulness",
        "perspective_distortion",
        "gender_class",
        "face_visible",
        "face_occlusion",
        "multiple_people",
        "main_subject_clear",
        "watermark_or_overlay",
        "prominent_readable_text",
        "image_medium",
        "mirror_selfie",
        "frame_subtype",
        "visual_style_type",
        "look_context",
        "hair_description",
        "hair_length",
        "beard_description",
        "glasses_description",
        "piercings_description",
        "makeup_description",
        "makeup_intensity",
        "makeup_style",
        "skin_tone",
        "eye_color",
        "eye_appearance",
        "body_build",
        "body_height_impression",
        "body_skin_visibility",
        "face_orientation_in_frame",
        "tattoos_visible",
        "tattoos_description",
        "clothing_description",
        "pose_description",
        "expression",
        "expression_category",
        "gaze_direction",
        "gaze_category",
        "head_pose_bucket",
        "occlusion_type",
        "background_description",
        "lighting_description",
        "lighting_type",
        "background_type",
        "hair_texture",
        "has_glasses_now",
        "glasses_frame_shape",
        "glasses_frame_material",
        "glasses_lens_type",
        "glasses_position",
        "costume_accessories",
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
        "caption_source",
        "caption_model",
        "caption_retry_count",
        "caption_validation_error",
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
        "subject_profile_normalizer_source": (subject_profile or {}).get("normalizer_source", ""),
        "subject_profile_normalizer_retry_count": (subject_profile or {}).get("normalizer_retry_count", 0),
        "subject_profile_normalizer_primary_error": (subject_profile or {}).get("normalizer_primary_error", ""),
        "subject_profile_sample_size": (subject_profile or {}).get("sample_size", 0),
        "subject_profile_total_usable_images": (subject_profile or {}).get("total_usable_images", 0),
        "training_target": normalize_training_target(TRAINING_TARGET),
        "caption_profile": caption_profile_for_training_target(TRAINING_TARGET),
        "audit_model": AI_MODEL,
        "krea_ai_captioning": bool(normalize_training_target(TRAINING_TARGET) == "krea2" and USE_KREA_AI_CAPTIONING),
        "krea_caption_model": KREA_CAPTION_MODEL if normalize_training_target(TRAINING_TARGET) == "krea2" else "",
        "krea_caption_repair_enabled": bool(normalize_training_target(TRAINING_TARGET) == "krea2" and USE_KREA_CAPTION_REPAIR),
        "krea_caption_repair_model": KREA_CAPTION_REPAIR_MODEL if normalize_training_target(TRAINING_TARGET) == "krea2" and USE_KREA_CAPTION_REPAIR else "",
        "caption_primary_count": sum(1 for r in selected_sorted if r.get("caption_source") == "gpt_primary"),
        "caption_repair_count": sum(1 for r in selected_sorted if r.get("caption_source") == "gpt_repair"),
        "caption_local_fallback_count": sum(1 for r in selected_sorted if r.get("caption_source") == "local_fallback"),
        "controlled_buckets": bool(USE_CONTROLLED_BUCKETS),
        "medium_rescue_crop_enabled": bool(ENABLE_MEDIUM_RESCUE_CROP),
    }
    openai_usage_summary = build_openai_usage_summary()
    summary.update({
        "openai_api_requests": openai_usage_summary.get("requests", 0),
        "openai_input_tokens": openai_usage_summary.get("input_tokens", 0),
        "openai_output_tokens": openai_usage_summary.get("output_tokens", 0),
        "openai_total_tokens": openai_usage_summary.get("total_tokens", 0),
        "openai_token_limit_total": openai_usage_summary.get("token_limit_total", 0),
        "openai_token_limit_reached": openai_usage_summary.get("token_limit_reached", False),
        "openai_estimated_cost_usd": (
            openai_usage_summary.get("estimated_cost_usd")
            if openai_usage_summary.get("estimated_cost_usd") is not None
            else "n/a"
        ),
    })

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
        "openai_usage": openai_usage_summary,
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
