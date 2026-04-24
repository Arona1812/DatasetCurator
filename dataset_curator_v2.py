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
import traceback
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


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
    import open_clip
    HAVE_CLIP = True
except ImportError:
    HAVE_CLIP = False


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
AI_MODEL = "gpt-5.4-nano"
TRIGGER_CHECK_MODEL = "gpt-5.4-nano"

# Optionale Eskalation für schwierige Fälle:
# Erstes Audit läuft mit AI_MODEL. Falls ein Bild im Grenzbereich liegt,
# ein Review ist oder lokale und AI-Heuristik widersprüchlich sind,
# kann optional ein zweites, stärkeres Modell entscheiden.
USE_REVIEW_ESCALATION = False
REVIEW_ESCALATION_MODEL = ""
REVIEW_ESCALATION_SCORE_MIN = 50
REVIEW_ESCALATION_SCORE_MAX = 65
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
FACE_BLUR_PADDING_FACTOR   = 0.15      # Face-Bbox um diesen Faktor erweitern vor Blur-Messung

# Belichtungs-Check per Histogramm-Median (PIL, kein OpenCV nötig).
# Zu dunkel: Median < DARK_THRESHOLD. Zu hell: Median > BRIGHT_THRESHOLD.
USE_EXPOSURE_FILTER       = False
HARD_MAX_DARK_MEDIAN      = 20        # Unter 30/255 -> zu dunkel
HARD_MIN_BRIGHT_MEDIAN    = 255       # Über 225/255 -> überbelichtet

# pHash-Vorfilter VOR der API: berechnet alle Hashes lokal und wirft
# pixelnahe Duplikate raus bevor ein einziger API-Call gemacht wird.
# Nutzt denselben PHASH_HAMMING_THRESHOLD wie der spätere Pass-2-Filter.
USE_EARLY_PHASH_DEDUP     = True

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
CLIP_DEVICE = "cuda" if HAVE_CLIP and torch.cuda.is_available() else "cpu"

# --------------------------------
# Session-/Outfit-Clusterung
# --------------------------------
USE_SESSION_OUTFIT_CLUSTERING = True  # Begrenzt zu viele ähnliche Bilder derselben Session / desselben Outfits
MAX_PER_OUTFIT_CLUSTER = 4  # Maximalzahl pro Outfit-Cluster im finalen Datensatz
MAX_PER_SESSION_CLUSTER = 5  # Maximalzahl pro Session-Cluster im finalen Datensatz
ENABLE_DIVERSITY_PENALTIES = True  # Bestraft zu ähnliche Kandidaten bei der Endauswahl

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

# --------------------------------
# Export
# --------------------------------
EXPORT_REVIEW_IMAGES = True  # Exportiert Review-Bilder zusätzlich in einen separaten Ordner
EXPORT_REJECT_IMAGES = True  # Exportiert Reject-Bilder physisch mit; oft aus = spart Platz
EXPORT_SMART_CROP_COMPARISON = True  # Exportiert Vergleichspaare (Original vs. Headshot-Crop) in 07_smart_crop_pairs

# --------------------------------
# Ausgabeordner
# --------------------------------
OUTPUT_ROOT = os.path.join(INPUT_FOLDER, f"curated_{TRIGGER_WORD}")
TRAIN_READY_DIR = os.path.join(OUTPUT_ROOT, "01_train_ready")
CAPTION_REMOVE_DIR = os.path.join(OUTPUT_ROOT, "02_caption_remove")
REVIEW_DIR = os.path.join(OUTPUT_ROOT, "03_review")
REJECT_DIR = os.path.join(OUTPUT_ROOT, "04_reject")
MANUAL_REVIEW_DIR = os.path.join(OUTPUT_ROOT, "05_needs_manual_review")
CACHE_DIR = os.path.join(OUTPUT_ROOT, "_cache")
CLIP_CACHE_DIR = os.path.join(CACHE_DIR, "clip")
TRIGGER_CACHE_DIR = os.path.join(CACHE_DIR, "trigger")
SMART_CROP_COMPARISON_DIR = os.path.join(OUTPUT_ROOT, "07_smart_crop_pairs")
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
SEND_TEXT_IMAGES_TO_CAPTION_REMOVE = True  # Bilder mit sichtbarem Text/Watermark -> 02_caption_remove statt train_ready
INTERACTIVE_CAPTION_OVERRIDE = True        # Pausiert nach der Caption-Regel-Analyse und fragt dich nach optionalen Overrides

# ── SMART PRE-CROP (Post-API Headshot-Zoom) ────────────────────────────────────────────────
# Nach dem API-Audit des Originals: wenn das Bild groß ist und das Gesicht klein,
# wird ein enger Headshot-Crop erzeugt und SEPARAT zur API geschickt.
# Beide Versionen (Original + Crop) werden bewertet; die bessere gewinnt das Dataset.
ENABLE_SMART_PRECROP = True                # Pre-Crop aktivieren
SMART_PRECROP_MIN_FACE_PX = 120            # Mindest-Pixelgröße des Gesichts (min(fw, fh)) für Pre-Crop. Unter diesem Wert zu klein.
SMART_PRECROP_TRIGGER_RATIO = 0.07         # Pre-Crop nur wenn Gesicht < 7% des Gesamtbildes. Größere Gesichter brauchen kein Zoom.
SMART_PRECROP_PADDING_FACTOR = 1.5         # Wie viel Rahmen um das Gesicht herum (Faktor der Gesichtsgröße)
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
# faelschlich wiederverwendet. Diese Liste wachst mit jedem internen Feld,
# das aus strukturellen Gruenden keine UI-Kontrolle haben soll.
_UI_PROTECTED_KEYS = {"IG_FRAME_CACHE_VERSION"}
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
    CAPTION_REMOVE_DIR = os.path.join(OUTPUT_ROOT, "02_caption_remove")
    REVIEW_DIR = os.path.join(OUTPUT_ROOT, "03_review")
    REJECT_DIR = os.path.join(OUTPUT_ROOT, "04_reject")
    MANUAL_REVIEW_DIR = os.path.join(OUTPUT_ROOT, "05_needs_manual_review")
    CACHE_DIR = os.path.join(OUTPUT_ROOT, "_cache")
    CLIP_CACHE_DIR = os.path.join(CACHE_DIR, "clip")
    TRIGGER_CACHE_DIR = os.path.join(CACHE_DIR, "trigger")
    SMART_CROP_COMPARISON_DIR = os.path.join(OUTPUT_ROOT, "07_smart_crop_pairs")
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
    CAPTION_REMOVE_DIR,
    REVIEW_DIR,
    CACHE_DIR,
    CLIP_CACHE_DIR,
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


# ============================================================
# 3) HILFSFUNKTIONEN
# ============================================================

def safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("utf-8", errors="replace").decode("utf-8"))


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


def normalize_caption_profile(value: Optional[str]) -> str:
    v = normalize_text(value)
    if v in {"ernie", "z_image_base", "custom"}:
        return v
    return "ernie"


def enforce_caption_policy_profile(profile: Optional[str], policy: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure profile-specific caption fields stay enabled."""
    normalized = normalize_caption_profile(profile)
    if normalized == "ernie":
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
    """
    if not ENABLE_SMART_PRECROP:
        return None
    try:
        import tempfile
        fx, fy, fw, fh = ai_face_bbox_abs
        pad = int(max(fw, fh) * SMART_PRECROP_PADDING_FACTOR)
        x1 = max(0, fx - pad)
        y1 = max(0, fy - pad)
        x2 = min(img_w, fx + fw + pad)
        y2 = min(img_h, fy + fh + pad)
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


def early_phash_dedup(image_paths: List[str]) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Berechnet pHash für alle Bilder und entfernt pixelnahe Duplikate
    BEVOR die API aufgerufen wird. Gibt (survivors, duplicates, phash_cache) zurück.
    phash_cache: {absoluter_pfad: phash_int} für Wiederverwendung in local_subject_metrics.
    Gewinner werden pro Duplikat-Gruppe anhand eines deterministischen,
    lokalen Qualitätsscores gewählt; Dateigröße ist nur Tie-Breaker.
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

    survivor_set = set()
    duplicate_set = set()
    no_hash_paths = [path for path, phash in hashes if phash is None]

    # pHash-fähige Bilder zuerst stabil nach Dateiname sortieren, damit die
    # Gruppenbildung reproduzierbar bleibt. Die eigentliche Auswahl erfolgt
    # anschließend über den lokalen Qualitätsscore.
    hashed_items = sorted(
        [(path, phash) for path, phash in hashes if phash is not None],
        key=lambda x: os.path.basename(x[0]).lower(),
    )

    groups: List[Dict[str, Any]] = []
    for path, phash in hashed_items:
        assigned = False
        for group in groups:
            if any(hamming_distance(phash, member_hash) <= PHASH_HAMMING_THRESHOLD for _, member_hash in group["members"]):
                group["members"].append((path, phash))
                assigned = True
                break
        if not assigned:
            groups.append({"members": [(path, phash)]})

    survivors = list(no_hash_paths)
    score_cache: Dict[str, Tuple[float, Dict[str, float]]] = {}
    for group in groups:
        members: List[Tuple[str, int]] = group["members"]
        ranked_members = []
        for member_path, _ in members:
            score_cache[member_path] = early_duplicate_pick_score(member_path)
            ranked_members.append((member_path, *score_cache[member_path]))

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

        winner = ranked_members[0][0]
        survivor_set.add(winner)
        survivors.append(winner)
        for member_path, _, _ in ranked_members[1:]:
            duplicate_set.add(member_path)

    survivors = [p for p in image_paths if p in survivor_set or (p in no_hash_paths and p not in duplicate_set)]
    duplicates = [p for p in image_paths if p in duplicate_set]

    safe_print(f"   ↳ Early pHash: kept {len(survivors)}, removed {len(duplicates)} duplicates\n")
    return survivors, duplicates, phash_cache


def local_subject_metrics(image_path: str, phash_cache: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    width, height = image_dimensions(image_path)
    metrics: Dict[str, Any] = {
        "width": width,
        "height": height,
        "file_size_mb": round(file_size_mb(image_path), 3),
        "face_count_local": 0,
        "main_face_bbox": None,
        "main_face_ratio": 0.0,
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
                best = max(boxes, key=lambda b: b[2] * b[3] * max(0.001, b[4]))
                metrics["main_face_bbox"] = [best[0], best[1], best[2], best[3]]
                metrics["main_face_ratio"] = bbox_area_ratio(metrics["main_face_bbox"], w, h)
        except Exception:
            pass

    if metrics["face_count_local"] == 0 and HAAR_CASCADE is not None:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = HAAR_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                metrics["face_count_local"] = len(faces)
                x, y, bw, bh = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                metrics["main_face_bbox"] = [int(x), int(y), int(bw), int(bh)]
                metrics["main_face_ratio"] = bbox_area_ratio(metrics["main_face_bbox"], w, h)
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


def audit_cache_key(base_hash: str, model: str, variant: str = "audit") -> str:
    raw = f"{variant}|{base_hash}|{(model or '').strip().lower()}"
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


def responses_api_call(model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
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
            return response.json()
        except Exception as e:
            last_error = e
            if attempt >= MAX_RETRIES:
                break
            sleep_s = RETRY_BASE_SECONDS * attempt
            safe_print(f"   ↳ API error, retry {attempt}/{MAX_RETRIES} in {sleep_s:.1f}s: {e}")
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
            "tattoos_visible": {"type": "boolean"},
            "tattoos_description": {"type": "string"},
            "clothing_description": {"type": "string"},
            "pose_description": {"type": "string"},
            "expression": {"type": "string"},
            "gaze_direction": {"type": "string"},
            "background_description": {"type": "string"},
            "lighting_description": {"type": "string"},
            "quality_sharpness": {"type": "number", "minimum": 0, "maximum": 100},
            "quality_lighting": {"type": "number", "minimum": 0, "maximum": 100},
            "quality_composition": {"type": "number", "minimum": 0, "maximum": 100},
            "quality_identity_usefulness": {"type": "number", "minimum": 0, "maximum": 100},
            "quality_total": {"type": "number", "minimum": 0, "maximum": 100},
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
            "tattoos_visible",
            "tattoos_description",
            "clothing_description",
            "pose_description",
            "expression",
            "gaze_direction",
            "background_description",
            "lighting_description",
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


def openai_audit_image(image_path: str, local_meta: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
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
You MUST score the image strictly out of 10 for each category! Use decimals if needed (e.g. 7.5 or 8.2).
- quality_sharpness: 1 to 10 (allows decimals)
- quality_lighting: 1 to 10 (allows decimals)
- quality_composition: 1 to 10 (allows decimals)
- quality_identity_usefulness: 1 to 10 (allows decimals)
- quality_total: the simple sum of the 4 scores above

Important:
- Flag prominent text, watermarks, overlays, or readable shirt/screen text.
- Flag multiple prominent people.
- Ignore brand names and exact text content. Just flag the presence.
- Describe visible tattoos only as a raw fact.
- Describe hair color, length, and texture PRECISELY (e.g. "long wavy blonde hair", "short dark brown curly hair"). Never return empty or vague values like "brown".
- Describe eye color PRECISELY if visible (e.g. "blue", "green", "gray-green", "hazel"). Return empty string only if eyes are not visible.
- Describe skin_tone as a neutral factual value (e.g. "fair", "light", "medium", "olive", "dark"). Never return empty.
- Describe beard/glasses/piercings/makeup only as visible raw facts.
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
        "max_output_tokens": 1400,
        "store": False,
        "temperature": 0.1,
    }

    data = responses_api_call(chosen_model, payload)
    if data.get("NSFW_BLOCKED"):
        return {"NSFW_BLOCKED": True}
    text = extract_response_text(data)
    return json.loads(text)


def normalize_audit_scores(audit: Dict[str, Any]) -> Dict[str, Any]:
    qs = float(audit.get("quality_sharpness", 0))
    ql = float(audit.get("quality_lighting", 0))
    qc = float(audit.get("quality_composition", 0))
    qi = float(audit.get("quality_identity_usefulness", 0))

    if qs <= 10.0 and ql <= 10.0 and qc <= 10.0 and qi <= 10.0:
        audit["quality_sharpness"] = round(qs * 10.0, 1)
        audit["quality_lighting"] = round(ql * 10.0, 1)
        audit["quality_composition"] = round(qc * 10.0, 1)
        audit["quality_identity_usefulness"] = round(qi * 10.0, 1)
        total = (qs * 4.0) + (ql * 2.5) + (qc * 2.0) + (qi * 1.5)
        audit["quality_total"] = round(min(100.0, total), 1)
    else:
        audit["quality_total"] = round(qs + ql + qc + qi, 1)
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
    v = normalize_text(val)
    if v in {"", "none", "no", "n/a", "unknown", "not visible", "not applicable"}:
        return ""
    return v


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

    if multiple_people:
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
                if face_var >= 0 and face_var < FACE_MIN_BLUR_VARIANCE:
                    item.setdefault("status_notes", []).append(
                        f"face_blur_variance_{face_var:.1f}_below_{FACE_MIN_BLUR_VARIANCE}"
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

    return penalty


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
    t = normalize_feature_value(text)
    if not t:
        return ""
    t = t.replace("visible ", "").strip()
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


def build_caption(item: Dict[str, Any], global_rules: Dict[str, Any]) -> str:
    shot_type = item.get("shot_type", "headshot")
    mirror_selfie = bool(item.get("mirror_selfie", False))
    photo_type = photo_type_phrase(shot_type, mirror_selfie)
    caption_profile = normalize_caption_profile(globals().get("CAPTION_PROFILE", "ernie"))

    gender_class = normalize_feature_value(item.get("gender_class")) or "person"
    hair_desc = compact_trait(item.get("hair_description"))
    beard_desc = compact_trait(item.get("beard_description"))
    glasses_desc = compact_trait(item.get("glasses_description"))
    piercings_desc = compact_trait(item.get("piercings_description"))
    makeup_desc = compact_trait(item.get("makeup_description"))
    skin_tone = compact_trait(item.get("skin_tone"))
    eye_color = compact_trait(item.get("eye_color"))          # ← NEU
    body_build = compact_trait(item.get("body_build"))
    tattoos_visible = bool(item.get("tattoos_visible", False))
    tattoos_desc = compact_trait(item.get("tattoos_description"))
    clothing = normalize_feature_value(item.get("clothing_description"))
    pose = normalize_feature_value(item.get("pose_description"))
    expression = normalize_feature_value(item.get("expression"))
    gaze = normalize_feature_value(item.get("gaze_direction"))
    background = normalize_feature_value(item.get("background_description"))
    lighting = normalize_feature_value(item.get("lighting_description"))

    hair_tag = None
    if CAPTION_POLICY["include_hair_always"] and hair_desc:
        hair_tag = hair_desc
    elif CAPTION_POLICY["include_hair_when_variable"]:
        hair_tag = build_hair_caption_tag(item, global_rules)

    # ── ERNIE-kompatibler Personen-Anker ────────────────────────────────
    # Haar, Augenfarbe und Hautton werden direkt nach dem Trigger-Word
    # als feste Anker eingebaut, damit ERNIE seinen Modell-Bias nicht
    # überschreibt. Reihenfolge: Haar → Augen → Haut.
    anchor_parts: List[str] = []
    if caption_profile == "ernie":
        if hair_tag:
            anchor_parts.append(hair_tag)
        if CAPTION_POLICY.get("include_eye_color") and eye_color:  # ← NEU
            anchor_parts.append(f"{eye_color} eyes")
        if CAPTION_POLICY["include_skin_tone"] and skin_tone:
            anchor_parts.append(f"{skin_tone} skin")

    first = f"A {photo_type} of {TRIGGER_WORD}"
    if CAPTION_POLICY["include_gender_class"] and gender_class:
        first += f", a {gender_class}"

    # Anker direkt nach dem Personentyp, vor den variablen Traits
    if anchor_parts:
        first += " with " + ", ".join(anchor_parts)

    trait_bits: List[str] = []

    if shot_type in {"medium", "full_body"} and CAPTION_POLICY["include_body_build"] and body_build:
        trait_bits.append(body_build)

    if caption_profile != "ernie" and hair_tag:
        trait_bits.append(hair_tag)

    # ── Bart-Tag ──────────────────────────────────────────────────────────
    beard_rule = global_rules.get("beard_description", {})
    beard_variable = beard_rule.get("variable", False)
    beard_mode = normalize_compact_text(beard_rule.get("mode", ""))

    if CAPTION_POLICY["include_beard_always"] and beard_desc:
        trait_bits.append(beard_desc)
    elif CAPTION_POLICY["include_beard_when_variable"]:
        if beard_variable and beard_desc:
            trait_bits.append(beard_desc)
        elif not beard_variable and beard_desc and beard_mode:
            item_beard = normalize_compact_text(item.get("beard_description", ""))
            if item_beard and item_beard != beard_mode:
                trait_bits.append(beard_desc)

    if CAPTION_POLICY["include_glasses"] and glasses_desc and glasses_desc not in {"none", "no glasses"}:
        trait_bits.append(glasses_desc)

    if CAPTION_POLICY["include_piercings"] and piercings_desc and piercings_desc not in {"none", "no piercings"}:
        trait_bits.append(piercings_desc)

    if CAPTION_POLICY["include_makeup"] and makeup_desc and makeup_desc not in {"none", "no makeup"}:
        trait_bits.append(makeup_desc)

    if CAPTION_POLICY["include_tattoos"] and tattoos_visible:
        trait_bits.append(tattoos_desc or "visible tattoos")

    if trait_bits:
        first += ", " + ", ".join(dict.fromkeys([t for t in trait_bits if t]))
    first += "."

    # ── Rest der Funktion bleibt unverändert ─────────────────────────────
    sentences = [first]
    pronoun = "They"
    if gender_class in ["woman", "girl"]:
        pronoun = "She"
    elif gender_class in ["man", "boy"]:
        pronoun = "He"

    if clothing:
        sentences.append(f"{pronoun} {'is' if pronoun in ['He', 'She'] else 'are'} wearing {clothing}.")

    pose_bits = []
    if pose and pose not in {"none", "unknown"}:
        pose_bits.append(pose)
    if CAPTION_POLICY["include_expression"] and expression and expression not in {"none", "unknown"}:
        pose_bits.append(f"with a {expression}")
    if CAPTION_POLICY["include_gaze"] and gaze and gaze not in {"none", "unknown"}:
        pose_bits.append(gaze)

    if pose_bits:
        sentences.append(f"{pronoun} {'is' if pronoun in ['He', 'She'] else 'are'} " + ", ".join(pose_bits) + ".")

    if CAPTION_POLICY["include_lighting"] and lighting:
        sentences.append(f"{lighting.capitalize()}.")

    if CAPTION_POLICY["include_background"] and background:
        sentences.append(f"{background.capitalize()}.")

    # Watermark/Text-Bilder gehen konsequent nach caption_remove.
    # Kein "Watermark, text on image." in der Caption, da das Modell sonst
    # den Zusammenhang von Wasserzeichen und Identitaet lernt.

    caption = " ".join(sentences)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption


# ============================================================
# 10) CROP
# ============================================================

def body_aware_crop(image_path: str, item: Dict[str, Any]) -> Image.Image:
    pil_img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    img = np.array(pil_img)

    h, w = img.shape[:2]

    # Smart-Crop-Rows: Den Pre-Crop-Bereich (Face + Padding) direkt als
    # quadratische Crop-Region verwenden, NICHT nochmal über die hohen
    # Multiplikatoren (4.5/5.0) des normalen Headshot-Branches gehen.
    # Das sorgt für einen tatsächlich engeren Zoom als das Original.
    if item.get("is_smart_crop") and item.get("smart_crop_bbox"):
        target_w, target_h = 1024, 1024
        fx, fy, fw, fh = item["smart_crop_bbox"]
        # Dieselbe Padding-Logik wie in generate_headshot_crop()
        pad = int(max(fw, fh) * SMART_PRECROP_PADDING_FACTOR)
        sc_x1 = max(0, fx - pad)
        sc_y1 = max(0, fy - pad)
        sc_x2 = min(w, fx + fw + pad)
        sc_y2 = min(h, fy + fh + pad)
        # Quadratisch machen (1:1 für 1024x1024 Output)
        sc_w = sc_x2 - sc_x1
        sc_h = sc_y2 - sc_y1
        size = max(sc_w, sc_h)
        # Mindestgröße: min(w,h)//5 damit Crop nicht zu winzig wird
        size = clamp_int(size, min(w, h) // 5, min(w, h))
        # Zentrieren auf Face-Mitte, leicht nach oben versetzt (0.45)
        cx = fx + fw // 2
        cy = fy + fh // 2
        sq_x1 = max(0, min(cx - size // 2, w - size))
        sq_y1 = max(0, min(cy - int(size * 0.45), h - size))
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
        if pose_bbox:
            px, py, pw, ph = pose_bbox
            crop_h = int(ph * 1.12)
            crop_w = int(crop_h * aspect)
            if crop_w > w:
                crop_w = w
                crop_h = int(crop_w / aspect)
            cx = px + pw // 2
            cy = py + ph // 2
            x1, y1, x2, y2 = crop_box(cx - crop_w // 2, cy - crop_h // 2, crop_w, crop_h)
        else:
            crop_h = h
            crop_w = int(h * aspect)
            if crop_w > w:
                crop_w = w
                crop_h = int(crop_w / aspect)
            # X: face-aware wenn BBox vorhanden, sonst Bildmitte
            if face_bbox:
                fx, fy, fw, fh = face_bbox
                cx = fx + fw // 2
                x_start = clamp_int(cx - crop_w // 2, 0, w - crop_w)
            else:
                x_start = (w - crop_w) // 2
            # Y: Gesichts-OBERKANTE + Haarpuffer als Crop-Start.
            # Ziel: Gesicht+Haare knapp oben im Crop, kein unnötiger Hintergrund darüber.
            if face_bbox:
                fy_top = face_bbox[1]        # obere Kante des Gesichts
                fh_val = face_bbox[3]
                # Puffer oberhalb des Gesichts: 0.8× Gesichtshöhe für Haare/Scheitel
                hair_headroom = int(fh_val * 0.8)
                y_start = clamp_int(fy_top - hair_headroom, 0, h - crop_h)
            else:
                y_start = (h - crop_h) // 2
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

    lines.append("============================================================\n")
    return "\n".join(lines)

def main() -> None:
    warnings: List[str] = []

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
                ig_cropped_path = detect_and_crop_ig_frame(image_path)
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
            local_meta = local_subject_metrics(image_path, phash_cache=phash_cache)
            row.update(local_meta)
            row["file_hash"] = file_hash

            clip_embedding = None
            if USE_CLIP_DUPLICATE_SCORING:
                clip_embedding = compute_clip_embedding(image_path, file_hash)
            row["clip_embedding"] = clip_embedding

            if cached:
                audit = cached["audit"] if "audit" in cached else cached
                safe_print("   ↳ Cache used")
            else:
                audit = openai_audit_image(image_path, local_meta, model=AI_MODEL)

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
                save_cached_audit(primary_audit_cache_key, {"audit": audit, "model": AI_MODEL})

            row.update(audit)

            local_status, local_reasons = local_status_override(row)
            api_status = row.get("suggested_status", "review")

            if should_escalate_audit(api_status, local_status, float(row.get("quality_total", 0))):
                escalation_cache_key = audit_cache_key(file_hash, REVIEW_ESCALATION_MODEL, "escalation_audit")
                cached_escalation = load_cached_audit(escalation_cache_key)
                if cached_escalation:
                    escalated_audit = cached_escalation.get("audit", cached_escalation)
                    safe_print(f"   ↳ Escalation cache used ({REVIEW_ESCALATION_MODEL})")
                else:
                    safe_print(f"   ↳ Escalating with {REVIEW_ESCALATION_MODEL}...")
                    escalated_audit = openai_audit_image(image_path, local_meta, model=REVIEW_ESCALATION_MODEL)
                    if not escalated_audit.get("NSFW_BLOCKED"):
                        escalated_audit = normalize_audit_scores(escalated_audit)
                        save_cached_audit(escalation_cache_key, {"audit": escalated_audit, "model": REVIEW_ESCALATION_MODEL})

                if not escalated_audit.get("NSFW_BLOCKED"):
                    row.update(escalated_audit)
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
                            crop_local_meta = local_subject_metrics(crop_path)
                            if cached_crop:
                                crop_audit = cached_crop["audit"] if "audit" in cached_crop else cached_crop
                                safe_print("   ↳ Crop cache used")
                            else:
                                crop_audit = openai_audit_image(crop_path, crop_local_meta, model=AI_MODEL)

                            if not crop_audit.get("NSFW_BLOCKED"):
                                crop_audit = normalize_audit_scores(crop_audit)

                                if not cached_crop:
                                    save_cached_audit(crop_primary_cache_key, {"audit": crop_audit, "model": AI_MODEL})

                                crop_score = float(crop_audit.get("quality_total", 0))
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
                                        escalated_crop_audit = openai_audit_image(crop_path, crop_local_meta, model=REVIEW_ESCALATION_MODEL)
                                        if not escalated_crop_audit.get("NSFW_BLOCKED"):
                                            crop_audit = normalize_audit_scores(escalated_crop_audit)
                                            save_cached_audit(crop_escalation_cache_key, {"audit": crop_audit, "model": REVIEW_ESCALATION_MODEL})
                                    crop_score = float(crop_audit.get("quality_total", 0))

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
                                            compute_clip_embedding(crop_path, crop_hash)
                                            if USE_CLIP_DUPLICATE_SCORING
                                            else None
                                        ),
                                    }
                                    crop_row.update(crop_audit)
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
    selected_names = {r["original_filename"] for r in selected}
    for row in all_rows:
        if row["original_filename"] in selected_names:
            row["selected"] = True

    # PASS 5: Speichern
    shot_order = {"headshot": 0, "medium": 1, "full_body": 2}
    selected_sorted = sorted(
        selected,
        key=lambda r: (shot_order.get(r.get("shot_type"), 9), -int(r.get("quality_total", 0)))
    )

    counters = {
        "train_ready": 1,
        "caption_remove": 1,
        "review": 1,
    }

    try:
        caption_rule_overview = get_caption_rule_overview(global_rules) if 'get_caption_rule_overview' in globals() else {"fixed": {}, "override": {}}
        if INTERACTIVE_CAPTION_OVERRIDE:
            print("\n================ CAPTION RULE OVERVIEW ================")
            print("FIXED (stabil erkannt, normalerweise nicht jedes Mal explizit captionen):")
            for k, v in caption_rule_overview.get("fixed", {}).items():
                print(f" - {k}: {v.get('mode','')} | counts={v.get('counts',{})}")
            print("OVERRIDE CANDIDATES (frequent, but not stable enough):")
            for k, v in caption_rule_overview.get("override", {}).items():
                print(f" - {k}: mode={v.get('mode','')} | candidates={v.get('candidates',[])} | counts={v.get('counts',{})}")
            ans = input("\nOverride jetzt anpassen? (j/n): ").strip().lower()
            if ans in {"j", "ja", "y", "yes"}:
                print("Format: feld=wert1,wert2   | leer = weiter")
                while True:
                    line = input("Override: ").strip()
                    if not line:
                        break
                    if '=' not in line:
                        print("Invalid. Example: hair_description=blonde,brunette")
                        continue
                    field, vals = line.split('=', 1)
                    field = field.strip()
                    values = [v.strip() for v in vals.split(',') if v.strip()]
                    if field in global_rules and isinstance(global_rules[field], dict):
                        global_rules[field]["override_candidates"] = values
                        global_rules[field]["variable"] = False # User mapped it manually, so we consider it fixed/stable now
                        global_rules[field]["mode"] = values[0] if values else ""
                        print(f"OK -> {field} set to: {values[0] if values else ''} (variable=False)")
                    else:
                        print("Unknown field.")
    except EOFError:
        pass

    for row in selected_sorted:
        needs_text_cleanup = bool(row.get("watermark_or_overlay") or row.get("prominent_readable_text"))

        if needs_text_cleanup and SEND_TEXT_IMAGES_TO_CAPTION_REMOVE:
            bucket = "caption_remove"
            out_dir = CAPTION_REMOVE_DIR
        else:
            bucket = "train_ready"
            out_dir = TRAIN_READY_DIR

        new_basename = f"{SAFE_TRIGGER}_{counters[bucket]:03d}"
        counters[bucket] += 1

        row["output_bucket"] = bucket
        row["new_basename"] = new_basename
        caption = build_caption(row, global_rules)
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
            needs_text_cleanup = bool(row.get("watermark_or_overlay") or row.get("prominent_readable_text"))

            if needs_text_cleanup and SEND_TEXT_IMAGES_TO_CAPTION_REMOVE:
                bucket = "caption_remove"
                out_dir = CAPTION_REMOVE_DIR
                new_basename = f"{SAFE_TRIGGER}_{counters['caption_remove']:03d}"
            else:
                bucket = "review"
                out_dir = REVIEW_DIR
                new_basename = f"{SAFE_TRIGGER}_review_{counters['review']:03d}"

            counters[bucket] += 1
            row["output_bucket"] = bucket
            row["new_basename"] = new_basename
            row["final_caption"] = build_caption(row, global_rules)

            try:
                cropped = body_aware_crop(row["original_path"], row)
                img_out = os.path.join(out_dir, f"{new_basename}.jpg")
                txt_out = os.path.join(out_dir, f"{new_basename}.txt")
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

            # Reason-Datei: alle verfügbaren Quellen zusammenführen
            try:
                reason_parts = []

                # 1) local_override_reasons (Liste oder String aus CSV-Roundtrip)
                lor = row.get("local_override_reasons", [])
                if isinstance(lor, str):
                    lor = [x.strip() for x in lor.split(",") if x.strip()]
                reason_parts.extend(lor)

                # 2) status_notes (Duplikat-Marker, Smart-Crop-Marker etc.)
                sn = row.get("status_notes", [])
                if isinstance(sn, str):
                    sn = [x.strip() for x in sn.split(",") if x.strip()]
                for note in sn:
                    if note not in reason_parts:
                        reason_parts.append(note)

                # 3) short_reason (hart vergebener Grund: too_small, NSFW, script_error …)
                sr = row.get("short_reason", "")
                if sr and sr not in reason_parts:
                    reason_parts.append(sr)

                # 4) Duplikat-Infos explizit
                dup_method = row.get("duplicate_method", "")
                dup_of = row.get("duplicate_of", "")
                if dup_method and dup_of:
                    dup_info = f"duplicate_of:{dup_of} (method:{dup_method})"
                    if dup_info not in reason_parts:
                        reason_parts.append(dup_info)

                # 5) API-Vorschlag wenn vorhanden
                api_status = row.get("suggested_status", "")
                api_reason = row.get("short_reason", "")
                if api_status == "reject" and api_reason and api_reason not in reason_parts:
                    reason_parts.append(f"api_reject: {api_reason}")

                reasons_str = ", ".join(reason_parts) if reason_parts else "unknown"

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
        "mirror_selfie",
        "hair_description",
        "beard_description",
        "glasses_description",
        "piercings_description",
        "makeup_description",
        "skin_tone",
        "body_build",
        "tattoos_visible",
        "tattoos_description",
        "clothing_description",
        "pose_description",
        "expression",
        "gaze_direction",
        "background_description",
        "lighting_description",
        "issues",
        "short_reason",
        "local_override_reasons",
        "duplicate_of",
        "duplicate_method",
        "duplicate_distance",
        "main_face_ratio",
        "face_count_local",
        "width",
        "height",
        "file_size_mb",
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

    report = {
        "summary": summary,
        "warnings": warnings,
        "global_rules": global_rules,
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
    if EXPORT_SMART_CROP_COMPARISON and crop_pairs:
        safe_print(f"Crop comparisons: {SMART_CROP_COMPARISON_DIR} ({len(crop_pairs)} pairs)")
    safe_print(f"Caption-remove:  {CAPTION_REMOVE_DIR}")
    if EXPORT_REVIEW_IMAGES:
        safe_print(f"Review:          {REVIEW_DIR}")
    safe_print("=" * 70)


if __name__ == "__main__":
    main()
