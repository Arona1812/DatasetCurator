import os
import json
import cv2
import numpy as np
from glob import glob
from insightface.app import FaceAnalysis
from sklearn.cluster import KMeans

# ==========================================
# 1. KONFIGURATION
# ==========================================
SOURCE_FOLDER = r"./00_videos"            # Dein Quellordner mit den Videos
TARGET_FOLDER = r"./00_input"             # Dein Zielordner für dataset_curator
REFERENCE_IMAGE = r"./referenz.jpg"       # Ein gutes, klares Bild der Zielperson

FRAMES_PER_MINUTE = 5                     # Anzahl der Bilder pro Video-Minute
SAMPLE_FPS = 2                            # Wie viele Frames pro Sekunde geprüft werden (spart Leistung)
SIMILARITY_THRESHOLD = 0.45               # Ab wann ein Gesicht als "richtige Person" gilt (0.4 bis 0.6 ist gut)
MIN_SHARPNESS = 50.0                      # Mindestschärfe (Laplace-Varianz), um Blur direkt zu verwerfen

# ── UI-Config Override ────────────────────────────────────────────────────────
_UI_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ui_video_config.json")
if os.path.exists(_UI_CONFIG_PATH):
    try:
        with open(_UI_CONFIG_PATH, "r", encoding="utf-8") as _f:
            _ui_cfg = json.load(_f)
        for _k, _v in _ui_cfg.items():
            if _k in globals() and not _k.startswith("_"):
                globals()[_k] = _v
    except Exception as _e:
        print(f"⚠️ UI-Config konnte nicht geladen werden: {_e}")
# ==========================================

def get_sharpness(img):
    """Berechnet die Schärfe über die Varianz des Laplace-Operators."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def compute_similarity(emb1, emb2):
    """Berechnet die Cosine Similarity zwischen zwei Face-Embeddings."""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def process_minute_chunk(chunk_frames, video_name, minute_idx):
    """Clustert die Frames einer Minute nach Winkel und speichert die besten 5."""
    if not chunk_frames:
        return
    
    # Wenn wir weniger oder exakt so viele Frames haben wie gefordert, alle nehmen
    if len(chunk_frames) <= FRAMES_PER_MINUTE:
        selected = chunk_frames
    else:
        # Clustering basierend auf Pose (Yaw, Pitch)
        features = np.array([[f['yaw'], f['pitch']] for f in chunk_frames])
        kmeans = KMeans(n_clusters=FRAMES_PER_MINUTE, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        selected = []
        for cluster_id in range(FRAMES_PER_MINUTE):
            # Alle Frames in diesem Cluster
            cluster_items = [chunk_frames[i] for i in range(len(chunk_frames)) if labels[i] == cluster_id]
            if cluster_items:
                # Wähle den Frame mit der höchsten Schärfe im jeweiligen Cluster
                best_frame = max(cluster_items, key=lambda x: x['sharpness'])
                selected.append(best_frame)
    
    # Speichere die ausgewählten Frames
    for idx, item in enumerate(selected):
        out_name = f"{video_name}_min{minute_idx:03d}_{idx+1}.jpg"
        out_path = os.path.join(TARGET_FOLDER, out_name)
        # 100% Qualität erzwingen, passend zum dataset_curator
        cv2.imwrite(out_path, item['frame'], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print(f"  -> Gespeichert: {out_name} (Schärfe: {item['sharpness']:.1f})")

def main():
    os.makedirs(TARGET_FOLDER, exist_ok=True)
    
    # InsightFace initialisieren
    print("Lade InsightFace Modell...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Referenzbild laden
    print(f"Lade Referenzbild: {REFERENCE_IMAGE}")
    ref_img = cv2.imread(REFERENCE_IMAGE)
    if ref_img is None:
        print("FEHLER: Referenzbild nicht gefunden!")
        return
        
    ref_faces = app.get(ref_img)
    if not ref_faces:
        print("FEHLER: Kein Gesicht auf dem Referenzbild gefunden!")
        return
    ref_embedding = ref_faces[0].normed_embedding
    print("Referenz-Embedding erfolgreich erstellt.\n")

    # Alle Videos suchen
    video_files = []
    for ext in ["*.mp4", "*.mov", "*.mkv", "*.avi"]:
        video_files.extend(glob(os.path.join(SOURCE_FOLDER, ext)))
        video_files.extend(glob(os.path.join(SOURCE_FOLDER, ext.upper())))
        
    print(f"Gefundene Videos: {len(video_files)}")

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\nVerarbeite Video: {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0 # Fallback
        
        frame_interval = int(fps / SAMPLE_FPS)
        if frame_interval < 1: frame_interval = 1
        
        minute_frames = int(fps * 60)
        current_minute_chunk = []
        minute_idx = 1
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Nur sample-frames analysieren (z.B. alle 0.5 Sekunden)
            if frame_count % frame_interval == 0:
                faces = app.get(frame)
                
                for face in faces:
                    sim = compute_similarity(ref_embedding, face.normed_embedding)
                    
                    if sim >= SIMILARITY_THRESHOLD:
                        sharpness = get_sharpness(frame)
                        
                        # Nur Frames behalten, die nicht massiv verschwommen sind
                        if sharpness > MIN_SHARPNESS:
                            yaw, pitch, roll = face.pose
                            current_minute_chunk.append({
                                'frame': frame.copy(),
                                'yaw': yaw,
                                'pitch': pitch,
                                'sharpness': sharpness
                            })
                        break # Wir haben unsere Zielperson in diesem Frame gefunden
            
            # Wenn eine Minute voll ist, Chunk verarbeiten
            if frame_count % minute_frames == 0:
                print(f" Analysiere Minute {minute_idx} ({len(current_minute_chunk)} gute Kandidaten gefunden)...")
                process_minute_chunk(current_minute_chunk, video_name, minute_idx)
                current_minute_chunk = []
                minute_idx += 1
                
        # Den letzten (angebrochenen) Chunk verarbeiten
        if current_minute_chunk:
            print(f" Analysiere Rest-Minute {minute_idx} ({len(current_minute_chunk)} gute Kandidaten gefunden)...")
            process_minute_chunk(current_minute_chunk, video_name, minute_idx)
            
        cap.release()

if __name__ == "__main__":
    main()