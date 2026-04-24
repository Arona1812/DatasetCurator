# LoRA Dataset Curator (DE)

Interaktive Toolchain zur automatischen Kuratierung von LoRA-Trainingsdaten aus Bildordnern und Videos.
Der Curator kombiniert lokale Filter (Schärfe, Auflösung, pHash), MediaPipe, CLIP und eine OpenAI-gestützte Bildanalyse, um einen kleinen, hochwertigen Datensatz mit konsistenten Captions zu erzeugen.

> Hinweis: Dieses Projekt stellt nur den **Code** zur Verfügung. Für die Nutzung externer Modelle und APIs (z. B. InsightFace-Modelle, OpenAI-API) gelten deren eigenen Lizenzbedingungen.

---

## Hintergrund

Dieses Projekt ist aus einem sehr praktischen Frust entstanden: Das manuelle Sortieren, Aufbereiten und Beschriften von Datensätzen war auf Dauer schlicht zu aufwendig. Also habe ich begonnen, gemeinsam mit ChatGPT nach Möglichkeiten zur Automatisierung dieses Workflows zu suchen. Nach rund 300 Prompts und dem Einsatz von 4 verschiedenen LLMs ist daraus das Tool entstanden, das du hier siehst. Im wahrsten Sinne des Wortes: 100 % vibe-coded.

---

## Voraussetzungen

Für die OpenAI-gestützten Funktionen dieses Projekts benötigst du einen eigenen OpenAI-API-Schlüssel sowie ausreichend API-Guthaben oder verfügbare kostenlose Tokens auf deinem Account.

---

## Features

- Gradio-Weboberfläche für Dataset Curator und Video Processor
- Persistente UI-Einstellungen mit automatischem Wiederherstellen sowie umschaltbare UI-Sprache auf Englisch/Deutsch
- Video-Frame-Extraktion per InsightFace-Referenzbild (`buffalo_l`)
- Lokale Vorfilter für Auflösung, Dateigröße, Unschärfe, Belichtung und frühe pHash-Deduplizierung vor API-Aufrufen
- Optionaler Subject-Sanity-/Gliedmaßen-Filter auf Basis erkannter Torso-Landmarks
- Duplikaterkennung mit pHash und OpenCLIP / CLIP für semantische Ähnlichkeit
- OpenAI-gestützte Bildbewertung für Qualität, Shot-Typ, Motivklarheit, Attribute, Text/Wasserzeichen und Caption-Metadaten
- Optionale Eskalation auf ein stärkeres Modell für schwierige Review-Fälle, Status-Konflikte und knappe Smart-Crop-Entscheidungen
- Optionale KI-Prüfung des Trigger-Worts
- Automatisches Captioning mit konfigurierbaren Caption-Profilen und optionalen Caption-Regel-Overrides
- Bilder mit Text/Wasserzeichen können automatisch nach `caption_remove` statt in den Train-Ready-Output einsortiert werden
- Session-/Outfit-Clustering und Diversity-Penalties für mehr Datensatzvielfalt
- Smart-Pre-Crop für Headshots aus weiteren Bildern inklusive Export von Original-vs.-Crop-Vergleichspaaren
- Entfernung von Instagram-/UI-Rändern bei Screenshots und Social-Media-Captures
- Bucket-taugliche Crop-Profile zur Reduzierung der Trainings-Buckets
- Strukturierte Ausgabeordner für Train-Ready-, Caption-Remove-, Review-, Reject- und Manual-Review-Bilder
- Audit-/Embedding-Cache sowie Retry-/Resume-Logik zum Sparen von Zeit und API-Kosten
- Integrierter Ergebnis-Viewer in der UI mit Bildgalerie, Captions und Vorschau des Dataset-Reports
- Ergebnis-Export mit Captions, CSV-, JSONL-Daten und Markdown-Dataset-Report

---

## Installation

### Schnellstart (Windows)

1. Repository klonen und in den Ordner wechseln:

```bash
git clone https://github.com/Arona1812/DatasetCurator.git
cd <dein-repo-ordner>
```

2. `start_curator.bat` doppelklicken.

Das Skript:
- erstellt die virtuelle Umgebung `curator_env`,
- installiert alle benötigten Pakete (requests, pillow, numpy, mediapipe, torch, torchvision, torchaudio, open_clip_torch, opencv-python, insightface, onnxruntime, scikit-learn, gradio),
- startet die Gradio-UI im Browser.

### Manuelle Installation (Beispiel Linux/macOS)

Versionen bitte an dein CUDA/PyTorch-Setup anpassen:

```bash
python3.10 -m venv curator_env
source curator_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install requests pillow numpy
pip install mediapipe==0.10.33
pip install "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" --index-url https://download.pytorch.org/whl/cu130
pip install open_clip_torch
pip install opencv-python insightface onnxruntime scikit-learn
pip install gradio
python dataset_curator_ui.py
```

> Bitte die konkreten Versionen an dein `start_curator.bat`-Setup und deinen CUDA-Treiber anpassen.

---

## Nutzung

### 1. Dataset Curator (Bilder)

1. UI starten:
   - Windows: `start_curator.bat` ausführen
   - Andere Plattformen: `python dataset_curator_ui.py` in der virtuellen Umgebung

2. Im Tab **Dataset Curator**:
   - `Trigger Word`: eindeutiges Token für deine Person (z. B. `aronaLora09`).
   - `Input-Ordner Bilder`: Ordner mit deinen Quellbildern (keine Unterordner-Suche).
   - `Ziel-Datensatzgröße`: gewünschte Anzahl finaler Trainingsbilder.
   - `OpenAI API Key`: dein eigener OpenAI-API-Schlüssel.
   - Qualitäts-Schwellen, Shot-Verteilung, Vorfilter, Duplikaterkennung, Smart-Crop, Clustering und Caption-Optionen nach Bedarf einstellen.

3. Der Curator schreibt temporäre Config-Dateien (`_ui_config.json`) und startet `dataset_curator_v2.py` im Hintergrund.

4. Ergebnisse liegen in `curated_<trigger>/` mit Ordnern wie `01_train_ready`, `02_caption_remove`, `03_review`, `04_reject`, `05_needs_manual_review`, `_cache` und `07_smart_crop_pairs`.

5. Nutze die Dateien aus `01_train_ready` und ausgewählte Bilder aus `03_review` für dein LoRA-Training. Prüfe zusätzlich `02_caption_remove` und `05_needs_manual_review`, falls Lieblingsbilder nur noch kleine manuelle Nacharbeit oder eine angepasste Caption brauchen.

### 2. Video Processor

1. Im Tab **Video Processor**:
   - `Video-Ordner`: Pfad mit deinen Video-Dateien (mp4, mov, mkv, avi).
   - `Ausgabe-Ordner`: Zielordner für extrahierte Frames (z. B. `r.00_input`).
   - `Referenzbild Zielperson`: klares Referenzfoto der Zielperson (frontal, gutes Licht).

2. Der Video Processor:
   - erkennt die Zielperson mit InsightFace (`buffalo_l`),
   - analysiert Frames mit einstellbarer FPS,
   - clustert Frames pro Minute nach Pose (Yaw/Pitch) und wählt die schärfsten Kandidaten.

Die extrahierten Frames können direkt dem Bild-Curator zugeführt werden.

---

## OpenAI-API

Dieses Projekt kann optional die OpenAI-API nutzen, um Bilder zu bewerten und strukturierte Metadaten zu erzeugen.

- Du benötigst einen **eigenen OpenAI-Account** und API-Key.
- Der API-Key wird entweder:
  - über das UI-Feld `OpenAI API Key` gesetzt oder
  - aus der Umgebungsvariablen `OPENAI_API_KEY` gelesen.
- Der Key wird nur lokal im Prozessumfeld verwendet und **nicht** im Repository gespeichert.
  Laufzeit-Konfigurationsdateien (wie `_ui_config.json`, `_ui_video_config.json`, `_ui_settings.json`) sind per `.gitignore` ausgeschlossen.

Mit der Nutzung der OpenAI-API erklärst du dich mit den OpenAI Terms of Use und dem OpenAI Services Agreement einverstanden.

---

## InsightFace-Modelle

Der Video Processor verwendet InsightFace zur Gesichtserkennung, insbesondere das Modell `buffalo_l`.

- Die **InsightFace-Python-Bibliothek** steht unter der MIT-Lizenz.
- Die **vortrainierten Modelle** aus dem InsightFace Model Zoo (inkl. `buffalo_l`) sind nur für **nicht-kommerzielle Forschungszwecke** freigegeben.
- Für **kommerzielle Nutzung** dieser Modelle ist eine gesonderte Lizenz direkt bei InsightFace erforderlich.

Dieses Repository enthält **keine** vortrainierten InsightFace-Modellgewichte; sie werden von der Bibliothek geladen oder müssen separat bezogen werden.

---

## Drittanbieter-Lizenzen

Der eigene Code in diesem Repository steht unter der **MIT-Lizenz**, siehe `LICENSE`.

Wichtige Bibliotheken und ihre Lizenzen:

- Gradio – Apache-2.0
- MediaPipe – Apache-2.0
- PyTorch – BSD-3-Clause
- OpenCV / opencv-python – Apache-2.0 (OpenCV), MIT (Wrapper)
- Pillow – HPND
- NumPy – BSD-3-Clause
- scikit-learn – BSD-3-Clause
- open_clip_torch / OpenCLIP – Apache-2.0 / MIT (versionsabhängig)
- InsightFace (Code) – MIT; Modelle non-commercial
- onnxruntime – MIT

Details findest du in `thirdparty-lic.md`.

---

## Spenden

Wenn dir dieses Projekt hilft und du die Entwicklung unterstützen möchtest, kannst du optional einen Kaffee spendieren:

- Buy me a coffee: https://buymeacoffee.com/arona1812

Spenden sind vollständig freiwillig und ändern **nicht** die Lizenzbedingungen oder Nutzungsbeschränkungen von Drittanbieter-Komponenten.

---

## Haftungsausschluss

Dieses Projekt wird "wie besehen" ohne jegliche Garantie bereitgestellt.

Du bist selbst verantwortlich für:
- die Einhaltung der Lizenzbedingungen der InsightFace-Modelle (non-commercial, ggf. kommerzielle Modelllizenz),
- die Einhaltung der Lizenz- und Nutzungsbedingungen der OpenAI-API und anderer externer Dienste.

Der Autor übernimmt keine Haftung für den Einsatz dieses Tools in produktiven oder kommerziellen Umgebungen.
