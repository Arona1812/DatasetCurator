# LoRA Dataset Curator

[English](README.md) · [MIT-Lizenz](LICENSE) · [Drittanbieter-Lizenzen](thirdparty-lic.md)

**LoRA Dataset Curator** ist eine lokale Web-UI, die aus einem Bildordner einen kuratierten und beschrifteten Trainingsdatensatz für **Character-LoRAs** erstellt. Das Tool ist für konsistente Personen-/Charakter-Datensätze optimiert und kombiniert Duplikaterkennung, Bildqualitätsprüfungen, optionale lokale Rahmenbereinigung, OpenAI-gestützte Bildaudits, Identitätskonsistenzprüfungen und datensatzbewusstes Captioning. Ein separater Video-Helfer kann geeignete Frames aus Quellvideos extrahieren.

> Dieses Repository enthält ausschließlich Code. Für externe APIs und Modellgewichte gelten die jeweiligen eigenen Nutzungs- und Lizenzbedingungen.

## Was das Tool kann

- Findet nicht lesbare Dateien, Bilder mit geringer Qualität und ähnliche Duplikate vor kostenpflichtigen API-Anfragen.
- Erkennt Screenshot-Rahmen und andere Layouts lokal; Originaldateien werden nie überschrieben.
- Prüft Bilder, wählt einen vielfältigen Datensatz aus und trennt fertige, zu prüfende und ausgeschlossene Ergebnisse.
- Erstellt optional ein **Subject Profile**, damit Identitätsmerkmale und Captions im gesamten Datensatz konsistent bleiben.
- Unterstützt Trainingsziel-Voreinstellungen für **ERNIE Image**, **Z-Image Base** und **Krea 2**.
- Exportiert Bilder, Captions, CSV, JSONL und einen Markdown-Dataset-Report.
- Extrahiert scharfe, posen-diverse Frames einer Referenzperson aus Videos (optionaler Video Processor).

## Ablauf

Der Bild-Curator folgt in der UI der tatsächlichen Verarbeitungsreihenfolge:

1. **Start / Project** — Bildordner, Trigger Word und API-Key festlegen.
2. **Preflight** — lokale Datei- und pHash-Duplikatprüfungen ausführen. Dabei wird keine OpenAI-Anfrage gestellt. Bereits bei der Projektinitialisierung erhält jedes Quellbild eine persistente ganzzahlige Bild-ID; die Rahmennavigation verwendet diese ID statt Dateinamen oder Galeriepositionen.
3. **Frames** *(optional)* — lokale Crop-Vorschläge prüfen, Original behalten oder manuellen Crop definieren.
4. **Audit & Selection** — Qualitäts-, Identitäts-, Diversitäts- und Captioning-Verarbeitung starten.
5. **Subject Profile** — bei `Profile then Caption` das Profil prüfen oder bei `Single Pass` automatisch fortfahren lassen.
6. **Results** — exportierte Trainings-, Review- und Nachbearbeitungsordner prüfen.

## Voraussetzungen

- **Python 3.10**
- Ein **OpenAI-API-Key mit verfügbarem Guthaben** für Audit- und Captioning-Pipeline. Lokaler Preflight und Frame-Analyse funktionieren ohne ihn.
- NVIDIA-GPU-Beschleunigung ist optional. Die Anwendung kann auch auf der CPU laufen.
- **InsightFace** ist für die normale Bildkuratierung optional, aber für Personenabgleich im Video Processor und den ArcFace Identity Check erforderlich.

Die UI zeigt die installierte Quellversion über `git describe --tags --always --dirty`. Dadurch wird der ausgecheckte Git-Tag inklusive weiterer Commits und lokaler Änderungen angezeigt, statt eine Versionsnummer manuell im Code zu pflegen.

## Schnellstart unter Windows

1. Repository klonen:

   ```bash
   git clone https://github.com/Arona1812/DatasetCurator.git
   cd DatasetCurator
   ```

2. `start_curator.bat` doppelklicken.

Der Launcher erstellt `curator_env`, installiert die Kernpakete, versucht die optionale InsightFace-Unterstützung zu installieren und öffnet die Gradio-UI im Browser.

> Der Windows-Launcher installiert derzeit CUDA-13.0-Builds von PyTorch und ONNX Runtime. Das Tool funktioniert auch ohne NVIDIA-GPU, für eine reine CPU-Installation ist die manuelle Installation aber schlanker.

## Manuelle Installation (Linux, macOS oder CPU-orientiertes Setup)

```bash
python3.10 -m venv curator_env
source curator_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional: erforderlich für Video Processor und ArcFace Identity Check.
pip install insightface

python dataset_curator_ui.py
```

`requirements.txt` verwendet portable Standardpakete. Für NVIDIA/CUDA-Beschleunigung installiere PyTorch- und ONNX-Runtime-Varianten passend zu Treiber und Plattform; siehe [PyTorch-Installationsanleitung](https://pytorch.org/get-started/locally/). Unter Windows kann InsightFace die [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) benötigen.

## Bild-Curator bedienen

1. UI starten und unter **Start / Project** den **Eingabeordner**, ein eindeutiges **Trigger Word** und den **OpenAI API Key** eingeben.
2. **Run preflight** anklicken. Nach Abschluss der lokalen Prüfungen werden die späteren Module freigeschaltet.
3. Optional in **Frames** einen lokalen Crop-Vorschlag auswählen, das Original beibehalten oder mit zwei Klicks auf diagonal gegenüberliegende Ecken einen rechteckigen manuellen Crop markieren.
4. In **Audit & Selection** Trainingsziel auswählen und die gewünschten Audit-, Identitäts-, Diversitäts- und Captioning-Einstellungen festlegen.
   - **Single Pass** erstellt das Profil und beendet den Workflow automatisch.
   - **Profile then Caption** pausiert nach der Profilerstellung, damit kanonische Merkmale, Cluster, Favoriten und Caption-Policies geprüft werden können.
5. Ergebnisse in `curated_<trigger>/` prüfen. Die wichtigsten Ordner sind:
   - `01_train_ready` — exportierte Trainingsbilder und Captions
   - `02_keep_unused` — gute, aber nicht für die Zielgröße ausgewählte Bilder
   - `03_caption_remove` — Bilder, deren Caption überarbeitet werden muss
   - `04_review`, `05_reject`, `06_needs_manual_review` — Bilder mit offenem Entscheidungsbedarf oder Ausschluss durch den Workflow

Der Projekt-Workspace speichert Reports und Caches ebenfalls unter `curated_<trigger>/`; Quellbilder bleiben unverändert.

## Trainingsziele

| Ziel | Captioning-Ansatz |
| --- | --- |
| **ERNIE Image** | Strukturierte Captions mit sichtbaren Identitätsankern. |
| **Z-Image Base** | Kompakte strukturierte Captions; das Trigger Token trägt die stabile Identität. |
| **Krea 2** | Datensatzbewusste Captions in natürlicher Sprache nach der finalen Bildauswahl. |

Die Ziel-Voreinstellungen liefern Startwerte. In der UI geänderte Einstellungen bleiben für den Lauf maßgeblich.

## Optionaler Video Processor

In der UI **Video Processor** öffnen und einen Videoordner, einen Ausgabeordner sowie ein klares Referenzbild der Zielperson angeben. Mit installiertem InsightFace werden unterstützte Videos (`mp4`, `mov`, `mkv`, `avi`) abgetastet, die Referenzperson abgeglichen und scharfe, posen-diverse Frames für den Bild-Curator gespeichert.

## API, Datenschutz und Lizenzen

- Das Projekt verwendet die OpenAI Responses API über `requests`; ein `openai`-Python-Paket ist nicht erforderlich.
- Den Key in der UI oder über `OPENAI_API_KEY` setzen. Er wird lokal verwendet und nicht in das Repository eingecheckt. Lokale UI-Einstellungen können ihn in ignorierten Laufzeitdateien speichern; diese Dateien nicht weitergeben.
- Das Projekt implementiert keinen Jugendschutz- oder NSFW-Eignungsfilter. Daten und Exporte vor dem Training selbst prüfen.
- Der eigene Code steht unter der [MIT-Lizenz](LICENSE). Details zu Abhängigkeiten stehen in [thirdparty-lic.md](thirdparty-lic.md).
- InsightFace-Code steht unter MIT, für die vortrainierten Modellgewichte einschließlich `buffalo_l` gelten jedoch separate Einschränkungen und Hinweise zur nicht-kommerziellen Forschung. Für eine kommerzielle Nutzung die erforderliche Lizenz direkt bei InsightFace beschaffen.

## Status und Unterstützung

Das Projekt ist in Entwicklung. Exportierte Bilder und Captions vor dem Training immer prüfen.

Wenn dir das Projekt hilft, kannst du die Entwicklung optional über [Buy Me a Coffee](https://buymeacoffee.com/arona1812) unterstützen. Spenden ändern keine Lizenzbedingungen.

## Haftungsausschluss

Die Software wird **ohne Gewähr** bereitgestellt. Du bist selbst für die Einhaltung der Bedingungen von OpenAI, InsightFace und allen weiteren verwendeten Diensten oder Modellen verantwortlich.