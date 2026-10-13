# DatasetCurator

**Aus einem Ordner mit Personenfotos wird ein prüfbarer Trainingsdatensatz für Personen-LoRAs.** DatasetCurator ist eine lokale Weboberfläche zum Bereinigen und Auswählen von Bildern, zum Bilden eines Personenprofils und zum Exportieren passender Trainingscaptions.

[English](README.md) · [Lizenz](LICENSE) · [Caption-Ablauf](CAPTION_PIPELINE.md)

![DatasetCurator-Ablauf: Bilder laden, prüfen, Profil und Captions erstellen, Datensatz exportieren](docs/assets/workflow.svg)

## Was das Tool kann

- **Bilder bereinigen:** Unlesbare Dateien, ähnliche Duplikate und Qualitätsprobleme vor dem Bildaudit erkennen.
- **Screenshot-Rahmen entfernen:** Lokale Zuschnittvorschläge prüfen, ähnliche Layouts gemeinsam bearbeiten oder ein Bild ausschließen. Die Quelldateien bleiben unverändert.
- **Geeignete Fotos auswählen:** Qualität und Identität prüfen und einen vielfältigen Mix aus Porträts, Halb- und Ganzkörperbildern zusammenstellen.
- **Subject Profile bilden:** Den häufigen Look der Person erkennen und sichtbare Veränderungen wie Haare, Brille, Piercings und Tattoos steuerbar halten.
- **Trainingscaptions erstellen:** Erstbeschreibungen von Florence 2, JoyCaption oder GPT-6 Luna speichern; finale Captions aus Bildaudit, Profil und gewählten Regeln zusammensetzen und vor dem Export ansehen.
- **Für Bild-LoRAs exportieren:** Neun Trainingsziele wählen, darunter ERNIE Image, Z-Image, Krea 2, Qwen-Image 2.1, FLUX.2 und experimentell Ideogram 4.

Der Schwerpunkt liegt auf Bildern. Ein optionaler Video Processor kann aus Videos geeignete Einzelbilder gewinnen.

## Voraussetzungen

- **Python 3.13**.
- **OpenAI-API-Key mit Guthaben** für Bildaudits, die optionale KI-Rahmensuche und GPT-6-Luna-Captions. Lokale Vorprüfung und Rahmenvorschläge funktionieren ohne Key.
- **NVIDIA-GPU mit CUDA** nur für lokale Florence-2- oder JoyCaption-Läufe. Die Hauptanwendung und die CPU-Installation benötigen keine GPU.
- **InsightFace** ist für die normale Bildkuratierung optional; ArcFace-Identitätsabgleich und Personenabgleich im Video Processor benötigen es.

Die Python-Paketliste steht in [requirements.txt](requirements.txt).

## Installation

### Windows

```powershell
git clone https://github.com/Arona1812/DatasetCurator.git
cd DatasetCurator
.\start_curator.bat
```

Du kannst `start_curator.bat` auch doppelt anklicken. Der Launcher legt eine Python-3.13-Umgebung an, installiert die App-Pakete und öffnet die Oberfläche. Für eine Installation ohne CUDA-Pakete: `.\start_curator.bat --cpu`.

Florence 2 und JoyCaption benötigen **separate lokale Modellgewichte und eine CUDA-GPU**. Der Launcher installiert diese Worker nicht. Hinweise stehen unter [lokale Caption-Einrichtung](CAPTION_PIPELINE.md#lokale-einrichtung). Ohne lokale Worker wähle unter *Audit & Auswahl* für die Erstbeschreibung **GPT-6 Luna**; dabei werden Bilder an OpenAI gesendet und API-Guthaben verbraucht.

### Linux oder macOS

```bash
git clone https://github.com/Arona1812/DatasetCurator.git
cd DatasetCurator
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python dataset_curator_ui.py
```

Der Windows-Launcher ist der gepflegte Installationsweg. Bei der manuellen Installation benötigen optionales InsightFace und GPU-Pakete eine plattformspezifische Einrichtung. Siehe [Installationshinweise](docs/INSTALLATION.md).

## Der erste Datensatz

1. Unter **Start / Projekt** den Bildordner wählen, ein eindeutiges Triggerwort festlegen und den API-Key eingeben.
2. **Vorprüfung** starten. Unter **Rahmen** mögliche Zuschnitte prüfen; du kannst das Original behalten oder ein Bild ausschließen.
3. Unter **Audit & Auswahl** Trainingsziel und Erstcaption-Modell wählen. Mit **Profil zuerst selbst prüfen** pausiert der Lauf vor dem Export; alternativ entscheidet das Profil automatisch.
4. **Subject Profile** und Caption-Vorschau prüfen, dann exportieren. Das Ergebnis liegt unter `curated_<trigger>/`.

In `01_train_ready` liegen die ausgewählten Bilder und gleichnamige `.txt`-Captions. Weitere Ordner trennen ungenutzte, zu prüfende und ausgeschlossene Bilder. Ein Bericht und Metadaten zum Trainingsziel werden ebenfalls exportiert. Der Eingabeordner wird nicht überschrieben.

## Vor dem Training

Die neun Trainingsziele sind Exportoptionen, keine neun bereits bewiesenen LoRA-Rezepte. ERNIE Image, Z-Image Base und Krea 2 Raw haben eigene Caption-Feld-Voreinstellungen; die übrigen starten derzeit mit allgemeinen, anpassbaren Regeln. Ideogram 4 verwendet einen experimentellen JSON-Export. **Prüfe Bilder und Captions vor dem Training.** Die Trainingsqualität wurde noch nicht für alle Ziele gemessen.

Mehr Details: [Caption-Ablauf](CAPTION_PIPELINE.md), [Rahmen prüfen](docs/FRAME_REVIEW.md) und [Projektstand](PROJECT_REVIEW.md).
Bei Fragen oder Problemen kannst du [ein GitHub-Issue eröffnen](https://github.com/Arona1812/DatasetCurator/issues).

## Datenschutz und Lizenz

Die App läuft lokal. Bildaudits und der optionale Luna-Captioner senden ausgewählte Bilder an OpenAI; lokale Vorprüfung, Rahmenvorschläge, Florence 2 und JoyCaption laufen auf deinem Rechner. API-Keys können in ignorierten lokalen Einstellungsdateien gespeichert werden; gib diese Dateien nicht weiter.

Der eigene Code steht unter der [MIT-Lizenz](LICENSE). Für Abhängigkeiten und Modellgewichte gelten eigene Bedingungen; siehe [Drittanbieter-Lizenzen](thirdparty-lic.md).
