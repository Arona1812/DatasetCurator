# DatasetCurator

**Aus einem Ordner mit Personenfotos wird ein prüfbarer Trainingsdatensatz für Personen-LoRAs.** DatasetCurator ist eine lokale Weboberfläche zum Bereinigen und Auswählen von Bildern, zum Bilden eines Personenprofils und zum Exportieren passender Trainingscaptions.

[English](README.md) · [Lizenz](LICENSE) · [Caption-Ablauf](docs/CAPTION_PIPELINE.md)

![DatasetCurator-Ablauf: Bilder laden, prüfen, Profil und Captions erstellen, Datensatz exportieren](docs/assets/workflow.svg)

## Was das Tool kann

- **Bilder bereinigen:** Unlesbare Dateien, ähnliche Duplikate und Qualitätsprobleme vor dem Bildaudit erkennen.
- **Screenshot-Rahmen entfernen:** Lokale Zuschnittvorschläge prüfen, ähnliche Layouts gemeinsam bearbeiten oder ein Bild ausschließen. Die Quelldateien bleiben unverändert. Der lokale Detektor findet auch Innenbilder in Story-Screenshots, Polaroid-Rahmen und Chat-Ansichten; ein Innenbild steht immer als erste Variante. Die Ansicht lässt sich auf **Nicht sicher automatisch beschnitten**, **Mehrere Entwürfe** oder **Alle Bilder** stellen.
- **Sticker und Stempel wegschneiden:** Erkennt das Bildaudit einen Musik-Sticker, Datumsstempel oder ein Wasserzeichen, sucht Luna die Stelle. Das Tool schneidet sie weg, sofern die Person vollständig und das Bild groß genug bleibt. Der Zuschnitt wird neu geprüft und ersetzt das Original nur, wenn er kein Overlay mehr zeigt; sonst landet das Bild wie bisher in `03_caption_remove`. Schalter unter *Audit & Auswahl → Qualität & Zuschnitt*.
- **Geeignete Fotos auswählen:** Qualität und Identität prüfen und einen vielfältigen Mix aus Porträts, Halb- und Ganzkörperbildern zusammenstellen. Mit InsightFace wird jedes Gesicht vor der Auswahl mit der Referenz verglichen (deine Favoriten und Core-Bilder, sonst alle Kandidaten): Ausreißer stehen im Profil als *Review*, zusammenpassende Bilder werden bevorzugt. Die Aufnahmeart aus dem Audit wird mit der gemessenen Gesichtsgröße abgeglichen: Selfies ab der Brust zählen als Porträt, Dreiviertelaufnahmen als Halbkörperbild, damit der Porträtanteil nicht verloren geht.
- **Subject Profile bilden:** Den häufigen Look der Person erkennen und sichtbare Veränderungen wie Haare, Brille, Piercings und Tattoos steuerbar halten.
- **Trainingscaptions erstellen:** GPT-6 Luna beschreibt jedes brauchbare Bild einmal; was in die finale Caption kommt, legen Caption-Regeln und Personenprofil fest; daraus entstehen ganze Sätze, lokal oder auf Wunsch von GPT-6 Luna aus dem exportierten Bild formuliert und gegen die Regeln geprüft. Vorschau vor dem Export.
- **Für Bild-LoRAs exportieren:** Neun Trainingsziele wählen, darunter ERNIE Image, Z-Image, Krea 2, Qwen-Image 2.1, FLUX.2 und experimentell Ideogram 4.

Der Schwerpunkt liegt auf Bildern. Ein optionaler Video Processor kann aus Videos geeignete Einzelbilder gewinnen.

## Voraussetzungen

- **Python 3.13**.
- **OpenAI-API-Key mit Guthaben** für Bildaudits, die optionale KI-Rahmensuche und GPT-6-Luna-Captions. Lokale Vorprüfung und Rahmenvorschläge funktionieren ohne Key.
- **InsightFace** ist für die normale Bildkuratierung optional; ArcFace-Identitätsabgleich und Personenabgleich im Video Processor benötigen es.
- **MediaPipe-Modelle** für die lokale Gesichts- und Posenerkennung (etwa 6 MB) werden beim ersten Gebrauch einmalig nach `_models/` geladen. Ohne sie arbeitet der Rahmen-Detektor weiter, prüft Zuschnitte aber nicht gegen Gesichter und Körper.
- **CLIP** für die Duplikatsuche: Ohne NVIDIA-GPU nimmt das Tool automatisch das schnelle Modell ViT-B-32, mit GPU ViT-L-14. Die Wahl steht unter *Audit & Auswahl → Duplikaterkennung*.

Die Python-Paketliste steht in [requirements.txt](requirements.txt).

## Installation

### Windows

```powershell
git clone https://github.com/Arona1812/DatasetCurator.git
cd DatasetCurator
.\start_curator.bat
```

Du kannst `start_curator.bat` auch doppelt anklicken. Der Launcher legt eine Python-3.13-Umgebung an, installiert die App-Pakete und öffnet die Oberfläche. Für eine Installation ohne CUDA-Pakete: `.\start_curator.bat --cpu`.

### Linux oder macOS

```bash
git clone https://github.com/Arona1812/DatasetCurator.git
cd DatasetCurator
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app/dataset_curator_ui.py
```

Der Windows-Launcher ist der gepflegte Installationsweg. Bei der manuellen Installation benötigen optionales InsightFace und GPU-Pakete eine plattformspezifische Einrichtung. Siehe [Installationshinweise](docs/INSTALLATION.md).

## Der erste Datensatz

1. Unter **Start / Projekt** den Bildordner wählen, ein eindeutiges Triggerwort festlegen und den API-Key eingeben.
2. **Vorprüfung** starten. Unter **Rahmen** mögliche Zuschnitte prüfen; du kannst das Original behalten oder ein Bild ausschließen. Brauchst du keine Zuschnitte, geht es mit **Rahmen überspringen · direkt zum Audit** weiter; das Audit nutzt dann die Originalbilder.
3. Unter **Audit & Auswahl** Trainingsziel und Erstcaption-Modell wählen. Mit **Profil zuerst selbst prüfen** pausiert der Lauf vor dem Export; alternativ entscheidet das Profil automatisch.
4. **Subject Profile** und Caption-Vorschau prüfen, dann exportieren. Das Ergebnis liegt unter `curated_<trigger>/`.

In `01_train_ready` liegen die ausgewählten Bilder und gleichnamige `.txt`-Captions. Weitere Ordner trennen ungenutzte, zu prüfende und ausgeschlossene Bilder. Ein Bericht und Metadaten zum Trainingsziel werden ebenfalls exportiert. Der Eingabeordner wird nicht überschrieben.

## Vor dem Training

Die neun Trainingsziele sind Exportoptionen, keine neun bereits bewiesenen LoRA-Rezepte. ERNIE Image, Z-Image Base und Krea 2 Raw haben eigene Caption-Feld-Voreinstellungen; die übrigen starten derzeit mit allgemeinen, anpassbaren Regeln. Ideogram 4 verwendet einen experimentellen JSON-Export. **Prüfe Bilder und Captions vor dem Training.** Die Trainingsqualität wurde noch nicht für alle Ziele gemessen.

Mehr Details: [Caption-Ablauf](docs/CAPTION_PIPELINE.md), [Rahmen prüfen](docs/FRAME_REVIEW.md) und [Projektstand](docs/PROJECT_REVIEW.md).
Bei Fragen oder Problemen kannst du [ein GitHub-Issue eröffnen](https://github.com/Arona1812/DatasetCurator/issues).

## Ordner

| Ordner | Inhalt |
|---|---|
| `app/` | das Programm (Oberfläche, Pipeline, Rahmen-Detektor, Captions) |
| `tests/` | Offline-Tests: `python -m unittest discover -s tests -t .` |
| `benchmark/` | Entwicklungswerkzeuge für Messungen am Rahmen-Detektor |
| `docs/` | Caption-Ablauf, Rahmenprüfung, Installation und Projektnotizen |

Einstellungen (`_ui_settings.json`), Laufkonfigurationen, `curator_env` und geladene Modelle bleiben im Hauptordner.

## Datenschutz und Lizenz

Die App läuft lokal. Bildaudits und der optionale Luna-Captioner senden ausgewählte Bilder an OpenAI; lokale Vorprüfung und Rahmenvorschläge laufen auf deinem Rechner. API-Keys können in ignorierten lokalen Einstellungsdateien gespeichert werden; gib diese Dateien nicht weiter.

Der eigene Code steht unter der [MIT-Lizenz](LICENSE). Für Abhängigkeiten und Modellgewichte gelten eigene Bedingungen; siehe [Drittanbieter-Lizenzen](thirdparty-lic.md).
