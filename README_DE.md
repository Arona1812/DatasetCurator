# LoRA Dataset Curator

[English](README.md) · [MIT-Lizenz](LICENSE) · [Drittanbieter-Lizenzen](thirdparty-lic.md)

**LoRA Dataset Curator** ist eine lokale Web-UI, die aus einem Bildordner einen kuratierten und beschrifteten Trainingsdatensatz für **Character-LoRAs** erstellt. Das Tool ist für konsistente Personen-/Charakter-Datensätze optimiert und kombiniert Duplikaterkennung, Bildqualitätsprüfungen, optionale lokale Rahmenbereinigung, OpenAI-gestützte Bildaudits, Identitätskonsistenzprüfungen und datensatzbewusstes Captioning. Ein separater Video-Helfer kann geeignete Frames aus Quellvideos extrahieren.

> Dieses Repository enthält ausschließlich Code. Für externe APIs und Modellgewichte gelten die jeweiligen eigenen Nutzungs- und Lizenzbedingungen.

## Was das Tool kann

- Findet nicht lesbare Dateien, Bilder mit geringer Qualität und ähnliche Duplikate vor kostenpflichtigen API-Anfragen.
- Erkennt Screenshot-Rahmen lokal und bietet zusätzlich eine KI-Erkennung des eigentlichen Fotobereichs für schwierige Layouts; Originaldateien werden nie überschrieben.
- Prüft Bilder, wählt einen vielfältigen Datensatz aus und trennt fertige, zu prüfende und ausgeschlossene Ergebnisse.
- Erstellt optional ein **Subject Profile**, damit Identitätsmerkmale und Captions im gesamten Datensatz konsistent bleiben.
- Unterstützt Trainingsziel-Voreinstellungen für **ERNIE Image**, **Z-Image Base** und **Krea 2**.
- Exportiert Bilder, Captions, CSV, JSONL und einen Markdown-Dataset-Report.
- Extrahiert scharfe, posen-diverse Frames einer Referenzperson aus Videos (optionaler Video Processor).

## Ablauf

Der Bild-Curator folgt in der UI der tatsächlichen Verarbeitungsreihenfolge:

1. **Start / Project** — Bildordner, Trigger Word und API-Key festlegen.
2. **Preflight** — lokale Datei- und pHash-Duplikatprüfungen ausführen. Dabei wird keine OpenAI-Anfrage gestellt.
3. **Frames** *(optional)* — nach Preflight automatisch erzeugte Vorschläge prüfen. Bei schwierigen Screenshots die zusätzliche KI-Erkennung starten; Vorschlag wählen oder mit zwei Klicks einen manuellen Crop setzen. Nach der Entscheidung folgt das nächste offene Bild.
4. **Audit & Selection** — Qualitäts-, Identitäts-, Diversitäts- und Captioning-Verarbeitung starten.
5. **Subject Profile** — bei `Profile then Caption` das Profil prüfen oder bei `Single Pass` automatisch fortfahren lassen.
6. **Results** — exportierte Trainings-, Review- und Nachbearbeitungsordner prüfen.

In **Audit & Auswahl** steht die Wahl zwischen automatischer Profilnutzung und eigener Profilprüfung direkt im Hauptbereich. Die lokale Laufübersicht zeigt Vorprüfungsstand, geladene Rahmenentscheidungen, Zielgröße, Audit-Modell und Ablauf. Sie aktualisiert sich beim Öffnen des Tabs und bei Änderungen; „Laufübersicht aktualisieren · lokal“ lädt sie erneut. Erst „Audit starten“ startet die Verarbeitung mit möglichen API-Kosten. Nachträglich geänderte Rahmenentscheidungen markieren einen vorhandenen Audit-Stand als aktualisierungsbedürftig.

## Voraussetzungen

- **Python 3.13**
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

Der Launcher erstellt `curator_env_py313`, installiert die Kernpakete, versucht die optionale InsightFace-Unterstützung zu installieren und öffnet die Gradio-UI im Browser.

> Der Windows-Launcher installiert standardmäßig CUDA-13.0-Builds von PyTorch und ONNX Runtime. Für eine portable Installation ohne CUDA `start_curator.bat --cpu` verwenden. Die frühere Umgebung `curator_env` bleibt erhalten.

Die lokale Gesicht- und Posenerkennung verwendet die aktuelle MediaPipe Tasks API. Beim ersten benötigten Einsatz werden öffentliche Modellgewichte heruntergeladen und im lokalen Cache gespeichert. Die Erkennung verarbeitet Bilder auf dem eigenen Rechner.

## Modellwahl und Rahmenfinder

Unter **Audit & Selection** stellt die Modell-Voreinstellung alle Rollen gemeinsam um: **Aktuelles ChatGPT** verwendet `chat-latest` für Routineaufgaben, **Sparsam** `gpt-6-luna`, **Qualität** `gpt-6-astra`. Profil- und Reparaturaufgaben verwenden je nach Voreinstellung `gpt-6.1-sol` oder `gpt-6-astra`. Bestehende gespeicherte Modellnamen werden nicht automatisch überschrieben. Einzelne Modelle und weitere Einstellungen bleiben in den erweiterten Bereichen verfügbar.

Der zusätzliche Rahmenfinder kombiniert Bildverständnis über die Responses API mit einer lokalen Verfeinerung an Bildkanten. Er kann beispielsweise das Foto innerhalb eines verschachtelten Screenshots oder vor einem unscharfen Hintergrund finden. Diese Funktion wird ausdrücklich gestartet, sendet ausgewählte Bilder an OpenAI und verursacht API-Kosten. Identische Anfragen werden aus dem Cache wiederverwendet. KI-Vorschläge müssen bestätigt werden; sie werden nicht automatisch zugeschnitten. Die Standard-Voreinstellung für neue Projekte zeigt auch lokale Vorschläge zur Prüfung an.

Siehe [Projektbewertung und nächste Schritte](PROJECT_REVIEW.md) für die Grenzen der bisherigen Validierung.

### Referenzbilder für den Rahmen-Benchmark

`frame_benchmark.py` erzeugt eine vollständig lokale Prüfseite mit Vorschau, zwei Klicks für Fotogrenzen und JSON-Export bestätigter Referenzen. Der Quellordner bleibt unverändert; Ergebnisse müssen in einem getrennten Ordner liegen. Optionale HEIC/HEIF-Referenzen benötigen `pip install -r requirements-benchmark.txt`. Dieser zusätzliche Decoder gilt für das Benchmark-Werkzeug; die normale Anwendung unterstützt bisher JPG, PNG, WebP und BMP.

```bash
python frame_benchmark.py Datasets/frames Datasets/frame_finder_review/benchmark
```

Die erzeugte `review.html` im Browser öffnen, Grenzen bestätigen oder korrigieren und anschließend die Referenzen als JSON speichern. Lokale Vorschläge und automatisch gebildete Layout-Gruppen zählen nicht als bestätigte Sollwerte. Beim erneuten Erzeugen bleibt die getrennte Referenzdatei erhalten; sie kann auf der Prüfseite wieder geladen werden. Es werden keine API-Aufrufe ausgeführt.

Mit `frame_benchmark_score.py` lassen sich die exportierten Referenzen gegen die lokalen Vorschläge vergleichen:

```bash
python frame_benchmark_score.py Datasets/frame_finder_review/benchmark/references.json Datasets/frame_finder_review/benchmark/frame_references_confirmed.json Datasets/frame_finder_review/benchmark
```

Der Vergleich trennt fehlende Vorschläge, korrigierte Fotogrenzen, beibehaltene Originale und Sonderfälle. Die geometrische Vergleichstoleranz beträgt pro Kante höchstens 3 Pixel oder 0,5 % der zugehörigen Bildseite. Diese Kennzahl ersetzt keine Prüfung abgeschnittener Körperteile und gibt keine automatische Zuschnittfreigabe.

Der lokale Rahmenfinder verfeinert bei Screenshot-/Canvas-Layouts nun den inneren Fotobereich anhand zusammenhängender Kanten und ruhiger Außenstreifen. Diese Vorschläge bleiben prüfpflichtig; der bisherige Zuschnitt bleibt als Alternative verfügbar. Runde Bilder ohne bestehenden Vorschlag werden nicht durch diese Verfeinerung zugeschnitten.

Für einen reproduzierbaren Vorher-Nachher-Vergleich den bisherigen Benchmark-Ordner aufbewahren und die neue Analyse in einem zweiten Ordner erzeugen. Beide Versionen gegen dieselbe exportierte Referenzdatei auswerten. Anschließend erzeugt dieses Werkzeug eine Bildvergleichsseite und einen Bericht:

```bash
python frame_benchmark_compare.py Datasets/frame_finder_review/benchmark/scores.json Datasets/frame_finder_review/benchmark_improved/scores.json Datasets/frame_finder_review/benchmark_improved/references.json Datasets/frame_finder_review/benchmark_improved
```

Auf den 54 für die Entwicklung bestätigten Handreferenzen steigen nahe Vorschläge von 16 auf 25, während Fälle mit mehr als 10 % verbliebener Außenfläche von 26 auf 13 sinken. Eine zusätzliche Prüfung erweitert schwach belegte innere Grenzen zu stärkeren äußeren Fotokanten: Fälle mit mehr als 5 % fehlender Fotofläche sinken von fünf auf vier. Unsichere Fotogrenzen werden auch bei aktivierter Automatik zur Prüfung zurückgehalten. Diese Entwicklungsmessung ist keine unabhängige Abnahme; Details stehen in [PROJECT_REVIEW.md](PROJECT_REVIEW.md).

### Ähnliche Rahmen gemeinsam prüfen

In **Rahmen** zuerst einen Vorschlag oder manuellen Zuschnitt bestätigen. Unter **Ähnliche Bilder gemeinsam prüfen** lässt sich dieser Zuschnitt als Vorlage anzeigen. Nach dem automatischen Weitergehen verwendet die Funktion den zuletzt bestätigten Zuschnitt; bei einem ausgewählten bestätigten Bild dessen Zuschnitt. Der Name der Vorlage wird angezeigt.

Die Funktion bietet bis zu zwölf offene Bilder mit gleichem Layout, gleichen Prüfflags, ähnlichen erkannten Grenzen (höchstens 2 % pro Kante) und passendem Seitenverhältnis (höchstens 1 % Abweichung) an. Unterschiedliche Auflösungen werden proportional umgerechnet. Die Ähnlichkeit ist keine Qualitätsfreigabe: alle Original-/Crop-Vorschauen prüfen und unpassende Bilder abwählen. **Ausgewählte Zuschnitte speichern** übernimmt nur die verbleibende Auswahl als manuelle Entscheidungen. Bereits bestätigte Bilder bleiben ausgeschlossen. Veränderte Dateien, Entscheidungen oder veraltete Vorschauen verhindern die Übernahme; betroffene vorhandene Audit-/Profil-Ergebnisse werden zur erneuten Prüfung markiert. Quellbilder werden nicht verändert, es werden keine KI-Anfragen gesendet.

Die Gruppenfunktion steht nun geöffnet am Anfang des Frames-Moduls. Nach dem Neustart wird der zuletzt bestätigte Zuschnitt aus den gespeicherten Entscheidungen wiederhergestellt. Ohne passende Vorlage oder ähnliche Bilder erscheint eine Erklärung; eine leere große Bildansicht wird ausgeblendet.

### Ausschluss und Automatik direkt in Frames

**Bild aus Datensatz ausschließen / Exclude image from dataset** speichert einen reversiblen Ausschluss für das ausgewählte Bild. Die Quelldatei bleibt erhalten. Ausgeschlossene Bilder verlassen die offene Warteschlange und werden vor weiteren Bildaudits sowie Trainingsauswahlen herausgefiltert, auch wenn Rahmenzuschnitte deaktiviert sind. Unter **Anzeige → Ausgeschlossen / View → Excluded** können sie wieder aufgerufen werden; **Original behalten** oder **Automatische Entscheidung wiederherstellen** nimmt sie wieder auf. Bereits vorhandene Exporte werden durch den Button nicht nachträglich gelöscht; die nächste vollständige Auswahl bzw. der nächste Export berücksichtigt den Ausschluss. Bestehende Profile werden als veraltet markiert, damit eine Fortsetzung keine alten Entscheidungen verwendet.

Unter **Automatik für lokale Zuschnitte / Automatic local crop decisions** zeigt Frames jetzt Modus und zugelassene Zuschnittarten. **Hohe Sicherheit übernehmen (empfohlen) / Apply high confidence (recommended)** aktiviert hohe Sicherheit für uniforme Canvas-Ränder und obere/untere Balken. Die Anzeige zählt und markiert die aktiven automatischen Zuschnitte sofort; deren Crops werden beim weiteren Lauf verwendet. Andere Zuschnittarten und der Modus mit mittlerer Sicherheit bleiben ausdrücklich wählbar. Unsichere Fotogrenzen, verschachtelte Layouts, KI-Vorschläge und bestehende manuelle Entscheidungen bleiben geschützt. Eine leere Typenauswahl aktiviert keinen wirkungslosen Automatikmodus. Änderungen werden in Projekt und Einstellungen gespeichert und mit den anderen Modulen synchronisiert.

Ein Ausschluss in Frames hat Vorrang vor früheren Prioritätsmarkierungen oder Trainingsrollen im Profil. Er greift bereits vor der Duplikatauswahl, damit ein ähnliches, weiterhin gewünschtes Bild nachrücken kann. Nach einer geänderten Entscheidung zunächst den normalen Curator-Lauf starten; die direkte Fortsetzung aus einem veralteten Profil wird gesperrt. Ein erfolgreicher neuer Export ersetzt den vorherigen Stand einschließlich alter Bild-/Caption-Dateien. Bei einem gescheiterten Export bleibt der letzte vollständige Stand erhalten.

Für die kostenpflichtige KI-Erkennung ist **GPT-6 Luna** die sparsame Voreinstellung, ohne Reasoning und mit höchstens 1.024 Ausgabetokens. **GPT-6.1 Sol** bleibt für schwierige Fälle wählbar. Die offiziellen Standardpreise pro Million Tokens liegen bei Luna bei $0,10 Eingang / $0,50 Ausgang und bei Sol bei $2 / $10; gleiche Tokenmengen kosten bei Luna ein Zwanzigstel. Bild-Tokenmengen und Qualität müssen separat verglichen werden. Luna wurde noch nicht mit den bestätigten Rahmenreferenzen live validiert. Die kostenfreie lokale Erkennung bleibt der erste Schritt. Quellen: [Luna](https://developers.openai.com/api/docs/models/gpt-6-luna), [Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol).

## Manuelle Installation (Linux, macOS oder CPU-orientiertes Setup)

```bash
python3.13 -m venv curator_env
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
3. Optional in **Frames** einen lokalen Crop-Vorschlag auswählen, das Original beibehalten oder einen manuellen Crop anwenden.
4. In **Audit & Selection** Trainingsziel auswählen und die gewünschten Audit-, Identitäts-, Diversitäts- und Captioning-Einstellungen festlegen.
   - **Single Pass** erstellt das Profil und beendet den Workflow automatisch.
   - **Profile then Caption** pausiert nach der Profilerstellung, damit kanonische Merkmale, Cluster, Priority Images und Caption-Policies geprüft werden können.
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
