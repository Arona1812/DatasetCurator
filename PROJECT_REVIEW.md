# Projektbewertung und Modernisierung

Stand: 11. Oktober 2026. Diese Bewertung beschreibt die lokal implementierten Änderungen und die verbleibenden Arbeiten. Sie bedeutet keine vollständige Abnahme aller Funktionen.

## Befund

Der Rahmenfinder hat ein grundsätzliches Problem: Geometrische Randheuristiken erkennen äußere Balken, unterscheiden aber ein vollständiges Foto nicht zuverlässig von einem Foto innerhalb eines Screenshots. Auch hohe interne Scores sind keine gemessene Erfolgswahrscheinlichkeit. Im lokalen Beispieldatensatz bleiben dadurch Bedienelemente, verschachtelte Rahmen oder unscharfe Kopien des Fotos erhalten.

Die Oberfläche bietet viele sinnvolle Einstellungen, zeigt diese aber zu früh und verlangt wiederholte Navigation. Außerdem vereinen die beiden Hauptmodule Oberfläche, Konfiguration, Netzwerkzugriffe, Zustandsverwaltung und Export in sehr großen Dateien. Das erschwert Änderungen und Fehlersuche.

Die Python-Migration betrifft auch tatsächliche Funktionen: MediaPipe 0.10.33 besitzt den bisherigen `mp.solutions`-Anschluss nicht mehr. Nur Paketinstallation und erfolgreiche Imports hätten diesen Ausfall nicht entdeckt.

## Implementiert

- Python 3.13 im Windows-Launcher, getrennte Umgebung, echte Importprüfung, portable CPU-Option und Fehlerweitergabe bei Installation.
- Aktuelle OpenAI-Modellfamilie und eine Voreinstellung für `chat-latest`; gemeinsame Modellkonfiguration und kompatible Responses-Parameter ohne Veränderung der ursprünglichen Anfrage. Gespeicherte Modellentscheidungen bleiben erhalten.
- Hauptaktionen und Fortschritt vor den erweiterten Einstellungen. Vollständige Optionen in vier thematischen Bereichen. Lokaler Projektstart ohne API-Key.
- Rahmenanalyse direkt nach Preflight, eine Warteschlange für offene Entscheidungen, automatisches Weitergehen nach Bestätigung und manueller Zuschnitt durch zwei Eckpunkte. UI und Pipeline verwenden dieselbe Entscheidungsregel.
- Neuer Fotobereich-Finder: Bildverständnis über OpenAI, begrenzte Vorschaugröße, strukturierte Ausgabe, Koordinatenprüfung, lokale Kantenverfeinerung und Cache. Verschachtelte Screenshots und unscharfe Hintergrundkopien werden als Bildinhalt verstanden. KI-Vorschläge erfordern Bestätigung und verändern niemals das Original.
- Lokale Gesichts- und Posenerkennung über MediaPipe Tasks, mit lokal gecachten Modellgewichten. Ein geänderter Cache-Stand verhindert die Wiederverwendung alter Analysewerte.

Die erste Voreinstellung zeigt Rahmenvorschläge zur Prüfung. Automatische lokale Verarbeitung bleibt wählbar; komplexe Layouts und semantische Vorschläge werden davon ausgeschlossen. Die tatsächlichen Fehler der bisherigen Erkennung rechtfertigen keine generelle automatische Freigabe.

## Validierung

- Vollständige portable Kernabhängigkeiten in einer frischen Python-3.13-Umgebung installiert; Paketprüfung ohne Konflikte.
- 21 automatisierte Regressionen für Anfrageparameter, Rahmenentscheidungen, Cache, unveränderte Originale, Navigation, Koordinaten und Aufbau der Oberfläche. Zusätzlich lokaler Vorlauf und Modell-Voreinstellung im Browser geprüft.
- MediaPipe Tasks mit echten Modellgewichten lokal an einem Beispiel ausgeführt: ein Gesicht und 33 Pose-Landmarks erkannt. Das ist ein Funktionstest, keine Genauigkeitsmessung.
- Sechs ausdrücklich freigegebene reale Beispielbilder mit `gpt-6.1-sol` geprüft, maximal 4.096 Ausgabetokens pro Anfrage. Alle sechs Antworten abgeschlossen, insgesamt 8.829 Tokens. Die Vergleichsansicht zeigt in allen sechs Fällen plausiblere Fotogrenzen als die lokale Ausgangserkennung, einschließlich eines zuvor nicht erkannten unscharfen Rahmens.

Die sechs Beispiele sind gezielt ausgewählte Problemfälle. Ohne manuell festgelegte Sollgrenzen ist daraus keine allgemeine Trefferquote ableitbar. Der vollständige kostenpflichtige Audit-/Captioning-Lauf, CUDA-Installation, InsightFace und Videoablauf sind noch nicht durchgängig geprüft.

## Nächste Entwicklungsstufen

1. **Messbarer Rahmen-Benchmark:** Sollgrenzen für die 28 vorhandenen Beispiele festlegen und um normale Fotos, Collagen, runde Bilder und Text über Personen ergänzen. Fotofläche, abgeschnittene Körperteile und unnötige Prüfentscheidungen getrennt messen. Automatik erst anhand dieser Ergebnisse freigeben.
2. **Mehrere Bilder gemeinsam prüfen:** Wiederkehrende Screenshot-Layouts gruppieren und bestätigte Grenzen als Vorlage anbieten. Andere Bildgrößen, Einblendungen und Abweichungen müssen erneut geprüft werden. Das kann mehr Handarbeit sparen als ein immer größerer Einzeldetektor.
3. **Lokale Alternative bewerten:** Den neuen Fotobereich-Ansatz gegen ein lokal ausgeführtes Vision-Modell auf denselben Sollfällen testen. Entscheidung anhand Qualität, Laufzeit, GPU-Speicher und Kosten treffen; keinen neuen Modellnamen ohne Vergleich als Verbesserung behandeln.
4. **Gezielte Maskenwerkzeuge:** Für störende Overlays und nicht rechteckige Bereiche eignet sich eine interaktive Segmentierung als eigener Arbeitsgang. SAM 2 ist hierfür ein Kandidat, ersetzt aber nicht automatisch das Finden des vollständigen Fotos. Retusche erzeugt neue Bildinhalte und sollte für Trainingsdaten bewusst getrennt bleiben.
5. **Wartbare Architektur:** Projektzustand, Modellzugriff, Auswahl, Captioning und Export in eigenständige Dienste auslagern. Die langen positionsabhängigen UI-Argumentlisten durch benannte Konfigurationsobjekte ersetzen. Importseitige Dateisystemänderungen entfernen und Jobs mit fortsetzbaren Ergebnissen modellieren.
6. **Reproduzierbare Installation:** Getrennte, geprüfte CPU- und CUDA-Abhängigkeitsstände sowie CI für Python 3.13 etablieren. Optionale Funktionen müssen ihren tatsächlichen Betriebszustand anzeigen.

## Quellen

- [OpenAI: aktueller Modellleitfaden](https://developers.openai.com/api/docs/guides/latest-model) und [ChatGPT-Alias](https://developers.openai.com/api/docs/models/chat-latest).
- [OpenAI: Bildverarbeitung und Grenzen räumlicher Genauigkeit](https://developers.openai.com/api/docs/guides/images-vision). Daraus folgt die Kombination mit lokaler Verfeinerung und manueller Bestätigung.
- [MediaPipe: Face Detector in Python](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector/python) und [Pose Landmarker in Python](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python).
- [SAM 2: offizielles Repository](https://github.com/facebookresearch/sam2).
