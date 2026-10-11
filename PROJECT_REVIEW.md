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
- 33 automatisierte Regressionen für Anfrageparameter, Rahmenentscheidungen, Cache, unveränderte Originale, Navigation, Koordinaten, Aufbau der Oberfläche, innere Fotogrenzen, Fotoflächen-Erweiterung und gemeinsame Vorlagenprüfung. Zusätzlich lokaler Vorlauf und Modell-Voreinstellung im Browser geprüft. Die Gruppenbedienung wurde mit vier echten Bildvorschauen in einem getrennten Testordner geprüft: ein Bild abgewählt, drei Zuschnitte gespeichert.
- MediaPipe Tasks mit echten Modellgewichten lokal an einem Beispiel ausgeführt: ein Gesicht und 33 Pose-Landmarks erkannt. Das ist ein Funktionstest, keine Genauigkeitsmessung.
- Sechs ausdrücklich freigegebene reale Beispielbilder mit `gpt-6.1-sol` geprüft, maximal 4.096 Ausgabetokens pro Anfrage. Alle sechs Antworten abgeschlossen, insgesamt 8.829 Tokens. Die Vergleichsansicht zeigt in allen sechs Fällen plausiblere Fotogrenzen als die lokale Ausgangserkennung, einschließlich eines zuvor nicht erkannten unscharfen Rahmens.

Die sechs Beispiele sind gezielt ausgewählte Problemfälle. Ohne manuell festgelegte Sollgrenzen ist daraus keine allgemeine Trefferquote ableitbar. Der vollständige kostenpflichtige Audit-/Captioning-Lauf, CUDA-Installation, InsightFace und Videoablauf sind noch nicht durchgängig geprüft.

## Nächste Entwicklungsstufen

Der Referenzbestand wurde anschließend auf 82 Bilder erweitert, einschließlich zweier HEIC-Dateien. Alle wurden lokal gelesen und analysiert; keine zusätzlichen OpenAI-Aufrufe. Neue Fälle umfassen weiche ovale Vignetten, dekorative Rahmen und weitere Story-/Screenshot-Einbettungen. Das Benchmark-Werkzeug erzeugt eine lokale Prüfseite mit zwei Klicks für Sollgrenzen und JSON-Export. 13 automatisch vorgeschlagene Layout-Gruppen sind zunächst nur Vergleichshinweise. Weder lokale Detektorwerte noch diese Gruppierung ersetzen bestätigte Sollgrenzen. Die Bilder und erzeugten Prüfdateien bleiben in den ignorierten lokalen Datensatzordnern.

Der Nutzer hat die 54 auf dieser Seite als neu geführten Referenzen geprüft und bestätigt. Sein Eindruck: ungefähr die Hälfte der lokalen Empfehlungen war brauchbar, weitere Fälle brauchten Korrekturen. Runde Rahmen wurden ohne Zuschnittsvorschlag gelassen und als nicht zuschneidbar bewertet. Seine Grenzen sind bewusst als ungefähre Handmarkierungen zu behandeln.

Der JSON-Export liegt vor und wurde ausgewertet: fünf beibehaltene Originale ohne Vorschlag, 49 bestätigte Zuschnitte. Für zwei dieser Zuschnitte fehlte ein Vorschlag. Von den 47 vorhandenen Vorschlägen liegen 16 nahe an den Referenzgrenzen; dieses Ergebnis bleibt bei Kantentoleranzen von 0,5 %, 1 %, 2 %, 3 % und 5 % der Bildseite unverändert. Damit erklären wenige Pixel Markierungsungenauigkeit die übrigen Abweichungen nicht. 26 Vorschläge enthalten mehr als 10 % zusätzliche Fläche außerhalb des markierten Fotos; fünf lassen mehr als 5 % der markierten Fotofläche weg. Diese beiden Kategorien können sich überschneiden. Das Hauptproblem sind verbliebene äußere Screenshot-/Canvas-Bereiche. Die Zahlen bewerten ausschließlich lokale Heuristiken, nicht den neuen KI-Fotobereich-Finder. Die restlichen 28 Referenzen sind noch unbestätigt.

1. **Messbarer Rahmen-Benchmark:** Sollgrenzen für die 28 vorhandenen Beispiele festlegen und um normale Fotos, Collagen, runde Bilder und Text über Personen ergänzen. Fotofläche, abgeschnittene Körperteile und unnötige Prüfentscheidungen getrennt messen. Automatik erst anhand dieser Ergebnisse freigeben.
2. **Mehrere Bilder gemeinsam prüfen:** Wiederkehrende Screenshot-Layouts gruppieren und bestätigte Grenzen als Vorlage anbieten. Andere Bildgrößen, Einblendungen und Abweichungen müssen erneut geprüft werden. Das kann mehr Handarbeit sparen als ein immer größerer Einzeldetektor.
3. **Lokale Alternative bewerten:** Den neuen Fotobereich-Ansatz gegen ein lokal ausgeführtes Vision-Modell auf denselben Sollfällen testen. Entscheidung anhand Qualität, Laufzeit, GPU-Speicher und Kosten treffen; keinen neuen Modellnamen ohne Vergleich als Verbesserung behandeln.
4. **Gezielte Maskenwerkzeuge:** Für störende Overlays und nicht rechteckige Bereiche eignet sich eine interaktive Segmentierung als eigener Arbeitsgang. SAM 2 ist hierfür ein Kandidat, ersetzt aber nicht automatisch das Finden des vollständigen Fotos. Retusche erzeugt neue Bildinhalte und sollte für Trainingsdaten bewusst getrennt bleiben.
5. **Wartbare Architektur:** Projektzustand, Modellzugriff, Auswahl, Captioning und Export in eigenständige Dienste auslagern. Die langen positionsabhängigen UI-Argumentlisten durch benannte Konfigurationsobjekte ersetzen. Importseitige Dateisystemänderungen entfernen und Jobs mit fortsetzbaren Ergebnissen modellieren.
6. **Reproduzierbare Installation:** Getrennte, geprüfte CPU- und CUDA-Abhängigkeitsstände sowie CI für Python 3.13 etablieren. Optionale Funktionen müssen ihren tatsächlichen Betriebszustand anzeigen.

## Lokale Verbesserung anhand der bestätigten Referenzen

Der lokale Finder sucht jetzt innerhalb eines bereits erkannten Screenshot-/Canvas-Bereichs nach langen, zusammenhängenden Fotokanten und ruhigen Außenstreifen. Er verfeinert nur durch mehrere passende Grenzen gestützte Vorschläge. Bereits vollständig begrenzte Fotos werden geschützt; ohne bestehenden Rahmenvorschlag entsteht kein neuer Zuschnitt. Bekannte Gesichtsbereiche werden zusätzlich geprüft. Die neue Variante ist in die normale Rahmenanalyse integriert und invalidiert alte Cache-Ergebnisse. Diese verschachtelten Vorschläge bleiben prüfpflichtig, der bisherige Vorschlag bleibt als Alternative verfügbar.

Vergleich auf denselben 54 bestätigten Handreferenzen:

| Messung | Bisher | Verbessert |
| --- | ---: | ---: |
| Alle Fotogrenzen innerhalb 0,5 % / mindestens 3 Pixel | 16 | 24 |
| Alle Fotogrenzen innerhalb 1 % / mindestens 3 Pixel | 16 | 29 |
| Mehr als 10 % zusätzliche Außenfläche | 26 | 13 |
| Mehr als 5 % fehlende Fotofläche | 5 | 5 |

15 Vorschläge wurden verändert, keiner verliert mehr als einen Prozentpunkt Überlappung mit der Handreferenz. Alle 16 zuvor nahe liegenden Vorschläge und alle fünf beibehaltenen Originale bleiben erhalten. 23 Vorschläge brauchen weiterhin Korrekturen; zwei zuschneidbare Fälle bleiben ohne Vorschlag. Die fünf Fälle mit fehlender Fotofläche sind noch nicht verbessert. Runde Rahmen erhalten weiterhin keinen Vorschlag.

Diese Referenzen wurden für die Entwicklung verwendet. Das Ergebnis ist ein Entwicklungsvergleich, keine unabhängige Abnahme. Die weiteren 28 Bilder haben noch keine bestätigten Grenzen. Es wurden keine weiteren API-Anfragen ausgeführt und keine Originalbilder geändert. Die lokale Vergleichsansicht unter `Datasets/frame_finder_review/benchmark_improved/comparison.html` zeigt alle 15 Änderungen nebeneinander mit den Handmarkierungen.

## Fotoinhalt schützen und Handarbeit reduzieren

Die nächste lokale Prüfung erweitert bestehende Vorschläge nach außen, wenn die bisherige Kante schwach belegt ist und eine stärkere Fotokante mit ruhigem Außenstreifen vorhanden ist. Sie verkleinert den Ausgangsvorschlag nicht; die anschließende innere Kantenverfeinerung bleibt ein eigener Schritt. Schwach belegte Fotogrenzen erhalten `photo_boundary_uncertain` und werden von der automatischen Zuschnittfreigabe ausgeschlossen. Diese Prüfung schützt auch die Automatik-Modi; manuelle Entscheidungen bleiben möglich.

Gegenüber der vorigen Verbesserung ändern sich auf den 54 Entwicklungsreferenzen zwei Vorschläge: bei `Duygu_00000011.jpg` wird links Fotoinhalt zurückgewonnen, bei `Kübra_00000018.jpg` unten. Nahe Vorschläge steigen von 24 auf 25, Fälle mit mehr als 5 % fehlender Fotofläche sinken von fünf auf vier. Keine zuvor nahe liegende Empfehlung und kein beibehaltenes Original gehen verloren; Fälle mit mehr als 10 % Außenfläche bleiben bei 13. `Duygu_00000010.jpg`, `Duygu_00000011.jpg` (oben), `Kübra_00000005.jpg` und die HEIC-Story vom 6. August 2024 bleiben offene Fälle. Die Bildgrenzen sind Handreferenzen, keine exakten Segmentierungen. Der spezielle Licht-/Herzrahmen in `Screenshot_20201011-081831_Instagram.jpg` wurde vom Nutzer als brauchbarer Sonderfall akzeptiert; seine geometrische Abweichung bleibt in den Zahlen sichtbar und wird nicht als zwingendes Optimierungsziel behandelt.

Die normale Oberfläche enthält außerdem **Ähnliche Bilder gemeinsam prüfen**. Ein ausgewählter oder zuletzt bestätigter Zuschnitt wird proportional auf höchstens zwölf ähnliche offene Layouts übertragen und zunächst als Original-/Crop-Vorschau gezeigt. Der Nutzer kann einzelne Bilder abwählen und speichert die verbleibende Auswahl ausdrücklich. Bestehende Entscheidungen, geänderte Quelldateien und veraltete Vorschauen werden geschützt. Betroffene bereits vorhandene Audit-/Profil-Ergebnisse werden gezielt als veraltet markiert. Die Auswahlfunktion wurde automatisiert und über die Browserbedienung geprüft; eine gemessene Zeitersparnis im normalen Nutzerablauf steht noch aus.

Die Referenzen bleiben als Entwicklungsvergleich ausgewiesen. Es wurden keine weiteren API-Aufrufe ausgeführt und keine Originalbilder verändert. Die letzten zwei Änderungen stehen unter `Datasets/frame_finder_review/benchmark_protected/comparison.html`.

## Frames: Ausschluss, sichtbare Gruppenfunktion und explizite Automatik

Ein neuer Ausschluss-Button speichert `exclude` als reversible Entscheidung. Die Quelle bleibt erhalten; der nächste vollständige Pipeline-Lauf filtert ausgeschlossene Dateien vor der Duplikatauswahl und vor dem Bildaudit heraus, unabhängig davon, ob Rahmenzuschnitte aktiviert sind. Der Fingerprint der verbleibenden Quellen verhindert, dass ein alter Preflight-Cache einen ausgeschlossenen Duplikatvertreter wieder einführt oder dessen Alternative weiterhin unterdrückt. Sie werden als manuelle Reject-Zeilen im Report geführt. Die UI entfernt sie aus offenen Entscheidungen und zeigt sie in einem eigenen Filter. Vorhandene Exporte bleiben bis zum erneuten Export bestehen. Vorhandene Profile/Caption-Stände werden zur erneuten Prüfung markiert.

Die Gruppenfunktion steht geöffnet direkt unter dem lokalen Status; der zuletzt bestätigte Crop wird auch nach einem Neustart wiedergefunden. Leere Vorschauflächen bleiben ausgeblendet. Frames enthält jetzt die Automatik-Steuerung samt erlaubten Typen und einen empfohlenen Knopf für sichere uniforme Canvas-Ränder und Story-Balken. Automatisch aktive Vorschläge werden im Viewer vorausgewählt. Modus und Typen werden im Workspace und in den Einstellungen gespeichert und mit Start/Audit synchronisiert. Unsichere Grenzen und komplexe Layouts bleiben ausgeschlossen; eine leere Typenliste wird beim Aktivieren erklärt.

Die beim Nutzer gelesene Workspace-Konfiguration enthielt `suggest_only` und keine freigegebenen Typen. 20 noch unentschiedene Analysen aus dem vorhandenen aktuellen Cache sind für die empfohlene Automatik geeignet; dies ist eine lokale Momentaufnahme, keine pauschale Freigabe aller hoch bewerteten Bilder. Die Nutzerkonfiguration wurde bei der Untersuchung nicht geändert.

39 lokale Regressionen bestehen. Die vollständige Oberfläche wurde zusätzlich mit vier getrennten Kopien echter Bilder geprüft: vier automatisch aktive Zuschnitte, danach ein Ausschluss mit automatisch aktualisiertem Status (drei automatische Zuschnitte, ein Ausschluss) und Navigation zum nächsten Bild. Die automatisierten Prüfungen decken Wiederaufnahme, unveränderte Quellen, Schutz manueller Entscheidungen, Profil-Veraltung und Ausschluss vor weiteren Bildaudits ab. Ein kostenpflichtiger Gesamt-Audit-/Exportlauf wurde dafür nicht gestartet. Automatisch vorausgewählte Zuschnitte lassen sich ausdrücklich als Vorlage bestätigen, ohne erst eine andere Radio-Option auswählen zu müssen.

Der KI-Finder verwendet nun GPT-6 Luna als sparsame Voreinstellung mit Reasoning `none` und maximal 1.024 Ausgabetokens; Sol bleibt auswählbar. Die offiziellen Modellseiten bestätigen Bildeingabe, strukturierte Ausgabe und Standardpreise von $0,10/$0,50 pro Million Eingabe-/Ausgabetokens für Luna gegenüber $2/$10 für Sol. Die Zwanzigstel-Relation gilt für gleiche Tokenmengen, nicht als garantierter Preis pro Bild. Luna-Qualität wurde bisher nur über simulierte Antworten/Parameter geprüft, nicht über neue kostenpflichtige Anfragen.

## Frames: Entscheidungen bis zum Export geprüft

44 lokale Regressionen bestehen unter Python 3.13. Fünf zusätzliche Integrationstests verwenden isolierte Bilddateien sowie kontrollierte Detektorvorschläge und Audit-Zeilen. Die tatsächlichen Crop-Dateien, lokalen Captions, Reports, gespeicherten Profile und transaktionalen Exporte werden durch den produktiven Backend-Code erzeugt. Netzwerkzugriffe sind für diese Exporttests ausdrücklich gesperrt; dies ist keine zusätzliche Validierung der KI- oder Detektorqualität.

Geprüft wurden automatische und manuelle Zuschnitte, Original behalten, Ausschluss, Wiederaufnahme und spätere Änderungen am Zuschnitt. Ein erneuter Export ersetzt alte Bilder und Caption-Dateien; ein gescheiterter Export erhält den letzten vollständigen Stand. Eine nach dem Audit in der UI geänderte Entscheidung sperrt die Fortsetzung mit veralteten Profildaten, bis ein aktualisierter Audit-Stand vorliegt. Alle Quellbytes bleiben unverändert.

Dabei wurden zwei Ablaufprobleme korrigiert: Ausschlüsse greifen jetzt vor der Duplikatauswahl und verändern deren Cache-Fingerprint. Frühere Prioritätsmarkierungen oder Trainingsrollen aus dem Profil können explizit in Frames ausgeschlossene Bilder nicht mehr in die Trainingsauswahl zurückholen. Zur Wiederaufnahme muss zuerst der Ausschluss in Frames aufgehoben werden.

Prüfaufruf: `python -m unittest test_curator_workflows test_photo_regions test_frame_group_review test_frames_controls test_frames_export`.

## Separate Referenzprüfung vorbereitet

Die separate Referenzprüfung liegt unter `Datasets/frame_finder_review/validation/review.html`: 28 bisher unbestätigte Rahmenbilder und 20 zusätzliche Kontrollkandidaten aus den Originalen des aktuellen Datensatzes. Alle 48 Bilder wurden lokal gelesen; keine Lesefehler und keine API-Aufrufe. Die 54 bestätigten Entwicklungsreferenzen sind nach Inhalts-Hash ausgeschlossen. Die Auswahl wird vor der Erkennung nach einer festen Dateinamen-Sortierung eingefroren; Kontrollkandidaten werden nicht anhand günstiger Detektorwerte gewählt und sind noch nicht als rahmenfrei bestätigt. Einige der 28 Rahmenbilder waren in früheren Versuchen bereits sichtbar: dies ist eine separate Sollreferenzprüfung, kein vollständig ungesehener Testkorpus.

Die Prüfseite verbirgt Vorschläge bis zur Bestätigung, verwendet einen eigenen Browser-Speicherschlüssel und verändert keine Quellen. Die bestätigten Referenzen müssen als JSON exportiert werden. Die Auswertung unterscheidet Vorschlagsfehler und Fehler der empfohlenen Automatik und trennt die beiden Bildgruppen. Drei zusätzliche Offline-Regressionen prüfen Ausschluss von Entwicklungsbildern/Exportkopien, veränderte Quellen gegenüber der eingefrorenen Auswahl und die Trennung unnötiger Vorschläge von unnötigen automatischen Zuschnitten. Die tatsächliche Qualität bleibt bis zur Referenzbestätigung offen.

## Bestätigte separate Referenzen ausgewertet

Der zweite Benutzerexport enthält alle 48 gültigen Referenzen; Inhalts-Hashes der Quellen stimmen weiterhin mit der eingefrorenen Auswahl überein. Die bestätigte Datei ist separat unter `Datasets/frame_finder_review/validation/frame_references_confirmed.json` gesichert, die Auswertung unter `validation/scored`. Der eingefrorene Vorschlagsstand bleibt unverändert. Die neue Bildvergleichsseite `validation/results.html` zeigt Original, lokalen Vorschlag und Handmarkierung und filtert standardmäßig die vier Automatikfälle mit Nacharbeit.

39 Referenzen verlangen einen Zuschnitt, neun behalten das Original. Für 38 der 39 Zuschnitte existiert ein Vorschlag: 13 liegen innerhalb der strengen Kantentoleranz von 0,5 %/mindestens 3 Pixeln, 18 innerhalb 1 % und 20 innerhalb 2 %. Zwölf Vorschläge enthalten mehr als 10 % zusätzliche Außenfläche; fünf lassen mehr als 5 % der markierten Fotofläche weg. Acht der neun Originale haben keinen Vorschlag, eines einen unnötigen Vorschlag, der durch die empfohlene Automatik nicht angewendet würde.

Die empfohlene Automatik (hohe Sicherheit, `uniform_canvas`/`story_bars`) würde elf Bilder zuschneiden: sieben liegen nahe den Referenzen, vier benötigen substanzielle Korrektur. Keines der neun bestätigten Originale würde automatisch zugeschnitten. Die vier Automatikfehler wurden visuell kontrolliert: zwei Instagram-Screenshots behalten App-Chrome beziehungsweise ein äußeres Bild mit eingebettetem inneren Foto, zwei Balkenbilder behalten den unteren schwarzen Rand. Diese vier Fälle sind keine bloße Folge ungenauer Handmarkierungen. Der nächste Verbesserungsbedarf liegt bei vollständigen unteren Balken und der Unterscheidung äußerer Screenshot-Bereiche vom eigentlichen Foto. Bis dahin ist diese Referenzprüfung keine Freigabe einer breiteren Automatik. Sobald die neuen Referenzen zur Anpassung verwendet werden, zählen sie als Entwicklungsdaten.

Der Nutzer hat die Referenzen mit möglichst vollständigem Bilderhalt gesetzt. Für weitere Verbesserungen hat vollständiger Erhalt der abgebildeten Person Vorrang: Haare, Körper, Kleidung und Accessoires dürfen durch das Entfernen von Instagram-Bedienelementen oder Schriftzügen nicht angeschnitten werden. Text darf durch einen Zuschnitt entfernt werden, wenn er vollständig außerhalb des zu erhaltenden Motivs liegt. Bei Überschneidung oder unsicherer Motivgrenze bleibt der Bereich erhalten beziehungsweise zur manuellen Prüfung offen. Dieses Kriterium ist eine Anforderung für die nächste Änderung; eine pixelgenaue Motivschutzgarantie ist durch die bisherige Referenzprüfung nicht nachgewiesen.

## Vollständige Balken und Instagram-Fotobereiche korrigiert

Detektorstand `smart-frame-v6.2-complete-frame` ergänzt bisher übersehene schwarze Außenbalken, wenn gegenüberliegende schwarze Ränder den vollständigen Fotobereich belegen. Bereits beschnittene Kanten werden dabei nicht enger gesetzt. Bei App-Screenshots werden zusätzlich die Fotogrenzen gegenüber Kopf-/Nachrichtenleisten und geschlossene farbige Innenbilder geprüft. Innenbilder erhalten einen kleinen äußeren Sicherheitssaum und bleiben manuell zu bestätigen. Fotointerne Schriftzüge werden nicht durch einen engeren Motivzuschnitt entfernt.

Die drei beanstandeten Beispiele sind im neuen Vergleich sichtbar: `wiki_to_20201013_211038_0.jpg` erhält den fehlenden unteren Abschluss und ist für die empfohlene Automatik geeignet. Der Instagram-Screenshot vom 23. Oktober 2024 erhält den vollständigen Fotobereich ohne App-Leisten; der vom 18. Januar 2025 das farbige Innenbild. Beide bleiben komplexe Prüffälle. Auch `pp (84).jpg` erhält den fehlenden unteren Abschluss; wegen der vollständigen Fotohöhe unter der konfigurierten Mindestgröße von 400 Pixeln bleibt er zur manuellen Entscheidung offen.

Auf den 48 jetzt zur Entwicklung verwendeten Referenzen ändern sich sechs Vorschläge: strenge Kantentreffer steigen von 13 auf 17, Fälle mit mehr als 10 % zusätzlicher Außenfläche sinken von zwölf auf sechs. Fälle mit mehr als 5 % fehlender Fotofläche bleiben bei fünf. Die empfohlene Automatik enthält acht nahe Zuschnitte und keinen Korrekturfall statt zuvor sieben nahen und vier fehlerhaften Zuschnitten; 40 Bilder bleiben ohne automatische Entscheidung. Auf den 54 bisherigen Entwicklungsreferenzen ändern sich fünf Vorschläge; Treffer-/Fehlerzahlen bleiben gleich. In beiden Gruppen fällt kein zuvor naher Vorschlag aus der Toleranz; keine Änderung verschlechtert die IoU oder den Anteil fehlender Fotofläche um mehr als einen Prozentpunkt. Dies ist ein Entwicklungsvergleich, keine unabhängige Qualitätsfreigabe und keine pixelgenaue Motivschutzgarantie.

Alle 102 Quell-Hashes sind unverändert. 52 lokale Tests unter Python 3.13 bestehen, darunter fünf neue Regressionen für vollständige schwarze Rahmen, dunklen Fotoinhalt, kleine Fotobereiche, App-Fotogrenzen und farbige Innenbilder. Keine zusätzlichen API-Aufrufe. Eingefrorene alte Vorschläge und Benutzerreferenzen bleiben erhalten; der neue Vergleich liegt unter `Datasets/frame_finder_review/validation_v62`, sichtbar auch über `validation/improved_comparison.html`. Die neue Detektorversion invalidiert alte Analyse-Caches beim erneuten lokalen Prüfen; gespeicherte manuelle Entscheidungen bleiben separat erhalten.

## Lokaler Ablauf mit 48 echten Bildkopien geprüft

Unter `Datasets/frame_finder_review/workflow_v62` wurde die Vorprüfung mit pHash, die tatsächliche lokale Frames-Analyse und die Wiederaufnahme über die produktiven UI-Funktionen ausgeführt. Acht Zuschnitte waren automatisch freigegeben. Ein bestätigter Vorlagenzuschnitt wurde nach Vorschau auf ein tatsächlich passendes weiteres Layout übertragen und gespeichert. Ein weiteres Bild wurde ausgeschlossen, ein Original ausdrücklich beibehalten. Beim erneuten lokalen Scan waren alle manuellen Entscheidungen unverändert wieder vorhanden; die 48 Eingaben blieben getrennt von den erzeugten Dateien.

Der produktive Caption-/Exportpfad erzeugte anschließend 47 Bilder und 47 nicht leere Caption-Dateien. Jede exportierte Bildgröße wurde gegen die tatsächlich gewählte Original-/Crop-Datei geprüft; der ausgeschlossene Eintrag fehlt. Alle 48 Original-Hashes und alle 48 Eingabekopien sind unverändert. Netzwerkaufrufe waren gesperrt; Audit-/Profildaten wurden kontrolliert vorgegeben. Damit ist der lokale Entscheidungs-/Exportablauf geprüft, nicht ein neuer KI-Audit oder die Caption-Qualität. Unsichere Fälle bleiben im normalen Ablauf prüfpflichtig. Einstellungen und Entscheidungen des echten Nutzerprojekts wurden nicht verändert.

Die Einzelbelege stehen in `workflow_v62/workflow_results.json`, der lesbare Bericht in `workflow_v62/WORKFLOW_REPORT.md`. Diese Prüfung hat keinen weiteren Produktcode geändert.

## Audit & Auswahl: sichtbarer Ablauf und lokale Laufübersicht

Die Entscheidung zwischen automatischer Profilnutzung und eigener Profilprüfung steht jetzt als direkte Auswahl im Hauptbereich statt in den erweiterten Caption-Einstellungen. Die gespeicherten Werte und die Position in der Backend-Konfiguration bleiben erhalten. Die Startaktion heißt „Audit starten“.

Eine lokale, nur lesende Laufübersicht aktualisiert sich beim Öffnen des Audit-Tabs und beim Ändern von Zielgröße, Audit-Modell oder Ablauf. Sie zeigt den aktuellen Vorprüfungsstand, automatische/offene/ausgeschlossene Frames des geladenen passenden Projekts, die gewählte Verarbeitungskette und vorhandene Audit-Stände. Veraltete Vorprüfung und fremde Frames-Zustände werden nicht als aktuelle Freigabe ausgegeben. In Stage oder Profil markierte nachträgliche Rahmenänderungen werden als aktualisierungsbedürftiger Audit angezeigt. Die Übersicht ersetzt keine erneute Prüfung beim Start und macht keine Aussage zur Wiederverwendbarkeit einzelner API-Caches.

55 lokale Regressionen bestehen. Im Browser wurde die isolierte 48-Bild-Testkopie geladen: acht automatische Zuschnitte, 27 offene Entscheidungen, ein Ausschluss; Wechsel zur eigenen Profilprüfung aktualisiert den angezeigten Ablauf. Die Vorschau auf Port 7876 enthält keinen API-Key und verändert keine Nutzerkonfiguration. Kein neuer KI-Audit und keine neue Qualitätsmessung. Screenshot: `Datasets/audit_review/audit_overview_verified.png`.

## Quellen

- [OpenAI: aktueller Modellleitfaden](https://developers.openai.com/api/docs/guides/latest-model) und [ChatGPT-Alias](https://developers.openai.com/api/docs/models/chat-latest).
- [OpenAI: Bildverarbeitung und Grenzen räumlicher Genauigkeit](https://developers.openai.com/api/docs/guides/images-vision). Daraus folgt die Kombination mit lokaler Verfeinerung und manueller Bestätigung.
- [MediaPipe: Face Detector in Python](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector/python) und [Pose Landmarker in Python](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python).
- [SAM 2: offizielles Repository](https://github.com/facebookresearch/sam2).
