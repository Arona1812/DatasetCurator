# Fünf Caption-Kandidaten für Personen-LoRAs

> **Historisch:** Die lokalen Provider Florence 2 und JoyCaption wurden aus der App entfernt; die Erstbeschreibung schreibt GPT-6 Luna. Dieser Vergleich bleibt als Begründung erhalten.

Stand: 12. Oktober 2026. Vergleich auf denselben zwölf eingefrorenen Bildkopien. Auf Wunsch wurden GPT-6 Luna und GPT-6.1 Sol als zwei zusätzliche Bildbeschreiber ausgeführt. Beide bekamen keine Ergebnisse der anderen Modelle. Es gab genau 24 OpenAI-Anfragen, keine Wiederholungen und keine zusätzlichen Bewertungsanfragen.

## Ergebnis und Empfehlung

**GPT-6 Luna ist in dieser kleinen Auswahl der sinnvollste Kandidat für einen sparsamen, allgemeinen Caption-Pfad.** Kleidung, Pose, Bildkontext und sichtbarer Schmuck sind deutlich brauchbarer als bei der schnellen Florence-Basis. Sol liefert die sorgfältigere und reichhaltigere Beschreibung, besonders bei Bildausschnitt, Hintergrund und unsicheren Details. Die Empfehlung ist eine qualitative Sichtprüfung dieser Auswahl, kein gemessener allgemeiner Genauigkeitswert und kein Trainingsnachweis.

| Kandidat | Median pro Bild | Konkrete Beobachtung |
| --- | ---: | --- |
| Florence More Detailed | 1,40 s | Sehr schnell; grundlegende Kleidung/Szene. Im lila Top-Bild werden beide Arme als angehoben beschrieben; im Graffiti-Bild wird das Bandana-Oberteil zur Halsbekleidung. |
| JoyCaption | 8,80 s | Gute längere Personenbeschreibung. Das App-Musik-Symbol wird zum Ringlicht; im Bauchnabel-Bild wird eine zusätzliche Schmuckkette behauptet. |
| Qwen3-VL 4B, freier Caption-Lauf | 38,89 s | Viele Details, aber vier Antworten am 512-Tokenlimit abgeschnitten. Im Hut-Bild fälschliche Piercing-Abwesenheit; im Graffiti-Bild falscher Blick und falscher Bildausschnitt. |
| GPT-6 Luna | 4,74 s | Erkennt das Septum-Piercing unter dem Hut, das Bandana-Oberteil und die Arm-/Handanordnung im lila Top-Bild. Beschreibt den Bildkontext sinnvoll. Augenfarben werden relativ häufig und selbstsicher als Braun angegeben. |
| GPT-6.1 Sol | 10,77 s | Präziser bei Framing, Blick, Kleidung und Hintergrund; trennt im Bauchnabel-Bild das Herz-Tattoo von der Halskette. Augenfarben bleiben bis auf eine Beobachtung offen. Kleine Schmuckmerkmale fehlen im ersten Portrait teilweise. |

Luna und Sol nennen in den vier gezielt angesehenen Merkmalen (Hut: Nase; schwarzes Crop-Top: Nase und Nabel; Rundspiegel: Nase) jeweils ein entsprechendes sichtbares Merkmal. Diese vier Ereignisse sind keine allgemeine Piercing-Erkennungsquote. Die sichtbaren Positionen und Typen wurden in den hochauflösenden Bildern angesehen; keine ungeprüfte Gleichsetzung von Schmuckmaterial, Körperseite oder feinen Tattoo-Motiven mit gesicherter Wahrheit.

Nach protokollierter Nutzung und den veröffentlichten Standard-Tarifen ergeben sich für die zwölf Bilder ungefähr **0,0048 USD für Luna** und **0,1105 USD für Sol**, zusammen etwa **0,1153 USD**. Dies ist eine Tokenschätzung, keine Rechnung. Quellen: [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna), [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol).

## Was der gekoppelte Profilweg gezeigt hat

Die Rohbeschreibungen sind nur die erste Stufe. Die Kandidaten wurden zusätzlich in den gemeinsamen vorhandenen Profil-/Caption-Aufbau geführt, mit den gespeicherten Krea-Caption-Schaltern und dem gewählten Modus für variable Merkmale. Es wurden keine Nutzer-Einstellungen verändert.

- Florence und JoyCaption benötigen in diesem Versuchsaufbau einen zusätzlichen lokalen Qwen-Textschritt, um strukturierte Merkmale für das bestehende Profil zu erzeugen. Das ist **keine reine Florence- bzw. JoyCaption-Lösung**. Die zusätzliche Strukturierung dauert median ungefähr 18,48 bzw. 20,13 Sekunden und kann Angaben verlieren. Eine kostenlose Roh-Caption allein belegt damit noch keinen kostenlosen oder gelungenen gemeinsamen Profilweg.
- JoyCaption konnte das angeforderte strukturierte Format in drei direkten Proben nicht durchgehend liefern. Diese Rohversuche bleiben separat gespeichert. Das ist ein Befund dieses Prompts, keine Behauptung, dass JoyCaption grundsätzlich kein JSON erzeugen könne.
- Der neuere direkte Qwen-Merkmalslauf lieferte auf allen zwölf Bildern begrenzte Ausgaben ohne Tokenlimit; JSON-Literale mussten teilweise verlustfrei normalisiert werden. Das verbessert die Verarbeitbarkeit, aber nicht automatisch die erkannte Bildwahrheit. Unter anderem gehen kleine Piercings verloren.
- Luna und Sol lieferten pro Bild eine freie Beschreibung und native strukturierte Bildmerkmale in derselben Anfrage. Alle 24 Antworten waren vollständig und formal gültig. Diese Mehrfachaufgabe unterscheidet sich vom früheren freien lokalen Caption-Lauf; die Zeiten vergleichen praktische Arbeitsweisen, nicht isolierte Modellarchitekturen.
- Modelltexte können feste Identitätsmerkmale in falsche Felder einschleusen, etwa Augenfarbe in Make-up. Solche Angaben werden für den experimentellen Profilweg nicht übernommen, mit einer Begründung protokolliert und in der Rohantwort erhalten. Textstrukturierung darf nur quellengestützte Passagen übernehmen; nicht gestützte Felder werden leer gelassen statt durch Vermutungen ersetzt.
- Unklare Sichtbarkeit bleibt unbekannt. Aus fehlendem Brillenwort in einer freien Caption wird keine Brillen-Abwesenheit. Unterschiedliche sichtbare Accessoires mit der unspezifischen Position `other` werden anhand ausdrücklich genannter Regionen wie Nostril/Ear getrennt; eine Körperseite wird dabei nicht erfunden.
- Auch eine gute Rohbeschreibung kann im gemeinsamen Satzaufbau verlieren: falsche Feldzuordnung, doppelte Angaben oder unpassende Grammatik sind zusätzliche Fehlerquellen. Der Adapter korrigiert einige reine Formulierungsfälle; die Caption-Syntax und die vollständige Profilabdeckung sind noch keine abgenommenen Produktfunktionen.

Für den festen Vergleich wird die einzige beschriebene Person bei höchstens einem lokal erkannten Gesicht als Vergleichssubjekt verwendet. Die Modell-Aussage zur Identitätssicherheit bleibt als Beobachtung erhalten. Das Bild mit zwei Personen geht bei allen fünf Varianten in die Prüfung und beeinflusst das Profil nicht. **Diese Testregel ist keine Identitätsprüfung und keine automatische Freigabe für den Nutzerdatensatz.** Die erzeugten Profile sind automatische, unbestätigte Vorschläge aus einer kleinen Bildauswahl. Das reduzierte Merkmalsformat deckt nicht das vollständige produktive Audit und alle Profilmerkmale wie Körperbau/Hauttyp ab.

Die Hauptanwendung wurde noch nicht auf einen neuen lokalen Caption-Provider umgestellt. Automatischer und manueller Profilmodus sowie die bestehenden Caption-Auswahlen bleiben erhalten. Die geprüften Regelkorrekturen aus `PROFILE_CAPTION_FLOW_REVIEW.md` sind im Backend umgesetzt; der neue lokale Merkmalsadapter ist bislang eine getestete Grundlage für die Vergleichsanbindung.

## Aktuelle Entscheidung des Nutzers

Florence, JoyCaption und GPT-6 Luna bleiben die Caption-Kandidaten. GPT-6.1 Sol und Qwen werden für die weitere Caption-Entwicklung vorerst verworfen. Damit entfällt auch Qwen als geplanter zusätzlicher Strukturierer für Florence/JoyCaption. Die gespeicherten Vergleichsergebnisse bleiben als Belege erhalten; diese Entscheidung betrifft die Caption-Kandidaten und ändert keine unabhängigen Audit-/Profileinstellungen.

Der Nutzer trennt ausdrücklich zwei Arbeitsschritte:

1. **Erstcaptioning:** Das Bild beschreiben und die Rohbeschreibung erhalten. Der gewählte Caption-Provider ist vom späteren Trainingsziel und Personenprofil getrennt. Ein geändertes Profil oder andere Caption-Auswahlen dürfen diesen Schritt nicht unnötig neu auslösen.
2. **Zusammensetzung in der Profilansicht:** Die vorhandenen Bildinformationen mit dem Personenprofil, Triggerwort und den gewählten Zielmodell-/Caption-Regeln zu einer Trainingscaption zusammenstellen. Im manuellen Ablauf gehören Prüfung, Änderungen und sichtbare Ergebnisse zusammen; im automatischen Ablauf wird die bestmögliche Profilschätzung genutzt.

Aktueller Produktstand nach dem Einbau am 12.10.2026: Florence/JoyCaption/Luna sind in der Hauptoberfläche auswählbar. Die Profilansicht bietet Vorschau und Zusammensetzung mit aktuellen Editorwerten. Erstbeschreibung und finale Caption sind getrennt gespeichert. Der frühere Vergleichsadapter mit Qwen ist keine Produktionsabhängigkeit. Details und verbleibende Qualitätsgrenzen stehen in `CAPTION_PIPELINE.md`.

## Sinnvoller nächster Produkt-Schritt

Florence, JoyCaption und Luna als auswählbare Erstcaption-Provider anbinden und die Zusammensetzung davon getrennt in der Profilansicht nachvollziehbar machen. Dafür den vorhandenen strukturierten Audit wiederverwenden, statt denselben Inhalt zusätzlich durch einen Qwen-Textschritt zu schicken. Rohbeschreibung und finale Caption getrennt erhalten. Jede Option muss ihre Profilbereinigung mit demselben bestätigten Personenprofil bewähren. Die vorherige Empfehlung, Sol für Caption-Sonderfälle zu verwenden, ist durch die Nutzerentscheidung überholt.

Vor einer Umstellung der Standards sind die vollständige Merkmalsabdeckung, der manuelle Profil-Fortsetzungsweg und die finale Formulierung gemeinsam zu prüfen. Modellabhängige Caption-Konventionen und gewünschte Steuerbarkeit werden weiterhin durch die vorhandenen Caption-Einstellungen geregelt. Dieser Vergleich entscheidet weder über LoRA-Trainingsqualität noch über die optimale Caption-Konvention jedes Zielmodells.

## Belege

### Produkt-Einbau am 12.10.2026

Florence/JoyCaption/Luna sind nun in der Hauptoberfläche als Erstcaption-Provider auswählbar, unabhängig vom Trainingsziel. Die produktiven Batch-Worker für Florence und JoyCaption wurden jeweils an einem echten eingefrorenen Referenzbild offline geprüft; beide lieferten vollständige Ausgaben. Rohtexte und Bild-/Provider-Fingerprints werden separat gespeichert. Die Profilansicht bietet eine lokale Vorschau mit ungespeicherten Editorwerten sowie dort einstellbare, mit Modul 2 synchronisierte Caption-Regeln. Lokal verwendet die finale Zusammensetzung Audit-Merkmale. Optional extrahiert Luna ausschließlich belegte Szenenpassagen aus den Rohtexten; die Profilbereinigung bleibt lokal. Zusätzliche Identitäts-/Schmuckmerkmale werden nicht über diesen Adapter übernommen. Seine Bildwahrheit und Qualität sind noch nicht im echten Modelltest belegt. Die vollständige Suite bestand mit 88 Tests; keine neuen OpenAI-Testanfragen oder Änderungen am Nutzerdatensatz. Umsetzung und Einrichtung stehen in `CAPTION_PIPELINE.md`.

Lokale Dateien unter `Datasets/caption_benchmark/`:

- `FIVE_MODEL_RAW_RESULTS.md`: alle 60 unveränderten Rohbeschreibungen mit Bildern und Kennzeichnung abgeschnittener Antworten.
- `five_model_summary.json`: Zeiten, Tokens, Kostenschätzung und SHA-256 der Ergebnisdateien.
- `gpt_luna_facts_v2.jsonl`, `gpt_sol_facts_v2.jsonl`: Rohantwort, freie Caption, Merkmale, konkrete Modellkennung und Nutzung pro Anfrage.
- `openai_caption_attempts_v1.json`: genau 24 eindeutige abgeschlossene Anfragen.
- `joint_results.json`: gemeinsame Profil-/Caption-Vorschläge, abgelehnte Felder und Format-Normalisierungen für fünf Varianten.
- `joint_local_metrics.json`: tatsächlich lokal gemessene Gesichts-/Farbmetriken, vorhandenes MediaPipe-Modell; keine erfundenen Gesichtsgrößen.

Die Bild-Hashes sind unverändert. Das Projektprofil und die Trainingsausgabe des Nutzerdatensatzes wurden nicht überschrieben. Es gab keinen LoRA-Trainingslauf. Nach zusätzlicher Absicherung der getrennten unbekannten Schmuckpositionen und weiterer Grammatikfälle bestanden 17 gezielte Adapter-/Profiltests und anschließend die vollständige Suite mit 76 Tests.
