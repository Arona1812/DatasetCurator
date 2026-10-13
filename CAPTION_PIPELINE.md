# Personen-LoRA: zwei Caption-Schritte

Im Audit-Modul stehen Florence 2, JoyCaption und GPT-6 Luna unabhängig vom Trainingsziel zur Auswahl. Florence bietet Caption, Detailed Caption und Very Detailed Caption; voreingestellt ist Very Detailed. GPT-6.1 Sol und Qwen sind keine Erstcaption-Provider. Unabhängige Audit-/Profilmodelle bleiben separat konfigurierbar.

## Erstbeschreibung

Nach der Bildauswahl und vor der Profilprüfung werden die effektiven Bilder beschrieben. Keep- und Review-Kandidaten werden vorbereitet, damit nachträglich ausgewählte Bilder verfügbar bleiben. Rejects und harte Identitätskonflikte sind ausgeschlossen. Rohtexte, Provider-/Prompt-Einstellungen, Modellrevisionen und Bild-Hashes bleiben im Cache und im gespeicherten Laufzustand erhalten. Profil, Trigger, Zielmodell und Caption-Schalter verändern den Erstcaption-Cache nicht. Andere Bildbytes, Florence-Stufen und Modellrevisionen erzeugen neue Einträge. Leere oder abgeschnittene Ausgaben stoppen den Lauf; es gibt keinen automatischen Wechsel zu einer kostenpflichtigen API.

Bei der Profilbildung werden positive, eindeutig benannte Merkmale aus der Erstbeschreibung eng begrenzt ausgelesen: Haar- und Augenfarbe, sichtbare Brille sowie ausdrücklich genannte Piercings/Ohrschmuck und Tattoos. Ein Vergleich mit dem Bildaudit teilt sie in Übereinstimmungen, Widersprüche und nicht bestätigte Hinweise. Nur Übereinstimmungen werden dem Profil-Normalizer als zusätzliche Evidenz für bereits auditierte Merkmale übergeben. Einzelne Textbehauptungen ersetzen niemals einen Bildaudit, ändern keine Häufigkeitszählung und erzeugen keine neue Bildbeobachtung. Widersprüche und zusätzliche Hinweise erscheinen in der Profil-Diagnostik mit Bildnamen. Mehrpersonenbilder liefern keine Erstcaption-Evidenz für das Profil. Der Profil-Cache berücksichtigt die Text- und Provider-Provenienz; ein geänderter Erstcaption-Provider kann daher eine neue Profil-Normalisierung auslösen.

Luna verwendet den bestehenden OpenAI-Schlüssel, Budgetschutz, Tokenzähler und Abbruchweg. Bilder werden an OpenAI gesendet. Lokale Provider werden einmal pro Batch geladen und arbeiten offline. Erfolgreiche Einzelbilder bleiben nach einem späteren Fehler im Cache.

## Zusammensetzung im Profil-Modul

Das automatische Profil wird direkt verwendet; bei manueller Prüfung pausiert der Lauf vor finalen Captions und Export. Die Vorschau nutzt aktuelle, auch ungespeicherte Editorwerte sowie die Caption-Schalter und das Zielmodell aus Modul 2. Sie macht keine Modellanfrage und verändert weder Profil noch Ausgabe. Der Export-Button bestätigt und speichert die Editorwerte vor dem Fortsetzen. Änderungen an Identitätsrollen, die einen Profil-Neuaufbau erfordern, blockieren die Vorschau bis zur Aktualisierung.

Zielmodell, Caption-Felder und Abweichungsmodus sind zusätzlich direkt in der Profilansicht einstellbar und mit Modul 2 synchronisiert. Nach einer Änderung wird die veraltete Vorschau geleert; sie muss bewusst aktualisiert werden.

Das Vorschaubild verwendet den tatsächlichen Export-Ausschnitt mit den aktuellen Trainingsformat-Einstellungen. Audit und Erstbeschreibung beziehen sich weiterhin auf das effektive Eingangsbild. Zusätzliche Format-Crops können deshalb Details entfernen, die in diesen Texten vorkommen; die Vorschau ermöglicht die Kontrolle, ersetzt aber keinen neuen Bildbefund über einen geänderten Ausschnitt.

Die lokale Zusammensetzung verwendet strukturierte **Audit-Beobachtungen** und das Personenprofil. Die freie Erstbeschreibung wird für die Profil-Gegenprüfung verwendet und bleibt zusätzlich als Referenz erhalten; ihre unbestätigten Bilddetails gehen nicht automatisch in die finale Caption ein. Es wird kein Qwen-Textschritt ergänzt. Rohtext einfach anzuhängen würde ausgeschaltete Caption-Felder, falsche Augenfarben oder unsichtbare Merkmale wieder einschleusen.

Optional ist `Luna-Textdetails + lokale Profilregeln` wählbar. Luna extrahiert dabei ausschließlich wörtliche Szenenpassagen für Kleidung, Pose, Ausdruck, Blick, Hintergrund, Licht und Make-up. Bilder und Personenprofile werden bei diesem Schritt nicht gesendet. Nicht im Rohtext belegte Angaben und Passagen mit geschützten Identitäts-/Schmuckmerkmalen werden verworfen und protokolliert. Audit-Beobachtungen bleiben unverändert erhalten; die Zusammensetzung arbeitet mit einer abgeleiteten Kopie. Sichtbarkeit, Augen-/Haaridentität, Brillen, Tattoos und Piercings werden weiterhin durch Audit und Profil geregelt. Zusätzliche kleine Schmuckmerkmale aus der Erstbeschreibung werden damit noch nicht automatisch als Profilbeobachtungen übernommen.

Im manuellen Ablauf bereitet der explizite Button `Luna-Textdetails vorbereiten · API-Kosten` diesen Schritt vor. Danach ist die Vorschau lokal; ohne vorbereitete Textdetails blockiert der gewählte Modus den Export vor dem Austausch vorhandener Trainingsdateien. Im automatischen Ablauf führt die ausdrückliche Auswahl dieses Modus den Textschritt nach der Erstbeschreibung aus. Textdetails sind nur vom Rohtext, Extraktionsmodell und Parser-/Prompt-Version abhängig; Profil-/Zieländerungen lösen keine erneute Textanfrage aus. Bei lokal erkannten Mehrpersonenbildern wird der Textschritt nicht verwendet.

Der optionale Adapter wurde mit simulierten API-Antworten auf Quellbindung, geschützte Merkmale und Cache-Wiederverwendung geprüft. Ein echter Qualitätsvergleich dieses neuen Textschritts steht noch aus. Formale Quellbindung belegt nicht, dass die Erstbeschreibung das Bild richtig erkannt oder ein Detail richtig zugeordnet hat.

Exportierte `.txt`-Dateien enthalten die Trainingscaption. Daneben dokumentiert `.caption.json` die unveränderte Erstbeschreibung und finale Caption. Alte gespeicherte Läufe ohne Erstbeschreibung bleiben fortsetzbar. Automatische und manuelle Läufe speichern beide `_caption_stage.json`. Eine geänderte Quelldatei mit bestehender Erstbeschreibung blockiert die Fortsetzung vor dem Austausch der Trainingsausgabe.

## Lokale Einrichtung

Validierung am 12.10.2026: 94 lokale Tests bestanden. Beide GPU-Worker wurden außerdem jeweils auf demselben echten eingefrorenen Referenzbild geprüft; beide Ausgaben waren vollständig. Die Profil-Gegenprüfung wurde zusätzlich auf jeweils zwölf bereits gespeicherten Florence- und JoyCaption-Beschreibungen gegen vorhandene Luna-Bildbefunde erprobt: Florence ergab sieben Übereinstimmungen ohne Widerspruch, JoyCaption 16 Übereinstimmungen, einen Augenfarben-Widerspruch und neun nicht bestätigte Zusatzhinweise. Diese Luna-Befunde sind **kein bestätigter Goldstandard** und ersetzen keine Prüfung am echten Nutzerdatensatz. Die optionale Luna-Textextraktion wurde ausschließlich mit simulierten Antworten geprüft. Es gab keine weiteren kostenpflichtigen Modelltests, keine Änderungen am Nutzerdatensatz und keinen LoRA-Trainingslauf.

Die CPU-Hauptanwendung bleibt unter Python 3.13. GPU-Provider verwenden eigene Python-Umgebungen:

- Florence: `.venv_florence/Scripts/python.exe`, CUDA PyTorch und Transformers 4.49 mit dem offiziellen Florence-Adapter.
- JoyCaption: `.venv_caption/Scripts/python.exe`, CUDA PyTorch, Transformers 4.57.6 und BitsAndBytes; Sprachmodell in NF4, Bildteil in BF16.

Gewichtsverweise liegen unter `_cache/caption_models/florence2_weights.json` beziehungsweise `joycaption_weights.json`, jeweils als Objekt mit `model`, `revision` und absolutem `snapshot`-Pfad. Bereits vorbereitete Vergleichsgewichte unter `Datasets/caption_benchmark` werden lokal ebenfalls gefunden. Gewichte werden nicht selbständig heruntergeladen.

Für andere Installationspfade können in der Laufkonfiguration `INITIAL_CAPTION_PYTHON` und `INITIAL_CAPTION_WEIGHTS` gesetzt werden. `INITIAL_CAPTION_ENGINE` ist `florence2`, `joycaption` oder `gpt-6-luna`; `INITIAL_CAPTION_FLORENCE_TASK` ist `<CAPTION>`, `<DETAILED_CAPTION>` oder `<MORE_DETAILED_CAPTION>`. `CAPTION_COMPOSITION_MODE` ist `local` (Standard) oder `luna_text`. Ohne Provider-Angabe behalten ältere CLI-Konfigurationen ihren bisherigen Ablauf.
