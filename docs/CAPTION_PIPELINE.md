# Personen-LoRA: zwei Caption-Schritte

Die Erstbeschreibung schreibt GPT-6 Luna, unabhängig vom Trainingsziel. Die lokalen Provider Florence 2 und JoyCaption wurden entfernt: Sie brachten bei deutlich längerer Rechenzeit keinen Mehrwert für die Trainingscaptions. Gespeicherte Einstellungen mit `florence2` oder `joycaption` wechseln automatisch zu Luna; bereits gespeicherte Luna-Beschreibungen bleiben gültig.

## Erstbeschreibung

Nach der Bildauswahl und vor der Profilprüfung werden die effektiven Bilder beschrieben. Keep- und Review-Kandidaten werden vorbereitet, damit nachträglich ausgewählte Bilder verfügbar bleiben. Rejects und harte Identitätskonflikte sind ausgeschlossen. Rohtexte, Provider-/Prompt-Einstellungen, Modellrevisionen und Bild-Hashes bleiben im Cache und im gespeicherten Laufzustand erhalten. Profil, Trigger, Zielmodell und Caption-Schalter verändern den Erstcaption-Cache nicht. Andere Bildbytes erzeugen neue Einträge. Leere oder abgeschnittene Ausgaben stoppen den Lauf; es gibt keinen automatischen Wechsel zu einer kostenpflichtigen API.

Bei der Profilbildung werden positive, eindeutig benannte Merkmale aus der Erstbeschreibung eng begrenzt ausgelesen: Haar- und Augenfarbe, sichtbare Brille sowie ausdrücklich genannte Piercings/Ohrschmuck und Tattoos. Ein Vergleich mit dem Bildaudit teilt sie in Übereinstimmungen, Widersprüche und nicht bestätigte Hinweise. Nur Übereinstimmungen werden dem Profil-Normalizer als zusätzliche Evidenz für bereits auditierte Merkmale übergeben. Einzelne Textbehauptungen ersetzen niemals einen Bildaudit, ändern keine Häufigkeitszählung und erzeugen keine neue Bildbeobachtung. Widersprüche und zusätzliche Hinweise erscheinen in der Profil-Diagnostik mit Bildnamen. Mehrpersonenbilder liefern keine Erstcaption-Evidenz für das Profil. Der Profil-Cache berücksichtigt die Text- und Provider-Provenienz; ein geänderter Erstcaption-Provider kann daher eine neue Profil-Normalisierung auslösen.

Luna verwendet den bestehenden OpenAI-Schlüssel, Budgetschutz, Tokenzähler und Abbruchweg. Bilder werden an OpenAI gesendet. Lokale Provider werden einmal pro Batch geladen und arbeiten offline. Erfolgreiche Einzelbilder bleiben nach einem späteren Fehler im Cache.

## Zusammensetzung im Profil-Modul

Das automatische Profil wird direkt verwendet; bei manueller Prüfung pausiert der Lauf vor finalen Captions und Export. Die Vorschau nutzt aktuelle, auch ungespeicherte Editorwerte sowie die Caption-Schalter und das Zielmodell aus Modul 2. Sie macht keine Modellanfrage und verändert weder Profil noch Ausgabe. Der Export-Button bestätigt und speichert die Editorwerte vor dem Fortsetzen. Änderungen an Identitätsrollen, die einen Profil-Neuaufbau erfordern, blockieren die Vorschau bis zur Aktualisierung.

Zielmodell, Caption-Felder und Abweichungsmodus sind zusätzlich direkt in der Profilansicht einstellbar und mit Modul 2 synchronisiert. Nach einer Änderung wird die veraltete Vorschau geleert; sie muss bewusst aktualisiert werden.

Das Vorschaubild verwendet den tatsächlichen Export-Ausschnitt mit den aktuellen Trainingsformat-Einstellungen. Audit und Erstbeschreibung beziehen sich weiterhin auf das effektive Eingangsbild. Zusätzliche Format-Crops können deshalb Details entfernen, die in diesen Texten vorkommen; die Vorschau ermöglicht die Kontrolle, ersetzt aber keinen neuen Bildbefund über einen geänderten Ausschnitt.

**Was in eine Caption kommt, entscheiden ausschließlich die Caption-Regeln (Modul 2) und das bestätigte Personenprofil.** Daraus entsteht pro Bild eine feste Faktenliste (`collect_caption_facts`): Fototyp, Pflichtphrasen mit genauem Wortlaut (z. B. Abweichungen vom Kanon, sichtbare Piercings, Make-up, Accessoires), die eingeschalteten Szenenteile (Kleidung, Pose, Ausdruck, Blick, Licht, Hintergrund) und die Identitätswerte, die beim Triggerwort bleiben und nicht genannt werden dürfen. Beide Zusammensetzungen formulieren nur diese Fakten; keine von beiden entscheidet selbst über Inhalte.

- **Lokal · Audit und Personenprofil** formt die Fakten ohne Modellanfrage zu ganzen englischen Sätzen (`caption_grammar.py`): Merkmale als Satz („She has a nose ring and wears dramatic eyeliner.“) statt Tag-Liste, passende Verbform zur Pose, ein Artikel, einheitliche Pronomen und großgeschriebene Satzanfänge.
- **Luna · natürliche Sätze aus Bild, Regeln und Profil** lässt beim Export das Caption-Modell (Standard `gpt-6-luna`) die Fakten für das gewählte Trainingsziel formulieren. Es sieht den tatsächlichen Export-Ausschnitt; die Erstbeschreibung dient nur als Nachschlagewerk für Szenendetails, Identitätsangaben daraus werden ignoriert. Jede Caption wird lokal geprüft: richtiger Anfang mit Triggerwort, alle Pflichtphrasen enthalten, nichts Ausgeschlossenes (z. B. Piercings bei abgeschalteter Regel, kanonische Haar- oder Augenfarbe), keine Tags oder Meta-Formulierungen. Bei einem Verstoß folgt genau ein Reparaturversuch (Standard `gpt-6.1-sol`), danach gilt die lokale Caption. Ergebnisse werden pro Bild, Faktenliste, Erstbeschreibung und Modell gespeichert; neue Anfragen gibt es nur, wenn sich eines davon ändert. Kosten mit Luna: etwa 0,0005 USD pro Trainingsbild. Ohne API-Schlüssel bricht der Export vor dem Austausch vorhandener Trainingsdateien ab.

**Mehrfarbige Haare:** Aus der Haarbeschreibung des Audits erkennt das Tool neben Highlights, Ombré und Balayage auch **Strähnen vorn** (z. B. weiß wie bei Rogue, „money piece“) und **zweigeteilte Haare** (Split Dye) samt Zweitfarbe. Zeigt mindestens die Hälfte der Bilder mit sichtbaren Haaren dasselbe Muster, wird es Kanon im Profil (Felder „Kanonisches Haarmuster“ und „Zweitfarbe“, editierbar). Das Muster folgt dann derselben Regel wie die Haarfarbe: Der Kanon gehört zum Triggerwort, Abweichungen werden genannt, etwa „black hair without white front streaks“ oder „split-dyed blonde and black hair“. Beschreibt das Audit ein Muster nicht, lässt es sich im Profil von Hand setzen.

Alle Trainingsziele außer Ideogram 4 erwarten natürliche Sprache. FLUX.2 erhält eine kompakte Form (25–60 Wörter), die übrigen Ziele eine ausführliche (40–110 Wörter); Ideogram 4 bekommt dieselbe Caption im JSON-Rahmen. Die Vorschau zeigt im Luna-Modus die gespeicherte Luna-Caption für genau diese Regeln und dieses Profil, sonst den lokalen Entwurf; sie macht nie eine Modellanfrage.

Der frühere Modus `Luna-Textdetails + lokale Profilregeln` (wörtliche Textausschnitte in einer Satzschablone) erzeugte grammatisch fehlerhafte Captions und wurde ersetzt. Gespeicherte Einstellungen mit `luna_text` wechseln automatisch zum neuen Luna-Modus.

Der Luna-Modus wurde ausschließlich mit simulierten API-Antworten geprüft. Ein Qualitätsvergleich mit echten Antworten am eigenen Datensatz steht noch aus.

Exportierte `.txt`-Dateien enthalten die Trainingscaption. Daneben dokumentiert `.caption.json` die unveränderte Erstbeschreibung und finale Caption. Alte gespeicherte Läufe ohne Erstbeschreibung bleiben fortsetzbar. Automatische und manuelle Läufe speichern beide `_caption_stage.json`. Eine geänderte Quelldatei mit bestehender Erstbeschreibung blockiert die Fortsetzung vor dem Austausch der Trainingsausgabe.


## Laufkonfiguration

`CAPTION_COMPOSITION_MODE` ist `local` (Standard) oder `luna_writer`; der alte Wert `luna_text` wird zu `luna_writer`. `INITIAL_CAPTION_ENGINE` ist `gpt-6-luna` oder leer (keine Erstbeschreibung); alte Werte `florence2` und `joycaption` werden zu `gpt-6-luna`.
