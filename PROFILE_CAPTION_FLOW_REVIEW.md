# Personenprofil und Captioning gemeinsam prüfen

Stand: 12. Oktober 2026. Zunächst Codeprüfung und vier isolierte Reproduktionen mit gesperrtem Netzwerk; anschließend die unten dokumentierten Regelkorrekturen. Kein neuer Bildaudit, kein API-Aufruf, keine Änderung am Nutzerdatensatz und kein LoRA-Training.

## Bestätigte Anforderungen

- Automatischer Ablauf: Das Tool erstellt und verwendet seine bestmögliche Profilschätzung.
- Manueller Ablauf: Das Profil wird vor Captions und Export durch den Nutzer geprüft und optimiert. Die letzte Korrektur des Nutzers ersetzt die vorherige Forderung nach Bestätigung in jedem Ablauf.
- Der normale Look wird zunächst anhand der Häufigkeit empfohlen. Semantisch ähnliche Werte wie Platin, Hellblond und Blond sollen hierfür zusammengezählt werden.
- Feste Merkmale, insbesondere die natürliche Augenfarbe, werden über den Datensatz konsolidiert. 30 blaue und zwei grüne Beobachtungen sollen nicht automatisch zwei echte Augenfarben begründen; seltene Fotografie-/Erkennungsfehler sind zu normalisieren.
- Piercings, Haarfarbe/Frisur, Make-up, Brille und Tattoos bleiben separat steuerbare Merkmale. Häufigkeit allein macht ein steuerbares Merkmal nicht zum Bestandteil des Triggerworts.
- Die vorhandenen Zielmodell-/Caption-Einstellungen bestimmen, welche Merkmale beschrieben werden müssen. Steuerbar bedeutet nicht automatisch, dass jedes Merkmal in jeder Caption erscheinen soll. Das Profil liefert den normalen Look und darf diese Auswahl nicht überschreiben.
- Brille: Ist sie nicht sichtbar, wird sie nicht captioniert. Die übliche sichtbare Brille kann bei einer auf Abweichungen ausgelegten Caption-Policy entfallen. Eine deutlich abweichende sichtbare Brille wird entsprechend dieser Policy beschrieben. Soll die Brille optional gelernt werden, beschreibt die dafür gewählte Einstellung auch die normale sichtbare Brille. Ein verdecktes Gesicht begründet weder eine Brillenbeschreibung noch `without glasses`.
- Allgemeine Caption-Qualität bleibt entscheidend: Kleidung, Pose, Ausdruck und Bildkontext müssen korrekt erkannt werden. Das Profil kann falsche Bildbeobachtungen nicht nachträglich zuverlässig ersetzen.

## Tatsächlicher Ablauf

1. Bildaudits liefern strukturierte Beobachtungen. Die lokale Caption-Benchmark liefert bislang freie Texte und ist noch nicht an diesen strukturierten Eingang angeschlossen.
2. Auswahl, Duplikatbehandlung und ArcFace-Prüfung bestimmen zunächst die Trainingsauswahl. Zusätzlich geeignete, ungenutzte Keep-Bilder fließen ins Profil ein.
3. Die Profilbildung verwendet strukturierte Audit-Daten, keine Bilder. Bei großen Datensätzen wird eine Teilmenge ausgewählt. Priority/Core/Body-Reference-Rollen beeinflussen Auswahl und Gewichtung; ein Textmodell konsolidiert die Angaben, bei Fehlern gibt es einen lokalen Rückfall.
4. Das Tool ergänzt Merkmale pro Bild, Häufigkeiten, Variationsregeln, Piercing-Inventar und Erscheinungsgruppen. Nutzer-Overrides werden übernommen.
5. `single_pass` arbeitet automatisch weiter. `profile_then_caption` speichert einen Zwischenstand und pausiert vor Captions und Export. Die UI bietet beide Wege bereits ausdrücklich an.
6. Im manuellen Fortsetzungsweg werden Rollen und Auswahl erneut berücksichtigt. Wenn Rollen eine Profil-Neuberechnung erforderlich machen, erfolgt diese beim Fortsetzen; bestätigte Werte werden danach wieder angewendet. Eine solche Neuberechnung kann einen Textmodell-Aufruf auslösen, auch wenn kein Bild neu auditiert wird.
7. Caption-Regeln kombinieren Profil, Beobachtungen pro Bild und Zielmodell. Der lokale Textaufbau nutzt weiterhin Audit-Felder für Kleidung, Pose und Kontext. Der Krea-Pfad verwendet zusätzlich einen bildgestützten Caption-Aufruf.

Code-Einstiege: `build_subject_profile`, `stratified_sample_for_profile`, `per_image_profile_traits`, `attach_profile_variability_policies`, `attach_piercing_inventory`, `continue_caption_from_profile` und `build_local_caption` in `dataset_curator_v2.py`; Ablaufwahl und Profileditor in `dataset_curator_ui.py`.

## Reproduzierte Abweichungen von den Anforderungen

| Fall | Aktuelles Ergebnis | Erforderliche Änderung |
| --- | --- | --- |
| 30 verlässliche blaue und zwei grüne Augenbeobachtungen | `_trait_variance` meldet Variation; `get_eye_feature_state` fordert für ein grünes Bild ausdrücklich `green eyes`. | Natürliches Identitätsmerkmal konsolidieren. Eine echte Änderung wie farbige Kontaktlinsen benötigt zusätzliche Evidenz; eine Minderheit von zwei Beobachtungen allein genügt nicht. |
| 20 Blond, 20 Platin, 20 Dunkelblond gegenüber 30 Braun; gleiche Gewichte | `_weighted_trait_mode` liefert `brown`, obwohl die blonde Familie insgesamt 60 Bilder umfasst. | Semantische Gruppen vor der Häufigkeitsentscheidung bilden. Bestehende Erscheinungsgruppen und Auswahlboni ersetzen diese Konsolidierung nicht. |
| Brille normalerweise vorhanden, Gesicht verdeckt, keine Brille erkannt | `get_glasses_feature_state` liefert `without glasses`, obwohl die Position `not_visible` ist. | Sichtbarkeit zuerst prüfen; eine verdeckte Brille erzeugt keine Brillenphrase. Für sichtbare Brillen entscheiden die vorhandene Caption-Policy, die Abweichung vom normalen Look und die ausdrücklich gewünschte optionale Steuerbarkeit. |
| Nasenpiercing in sechs von zehn Bildern, keine manuelle Rolle gesetzt | `attach_piercing_inventory` setzt `canonical`; ein identitätsorientierter Caption-Pfad lässt das sichtbare Piercing daraufhin weg. | Steuerbarkeit unabhängig von Häufigkeit festlegen. Sichtbare Piercings standardmäßig separat beschreiben; die Häufigkeit ist eine Empfehlung für den normalen Look, keine Erlaubnis zur Auslassung. |

Die Reproduktionen rufen die echten Backend-Funktionen mit kontrollierten Eingaben auf. Der Import wurde über die bestehende isolierte Fixture aus `test_frames_export.py` mit leerem API-Key und temporärem Eingabeordner ausgeführt; HTTP-Anfragen waren gesperrt. Dies bestätigt die Regelabweichungen, nicht die Erkennungsqualität auf realen Fotos.

## Konsequenz für den gemeinsamen Aufbau

Das Profil braucht getrennte Entscheidungen für **normalen Look**, **feste Identität**, **Steuerbarkeit** und **Sicherheit der Beobachtung**. Ein einzelnes Feld wie `canonical` vermischt diese Zwecke derzeit. Beispiel: Ein sehr häufiges Nasenpiercing kann zum normalen Look gehören und trotzdem separat captioniert werden, damit es steuerbar bleibt.

Die vorhandene Caption-Auswahl bleibt die maßgebliche Einstellung für den Inhalt. Die Reihenfolge lautet: Sichtbarkeit feststellen → normalen Look aus dem Profil vergleichen → gewählte Zielmodell-/Caption-Policy anwenden. Der häufigste Look ist eine Referenz, keine pauschale Caption-Pflicht. Insbesondere darf aus der gewünschten optionalen Brille keine Behauptung über eine verdeckte Brille entstehen.

Die Rohbeobachtung bleibt erhalten. Eine normalisierte Augenfarbe wird als Profilentscheidung mit Begründung geführt, nicht rückwirkend als neue Bildbeobachtung ausgegeben. Bei verdeckten Augen darf das Profil keine sichtbaren Augen erfinden. Bei glaubhaft erkannten farbigen Kontaktlinsen muss eine echte Bildabweichung erhalten bleiben.

Häufigkeiten brauchen einen nachvollziehbaren Nenner: unbekannte oder verdeckte Merkmale zählen nicht als Abwesenheit. Rollen-Gewichte sind eine zusätzliche, erkennbare Nutzerentscheidung und dürfen eine gewöhnliche Häufigkeitsempfehlung nicht unbemerkt ersetzen. Für den Profilvorschlag sollte die vollständige geeignete Datengrundlage deterministisch gezählt werden; die Modell-Teilmenge dient ergänzender Konsolidierung.

Im manuellen Ablauf muss die Vorschau das Profil zeigen, das tatsächlich beim Captioning verwendet wird. Rollenänderungen sollten die Vorschau vor dem Fortsetzen aktualisieren; eine relevante Neuberechnung nach der Prüfung darf nicht unsichtbar zu einem anderen Profil führen. Der Startknopf kann die ausdrückliche Freigabe des angezeigten Stands übernehmen. Im automatischen Ablauf wird ohne zusätzliche Nutzerfreigabe die bestmögliche Schätzung verwendet.

## Nächster sinnvoller Qualitätsvergleich

1. Zuerst die vier oben belegten Regeln korrigieren und mit kleinen, gezielten Regressionen absichern.
2. Einen gemeinsamen strukturierten Bildfakten-Eingang für lokale Captioner schaffen: Kleidung, Pose, Kontext sowie Personenmerkmale mit Sichtbarkeit und Sicherheit. Freie Texte allein genügen für den vorhandenen Profilweg nicht.
3. Auf denselben eingefrorenen zwölf Bildern den vollständigen Weg vergleichen: lokale Beobachtung → Profilvorschlag → automatische oder manuelle Profilentscheidung → zielabhängige Caption.
4. Getrennt bewerten: korrekte Bildbeschreibung, richtige Profilkonsolidierung, Erhalt echter Abweichungen, keine erfundenen Details, Steuerbarkeit und angemessenes Ausgabeformat. Caption-Länge oder die Anzahl erkannter Piercings allein sind kein Gesamtqualitätsmaß.

Der bisherige Vergleich von Florence, JoyCaption und Qwen bleibt ein Roh-Caption-Vergleich. Es liegt hier noch keine reale Abnahme des zusammenhängenden Profil-/Caption-Wegs und kein Trainingsnachweis vor. Im aktuell geprüften echten Projektordner war kein gespeichertes Subject Profile bzw. Caption-Stage vorhanden, auf dem sich eine solche Abnahme nachholen ließe.

## Umgesetzte Regelkorrekturen

- Die Profil-Haarfarbe wird aus den vollständigen geeigneten Bildmerkmalen nach Häufigkeit empfohlen. Blond/Platin/Dunkelblond sowie nahe Brauntöne werden jeweils zusammengezählt; Schwarz bleibt von Braun unterscheidbar. Die ursprünglichen Farbangaben pro Bild und deren Rohhäufigkeiten bleiben erhalten. Auswahl und Abweichungsprüfung verwenden dieselbe Gruppierung.
- Die natürliche Augenfarbe wird anhand geeigneter Beobachtungen konsolidiert. Minderheitsbeobachtungen bleiben als Evidenz erhalten und begründen allein keine variable Augenfarbe. Erkannte farbige Kontaktlinsen bleiben eine Bildabweichung; ein bloßes Urteil „unnatürliche Farbe“ genügt dafür nicht. Verdeckte/geschlossene Augen erzeugen keine Augenfarbphrase. Der lokale Caption-Aufbau und der Krea-Aufruf verwenden dieselbe Merkmalsentscheidung.
- Brillen haben getrennte Beobachtungszustände. Verdeckt/unsicher zählt nicht als Abwesenheit und erzeugt keine Brillenphrase. Die Häufigkeit und Empfehlung zur regelmäßig getragenen Brille beruhen auf beurteilbaren Bildern. Die vorhandenen Schalter für immer beschreiben, Abweichungen und variable Merkmale bleiben maßgeblich.
- Häufigkeit setzt Piercings nicht mehr automatisch auf `canonical`; auch eine solche unbestätigte Modell-Einstufung wird zu `variable`. Explizit bestätigte Rollen und deaktivierte Caption-Merkmale bleiben erhalten.
- Bestätigte Profilwerte werden bei Konsolidierung geschützt. Auch das Speichern über den JSON-Editor hält die bewussten Entscheidungen für spätere Neuberechnungen fest.
- Profil-Schema v15 und Krea-Caption-Prompt v7 verhindern die unbemerkte Wiederverwendung alter Profil-/Caption-Caches. Der Bildaudit-Cache wurde durch diese Regeländerung nicht versioniert. Die bestehende lokale Migration eines gespeicherten Caption-Zwischenstands nutzt dessen vorhandene Audits.

Validierung: acht gezielte Profil-/Caption-Regressionen in `test_profile_caption_policy.py`, darunter tatsächliche lokale Captions für ERNIE, Z-Image und Krea. Der vollständige Lauf mit diesen Tests bestand mit 67 Tests; nach Ergänzung des Kontaktlinsen-Grenzfalls bestanden die acht gezielten Tests erneut. Exportprüfungen umfassen alle neun Zielmodelle und unveränderte Quelldateien. Der UI-Aufbau hing zunächst beim lokalen Windows-Socket in der Sandbox; derselbe vollständige Testlauf außerhalb dieser Einschränkung bestand. Kein neuer visueller Qualitätsvergleich und keine neue Modell-Inferenz.

Noch offen sind der strukturierte Eingang für lokale Captioner, der gemeinsame Vergleich auf den zwölf realen Benchmarkbildern und die Vorschau-Aktualisierung bei rollenbedingter Neuberechnung im manuellen Ablauf.
