# Caption-Konventionen fuer Personen-LoRAs

Recherche: 11.10.2026. Fokus: eine konkrete Person in neuen Posen, Kleidungen und Umgebungen reproduzieren. Kein Ausbau auf Stil-/Objekt-LoRAs oder Video. Dieses Dokument ist eine Forschungsgrundlage; Produktlogik wurde dafuer nicht weitergebaut, Modelle wurden nicht installiert und keine Bilder an APIs gesendet.

Nachtrag 12.10.2026: Nach Freigabe des GPU-Speichers wurden die lokalen Captioner tatsaechlich auf 12 unveraenderten Bildkopien verglichen, einschliesslich dreier Florence-Stufen und Large-ft-Kontrolle. [Messwerte und begrenzte Sichtbewertung](../Datasets/caption_benchmark/ASSESSMENT.md). Kein LoRA-Training; keine API-Bilduebertragung.

## Evidenz richtig einordnen

Grundmodell-Pretraining, Generierungs-Prompts und Personen-LoRA-Finetuning sind verschiedene Aufgaben. Ein Prompt-Guide liefert Hinweise auf die verstandene Sprache, aber keinen Beweis fuer die optimale LoRA-Caption. Ein automatisch vorgeschalteter Prompt Enhancer kann zudem verbergen, dass der Bildgenerator eine deutlich andere Beschreibung bekommt als der Nutzer eingegeben hat. Trainingsempfehlungen haben deshalb Vorrang; offene Fragen bleiben als solche markiert.

## Vergleich der Ziele

| Ziel | Belegte Konvention | Konsequenz fuer Personen-LoRAs / offener Punkt |
| --- | --- | --- |
| FLUX.2 Klein 4B / 9B Base | BFL empfiehlt beschreibende, detaillierte Captions und konsistente Trigger; zu lernendes Konzept ausnehmen. Keine separaten Caption-Regeln fuer 4B und 9B dokumentiert. | Gemeinsamer Text-Ablauf. Variable Merkmale erfassen; eine Pflicht zu kuerzeren 4B-Captions ist nicht belegt. [BFL Training](https://docs.bfl.ai/flux_2/flux2_klein_training) |
| FLUX.2 dev | AI Toolkit verwendet Mistral statt der Qwen3-Encoder der Klein-Varianten. Text-Captions sind verwendbar. | Gemeinsamer FLUX-Ablauf ist eine sinnvolle Ausgangshypothese, keine bewiesene Gleichheit der besten Personen-Caption. Florence-Erfahrung mit FLUX.1 nicht automatisch auf FLUX.2 uebertragen. [AI Toolkit dev](https://ostris.com/docs/ai-toolkit/models/flux2), [4B](https://ostris.com/docs/ai-toolkit/models/flux2-klein-4b), [9B](https://ostris.com/docs/ai-toolkit/models/flux2-klein-9b) |
| Krea 2 Raw / Turbo | Technischer Bericht: reichhaltige Beschreibungen, ueberwiegend lange Captions, daneben kuerzere Varianten. | Gemeinsam objektive, ausreichend detaillierte Texte. Kein separater Raw-/Turbo-Personen-Caption-Standard gefunden. [Krea Bericht](https://www.krea.ai/blog/krea-2-technical-report) |
| Z-Image Base | Bericht beschreibt fuenf Caption-Arten: lang, mittel, kurz, Tags und simulierte Nutzerprompts. | Ausfuehrliche Beschreibung wird verstanden; "immer alles" ist keine nachgewiesene Pflicht. Diffusers zeigt sogar DreamBooth mit kurzem Instanzprompt, allerdings fuer einen Hund und ohne Personen-Vergleich. [Z-Image Bericht](https://arxiv.org/html/2511.22699v1), [Diffusers Training](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_z_image.md) |
| ERNIE Image | Detaillierte VLM-Beschreibungen im Pretraining; SFT variiert Stil, Granularitaet und Laenge. Ein Prompt Enhancer erweitert Nutzertexte. | Detaillierter Text-Ablauf als begruendeter Start. Keine verifizierte Regel, alle stabilen Gesichtsmerkmale in jede Personen-LoRA-Caption zu schreiben. [ERNIE Bericht](https://arxiv.org/html/2605.25347v1) |
| Qwen-Image 2.1 | Text-/Bild-Konditionierung in AI Toolkit; keine JSON-Pflicht dokumentiert. | Beschreibender Text-Ablauf vorlaeufig. Der aeltere Qwen-Image-Bericht betont fuer Portraitdaten Gesicht, Ausdruck, Kleidung und Kontext; das belegt nicht die optimale LoRA-Policy von 2.1. [AI Toolkit 2.1](https://ostris.com/docs/ai-toolkit/models/qwen-image-2), [Qwen-Image Bericht, aeltere Generation](https://arxiv.org/html/2508.02324v1) |
| Ideogram 4 | Ausschliesslich strukturierte JSON-Captions im Grundtraining, detaillierte Elementbeschreibungen. | Eigener Serializer und Schema-Pruefung. Format ist belegt; optimale Identitaetsbeschreibung fuer Personen-LoRAs bleibt offen. [Ideogram Technik](https://ideogram.ai/blog/ideogram-4.0/), [Schema](https://github.com/ideogram-oss/ideogram4/blob/main/docs/prompting.md) |

Die Herkunft eines Modells ist keine belastbare Caption-Regel. Auch Krea und Ideogram verwenden reichhaltige Annotationen; Z-Image und ERNIE kennen kurze Formen.

## Identitaet und steuerbare Abweichungen

Krea empfiehlt direkt fuer LoRAs, Details zu benennen, die nicht in das gelernte Konzept eingehen sollen. Fuer Charaktere nennt Krea Variation von Blickwinkel, Ausdruck, Licht und Hintergrund. Das stuetzt die gemeinsame Strategie fuer wechselnde Kleidung, Frisuren, Make-up und Zubehoer. [Krea LoRA-Anleitung](https://www.krea.ai/blog/krea-2-lora-training)

Vorgeschlagene gemeinsame Personen-Policy:

- Trigger bezeichnet die Person. Stabile Identitaet wird im Profil erfasst, aber nicht automatisch vollstaendig in jede exportierte Caption kopiert.
- Pose, Ausdruck, Blickrichtung, Bildausschnitt, Kleidung, sichtbares Zubehoer, Frisur, Hintergrund und relevante Lichtverhaeltnisse sind separat beschreibbar.
- Piercings, Brillen und Schmuck nur bei gesicherter Sichtbarkeit nennen. Ein Profilinventar ist kein Beweis fuer ihre Praesenz im Einzelbild.
- "Nicht sichtbar" und "sichtbar nicht vorhanden" unterscheiden; keine erfundenen negativen Aussagen bei verdeckten Stellen.
- Captioning allein garantiert keine Entkopplung: Wenn jedes Bild dasselbe Piercing zeigt, fehlen Gegenbeispiele. Datensatzvariation muss mitbewertet werden. Dies ist eine methodische Schlussfolgerung, keine modelluebergreifende Erfolgsgarantie.
- Nicht aus Fotos vermeintlich exakte Kamera-, Objektiv- oder Belichtungswerte erfinden.

## Lokale Captioner

Florence-2 ist ein kleiner spezialisierter Vision-Language-Captioner mit festen Caption-, Detail-, Region- und OCR-Aufgaben. Large umfasst etwa 0,77B Parameter. Es benoetigt keinen externen Chat-LLM-Dienst. Empfehlenswert als kostenguenstige Vergleichsbasis; eine perfekte Erkennung kleiner Piercings oder eine eigene Personen-Policy ist damit noch nicht nachgewiesen. [Microsoft Modellkarte](https://huggingface.co/microsoft/Florence-2-large)

JoyCaption ist gezielt fuer Diffusions-Trainingscaptions entwickelt. Es bietet Namensvorgaben und eine Option zum Auslassen unveraenderlicher Personenmerkmale, sowie objektive und ausfuehrlichere Modi. Native Gewichte benoetigen laut Entwickler etwa 17 GB VRAM; Quantisierung ist moeglich. Der Entwickler nennt Grenzen bei Instruktionen, mehreren Personen, Links/Rechts und OCR. Kandidat fuer Personen-Policy, kein ungepruefter Standard. [JoyCaption Entwicklerdokumentation](https://github.com/fpgaminer/joycaption)

Qwen3-VL ist in AI Toolkit als lokaler Bildcaptioner mit 2B-, 4B- und 8B-Optionen vorhanden. AI Toolkit hat ausserdem einen eigenen Ideogram-4-Captioner auf Qwen3-VL-Basis. Das erlaubt denselben lokalen Unterbau fuer Text und JSON; eine API ist dafuer nicht grundsaetzlich erforderlich. Kleine Bilddetails muessen mit geeigneter Bildaufloesung und gegebenenfalls zusaetzlichen Ausschnitten geprueft werden. [AI Toolkit Caption-Auswahl](https://github.com/ostris/ai-toolkit/blob/main/ui/src/helpers/captionOptions.ts)

## Zusammengefasster Ablauf als Umsetzungsvorschlag

1. Einmal sichtbare Bildfakten erfassen und mit dem Personenprofil abgleichen.
2. Gemeinsame Personen-Policy anwenden: Identitaet vs. steuerbare Merkmale, Sichtbarkeit und konsistente Begriffe.
3. Zwei Textvarianten anbieten: kompakte variable Beschreibung (FLUX-Ausgangsbasis) und detaillierte objektive Beschreibung (Krea/Z/ERNIE/Qwen-Ausgangsbasis). Die Zuordnung ist vorlaeufig und uebersteuerbar.
4. Ideogram separat in sein offizielles JSON-Schema ausgeben. `compositional_deconstruction` benoetigt `background` und `elements`; Schluesselreihenfolge beachten. Bounding Boxes sind optional und duerfen nicht geraten werden. JSON laesst sich aus strukturierten Fakten ohne weiteren LLM-Aufruf serialisieren, aber die Fakten muessen dafuer ausreichend gegliedert sein.
5. Unsichere Merkmale gezielt nachpruefen statt alle Bilder mit einem grossen API-Modell erneut zu beschreiben.

Aktueller Produktstand: Die FLUX-Ziele teilen allgemeine Voreinstellungen; Florence/JoyCaption/Qwen3-VL sind noch nicht integriert. Der Ideogram-Export verpackt bisher die gesamte Caption als ein Element und enthaelt keinen getrennten `background`. Er entspricht damit noch nicht vollstaendig der offiziellen Verifier-Konvention. Ein formal lesbares JSON ist keine vollstaendige, hochwertige Ideogram-Caption. Vor weiterer Optimierung muss das korrigiert und validiert werden.

Naechste sachliche Pruefung: Florence-2, JoyCaption und Qwen3-VL auf einer festen, manuell bewerteten Auswahl vergleichen (Pose, Kleidung, Schmuck/Piercings, Verdeckung, Hintergrund, erfundene Inhalte und Laufzeit). Erst danach lokale Standard-Engine waehlen. Das bewertet Caption-Fakten, noch nicht LoRA-Ergebnisse. Ein anschliessender begrenzter Trainingsvergleich prueft Identitaet, neue Posen/Kleidung und Generierung mit/ohne Zubehoer. Keine allgemein optimalen Wortzahlen oder modellabhaengigen Personen-Regeln behaupten, bevor dieser Vergleich stattgefunden hat.
