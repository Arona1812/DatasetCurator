"""Compare local proposals with explicitly confirmed reference boundaries."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def region_metrics(proposal, reference):
    def area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    intersection = max(0, min(proposal[2], reference[2]) - max(proposal[0], reference[0])) * max(0, min(proposal[3], reference[3]) - max(proposal[1], reference[1]))
    actual, target = area(proposal), area(reference)
    return {"iou": intersection / (actual + target - intersection),
            "missing_photo_fraction": 1 - intersection / target,
            "extra_area_fraction": 1 - intersection / actual,
            "edge_error_pixels": [abs(a-b) for a,b in zip(proposal, reference)]}


def valid_box(box, size):
    return (isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box)
            and 0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1])


def score(manifest, confirmed):
    if manifest.get("schema") != "frame-reference-v1" or confirmed.get("schema") != "frame-reference-v1":
        raise ValueError("Unsupported reference schema")
    sources = {r["id"]: r for r in manifest["records"]}
    results, rejected, seen = [], [], set()
    for reference in confirmed.get("records", []):
        source = sources.get(reference.get("id"))
        decision = reference.get("reference_decision")
        if (not source or reference.get("reference_confirmed") is not True
                or reference.get("size") != source["size"] or reference.get("id") in seen
                or decision not in {"crop", "keep_original", "manual_review"}):
            rejected.append({"filename": reference.get("filename"), "reason": "unknown_or_invalid_reference"})
            continue
        target = reference.get("reference_bbox")
        if decision == "crop" and not valid_box(target, source["size"]):
            rejected.append({"filename": reference.get("filename"), "reason": "invalid_reference_box"})
            continue
        seen.add(reference["id"])
        proposal = source.get("local_bbox")
        row = {"filename": source["filename"], "id": source["id"], "layout": source["layout"],
               "cohort": source.get("cohort", "reference"),
               "confidence_level": source["confidence_level"], "reference_decision": decision,
               "reference_bbox": target, "local_bbox": proposal}
        if decision == "crop":
            if not valid_box(proposal, source["size"]):
                row["outcome"] = "missing_proposal"
            else:
                row.update(region_metrics(proposal, target))
                # A disclosed comparison tolerance, not a safety/automatic-crop gate.
                limits = [max(3, source["size"][i % 2] * .005) for i in range(4)]
                row["outcome"] = "near_reference" if all(error <= limit for error, limit in zip(row["edge_error_pixels"], limits)) else "needs_correction"
        elif decision == "keep_original":
            row["outcome"] = "correct_keep" if not proposal else "unnecessary_proposal"
        else:
            row["outcome"] = "special_case_without_proposal" if not proposal else "special_case_with_proposal"
        automatic = source.get("automatic_bbox")
        if "automatic_bbox" in source:
            row["automatic_bbox"] = automatic
            if not automatic:
                row["automatic_outcome"] = "no_automatic_crop"
            elif decision == "keep_original":
                row["automatic_outcome"] = "unnecessary_automatic_crop"
            elif decision == "manual_review":
                row["automatic_outcome"] = "special_case_automatic_crop"
            elif valid_box(automatic, source["size"]):
                metrics = region_metrics(automatic, target)
                limits = [max(3, source["size"][i % 2] * .005) for i in range(4)]
                row["automatic_outcome"] = "near_reference" if all(
                    error <= limit for error, limit in zip(metrics["edge_error_pixels"], limits)) else "needs_correction"
            else:
                row["automatic_outcome"] = "invalid_automatic_crop"
        results.append(row)
    outcomes = Counter(r["outcome"] for r in results)
    groups, cohorts, automatic_outcomes = defaultdict(Counter), defaultdict(Counter), Counter()
    for row in results:
        groups[row["layout"]][row["outcome"]] += 1
        cohorts[row["cohort"]][row["outcome"]] += 1
        if row.get("automatic_outcome"):
            automatic_outcomes[row["automatic_outcome"]] += 1
    crop_rows = [r for r in results if "edge_error_pixels" in r]
    sensitivity = {}
    for percent in (0.5, 1, 2, 3, 5):
        sensitivity[str(percent)] = sum(
            all(error <= max(3, sources[r["id"]]["size"][i % 2] * percent / 100)
                for i, error in enumerate(r["edge_error_pixels"])) for r in crop_rows
        )
    return {"schema": "frame-benchmark-score-v1", "detector": "local_heuristics",
            "confirmed": len(results), "unconfirmed": len(sources)-len(seen), "outcomes": dict(outcomes),
            "by_layout": {k:dict(v) for k,v in groups.items()}, "results": results, "rejected": rejected,
            "by_cohort": {k:dict(v) for k,v in cohorts.items()}, "automatic_outcomes": dict(automatic_outcomes),
            "crop_proposals_compared": len(crop_rows), "edge_tolerance_sensitivity": sensitivity,
            "substantial_extra_area": sum(r["extra_area_fraction"] > 0.10 for r in crop_rows),
            "substantial_missing_photo": sum(r["missing_photo_fraction"] > 0.05 for r in crop_rows),
            "tolerance": "Each edge within max(3 pixels, 0.5% of its source dimension). This is not a measured safety guarantee."}


def write_report(result, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    labels = {"near_reference": "Fotogrenzen nahe der Referenz", "needs_correction": "Vorschlag benötigt Korrektur",
              "missing_proposal": "Zuschneidbares Foto ohne Vorschlag", "correct_keep": "Original korrekt ohne Vorschlag",
              "unnecessary_proposal": "Vorschlag trotz bestätigtem Original", "special_case_without_proposal": "Sonderfall ohne Zuschnittsvorschlag",
              "special_case_with_proposal": "Sonderfall mit Zuschnittsvorschlag"}
    lines = ["# Vergleich mit bestätigten Fotogrenzen", "", f"{result['confirmed']} bestätigte Referenzen ausgewertet; {result['unconfirmed']} noch offen.", "",
             "Dieser Vergleich bewertet ausschließlich die lokalen Heuristiken. Es wurden keine weiteren KI-Anfragen ausgeführt.", "",
             "| Ergebnis | Bilder |", "| --- | ---: |"]
    lines.extend(f"| {labels[k]} | {v} |" for k,v in result["outcomes"].items())
    if result.get("automatic_outcomes"):
        lines += ["", "## Empfohlene Automatik separat", "",
                  "Hohe Sicherheit für uniforme Canvas-Ränder und Story-Balken. Ein bloßer Vorschlag zählt hier nicht als automatischer Zuschnitt.", "",
                  "| Automatik-Ergebnis | Bilder |", "| --- | ---: |"]
        auto_labels = {"no_automatic_crop": "Kein automatischer Zuschnitt", "unnecessary_automatic_crop": "Unnötiger automatischer Zuschnitt",
                       "special_case_automatic_crop": "Automatischer Zuschnitt bei Sonderfall", "near_reference": "Automatischer Zuschnitt nahe Referenz",
                       "needs_correction": "Automatischer Zuschnitt benötigt Korrektur", "invalid_automatic_crop": "Ungültiger automatischer Zuschnitt"}
        lines.extend(f"| {auto_labels[k]} | {v} |" for k,v in result["automatic_outcomes"].items())
    lines += ["", "Als nahe gelten Grenzen mit höchstens 3 Pixeln oder 0,5 % der jeweiligen Bildseite Abweichung. "
              "Die Toleranz dient dem Vergleich und bestätigt keine sichere Automatik. Fehlende Fotofläche und zusätzlich enthaltene Fläche sind separat in scores.json erfasst; Körperteile werden dabei nicht gesondert bewertet.", "",
              "## Einfluss ungenauer Handmarkierungen", "",
              "Die Referenzen sind per Hand gesetzte Sollgrenzen. Deshalb wird zusätzlich mit mehreren Kantentoleranzen verglichen. "
              "Kleine Unterschiede sind kein Beleg für einen fehlerhaften Zuschnitt.", "",
              f"{result['crop_proposals_compared']} vorhandene Vorschläge für bestätigte Zuschnitte werden in der folgenden Tabelle verglichen. Beibehaltene Originale und fehlende Vorschläge sind aus diesem Teilvergleich ausgeschlossen.", "",
              "| Erlaubte Abweichung je Kante | Vorschläge innerhalb der Toleranz |", "| --- | ---: |"]
    lines.extend(f"| {percent} % der Bildseite, mindestens 3 Pixel | {count} / {result['crop_proposals_compared']} |" for percent,count in result["edge_tolerance_sensitivity"].items())
    lines += ["", f"Bei {result['substantial_extra_area']} Vorschlägen liegen mehr als 10 % der vorgeschlagenen Fläche außerhalb des bestätigten Fotos. "
              f"Bei {result['substantial_missing_photo']} Vorschlägen fehlen mehr als 5 % der bestätigten Fotofläche. Ein Bild kann beide Kriterien erfüllen. "
              "Diese Flächenvergleiche beziehen sich auf die Handmarkierungen und sind keine automatische Freigabe.", "",
              "## Größte Abweichungen", "", "| Bild | Fehlende Fotofläche | Zusätzliche Fläche im Vorschlag |", "| --- | ---: | ---: |"]
    errors = sorted((r for r in result["results"] if r["outcome"] == "needs_correction"), key=lambda r:r["missing_photo_fraction"]+r["extra_area_fraction"], reverse=True)
    lines.extend(f"| {r['filename'].replace('|',' ')} | {r['missing_photo_fraction']:.1%} | {r['extra_area_fraction']:.1%} |" for r in errors[:15])
    if result["rejected"]:
        lines += ["", f"{len(result['rejected'])} unpassende oder ungültige Referenzen wurden ausgeschlossen."]
    (output / "SCORE_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("confirmed", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = score(json.loads(args.manifest.read_text(encoding="utf-8")), json.loads(args.confirmed.read_text(encoding="utf-8")))
    write_report(result, args.output)
    print(json.dumps({k:v for k,v in result.items() if k in {"confirmed", "unconfirmed", "outcomes", "by_layout"}}, ensure_ascii=False))
