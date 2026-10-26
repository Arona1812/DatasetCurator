"""Build an offline reference review; never calls an API or changes sources."""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from html import escape
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw
try:
    from pillow_heif import register_heif_opener
except ImportError:
    register_heif_opener = None

from frame_cleanup import DetectorSettings, FRAME_DETECTOR_SCHEMA_VERSION, IMAGE_EXTENSIONS, analyze_frame_cleanup, automatic_crop_bbox, file_sha1, load_bounded_rgb


def build_review(source: Path, output: Path, previous: Path | None = None,
                 selection: Path | None = None, blind: bool = False):
    if register_heif_opener is not None:
        register_heif_opener(thumbnails=False)
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Place the benchmark output outside the source image folder")
    output.mkdir(parents=True, exist_ok=True)
    prior = json.loads(previous.read_text(encoding="utf-8")) if previous and previous.exists() else []
    if isinstance(prior, dict):
        prior = prior.get("records", [])
    known = {item["filename"] for item in prior}
    selected = json.loads(selection.read_text(encoding="utf-8")) if selection else None
    entries = selected.get("images", []) if selected else [
        {"source_path": str(path)} for path in sorted(source.iterdir(), key=lambda p: p.name.casefold())]
    records, failures, seen = [], [], set()
    settings = DetectorSettings()
    for entry in entries:
        path = Path(entry["source_path"]).resolve()
        if path.parent == output or path.parent in output.parents:
            raise ValueError("Place benchmark output outside every selected source folder")
        if not path.is_file() or path.suffix.lower() not in (*IMAGE_EXTENSIONS, ".heic", ".heif"):
            continue
        try:
            digest = file_sha1(str(path))
            if entry.get("id") and entry["id"] != digest:
                raise ValueError("Selected source changed after sampling")
            if digest in seen:
                raise ValueError("Duplicate source content in selection")
            seen.add(digest)
            result = analyze_frame_cleanup(str(path), str(output / "_cache"), digest, settings=settings)
            preview = load_bounded_rgb(str(path), 900)
            buffer = io.BytesIO()
            preview.save(buffer, format="JPEG", quality=85)
            bbox = result.get("candidate_bbox")
            width, height = result["original_size"]
            # These are tentative layout families, never instructions to apply a crop.
            normalized = [round(v / (width if i % 2 == 0 else height) * 1000) for i, v in enumerate(bbox)] if bbox else None
            group_key = (width, height, result.get("layout_class"), result.get("crop_type"),
                         tuple(round(v / 20) for v in normalized) if normalized else ())
            records.append({"id": digest, "filename": path.name, "source_path": str(path),
                            "size": [width, height], "new_reference": path.name not in known,
                            "local_bbox": bbox, "layout": result.get("layout_class", "unknown"),
                            "crop_type": result.get("crop_type", "unknown"),
                            "confidence_level": result.get("confidence_level", "low"),
                            "cohort": entry.get("cohort", "reference"),
                            "automatic_bbox": automatic_crop_bbox(result, "auto_high_review_medium", ["uniform_canvas", "story_bars"]),
                            "group_key": repr(group_key), "reference_confirmed": False,
                            "reference_decision": None, "reference_bbox": None,
                            "preview": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")})
        except Exception as exc:
            failures.append({"filename": path.name, "error": type(exc).__name__})
    families = defaultdict(list)
    for record in records:
        families[record.pop("group_key")].append(record)
    groups = []
    for index, members in enumerate(families.values(), 1):
        for record in members:
            record["group"] = index
        if len(members) > 1:
            groups.append({"group": index, "count": len(members), "files": [r["filename"] for r in members]})
    manifest = {"schema": "frame-reference-v1", "source": str(source),
                "review_id": str(output), "blind": blind,
                "detector_schema": FRAME_DETECTOR_SCHEMA_VERSION,
                "selection_notes": selected.get("notes", "") if selected else "",
                "records": [{k:v for k,v in r.items() if k != "preview"} for r in records],
                "tentative_groups": groups, "failures": failures}
    (output / "references.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    template = Path(__file__).with_name("frame_benchmark_template.html").read_text(encoding="utf-8")
    payload = json.dumps({**manifest, "records": records}, ensure_ascii=False).replace("<", "\\u003c")
    (output / "review.html").write_text(template.replace("__REFERENCE_DATA__", payload), encoding="utf-8")
    new = [r for r in records if r["new_reference"]]
    for start in range(0, len(new), 16):
        sheet = Image.new("RGB", (1200, 1200), "#202329")
        draw = ImageDraw.Draw(sheet)
        for slot, record in enumerate(new[start:start + 16]):
            x, y = (slot % 4) * 300, (slot // 4) * 300
            preview = load_bounded_rgb(record["source_path"], 260)
            preview.thumbnail((280, 250))
            sheet.paste(preview, (x + (300-preview.width)//2, y+28))
            draw.text((x+8, y+5), f"{start+slot+1}: {record['filename'][:29]}", fill="white")
            draw.text((x+8, y+280), f"{record['size']} / {record['layout']}", fill="white")
        sheet.save(output / f"new_references_{start//16+1}.jpg")
    counts = Counter(r["layout"] for r in records)
    cohorts = Counter(r["cohort"] for r in records)
    intro = (f"{len(records)} Bilder für die separate Referenzprüfung: "
             f"{cohorts['unconfirmed_frames']} bisher unbestätigte Rahmenbilder und "
             f"{cohorts['control_candidates']} zusätzliche Kontrollkandidaten. Vorschläge bleiben bis zur Bestätigung verborgen. "
             "Die Kontrollbilder sind noch nicht als rahmenfrei bestätigt. Einige Rahmenbilder waren bereits in früheren Versuchen sichtbar. "
             if blind else f"{len(records)} Bilder, davon {len(new)} seit dem bisherigen Vergleich hinzugekommen. ")
    summary = (f"# Rahmenreferenzen\n\n{intro}"
               f"{len(failures)} Lesefehler. Alle Sollentscheidungen sind noch unbestätigt.\n\n"
               "[Referenzen prüfen](review.html). Die Vorschläge stammen ausschließlich aus der lokalen Erkennung. "
               "Es wurden keine Bilder hochgeladen und keine API-Aufrufe ausgeführt.\n\n"
               "## Lokale Layout-Einteilung\n\n" + "\n".join(f"- {escape(k)}: {v}" for k,v in counts.items()) +
               f"\n\n{len(groups)} mögliche Layout-Gruppen mit mehreren Bildern. Diese Gruppen müssen visuell geprüft werden; "
               "gleichartige Detektorfehler können ebenfalls ähnliche Vorschläge erzeugen.\n")
    (output / "REPORT.md").write_text(summary, encoding="utf-8")
    return {"images": len(records), "new": len(new), "failures": len(failures), "groups": len(groups), "layouts": dict(counts)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--selection", type=Path, help="Frozen JSON sampling manifest with images/source_path/id/cohort")
    parser.add_argument("--blind", action="store_true", help="Hide detector proposals until a reference is confirmed")
    args = parser.parse_args()
    print(json.dumps(build_review(args.source, args.output, args.previous, args.selection, args.blind), ensure_ascii=False))
