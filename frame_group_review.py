"""Conservative layout matches for explicitly reviewed crop templates."""
from __future__ import annotations


def normalized_box(record, key):
    size, box = record.get("original_size"), record.get(key)
    if not size or len(size) != 2 or min(size) <= 0 or not box or len(box) != 4:
        return None
    w, h = size
    if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
        return None
    return [value / size[index % 2] for index, value in enumerate(box)]


def build_group_plan(records, source_hash, limit=12):
    source = next((r for r in records if r.get("source_hash") == source_hash), None)
    if not source or source.get("user_decision") not in {"manual", "accept"}:
        return {}
    template = normalized_box(source, "display_bbox")
    detected = normalized_box(source, "candidate_bbox")
    layout = source.get("layout_class")
    if not template or not detected or layout not in {
        "uniform_canvas", "vertical_story_canvas", "nested_app_screenshot"
    }:
        return {}
    sw, sh = source["original_size"]
    targets = []
    for record in records:
        if not record.get("source_hash") or record["source_hash"] == source_hash:
            continue
        if record.get("user_decision", "auto") != "auto" or record.get("layout_class") != layout:
            continue
        if sorted(record.get("layout_flags", [])) != sorted(source.get("layout_flags", [])):
            continue
        candidate = normalized_box(record, "candidate_bbox")
        if not candidate or max(abs(a-b) for a, b in zip(candidate, detected)) > .02:
            continue
        w, h = record["original_size"]
        if abs((w/h) / (sw/sh) - 1) > .01:
            continue
        box = [round(v * (w if i % 2 == 0 else h)) for i, v in enumerate(template)]
        if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
            continue
        targets.append({"source_hash": record["source_hash"], "bbox": box})
        if len(targets) >= limit:
            break
    return {"source_hash": source_hash, "source_bbox": list(source["display_bbox"]), "targets": targets}


def validate_group_selection(records, plan, selected):
    """Reject stale previews or attempts to include an unpreviewed image."""
    if not plan or not selected:
        raise ValueError("no reviewed selection")
    current = build_group_plan(records, plan.get("source_hash"))
    if current != plan:
        raise ValueError("group preview is stale")
    choices = {r["source_hash"]: r for r in plan["targets"]}
    if len(set(selected)) != len(selected) or any(key not in choices for key in selected):
        raise ValueError("invalid group selection")
    return [choices[key] for key in selected]
