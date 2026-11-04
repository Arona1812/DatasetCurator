"""Shows where headshots, medium and full-body shots drop out in a finished run.

Reads dataset_audit_<trigger>.jsonl from the curated_<trigger> folder and
prints one count per stage and shot type. Only shot types, statuses and
reasons are printed, never image descriptions.

    python benchmark/shot_type_report.py "C:\\path\\to\\curated_Name"
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

from shot_types import audit_shot_type, face_height_ratio, reconcile_shot_type

SHOTS = ("headshot", "medium", "full_body")
TRAIN_ROLES = {"core", "variation", "body_reference"}


def load_rows(folder: str) -> list:
    paths = sorted(glob.glob(os.path.join(folder, "dataset_audit_*.jsonl")))
    if not paths:
        sys.exit(f"No dataset_audit_*.jsonl in {folder}")
    rows = []
    with open(paths[-1], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def text(row: dict, key: str) -> str:
    return str(row.get(key, "") or "").strip().lower()


def stage_counts(rows: list) -> dict:
    stages = defaultdict(Counter)
    for row in rows:
        shot = text(row, "shot_type") or "(none)"
        stages["audited"][shot] += 1
        if row.get("is_smart_crop"):
            stages["  of which headshot crops"][shot] += 1
        status = text(row, "base_status")
        stages[f"status {status or '(none)'}"][shot] += 1
        if row.get("duplicate_of"):
            stages[f"duplicate ({text(row, 'duplicate_method') or '?'})"][shot] += 1
        if text(row, "arcface_flag") == "hard":
            stages["face outlier (ArcFace hard)"][shot] += 1
        if row.get("overlay_crop_available"):
            stages["replaced by overlay crop"][shot] += 1
        role = text(row, "identity_cluster_role")
        if status != "reject":
            stages[f"profile role {role if role else '(none)'}"][shot] += 1
            if role in TRAIN_ROLES or row.get("priority_image"):
                stages["training candidates"][shot] += 1
        if row.get("selected"):
            stages["selected"][shot] += 1
        if text(row, "output_bucket") == "train_ready":
            stages["01_train_ready"][shot] += 1
    return stages


STAGE_ORDER = (
    "audited", "  of which headshot crops", "status keep", "status review", "status reject",
    "duplicate", "face outlier", "replaced by overlay crop", "profile role",
    "training candidates", "selected", "01_train_ready",
)


def stage_rank(name: str) -> tuple:
    for index, prefix in enumerate(STAGE_ORDER):
        if name.startswith(prefix):
            return index, name
    return len(STAGE_ORDER), name


def print_table(stages: dict) -> None:
    shots = list(SHOTS) + sorted({s for c in stages.values() for s in c} - set(SHOTS))
    width = max(len(name) for name in stages) + 2
    print("".ljust(width) + "".join(s.rjust(11) for s in shots))
    for name in sorted(stages, key=stage_rank):
        counter = stages[name]
        print(name.ljust(width) + "".join(str(counter.get(s, 0)).rjust(11) for s in shots))


def print_subtypes(rows: list) -> None:
    print("\nframe_subtype per shot_type (audited):")
    pairs = Counter((text(r, "shot_type"), text(r, "frame_subtype") or "-") for r in rows)
    for (shot, subtype), n in sorted(pairs.items()):
        print(f"  {shot or '(none)':<10} {subtype:<20} {n}")


def print_reject_reasons(rows: list) -> None:
    print("\nTop reject reasons per shot_type:")
    for shot in SHOTS:
        reasons = Counter()
        for row in rows:
            if text(row, "shot_type") != shot or text(row, "base_status") != "reject":
                continue
            raw = row.get("local_override_reasons") or row.get("issues") or ""
            items = raw if isinstance(raw, list) else str(raw).split("|")
            for item in items:
                item = str(item).strip()
                if item:
                    reasons[item.split(":")[0][:60]] += 1
        top = ", ".join(f"{r} ({n})" for r, n in reasons.most_common(6)) or "-"
        print(f"  {shot:<10} {top}")


def print_reconciled(rows: list) -> None:
    print("\nshot_type from the audit -> shot_type in the next run (face size and subtype):")
    moves = Counter((audit_shot_type(r) or "(none)", reconcile_shot_type(r) or "(none)") for r in rows)
    for (before, after), n in sorted(moves.items()):
        print(f"  {before:<10} -> {after:<10} {n}")
    after = Counter(reconcile_shot_type(r) for r in rows if text(r, "base_status") != "reject")
    print("  usable (not reject) after correction: " + ", ".join(f"{s} {after.get(s, 0)}" for s in SHOTS))


def print_face_heights(rows: list) -> None:
    print("\nface height / image height per audit shot_type and subtype (median, n):")
    groups = defaultdict(list)
    for row in rows:
        height = face_height_ratio(row)
        if height is not None:
            groups[(audit_shot_type(row), text(row, "frame_subtype") or "-")].append(height)
    for (shot, subtype), values in sorted(groups.items()):
        values.sort()
        print(f"  {shot or '(none)':<10} {subtype:<20} {values[len(values) // 2]:.2f}  ({len(values)})")


def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = load_rows(folder)
    print_table(stage_counts(rows))
    print_subtypes(rows)
    print_reject_reasons(rows)
    print_face_heights(rows)
    print_reconciled(rows)


if __name__ == "__main__":
    main()
