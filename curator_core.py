#!/usr/bin/env python3
"""Shared configuration, persistence, cache-fingerprint and export helpers.

This module is deliberately dependency-light so both the Gradio UI and the
backend can use exactly the same normalization rules.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional


def _git_describe_version() -> str:
    """Return the installed source version without maintaining it manually.

    A tagged Git checkout reports its nearest tag, commit distance and dirty
    state. Source archives and installations without Git remain usable and
    receive a neutral fallback instead.
    """
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            check=True,
            text=True,
            timeout=1,
        )
        version = result.stdout.strip()
        return version or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


APP_VERSION = _git_describe_version()
RUN_CONFIG_SCHEMA_VERSION = "run-config-v1"
CAPTION_POLICY_SCHEMA_VERSION = "caption-policy-v2"
RUN_MANIFEST_SCHEMA_VERSION = "run-manifest-v1"


def natural_sort_key(value: Any) -> tuple:
    """Case-insensitive human ordering for names containing numbers.

    Example: Bild 2 sorts before Bild 10 instead of after it.
    """
    text = os.path.basename(str(value or "")).casefold()
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text))


def normalize_asset_id(value: Any) -> int:
    """Return one positive integer asset ID or ``0`` for legacy/invalid data."""
    if isinstance(value, Mapping):
        for key in ("asset_id", "source_asset_id"):
            normalized = normalize_asset_id(value.get(key))
            if normalized > 0:
                return normalized
        return 0
    try:
        normalized = int(str(value or "").strip())
    except (TypeError, ValueError):
        return 0
    return normalized if normalized > 0 else 0


def asset_id_for_row(row: Optional[Mapping[str, Any]]) -> int:
    """Read the authoritative project image ID from a pipeline row."""
    return normalize_asset_id(row or {})


def assign_asset_id(row: MutableMapping[str, Any], asset_id: Any) -> int:
    """Write the canonical ID and its legacy alias to one mutable row."""
    normalized = normalize_asset_id(asset_id)
    if normalized > 0:
        row["asset_id"] = normalized
        # Retained while old caches/stages are migrated. New logic never uses
        # this alias as a separate identity.
        row["source_asset_id"] = normalized
    return normalized


def asset_id_key(value: Any) -> str:
    """Stable JSON/dictionary key for one project asset."""
    normalized = normalize_asset_id(value)
    return str(normalized) if normalized > 0 else ""


def row_identity_key(row: Optional[Mapping[str, Any]]) -> str:
    """Authoritative row key; asset ID first, legacy fallback only for migration."""
    asset_key = asset_id_key(row or {})
    if asset_key:
        return asset_key
    legacy = row or {}
    for key in ("profile_image_id", "file_hash", "original_path", "original_filename"):
        value = str(legacy.get(key) or "").strip()
        if value:
            return value
    return ""


def normalize_training_target(value: Any) -> str:
    v = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if v in {"z_image_base", "zimage", "z_image"}:
        return "z_image_base"
    if v in {"krea2", "krea_2", "krea2_character", "krea_2_character"}:
        return "krea2"
    return "ernie"


def normalize_pipeline_mode(value: Any) -> str:
    v = str(value or "").strip().lower()
    return "profile_then_caption" if v == "profile_then_caption" else "single_pass"


def normalized_caption_policy(policy: Optional[Mapping[str, Any]]) -> Dict[str, bool]:
    """Return a stable, complete boolean policy snapshot.

    Every user-facing caption switch is authoritative. Presets only populate
    these values in the UI and never override them here.
    """
    raw = dict(policy or {})
    keys = sorted(k for k in raw if str(k).startswith("include_"))
    return {str(k): bool(raw.get(k)) for k in keys}


def normalize_run_config_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    data = dict(payload or {})
    target = normalize_training_target(data.get("TRAINING_TARGET") or data.get("CAPTION_PROFILE"))
    data["TRAINING_TARGET"] = target
    data["CAPTION_PROFILE"] = "krea2_character" if target == "krea2" else target
    data["PIPELINE_MODE"] = normalize_pipeline_mode(data.get("PIPELINE_MODE"))
    data["CAPTION_POLICY"] = normalized_caption_policy(data.get("CAPTION_POLICY"))
    data["RUN_CONFIG_SCHEMA_VERSION"] = RUN_CONFIG_SCHEMA_VERSION
    data["ACTIVE_CAPTION_POLICY_SNAPSHOT"] = dict(data["CAPTION_POLICY"])
    data["RUN_CONFIG_FINGERPRINT"] = stable_hash({
        "schema": RUN_CONFIG_SCHEMA_VERSION,
        "target": target,
        "pipeline": data["PIPELINE_MODE"],
        "caption_policy": data["CAPTION_POLICY"],
        "variable_feature_mode": data.get("VARIABLE_FEATURE_CAPTION_MODE", "canonical_deviations"),
        "audit_model": data.get("AI_MODEL", ""),
        "profile_model": data.get("SUBJECT_PROFILE_NORMALIZER_MODEL", ""),
        "caption_model": data.get("KREA_CAPTION_MODEL", ""),
        "repair_model": data.get("KREA_CAPTION_REPAIR_MODEL", ""),
    })
    return data


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def replace_file_with_retry(tmp: str, path: str) -> None:
    """Replace ``path`` while tolerating short Windows/cloud-sync locks.

    Dropbox, antivirus scanners and preview/indexing services can briefly open a
    JSON file without delete sharing.  On Windows that makes an otherwise valid
    ``os.replace`` fail with WinError 5 or 32.  Retry the atomic operation first;
    only after the lock persists use a direct, fully flushed copy of the already
    completed temporary file.
    """
    delays = [0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00, 1.25, 1.50]
    last_error: Optional[OSError] = None
    for delay in delays:
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_error = exc
        except OSError as exc:
            # WinError 5 = access denied; WinError 32 = sharing violation.
            if getattr(exc, "winerror", None) not in {5, 32} and exc.errno not in {13, 16}:
                raise
            last_error = exc
        time.sleep(delay)

    # Some sync clients block rename/delete sharing longer than normal writes.
    # The temp file is complete and fsynced, so a direct copy is a safe final
    # compatibility fallback and avoids losing an otherwise finished audit.
    try:
        with open(tmp, "rb") as source, open(path, "wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            try:
                os.fsync(target.fileno())
            except OSError:
                pass
        os.remove(tmp)
        return
    except OSError as fallback_error:
        if last_error is not None:
            raise PermissionError(
                f"Could not replace or update '{path}' after repeated retries. "
                "A cloud-sync client, antivirus scanner or another program may "
                "still be holding the file open."
            ) from fallback_error
        raise


def atomic_write_json(path: str, payload: Any) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        replace_file_with_retry(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def atomic_write_text(path: str, text: str, *, newline: Optional[str] = None) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    try:
        with open(tmp, "w", encoding="utf-8", newline=newline) as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        replace_file_with_retry(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def caption_relevant_profile_snapshot(
    profile: Optional[Mapping[str, Any]],
    image_id: str,
) -> Dict[str, Any]:
    """Extract only data capable of changing a caption.

    Bucket layout, priority flags, diagnostics and normalizer notes are omitted
    intentionally so they do not invalidate valid captions.
    """
    p = dict(profile or {})
    identity = dict(p.get("identity_markers") or {})
    relevant_identity = {
        "glasses": identity.get("glasses", {}),
        "freckles": identity.get("freckles", {}),
        "tattoo_inventory": identity.get("tattoo_inventory", []),
        "piercing_inventory": identity.get("piercing_inventory", []),
        "piercing_baseline": identity.get("piercing_baseline", []),
    }
    per_image = p.get("per_image_traits") or {}
    image_traits = per_image.get(image_id, {}) if isinstance(per_image, Mapping) else {}
    return {
        "stable_identity": p.get("stable_identity", {}),
        "canonical_features": p.get("canonical_features", {}),
        "profile_policies": p.get("profile_policies", {}),
        "profile_variability_stats": p.get("profile_variability_stats", {}),
        "identity_markers": relevant_identity,
        "image_traits": image_traits,
    }


def build_caption_fingerprint(
    *,
    source_key: str,
    image_id: str,
    training_target: str,
    trigger_word: str,
    prompt_version: str,
    primary_model: str,
    primary_reasoning: str,
    repair_enabled: bool,
    repair_model: str,
    repair_reasoning: str,
    variable_feature_mode: str,
    caption_policy: Mapping[str, Any],
    subject_profile: Optional[Mapping[str, Any]],
    crop_variant: str,
) -> str:
    payload = {
        "schema": CAPTION_POLICY_SCHEMA_VERSION,
        "source": source_key,
        "image_id": image_id,
        "target": normalize_training_target(training_target),
        "trigger": trigger_word,
        "prompt": prompt_version,
        "primary": [primary_model, primary_reasoning],
        "repair": [bool(repair_enabled), repair_model, repair_reasoning],
        "variable_feature_mode": variable_feature_mode,
        "caption_policy": normalized_caption_policy(caption_policy),
        "profile": caption_relevant_profile_snapshot(subject_profile, image_id),
        "crop_variant": crop_variant or "original",
    }
    return stable_hash(payload)


@dataclass
class OutputTransaction:
    """Stage export folders and atomically swap them after validation.

    Previous completed output folders are untouched until ``commit``. If the
    process is cancelled or crashes, the staging directory can simply be
    removed on the next run.
    """

    output_root: str
    final_dirs: Mapping[str, str]
    expected_train_count: Optional[int] = None
    run_kind: str = "export"
    tx_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    staging_root: str = field(init=False)
    staged_dirs: Dict[str, str] = field(init=False, default_factory=dict)
    original_values: Dict[str, str] = field(init=False, default_factory=dict)
    manifest_path: str = field(init=False)

    def __post_init__(self) -> None:
        self.output_root = os.path.abspath(self.output_root)
        self.staging_root = os.path.join(self.output_root, f"._export_staging_{self.tx_id}")
        self.manifest_path = os.path.join(self.output_root, "_run_manifest.json")
        for global_name, final_path in self.final_dirs.items():
            staged = os.path.join(self.staging_root, os.path.basename(final_path.rstrip(os.sep)))
            self.staged_dirs[global_name] = staged

    def begin(self, namespace: MutableMapping[str, Any]) -> None:
        self.cleanup_abandoned(self.output_root)
        os.makedirs(self.staging_root, exist_ok=True)
        for global_name, staged in self.staged_dirs.items():
            os.makedirs(staged, exist_ok=True)
            self.original_values[global_name] = str(namespace.get(global_name, ""))
            namespace[global_name] = staged
        self._write_manifest("running")

    def _write_manifest(self, status: str, **extra: Any) -> None:
        # Preserve the run context written at process startup (effective config,
        # input fingerprint, schema/dependency versions) while the export
        # transaction updates its own status and counts.
        payload: Dict[str, Any] = {}
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if isinstance(existing, dict):
                payload.update(existing)
        except Exception:
            pass
        payload.update({
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "app_version": APP_VERSION,
            "transaction_id": self.tx_id,
            "run_kind": self.run_kind,
            "status": status,
            "staging_root": self.staging_root,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        payload.update(extra)
        atomic_write_json(self.manifest_path, payload)

    def validate(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for global_name, staged in self.staged_dirs.items():
            image_count = 0
            text_count = 0
            if os.path.isdir(staged):
                for name in os.listdir(staged):
                    low = name.lower()
                    if low.endswith((".jpg", ".jpeg", ".png", ".webp")):
                        image_count += 1
                    elif low.endswith(".txt"):
                        text_count += 1
            counts[f"{global_name}_images"] = image_count
            counts[f"{global_name}_texts"] = text_count
            if global_name == "TRAIN_READY_DIR" and image_count != text_count:
                raise RuntimeError(
                    f"transaction validation failed: train-ready images={image_count}, captions={text_count}"
                )
        if self.expected_train_count is not None:
            actual = counts.get("TRAIN_READY_DIR_images", 0)
            if actual < int(self.expected_train_count):
                raise RuntimeError(
                    f"transaction validation failed: expected at least {self.expected_train_count} train images, got {actual}"
                )
        return counts

    def commit(self, namespace: MutableMapping[str, Any]) -> Dict[str, int]:
        counts = self.validate()
        backups: Dict[str, str] = {}
        try:
            for global_name, final_path in self.final_dirs.items():
                staged = self.staged_dirs[global_name]
                final_path = os.path.abspath(final_path)
                backup = f"{final_path}._previous_{self.tx_id}"
                if os.path.exists(backup):
                    shutil.rmtree(backup, ignore_errors=True)
                if os.path.exists(final_path):
                    os.replace(final_path, backup)
                    backups[final_path] = backup
                os.replace(staged, final_path)
                namespace[global_name] = final_path
            for backup in backups.values():
                shutil.rmtree(backup, ignore_errors=True)
            shutil.rmtree(self.staging_root, ignore_errors=True)
            self._write_manifest("complete", counts=counts)
            return counts
        except Exception as exc:
            # Roll back any already-swapped directory.
            for final_path, backup in backups.items():
                if os.path.exists(backup):
                    if os.path.exists(final_path):
                        shutil.rmtree(final_path, ignore_errors=True)
                    os.replace(backup, final_path)
            self._write_manifest("failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            for global_name, original in self.original_values.items():
                namespace[global_name] = original

    def cancel(self, namespace: MutableMapping[str, Any], error: str = "cancelled") -> None:
        for global_name, original in self.original_values.items():
            namespace[global_name] = original
        shutil.rmtree(self.staging_root, ignore_errors=True)
        self._write_manifest("cancelled", error=error)

    @staticmethod
    def cleanup_abandoned(output_root: str) -> int:
        count = 0
        if not os.path.isdir(output_root):
            return 0
        for name in os.listdir(output_root):
            if name.startswith("._export_staging_"):
                shutil.rmtree(os.path.join(output_root, name), ignore_errors=True)
                count += 1
        return count
