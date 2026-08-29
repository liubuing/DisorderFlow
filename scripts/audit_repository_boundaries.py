#!/usr/bin/env python
"""Inventory repository zones and report architecture debt without moving history."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_ABSOLUTE = re.compile(r"[A-Za-z]:[/\\]")


def tracked_files():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True)
    return [Path(line) for line in result.stdout.splitlines() if line]


def classify(path, zones):
    text = path.as_posix()
    for zone, prefixes in zones.items():
        for prefix in prefixes:
            prefix = str(prefix).replace("\\", "/")
            if text == prefix or text.startswith(prefix):
                return zone
    return "unclassified"


def audit(config_path, output_path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    files = tracked_files()
    counts = {}
    root_scripts = []
    absolute_path_sources = []
    core_output_imports = []
    for path in files:
        zone = classify(path, config["zones"])
        counts[zone] = counts.get(zone, 0) + 1
        if len(path.parts) == 1 and path.suffix in {".py", ".sh", ".bat"}:
            root_scripts.append(path.as_posix())
        if path.suffix in {".py", ".yml", ".yaml", ".sh", ".bat", ".ps1"}:
            try:
                text = (ROOT / path).read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if WINDOWS_ABSOLUTE.search(text) or "/mnt/c/" in text:
                absolute_path_sources.append(path.as_posix())
            if zone == "core_package" and (
                    "reviewer_outputs" in text or "outputs/" in text):
                core_output_imports.append(path.as_posix())
    missing_entrypoints = [
        path for path in config["stable_entrypoints"] if not (ROOT / path).is_file()
    ]
    payload = {
        "schema_version": 1,
        "status": "repository_boundary_audit_complete",
        "zone_counts": counts,
        "tracked_files": len(files),
        "grandfathered_root_script_count": len(root_scripts),
        "grandfathered_root_scripts": sorted(root_scripts),
        "absolute_path_source_count": len(absolute_path_sources),
        "absolute_path_sources": sorted(absolute_path_sources),
        "core_output_import_violations": sorted(set(core_output_imports)),
        "missing_stable_entrypoints": missing_entrypoints,
        "hard_failures": [
            *[f"missing entrypoint: {path}" for path in missing_entrypoints],
            *[f"core imports generated output: {path}"
              for path in sorted(set(core_output_imports))],
        ],
        "decision": (
            "keep_history_in_place; route_new_work_through_scripts_and_core_package"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/repository_layout.yml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = ROOT / (args.output or Path(config["output"]))
    result = audit(config_path, output)
    print(json.dumps({
        key: result[key] for key in (
            "status", "tracked_files", "grandfathered_root_script_count",
            "absolute_path_source_count", "hard_failures", "decision")
    }, indent=2))


if __name__ == "__main__":
    main()
