#!/usr/bin/env python3
"""Audit prose documents against docs/EVIDENCE_REGISTRY.yml.

The registry lists canonical values and the values they superseded. This
script is read-only: it never edits documents or frozen artifacts. It reports
every stale value that appears without an allowed marker nearby, so a document
that quotes a voided number must also say that the number is void.

Exit codes:
  0  no unmarked stale values (or --fail-on-stale not set)
  1  unmarked stale values found and --fail-on-stale was set
  2  registry missing or unreadable
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a declared dependency
    print("PyYAML is required to read docs/EVIDENCE_REGISTRY.yml", file=sys.stderr)
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "docs" / "EVIDENCE_REGISTRY.yml"
DEFAULT_TARGETS = (
    "README.md",
    "PUBLICATION_PROTOCOL.md",
    "docs/**/*.md",
    "release/**/*.md",
    "publication/**/*.md",
)
DEFAULT_EXCLUDES = (
    "docs/archive/**",
    "release/zenodo_v1/upload/**",
    "release/ecls_v1/prepared/**",
    "**/node_modules/**",
)
MARKER_WINDOW = 25


def load_registry(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    if not isinstance(registry, dict):
        raise ValueError(f"Registry is not a mapping: {path}")
    return registry


def compile_value_pattern(value: str) -> re.Pattern:
    """Match a literal value with numeric/word boundaries where appropriate."""
    escaped = re.escape(value)
    if re.fullmatch(r"[0-9.]+", value):
        return re.compile(rf"(?<![\d.]){escaped}(?![\d])")
    return re.compile(escaped, re.IGNORECASE)


def iter_documents(root: Path, targets, excludes):
    seen = set()
    for pattern in targets:
        try:
            paths = list(root.glob(pattern))
        except OSError:
            # A broken symlink (for example publication/ on a clean checkout)
            # is not a reason to fail the audit.
            continue
        for path in paths:
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if any(path.match(pattern) for pattern in excludes):
                continue
            if relative in seen:
                continue
            seen.add(relative)
            yield path


def nearby_marker(lines, index: int, markers) -> bool:
    """Check the hit line, the preceding window, and the document header."""
    lower_markers = [marker.lower() for marker in markers]
    window_start = max(0, index - MARKER_WINDOW)
    candidates = lines[window_start:index + 1] + lines[:15]
    for line in candidates:
        lowered = line.lower()
        if any(marker in lowered for marker in lower_markers):
            return True
    return False


def find_stale_hits(text: str, stale_entries, markers):
    lines = text.splitlines()
    hits = []
    for entry in stale_entries:
        patterns = [(value, compile_value_pattern(value)) for value in entry["values"]]
        context = entry.get("context_required")
        context_pattern = re.compile(context) if context else None
        for index, line in enumerate(lines):
            if context_pattern is not None and not context_pattern.search(line):
                continue
            for value, pattern in patterns:
                if not pattern.search(line):
                    continue
                hits.append({
                    "stale_id": entry["id"],
                    "value": value,
                    "line": index + 1,
                    "marked": nearby_marker(lines, index, markers),
                    "context": line.strip()[:220],
                })
    return hits


def check_sources(root: Path, registry: dict):
    missing = []
    for claim_id, claim in registry.get("claims", {}).items():
        source = claim.get("source")
        if not source:
            continue
        if not (root / source).exists():
            missing.append({"claim": claim_id, "source": source})
    return missing


def build_report(root: Path, registry: dict, targets, excludes):
    markers = registry.get("allowed_markers", [])
    stale_entries = registry.get("stale_values", [])
    documents = []
    for path in iter_documents(root, targets, excludes):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")
        hits = find_stale_hits(text, stale_entries, markers)
        if not hits:
            continue
        documents.append({
            "path": path.relative_to(root).as_posix(),
            "unmarked": [hit for hit in hits if not hit["marked"]],
            "marked": [hit for hit in hits if hit["marked"]],
        })
    return {
        "registry": str(DEFAULT_REGISTRY.relative_to(ROOT)),
        "documents_with_stale_values": len(documents),
        "unmarked_hits": sum(len(doc["unmarked"]) for doc in documents),
        "marked_hits": sum(len(doc["marked"]) for doc in documents),
        "documents": documents,
        "missing_sources": check_sources(root, registry),
    }


def print_text(report: dict) -> None:
    print(f"registry: {report['registry']}")
    print(
        "documents with stale values: "
        f"{report['documents_with_stale_values']} | "
        f"unmarked hits: {report['unmarked_hits']} | "
        f"marked hits: {report['marked_hits']}"
    )
    for document in report["documents"]:
        unmarked = document["unmarked"]
        if not unmarked:
            continue
        print(f"\n{document['path']}")
        for hit in unmarked:
            print(
                f"  L{hit['line']}: [{hit['stale_id']}] {hit['value']} "
                f"-> {hit['context']}"
            )
    if report["missing_sources"]:
        print("\nmissing claim sources:")
        for item in report["missing_sources"]:
            print(f"  [{item['claim']}] {item['source']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="Glob to scan; repeatable. Defaults to README, protocol, docs, release, publication.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Glob to exclude; repeatable.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--fail-on-stale",
        action="store_true",
        help="Exit 1 when an unmarked stale value is found.",
    )
    parser.add_argument(
        "--fail-on-missing-source",
        action="store_true",
        help="Exit 1 when a claim source path is missing.",
    )
    args = parser.parse_args(argv)

    registry_path = args.registry if args.registry.is_absolute() else ROOT / args.registry
    if not registry_path.exists():
        print(f"Registry not found: {registry_path}", file=sys.stderr)
        return 2
    registry = load_registry(registry_path)

    targets = tuple(args.target) if args.target else DEFAULT_TARGETS
    excludes = tuple(args.exclude) if args.exclude else DEFAULT_EXCLUDES
    report = build_report(ROOT, registry, targets, excludes)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_text(report)

    failed = False
    if args.fail_on_stale and report["unmarked_hits"]:
        failed = True
    if args.fail_on_missing_source and report["missing_sources"]:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
