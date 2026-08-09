#!/usr/bin/env python
"""Apply the frozen metadata gate to a future exact-new snapshot discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


MINIMUM_ENTRIES = 12


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(discovery_path, output, now=None):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite readiness decision: {output}")
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    if discovery.get("classification") != "model_free_exact_ID_discovery":
        raise ValueError("Readiness requires an exact-ID, model-free discovery artifact")
    ids = sorted(set(discovery.get("candidate_entry_ids", [])))
    passed = len(ids) >= MINIMUM_ENTRIES
    payload = {
        "schema_version": 1,
        "status": "future_snapshot_ready" if passed else "future_snapshot_not_ready",
        "checked_at_utc": (now or datetime.now(timezone.utc)).astimezone(
            timezone.utc).isoformat().replace("+00:00", "Z"),
        "classification": "metadata_only_future_confirmatory_readiness",
        "source_discovery": str(discovery_path),
        "source_discovery_sha256": sha256(discovery_path),
        "minimum_exact_new_entries": MINIMUM_ENTRIES,
        "observed_exact_new_entries": len(ids),
        "ready": passed,
        "candidate_entry_ids_sha256": hashlib.sha256(
            "\n".join(ids).encode("ascii")).hexdigest(),
        "decision": (
            "eligible_to_freeze_a_new_confirmatory_protocol_before structure access"
            if passed else
            "wait_for_a_later_snapshot; structure_and_checkpoint_access_forbidden"),
        "claim_boundary": "metadata readiness only; not a performance result",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.discovery, args.output), indent=2))


if __name__ == "__main__":
    main()
