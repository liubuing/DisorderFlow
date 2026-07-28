#!/usr/bin/env python3
"""Build or verify the provenance-aware state-contrast dataset shards."""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from disorderflow.data_factory import build_dataset, verify_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/statecontrast/statecontrast_factory_v1.yml")
    parser.add_argument("--output", default="data/statecontrast_factory_v1")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--refresh-audit", action="store_true")
    args = parser.parse_args()

    output = (ROOT / args.output).resolve()
    if args.verify_only:
        result = verify_dataset(output)
        if args.refresh_audit:
            (output / "audit.json").write_text(
                json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    else:
        if args.refresh_audit:
            parser.error("--refresh-audit requires --verify-only")
        config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
        result = build_dataset(config, ROOT, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
