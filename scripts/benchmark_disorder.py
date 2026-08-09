#!/usr/bin/env python3
"""Deprecated synthetic disorder smoke test.

This legacy filename is retained only to redirect callers. The removed workflow
was not a CAID benchmark: it assigned proxy labels and fabricated an AF2 pLDDT
comparison. Publication evaluation must use benchmark_caid.py with homology
isolation metadata and real benchmark labels.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parser.error(
        "deprecated synthetic workflow; use scripts/benchmark_caid.py with "
        "--cluster-manifest and --training-clusters"
    )


if __name__ == "__main__":
    main()
