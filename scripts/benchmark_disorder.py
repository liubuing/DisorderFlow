#!/usr/bin/env python3
"""Deprecated synthetic disorder smoke test.

This legacy filename is retained only to redirect callers. The removed workflow
was not a CAID benchmark: it assigned proxy labels and fabricated an AF2 pLDDT
comparison. Publication evaluation must use benchmark_caid.py with homology
isolation metadata and real benchmark labels.
"""

import argparse


def main():
    # 弃用占位入口：无条件触发 parser.error 并给出替代脚本指引。
    # 旧流程使用代理标签并伪造 AF2 pLDDT 对比，不属于 CAID 基准；
    # 发表级评估必须改用 benchmark_caid.py（带同源性隔离元数据与真实标签）。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parser.error(
        "deprecated synthetic workflow; use scripts/benchmark_caid.py with "
        "--cluster-manifest and --training-clusters"
    )


if __name__ == "__main__":
    main()
