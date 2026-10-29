#!/usr/bin/env python
"""Build native-controlled VH:VL:IDP epitope complex fold panels."""
# 本脚本为非 Aβ 靶标构建"原生对照 + 设计候选"的三元复合物折叠面板：
# 每个靶标先放一条 native_control（参考 PDB 的 VH/VL 可变区 + 表位肽），
# 再加入该靶标全部设计候选的 VH/VL 序列；统一以 VH:VL:IDP 三段序列
# 写出 CSV 面板与折叠队列 FASTA，供复合物折叠（complex fold）测试
# 三元复合物可行性——仅为序列层面的合理性检验，不针对特定实验构象。

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "modules"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipeline"))

from analyze_shortlist_variable_regions import find_variable_region  # noqa: E402
from export_full_chain_constructs import pdb_chain_sequences  # noqa: E402


def main():
    # 主流程：读靶标清单与候选 FV 面板 → 逐靶标生成原生对照与候选行 →
    # 写 CSV 面板、折叠队列 FASTA（header 携带靶标/家族/链式标注）与摘要
    with open(
        PROJECT_ROOT / "outputs/non_abeta_idp_ensemble_prep_v1/ensemble_manifest.json",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)
    with open(
        PROJECT_ROOT / "outputs/non_abeta_idp_candidate_audit_v1/fv_fold_panel.csv",
        newline="", encoding="utf-8",
    ) as handle:
        panel = list(csv.DictReader(handle))
    out_dir = PROJECT_ROOT / "outputs/non_abeta_idp_complex_fold_panel_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for target in manifest["targets"]:
        chains = pdb_chain_sequences(PROJECT_ROOT / target["trimmed_reference_pdb"])
        heavy_sequence = "".join(row["aa"] for row in chains[target["heavy_chain"]])
        light_sequence = "".join(row["aa"] for row in chains[target["light_chain"]])
        native_heavy = find_variable_region(heavy_sequence, "heavy")["variable_seq"]
        native_light = find_variable_region(light_sequence, "light")["variable_seq"]
        rows.append({
            "construct_id": f"{target['reference_id']}_native",
            "target": target["target"],
            "antibody_family": target["reference_id"],
            "construct_type": "native_control",
            "vh_sequence": native_heavy,
            "vl_sequence": native_light,
            "antigen_sequence": target["epitope"],
            "source_candidate_uid": "",
        })
        for candidate in panel:
            if candidate["target"] != target["target"]:
                continue
            rows.append({
                "construct_id": candidate["candidate_uid"],
                "target": target["target"],
                "antibody_family": target["reference_id"],
                "construct_type": "designed_candidate",
                "vh_sequence": candidate["vh_sequence"],
                "vl_sequence": candidate["vl_sequence"],
                "antigen_sequence": target["epitope"],
                "source_candidate_uid": candidate["candidate_uid"],
            })

    with open(out_dir / "complex_fold_panel.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "complex_fold_queue.fasta", "w", encoding="ascii") as handle:
        for row in rows:
            handle.write(
                f">{row['construct_id']}|target={row['target']}|family={row['antibody_family']}|VH_VL_IDP\n"
                f"{row['vh_sequence']}:{row['vl_sequence']}:{row['antigen_sequence']}\n"
            )
    summary = {
        "schema_version": "nonabeta.complex_fold_panel.v1",
        "status": "pass",
        "constructs": len(rows),
        "native_controls": sum(row["construct_type"] == "native_control" for row in rows),
        "designed_candidates": sum(row["construct_type"] == "designed_candidate" for row in rows),
        "chains": "VH:VL:IDP_epitope",
        "claim_boundary": (
            "Sequence-only complex folding tests ternary complex plausibility. "
            "It does not condition on a specific experimental IDP conformer."
        ),
    }
    with open(out_dir / "complex_fold_panel_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Non-A-beta complex fold panel: constructs={len(rows)}")


if __name__ == "__main__":
    main()
