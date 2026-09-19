"""Summarize hard controls and audit old joint rankings under explicit seeds."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_ecls_pae_joint import digest, joint_scores, read, screening, write
from scripts.score_ecls_hard_controls import OUT, ecls_scores


def main():
    report = read(OUT / "report.json")
    diagnosis = read(OUT / "numeric_diagnosis.json")
    old_path = ROOT / "results/ecls_pae_joint_dev_v1/scores.json"
    old = read(old_path)
    audits = []
    for c in report["components"]:
        rows = [r for r in old["rows"] if r["scaffold"] == c["scaffold"]]
        for run in c["runs"]:
            arrays = []
            for state in ("complex", "apo"):
                path = OUT / "arrays" / f"{c['scaffold']}_{run['seed']}_{state}.npz"
                if digest(path) != report["arrays_sha256"][path.relative_to(ROOT).as_posix()]:
                    raise ValueError("Scoring array changed")
                with np.load(path) as z:
                    arrays.append(z["baseline_logp"])
                    indices = z["h3_indices"]
            values = ecls_scores(*arrays, indices, [r["h3_sequence"] for r in rows])
            legacy = np.array([r["ecls"] for r in rows])
            selected = np.array([r["entity_type"] == "candidate" for r in rows])
            y = [r["target"] for r in rows if r["entity_type"] == "candidate"]
            pae = [r["pae"] for r in rows if r["entity_type"] == "candidate"]
            audits.append({"scaffold": c["scaffold"], "seed": run["seed"],
                "max_abs_ECLS_difference_all_14": float(np.max(np.abs(values - legacy))),
                "candidate_midranks_unchanged": bool(np.array_equal(rankdata(values[selected]), rankdata(legacy[selected]))),
                "ecls_recall_at_3_of_12": screening(-values[selected], y, .2),
                "joint_recall_at_3_of_12": screening(joint_scores(values[selected], pae), y, .2)})
    audit = {"status": "development_seed_semantics_audit", "original_scores_sha256": digest(old_path), "rows": audits,
        "max_abs_ECLS_difference": max(a["max_abs_ECLS_difference_all_14"] for a in audits),
        "all_candidate_midranks_unchanged": all(a["candidate_midranks_unchanged"] for a in audits),
        "mean_ecls_recall": float(np.mean([a["ecls_recall_at_3_of_12"] for a in audits])),
        "mean_joint_recall": float(np.mean([a["joint_recall_at_3_of_12"] for a in audits]))}
    write(OUT / "legacy_seed_audit.json", audit)
    ci = report["descriptive_component_bootstrap_95_interval"]
    lines = ["# ECLS 近天然困难对照开发结果", "", "完成 6 组件 × 20 条对照 × 3 个显式种子的评分，以及抗原序列置换与 complex/apo 刚体变换检查。", "",
        "控制仅交换两个不同残基，保持 H3 长度和组成；120 条控制不等于 120 个独立样本。", "",
        "|组件|天然 ECLS − 近天然对照均值|三种子效应极差|天然胜过对照比例（种子均值，同分半分）|原始严格不变性检查|",
        "|---|---:|---:|---:|---|"]
    for c in report["components"]:
        fraction = float(np.mean([r["native_beats_control_fraction"] for r in c["runs"]]))
        passed = all(r["invariance_pass"] for r in c["runs"])
        lines.append(f"|{c['scaffold']}|{c['mean_native_advantage']:.6f}|{c['advantage_seed_range']:.2e}|{fraction:.1%}|{'通过' if passed else '未通过，见数值诊断'}|")
    lines += ["", f"组件等权平均优势 **{report['mean_native_advantage']:.6f}**，正向 **{report['positive_components']}/6**。组件 bootstrap 描述性 95% 区间 [{ci[0]:.6f}, {ci[1]:.6f}]。只有六个已查看的开发组件，不能用该区间宣称独立验证成功。", "",
        "## 实现检查", "", "预设绝对容差 1e-5，同时检查 H3 各位置/氨基酸 log probability 和 ECLS；任何超限均保留为原始失败。抗原序列置换属于 backbone-only 模式的负对照，不代表特异性验证。", ""]
    for c in diagnosis["components"]:
        lines.append(f"- {c['scaffold']}：双精度刚体变换最大 log probability 差 {max(c['float64_rigid_max_abs_H3_logp'].values()):.2e}；最大 ECLS 差 {c['float64_rigid_max_abs_ecls']:.2e}；原生优势相对单精度变化 {c['difference_from_float32_native_advantage']:.2e}。这是事后数值诊断，不修改最初判定。")
    max_shuffle = max(r["invariance"]["antigen_shuffle_max_abs_logp"] for c in report["components"] for r in c["runs"])
    lines += ["", f"抗原序列置换最大 log probability 差：{max_shuffle:.2e}。", "",
        "## 历史 seed=0 的复现审计", "", "原 ProteinMPNN CLI 将 seed=0 解释为随机生成种子，旧运行的实际种子未记录。这次绕过 CLI，显式设置 Python/NumPy/PyTorch 种子（包括 literal 0），使用 CPU、关闭骨架噪声并启用确定性算法。", "",
        f"对旧 84 条序列在三个显式种子下重新计算 ECLS，最大分差 {audit['max_abs_ECLS_difference']:.2e}；所有设计候选组的中秩保持一致：{audit['all_candidate_midranks_unchanged']}。新分数下 ECLS 召回 {audit['mean_ecls_recall']:.1%}、等权联合召回 {audit['mean_joint_recall']:.1%}（每组件选 3/12）。旧缓存及文件未修改。", "",
        "## 证据边界与下一步", "", "近天然对照没有已知结合标签；结果只检验固定骨架上的序列相容性。种子重复不是新的生物学样本。当前仍没有新增独立组件，也没有证明联合筛选优于 PAE/Ridge。", "",
        "先根据困难对照效果决定是否继续结构环境敏感性开发；接触/远端配对目前只有 2 个组件可用。投稿主张保留为评分的能力与边界；正式提交前仍需独立验证、清晰创新性和完整可复现材料。JCIM 非 OA 为投稿意向，非接收保证。", "",
        "复现顺序：score_ecls_hard_controls.py → diagnose_ecls_numeric_invariance.py → report_ecls_hard_controls.py。细节见本目录 protocol.json、report.json、numeric_diagnosis.json、legacy_seed_audit.json；评分数组有 SHA-256。", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(audit)


if __name__ == "__main__":
    main()
