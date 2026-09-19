"""Render all prespecified cells without choosing the best seed or stratum."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_aayl_learnability import OUT, digest, load, metrics
from scripts.build_aayl_original_endpoint_cohort import freeze


def main():
    report = json.loads((OUT / "report.json").read_text())
    data, rows, changes = load()
    manifest = json.loads((OUT / "splits.json").read_text())
    predictions = json.loads((OUT / "predictions.json").read_text())["records"]
    # Verify saved metrics independently against candidate-level predictions.
    groups = {}
    for p in predictions:
        key = (p["family"], p["budget"], p["seed"], p["model"])
        groups.setdefault(key, {})[p["poi"]] = p
    splits = {(s["family"], s["budget"], s["seed"]): s for s in manifest["splits"]}
    for r in report["metrics"]:
        s = splits[(r["family"], r["budget"], r["seed"])]
        g = groups[(r["family"], r["budget"], r["seed"], r["model"])]
        assert set(g) == {rows[i]["poi"] for i in s["test"]}
        indices = [i for i in s["test"] if len(changes[i]) == r["degree"] and
                   (r["stratum"] == "all" or ((i in s["covered"]) == (r["stratum"] == "covered")))]
        v = metrics([rows[i]["endpoint"] for i in indices], [g[rows[i]["poi"]]["prediction"] for i in indices])
        for k, value in v.items():
            assert r[k] is None if value is None else np.isclose(r[k], value)
    freeze(OUT / "verification.json", {"metric_cells_recomputed": len(report["metrics"]),
        "prediction_records": len(predictions), "unique_candidate_count": len(rows),
        "artifact_sha256": {f: digest(OUT / f) for f in ["protocol.json", "splits.json", "report.json", "fits.json", "predictions.json", "embedding_receipt.json"]},
        "compatibility": "first launch stopped before fitting: Torch weights_only rejected argparse.Namespace; rerun with this stdlib class allowlisted; unchanged protocol and benchmark script",
        "runner_sha256": digest(ROOT / "scripts/run_aayl_learnability.py")})
    lines = ["# AAYL 亲和力排序：第一阶段可学习性结果", "", "日期：2026-09-16。已完成开发实验；不是独立验证或投稿达标证明。", "",
        "## 结论", "", "当前单突变监督向双、三突变推广的信号较弱。冻结小型 ESM2 表示没有稳定超过位置编码 Ridge，预设复杂模型投入门槛未通过。暂不据此开展昂贵结构融合或大模型训练。这个结果只约束本轮数据、表示和任务，不能排除其他方法可行。", "",
        "## 冻结设计与信息条件", "", "原始 AlphaSeq 完整三重复读数共 508 个候选；端点为 9 − median(log10 estimated Kd[nM])。单突变训练，双突变、三突变分别测试。训练预算 20、50、全部；20/50 各 5 个预设随机子集，全部预算只运行一次。89/55 个单突变不支持 100 标签预算。", "",
        "三种监督基线：位置×氨基酸 Ridge、同编码 RBF 核 Ridge、冻结 ESM2 t6 8M 的 H3 平均表示＋Ridge。模型选择仅使用训练内五折 CV，目标中心化和 ESM 特征缩放均在折内拟合。三种原始 ProteinMPNN 分数为零样本对照，信息条件不同。ECLS 此处用 apo NLL − complex NLL（越高越好）。", "",
        "候选序列无训练/测试重叠，按谱系独立训练。无 WT 实验锚点，不报告校准亲和力，也不混合不同突变阶数评价。蛋白预训练序列暴露未排除。", "",
        "## 全部单突变预算：主要结果", "", "Spearman 越高越好；top20 recall 表示预测前 ceil(20%×n) 中找回的实测前同等数量候选比例，边界并列按分数权重处理。", "",
        "| 谱系 | 测试 | n | 方法 | Spearman | top20 recall | 富集倍数 |", "|---|---|---:|---|---:|---:|---:|"]
    for r in report["primary"]:
        lines.append(f"| {r['family']} | {r['degree']} 突变 | {r['n']} | {r['model']} | {r['spearman']:.3f} | {r['top20_recall']:.3f} | {r['enrichment']:.2f} |")
    lines += ["", "## 学习曲线", "", "下表为全体相应测试候选的 Spearman 均值 [最小, 最大]。子集之间共享测试集；该范围不是置信区间，不构成五次独立验证。", "",
              "| 谱系 | 测试 | 标签预算 | 方法 | Spearman 均值 [范围] |", "|---|---|---|---|---|"]
    for family in sorted(data["structures"]):
        for degree in [2, 3]:
            for budget in [20, 50, "all"]:
                for model in ["position_ridge", "position_rbf", "esm2_ridge"]:
                    v = [r["spearman"] for r in report["metrics"] if r["family"] == family and r["degree"] == degree and r["budget"] == budget and r["model"] == model and r["stratum"] == "all"]
                    lines.append(f"| {family} | {degree} | {budget} | {model} | {np.mean(v):.3f} [{min(v):.3f}, {max(v):.3f}] |")
    lines += ["", "## 替换覆盖分层", "", "covered 要求每一个具体位点＋替换氨基酸都在训练出现过。以下全部预算结果显示纯组合外推样本很少，尤其 AAYL51；n=3 的相关性不宜用于科学结论。", "",
              "| 谱系 | 阶数 | 分层 | n | Ridge ρ | ESM2 ρ |", "|---|---|---|---:|---:|---:|"]
    for family in sorted(data["structures"]):
        for degree in [2, 3]:
            for stratum in ["covered", "uncovered"]:
                values = {r["model"]: r for r in report["metrics"] if r["family"] == family and r["degree"] == degree and r["budget"] == "all" and r["stratum"] == stratum}
                a, b = values["position_ridge"], values["esm2_ridge"]
                lines.append(f"| {family} | {degree} | {stratum} | {a['n']} | {a['spearman']:.3f} | {b['spearman']:.3f} |")
    lines += ["", "## 门槛与下一步", "",
        "事先设定的资源投入门槛：ESM 在四个谱系×阶数组合的全预算 Spearman 均严格高于 Ridge，且至少三个组合的 top20 recall 不下降。本轮未通过：ESM 仅在两个三突变组的相关性略有提高，双突变组下降约 0.180 和 0.169；top20 recall 仅一个组合提高。该门槛是资源决策规则，不是显著性检验。", "",
        "建议下一轮先独立冻结一个更符合当前数据覆盖的任务：单＋双突变训练、三突变测试，同预算比较简单模型与 ESM，并继续区分已见/未见替换。双突变标签现已暴露，这依然只能是开发实验，不能称新独立验证。", "",
        "同时核查重复测量一致性和无读数选择偏差，增加原始标签及更广的替换覆盖。只有出现稳定可学习信号，才检查可靠性加权或抗原界面特征的增量贡献。若仍无稳定增益，应优先扩展数据或调整任务。", "",
        "目前只有一个抗原、两个谱系；完整读数筛选排除了大量候选。没有跨抗原能力、新方法优势或一区/二区录用的证据。JCIM 非 OA 目标保留为后续选择，冻结 ECLS 论文结果未改动。", "",
        "## 复现与核查", "", "运行 `python scripts/run_aayl_learnability.py --prepare`、`python scripts/run_aayl_learnability.py --run`，再运行 `python scripts/report_aayl_learnability.py`。协议及已有输出有内容一致性保护，不允许静默覆盖。", "",
        "ESM 表示共 508×320；使用本机缓存权重，CPU 推理，未微调蛋白模型。协议记录权重、输入和脚本哈希。PyTorch 兼容入口仅允许反序列化标准库 argparse.Namespace，未关闭 weights_only 安全加载。", "",
        f"已从 {len(predictions)} 条逐候选预测重算核对 {len(report['metrics'])} 个指标单元。拆分、覆盖和并列边界测试通过。机器可读产物位于 `results/aayl_learnability_v1/`。", ""]
    path = ROOT / "docs/AAYL_LEARNABILITY_STAGE1_RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
