#!/usr/bin/env python3
"""Generate the ECLS ablation and replication figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FACT = ROOT / "results" / "ecls_factorial_v1"
FACT_M2 = ROOT / "results" / "ecls_factorial_v1_m2"
SCALE = ROOT / "results" / "ecls_cross_arch_v1" / "m3_scale_eval" / "scale_curve.json"
RANKER = FACT / "task_ranker.json"
M0 = ROOT / "results" / "publication" / "h3_ecls_temporal_final_v1" / "results.json"
M2 = ROOT / "results" / "ecls_cross_arch_v1" / "antifold" / "antifold_summary.json"
M3 = ROOT / "results" / "ecls_cross_arch_v1" / "m3_full10_ecls" / "results.json"
OUT = ROOT / "publication" / "figures"
COLORS = {"M0": "#4c72b0", "M2": "#dd8452", "M3": "#55a868"}
LABELS = {"M0": "ProteinMPNN", "M2": "AntiFold", "M3": "M3 scratch"}


def figure_context():
    m0 = json.loads((FACT / "summary.json").read_text(encoding="utf-8"))["summary"]
    m2 = json.loads((FACT_M2 / "summary.json").read_text(encoding="utf-8"))["summary"]
    models = ["M0", "M2", "M3"]
    context = [m0["M0"]["mean_delta_context"], m2["M2"]["mean_delta_context"],
               m0["M3"]["mean_delta_context"]]
    peptide = [m0["M0"]["mean_delta_peptide"], m2["M2"]["mean_delta_peptide"],
               m0["M3"]["mean_delta_peptide"]]
    fractions = [m0["M0"]["fraction_context_positive"],
                 m2["M2"]["fraction_context_positive"],
                 m0["M3"]["fraction_context_positive"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    x = np.arange(len(models))
    ax.bar(x, context, color=[COLORS[m] for m in models], width=0.62)
    for index, value in enumerate(context):
        ax.text(index, value + 0.006, f"{value:.3f}", ha="center", fontsize=9)
        ax.text(index, 0.012, f"{fractions[index]:.0%} +", ha="center",
                fontsize=8, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in models], fontsize=9)
    ax.set_ylabel("NLL(apo) − NLL(complex)  (nats)", fontsize=9)
    ax.set_title("Peptide presence is read", fontsize=11)
    ax.set_ylim(0, max(context) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.bar(x, peptide, color=[COLORS[m] for m in models], width=0.62)
    for index, value in enumerate(peptide):
        ax.annotate(f"{value:.1e}", (index, value), textcoords="offset points",
                    xytext=(0, 4 if value >= 0 else -12), ha="center", fontsize=8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in models], fontsize=9)
    ax.set_ylabel("NLL(peptide shuffle) − NLL(native)  (nats)", fontsize=9)
    ax.set_title("Peptide identity is not read", fontsize=11)
    ax.set_ylim(-2.5e-7, 2.5e-7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "ecls_fig1_context_identity.png", dpi=300)
    fig.savefig(OUT / "ecls_fig1_context_identity.pdf")
    plt.close(fig)


def figure_scale():
    curve = json.loads(SCALE.read_text(encoding="utf-8"))
    epochs = sorted(int(key) for key in curve)
    means = [curve[str(epoch)]["mean_ecls_advantage"] for epoch in epochs]
    lower = [curve[str(epoch)]["mean_ecls_advantage"]
             - curve[str(epoch)]["ci95"][0] for epoch in epochs]
    upper = [curve[str(epoch)]["ci95"][1]
             - curve[str(epoch)]["mean_ecls_advantage"] for epoch in epochs]
    passes = [curve[str(epoch)]["gate_pass"] for epoch in epochs]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = ["#55a868" if passed else "#bdbdbd" for passed in passes]
    ax.errorbar(epochs, means, yerr=[lower, upper], fmt="none",
                ecolor="#555555", capsize=3, linewidth=1)
    ax.scatter(epochs, means, s=70, color=colors, edgecolor="#333333", zorder=3)
    for epoch, mean, passed in zip(epochs, means, passes):
        ax.annotate(f"{mean:.3f}\n{'pass' if passed else 'fail'}",
                    (epoch, mean), textcoords="offset points", xytext=(6, 6),
                    fontsize=7)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("training epoch", fontsize=9)
    ax.set_ylabel("mean ECLS advantage", fontsize=9)
    ax.set_title("ECLS is training-sensitive, not monotonic", fontsize=11)
    ax.set_xticks(epochs)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "ecls_fig2_scale_curve.png", dpi=300)
    fig.savefig(OUT / "ecls_fig2_scale_curve.pdf")
    plt.close(fig)


def figure_ranker():
    report = json.loads(RANKER.read_text(encoding="utf-8"))
    families = sorted(report)
    features = ["ecls", "composition", "ecls_plus_composition"]
    labels = ["ECLS", "composition", "ECLS + composition"]
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    width = 0.24
    positions = np.arange(len(families))
    for index, (feature, label, color) in enumerate(zip(features, labels, colors)):
        means = [report[family]["mean"][feature] for family in families]
        errors = [report[family]["sd"][feature] for family in families]
        ax.bar(positions + (index - 1) * width, means, width=width,
               yerr=errors, capsize=3, color=color, label=label)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(families, fontsize=9)
    ax.set_ylabel("held-out Spearman with endpoint", fontsize=9)
    ax.set_title("Task-specific pairwise ranker on AAYL", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "ecls_fig3_task_ranker.png", dpi=300)
    fig.savefig(OUT / "ecls_fig3_task_ranker.pdf")
    plt.close(fig)


def units():
    m0 = {unit["unit"]: unit["mean_ecls_advantage"]
          for unit in json.loads(M0.read_text(encoding="utf-8"))["inference_units"]}
    m2 = {unit["unit"]: unit["mean_ecls_advantage"]
          for unit in json.loads(M2.read_text(encoding="utf-8"))["unit_results"]}
    m3 = {unit["unit"]: unit["mean_ecls_advantage"]
          for unit in json.loads(M3.read_text(encoding="utf-8"))["inference_units"]}
    return m0, m2, m3


def figure_agreement():
    m0, m2, m3 = units()
    keys = sorted(m0)
    pairs = [("M0", "M2", m0, m2), ("M0", "M3", m0, m3), ("M2", "M3", m2, m3)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
    for ax, (left, right, first, second) in zip(axes, pairs):
        x = np.array([first[key] for key in keys])
        y = np.array([second[key] for key in keys])
        r = float(np.corrcoef(x, y)[0, 1])
        ax.scatter(x, y, s=45, color="#4c72b0", alpha=0.8, edgecolor="white")
        low = min(x.min(), y.min())
        high = max(x.max(), y.max())
        ax.plot([low, high], [low, high], color="#999999", linestyle="--",
                linewidth=0.8)
        ax.set_xlabel(LABELS[left], fontsize=9)
        ax.set_ylabel(LABELS[right], fontsize=9)
        ax.set_title(f"r = {r:.2f}", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Per-antigen-cluster ECLS agreement", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "ecls_fig4_cluster_agreement.png", dpi=300)
    fig.savefig(OUT / "ecls_fig4_cluster_agreement.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figure_context()
    figure_scale()
    figure_ranker()
    figure_agreement()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
