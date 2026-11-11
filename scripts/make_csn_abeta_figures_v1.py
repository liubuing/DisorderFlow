#!/usr/bin/env python3
"""Generate the three CSN Abeta manuscript figures from frozen tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "csn_abeta_v1"
FIGDIR = ROOT / "publication" / "figures"
LINEAGE_ORDER = ["L1", "L2", "L3", "L4", "L5", "L7", "L9", "L8", "L6"]
TIGHT = {"L1", "L2", "L3", "L4", "L5", "L7"}
COLORS = {"R0": "#2c7fb8", "R1": "#e6550d"}


def load_data():
    entries = pd.read_csv(RESULTS / "abeta_entries.csv")
    pairs = pd.read_csv(RESULTS / "rmsd_pairs_primary.csv")
    free = pd.read_csv(RESULTS / "free_state_models.csv")
    return entries, pairs, free


def register_for(sequence: str) -> str:
    if sequence.startswith("DAEFRH"):
        return "R1"
    return "R0"


def figure1(entries: pd.DataFrame):
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 5.4))
    cohort = {
        "L1": ["2IPU", "2R0W"], "L2": ["3BAE", "3BKJ"], "L3": ["3IFL", "3IFN"],
        "L4": ["3IFO"], "L5": ["3IFP"], "L6": ["4HIX", "4OJF", "4ONF"],
        "L7": ["5CSZ"], "L8": ["5W3P"], "L9": ["6CO3"],
    }
    index = entries.set_index("pdb")
    for row, lineage in enumerate(LINEAGE_ORDER):
        for column, pdb in enumerate(cohort[lineage]):
            sequence = str(index.loc[pdb, "peptide_sequences"]).split("|")[0]
            register = register_for(sequence)
            box = FancyBboxPatch(
                (column - 0.34, row - 0.26), 0.68, 0.52,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=COLORS[register], edgecolor="white", linewidth=1.2,
                zorder=3,
            )
            ax.add_patch(box)
            ax.text(column, row - 0.02, pdb, ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white", zorder=4)
            ax.text(column, row + 0.42, sequence, ha="center", va="center",
                    fontsize=7, color="#333333", zorder=4)
    ax.set_yticks(range(len(LINEAGE_ORDER)))
    ax.set_yticklabels(LINEAGE_ORDER, fontsize=10)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["entry 1", "entry 2", "entry 3", "entry 4"], fontsize=9)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.6, len(LINEAGE_ORDER) - 0.2)
    ax.invert_yaxis()
    ax.set_title("Aβ N-terminal cohort: 9 lineages, 14 entries", fontsize=13)
    ax.set_xlabel("entries within lineage", fontsize=9)
    handles = [plt.Line2D([0], [0], marker="s", color="w",
                          markerfacecolor=COLORS[key], markersize=11, label=label)
               for key, label in [("R0", "A2-start (AEFRH…)"),
                                  ("R1", "D1-start (DAEFRH…)")]]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "csn_abeta_fig1_cohort.png", dpi=300)
    fig.savefig(FIGDIR / "csn_abeta_fig1_cohort.pdf")
    plt.close(fig)


def figure2(pairs: pd.DataFrame):
    cross = pairs[pairs["pair_type"] == "cross_lineage"]
    within = pairs[pairs["pair_type"] == "within_lineage"]
    matrix = pd.DataFrame(np.nan, index=LINEAGE_ORDER, columns=LINEAGE_ORDER)
    for (first, second), group in cross.groupby(
            [cross["lineage_i"], cross["lineage_j"]]):
        value = float(np.median(group["rmsd"]))
        matrix.loc[first, second] = value
        matrix.loc[second, first] = value
    for (first, second), group in within.groupby(
            [within["lineage_i"], within["lineage_j"]]):
        value = float(np.mean(group["rmsd"]))
        if np.isnan(matrix.loc[first, second]):
            matrix.loc[first, second] = value
            matrix.loc[second, first] = value

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    data = matrix.values.astype(float)
    image = ax.imshow(data, cmap="viridis_r", vmin=0, vmax=3.5)
    for row in range(len(LINEAGE_ORDER)):
        for column in range(len(LINEAGE_ORDER)):
            value = data[row, column]
            if np.isnan(value):
                ax.text(column, row, "—", ha="center", va="center",
                        fontsize=9, color="#888888")
            else:
                ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if value > 1.6 else "black")
    ax.set_xticks(range(len(LINEAGE_ORDER)))
    ax.set_yticks(range(len(LINEAGE_ORDER)))
    ax.set_xticklabels(LINEAGE_ORDER, fontsize=9)
    ax.set_yticklabels(LINEAGE_ORDER, fontsize=9)
    ax.set_title("Median cross-lineage Cα RMSD (Å)", fontsize=12)
    ax.add_patch(Rectangle((6.5, 6.5), 2, 2, fill=False, edgecolor="#d62728",
                           linewidth=2))
    ax.add_patch(Rectangle((-0.5, -0.5), 6, 6, fill=False, edgecolor="#2ca02c",
                           linewidth=2))
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RMSD (Å)", fontsize=9)
    fig.text(0.5, 0.02,
             "Green: tight cluster (L1–L5, L7)   |   Red: distinct block (L8, L6)   |   L9: boundary",
             ha="center", fontsize=8, color="#333333")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIGDIR / "csn_abeta_fig2_lineage_matrix.png", dpi=300)
    fig.savefig(FIGDIR / "csn_abeta_fig2_lineage_matrix.pdf")
    plt.close(fig)


def figure3(free: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    order = ["1AMC", "1BJB", "1BJC", "1ZE7", "1ZE9", "2BP4"]
    primary = {"1AMC", "1BJB", "1BJC", "1ZE7"}
    rng = np.random.default_rng(20261111)
    for position, entry in enumerate(order):
        subset = free[free["nmr_entry"] == entry]
        jitter = rng.uniform(-0.12, 0.12, size=len(subset))
        color = "#2c7fb8" if entry in primary else "#bdbdbd"
        ax.scatter(np.full(len(subset), position) + jitter,
                   subset["min_tight_rmsd"], s=28, color=color, alpha=0.8,
                   edgecolor="white", linewidth=0.4)
    for level in (0.5, 1.0, 1.5):
        ax.axhline(level, color="#999999", linestyle="--", linewidth=0.8)
        ax.text(len(order) - 0.4, level + 0.05, f"{level:.1f} Å", fontsize=7,
                color="#666666", ha="right")
    near = free.loc[free.groupby("nmr_entry")["min_tight_rmsd"].idxmin()]
    near = near[near["min_tight_rmsd"] < 1.0]
    for _, row in near.iterrows():
        ax.annotate(f"{row['nmr_entry']} {row['min_tight_rmsd']:.2f} Å",
                    xy=(order.index(row["nmr_entry"]), row["min_tight_rmsd"]),
                    xytext=(-0.15, 0.55), textcoords="offset points",
                    fontsize=8, color="#d62728", ha="right")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="#2c7fb8", markersize=7,
                          label="aqueous (primary)"),
               plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="#bdbdbd", markersize=7,
                          label="sensitivity")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("best RMSD to bound tight cluster (Å)", fontsize=9)
    ax.set_title("Bound conformer in free-state NMR ensembles", fontsize=11)
    ax.set_ylim(0, 3.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    spread = free.groupby("nmr_entry")["min_tight_rmsd"].median().reindex(order)
    bar_colors = ["#2c7fb8" if entry in primary else "#bdbdbd" for entry in order]
    ax2.barh(range(len(order)), spread.values, color=bar_colors, alpha=0.8,
             height=0.6)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(order, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("median best RMSD (Å)", fontsize=9)
    ax2.set_title("Per-entry median", fontsize=10)
    ax2.set_xlim(0, 3.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "csn_abeta_fig3_free_state.png", dpi=300)
    fig.savefig(FIGDIR / "csn_abeta_fig3_free_state.pdf")
    plt.close(fig)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    entries, pairs, free = load_data()
    figure1(entries)
    figure2(pairs)
    figure3(free)
    print(f"wrote figures to {FIGDIR}")


if __name__ == "__main__":
    main()
