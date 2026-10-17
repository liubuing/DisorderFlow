"""Figure 1 for the PAE analogue-transfer manuscript (v30 frozen artifacts).

Panel A: CA-trace view of the first evaluation component by identifier
(1lp9_E_F_C; no selection by effect size), TCR beta / TCR alpha / peptide with
the native CDR3-beta region highlighted. Panel B: per-family recall at the
20% budget. Panel C: component-equal means.

Reads only frozen v30 artifacts; no model inference. Emits PNG (300 dpi),
PDF, SVG and a plot-data JSON recording every plotted number.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = "results/pae_analog_evaluation_v30"
PDB = "results/pae_analog_structures_v26/pdb/1lp9_E_F_C.pdb"
OUT = "publication/figures/pae_analog_figure1_v30"
STRUCT_INSTANCE = "1lp9_E_F_C"

INK = "#22252A"
ALPHA_C = "#B9BEC7"   # TCR alpha
BETA_C = "#3E6DA8"    # TCR beta
REGION_C = "#E87A2C"  # CDR3-beta region
PEP_C = "#D9A21B"     # peptide
RANDOM_C = "#5A5F66"

THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
         "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
         "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
         "TYR": "Y", "VAL": "V", "MSE": "M"}


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_ca(path):
    """CA coordinates and one-letter sequences per chain (residue-deduped)."""
    chains = {}
    last = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            ch = line[21]
            key = line[22:27]
            if last.get(ch) != key:
                rec = chains.setdefault(ch, {"seq": [], "ca": []})
                rec["seq"].append(THREE.get(line[17:20].strip(), "X"))
                rec["ca"].append([float(line[30:38]), float(line[38:46]),
                                  float(line[46:54])])
                last[ch] = key
    for rec in chains.values():
        rec["seq"] = "".join(rec["seq"])
        rec["ca"] = np.asarray(rec["ca"])
    return chains


def main():
    evaluation = read_json(f"{ROOT}/evaluation.json")
    summary = read_json(f"{ROOT}/summary.json")
    geometry = read_json(f"{ROOT}/geometry.json")

    # evaluation components in frozen order (values follow this order everywhere)
    instances = list(evaluation["per_component"].keys())
    assert len(instances) == 6, instances

    ridge = {i: evaluation["per_component"][i]["ridge"] for i in instances}
    comp_only = {i: evaluation["per_component"][i]["composition_only"] for i in instances}
    nll = {i: evaluation["per_component"][i]["nll"] for i in instances}
    random = {i: evaluation["per_component"][i]["random"] for i in instances}

    mean = lambda d: float(np.mean(list(d.values())))
    means = {"ridge": mean(ridge), "comp_only": mean(comp_only),
             "nll": mean(nll), "random": mean(random)}
    assert abs(means["ridge"] - summary["ridge_recall"]["mean"]) < 1e-9
    assert abs(means["ridge"] - 0.5) < 1e-9 and abs(means["nll"] - 1 / 3) < 1e-9
    assert abs(means["random"] - 3 / 14) < 1e-9

    prereg_json = read_json("results/pae_analog_preregistration_v27/preregistration.json")
    fam_blob = json.dumps(prereg_json)
    fams = fam_blob[fam_blob.index('"eval_families"'):].split("]")[0]
    families = fams.split("[")[1].replace('"', "").split(",")
    labels = [f"{inst.split('_')[0]}\n({fam})" for inst, fam in zip(instances, families)]

    # ---- structure panel data ------------------------------------------------
    chains = parse_ca(PDB)
    gen = read_json("results/pae_analog_generation_v28/generation.json")
    entry = next(c for c in gen["components"] if c["instance"] == STRUCT_INSTANCE)
    region_seq = entry["native_region"]
    b_seq = chains["B"]["seq"]
    i0 = b_seq.index(region_seq)
    assert b_seq[i0:i0 + len(region_seq)] == region_seq
    region_idx = np.arange(i0, i0 + len(region_seq))
    geom = geometry["instances"][STRUCT_INSTANCE]

    # ---- figure --------------------------------------------------------------
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK, "ytick.color": INK})
    fig = plt.figure(figsize=(7.2, 3.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], wspace=0.32,
                          hspace=0.55, left=0.03, right=0.985,
                          top=0.93, bottom=0.13)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")

    def trace(rec, color, lw, z=0):
        ca = rec["ca"]
        ax3d.plot(ca[:, 0], ca[:, 1], ca[:, 2], color=color, lw=lw, zorder=z)

    trace(chains["A"], ALPHA_C, 1.1)
    trace(chains["B"], BETA_C, 1.1)
    reg = chains["B"]["ca"][region_idx]
    ax3d.plot(reg[:, 0], reg[:, 1], reg[:, 2], color=REGION_C, lw=3.2,
              zorder=5, solid_capstyle="round")
    ax3d.scatter(reg[:, 0], reg[:, 1], reg[:, 2], color=REGION_C, s=16, zorder=6)
    pep = chains["P"]["ca"]
    ax3d.scatter(pep[:, 0], pep[:, 1], pep[:, 2], color=PEP_C, s=46,
                 edgecolor="white", linewidth=0.5, zorder=7)

    r_mid = reg.mean(axis=0)
    p_mid = pep.mean(axis=0)
    ax3d.plot([r_mid[0], p_mid[0]], [r_mid[1], p_mid[1]], [r_mid[2], p_mid[2]],
              "--", color=RANDOM_C, lw=1.0, zorder=4)

    allpts = np.vstack([chains["A"]["ca"], chains["B"]["ca"], pep])
    mid = allpts.mean(axis=0)
    half = (allpts.max(axis=0) - allpts.min(axis=0)).max() * 0.42
    ax3d.set_xlim(mid[0] - half, mid[0] + half)
    ax3d.set_ylim(mid[1] - half, mid[1] + half)
    ax3d.set_zlim(mid[2] - half, mid[2] + half)
    ax3d.set_box_aspect((1, 1, 0.85))
    ax3d.view_init(elev=30, azim=-95)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis.line.set_color((1, 1, 1, 0))
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])

    ax3d.text2D(0.16, 0.78, "peptide", color="#9A7812", fontsize=8.5,
                weight="bold", transform=ax3d.transAxes)
    ax3d.text2D(0.05, 0.55, "CDR3-$\\beta$", color=REGION_C, fontsize=8.5,
                weight="bold", transform=ax3d.transAxes)
    ax3d.text2D(0.66, 0.30, "TCR-$\\alpha$", color="#6E747D", fontsize=8.5,
                weight="bold", transform=ax3d.transAxes)
    ax3d.text2D(0.10, 0.14, "TCR-$\\beta$", color=BETA_C, fontsize=8.5,
                weight="bold", transform=ax3d.transAxes)
    ax3d.text2D(0.02, 0.045,
                f"closest CDR3-$\\beta$-peptide\ndistance {geom[3]:.1f} A (frozen geometry)",
                color=RANDOM_C, fontsize=7, transform=ax3d.transAxes)
    ax3d.set_title("A   TCR-pMHC component 1lp9 (E F C)",
                   loc="left", fontsize=8.5, weight="bold", pad=2)

    # Panel B: per-family recall
    axb = fig.add_subplot(gs[0, 1])
    x = np.arange(len(instances))
    w = 0.36
    axb.bar(x - w / 2, [ridge[i] * 100 for i in instances], w,
            color=BETA_C, label="Ridge recipe")
    axb.bar(x + w / 2, [nll[i] * 100 for i in instances], w,
            color="#9AA3AD", label="ProteinMPNN NLL")
    axb.axhline(mean(random) * 100, ls="--", lw=1.1, color=RANDOM_C)
    axb.text(1.5, mean(random) * 100 + 2.6,
             f"random {mean(random)*100:.1f}%", color=RANDOM_C, fontsize=7.5,
             ha="center")
    axb.set_xticks(x, labels, fontsize=7.2)
    axb.set_ylabel("recall of teacher top-20% (%)")
    axb.set_ylim(0, 84)
    axb.set_title("B   per-family recall, 20% budget (3 of 14)",
                  loc="left", fontsize=8.5, weight="bold")
    axb.spines[["top", "right"]].set_visible(False)
    axb.legend(frameon=False, fontsize=7.3, loc="upper left",
               handlelength=1.2, borderaxespad=0.1)

    # Panel C: component-equal means
    axc = fig.add_subplot(gs[1, 1])
    bars = [("Ridge recipe", means["ridge"] * 100, BETA_C),
            ("composition-only", means["comp_only"] * 100, "#7FA3CE"),
            ("MPNN NLL", means["nll"] * 100, "#9AA3AD"),
            ("random (exact ties)", means["random"] * 100, "white")]
    y = np.arange(len(bars))[::-1]
    for yi, (lab, val, col) in zip(y, bars):
        axc.barh(yi, val, height=0.62, color=col,
                 edgecolor=RANDOM_C if col == "white" else "none", lw=0.9)
        axc.text(val + 1.4, yi, f"{val:.1f}%", va="center", fontsize=7.6)
    axc.set_yticks(y, [b[0] for b in bars], fontsize=7.8)
    axc.set_xlim(0, 66)
    axc.set_xlabel("component-equal mean recall (%)")
    axc.set_title("C   component-equal means", loc="left", fontsize=8.5,
                  weight="bold")
    axc.spines[["top", "right"]].set_visible(False)

    import os
    os.makedirs("publication/figures", exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{OUT}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)

    plot_data = {
        "figure": OUT,
        "generated_by": "scripts/plot_pae_analog_figure1_v30.py (deterministic, no RNG)",
        "frozen_sources": [f"{ROOT}/evaluation.json", f"{ROOT}/summary.json",
                           f"{ROOT}/geometry.json", PDB,
                           "results/pae_analog_generation_v28/generation.json"],
        "structure_panel": {
            "instance": STRUCT_INSTANCE,
            "selection_rule": "first evaluation component by identifier; no selection by effect size",
            "chains": {c: {"length": int(len(chains[c]["seq"]))} for c in chains},
            "region_zero_based_in_B": [int(region_idx[0]), int(region_idx[-1])],
            "region_sequence": region_seq,
            "geometry_instance_fields_as_frozen": [float(g) for g in geom],
        },
        "per_component": {i: {"ridge": ridge[i], "composition_only": comp_only[i],
                              "nll": nll[i], "random": random[i]}
                          for i in instances},
        "component_equal_means": means,
        "families": dict(zip(instances, families)),
    }
    with open(f"{OUT}_plot_data.json", "w", encoding="utf-8") as fh:
        json.dump(plot_data, fh, indent=1, ensure_ascii=False)
    print("wrote", OUT + ".{png,pdf,svg}", "and plot-data json")
    print("means:", {k: round(v * 100, 1) for k, v in means.items()})


if __name__ == "__main__":
    main()
