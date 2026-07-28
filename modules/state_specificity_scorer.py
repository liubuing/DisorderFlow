#!/usr/bin/env python
"""State-specific IDP antibody candidate design and scoring.

This module is a CPU-only MVP for the strategy:

    therapeutic epitope + disease-state specificity + paratope chemistry

It deliberately avoids depending on a fixed antibody template. The generated
sequence is a CDR/paratope candidate that can later be grafted onto a scaffold
or replaced by MPNN/BFN-generated candidates. The scorer ranks candidates by
positive-state complementarity minus negative-state cross-reactivity, with
basic developability filters.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


AA = "ACDEFGHIKLMNPQRSTVWY"
AROMATIC = set("FWY")
HYDROPHOBIC = set("AILMFWYV")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
POLAR = set("STNQYH")
PROLINE_GLY = set("PG")

KD_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


THERAPEUTIC_PRESETS: Dict[str, Dict] = {
    "abeta_core": {
        "target": "abeta42",
        "epitope_id": "core_16_24",
        "epitope_seq": "KLVFFAED",
        "positive_state": "oligomer_or_fibril_core_exposed",
        "negative_epitopes": ["DAEFRHDSGY", "GAIIGLMVGGVVIA", "KLVGGSED"],
        "goal": "block aggregation core and reduce toxic oligomer formation",
        "prior": "aromatic_clamp_with_polar_charge_anchors",
    },
    "abeta_nterm": {
        "target": "abeta42",
        "epitope_id": "n_term_1_10",
        "epitope_seq": "DAEFRHDSGY",
        "positive_state": "n_terminal_plaque_exposed",
        "negative_epitopes": ["KLVFFAED", "GAIIGLMVGGVVIA", "DAEAAHDSGA"],
        "goal": "recognize clearance-associated N-terminal A-beta epitope",
        "prior": "charge_guided_polar_aromatic_recognition",
    },
    "abeta_cterm": {
        "target": "abeta42",
        "epitope_id": "c_term_33_42",
        "epitope_seq": "GAIIGLMVGGVVIA",
        "positive_state": "c_terminal_fibril_or_membrane_exposed",
        "negative_epitopes": ["DAEFRHDSGY", "KLVFFAED", "GGGGGGGGGG"],
        "goal": "recognize hydrophobic C-terminal disease-associated exposure",
        "prior": "controlled_hydrophobic_patch_with_polar_edges",
    },
}


@dataclass(frozen=True)
class DesignSpec:
    target: str
    epitope_id: str
    epitope_seq: str
    positive_state: str
    negative_epitopes: Sequence[str]
    goal: str = "state-specific IDP antibody design"
    prior: str = "state_specific_paratope"


def load_preset(name: str) -> DesignSpec:
    """Return a built-in therapeutic design spec."""
    if name not in THERAPEUTIC_PRESETS:
        valid = ", ".join(sorted(THERAPEUTIC_PRESETS))
        raise ValueError(f"Unknown preset {name!r}. Valid presets: {valid}")
    p = THERAPEUTIC_PRESETS[name]
    return DesignSpec(**p)


def sequence_complexity(seq: str) -> float:
    """Return sequence complexity in [0, 1], higher is better."""
    seq = _clean(seq)
    if len(seq) < 3:
        return 1.0
    counts = {a: seq.count(a) for a in set(seq)}
    n = len(seq)
    entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
    entropy /= math.log(min(20, max(2, n)))
    max_freq = max(counts.values()) / n
    run_penalty = 0.0
    run_len = 1
    for i in range(1, n):
        if seq[i] == seq[i - 1]:
            run_len += 1
        else:
            if run_len >= 4:
                run_penalty += (run_len - 3) / n
            run_len = 1
    if run_len >= 4:
        run_penalty += (run_len - 3) / n
    score = entropy - max(0.0, max_freq - 0.45) - run_penalty
    return float(max(0.0, min(1.0, score)))


def immunogenicity_proxy(seq: str) -> float:
    """No-dependency immunogenicity/developability proxy in [0, 1]."""
    seq = _clean(seq)
    if not seq:
        return 0.0
    win = min(9, len(seq))
    hydros = [KD_HYDROPATHY[a] for a in seq]
    max_hydro = max(
        sum(hydros[i:i + win]) / win for i in range(max(1, len(seq) - win + 1))
    )
    hydro_score = max(0.0, min(1.0, (max_hydro + 1.0) / 4.5))
    aromatic_load = sum(1 for a in seq if a in AROMATIC) / len(seq)
    cys_penalty = 0.3 if "C" in seq else 0.0
    return float(max(0.0, min(1.0, 0.55 * hydro_score + 0.25 * aromatic_load + cys_penalty)))


def complementarity_score(paratope: str, epitope: str) -> float:
    """Heuristic paratope-epitope chemistry score in [0, 1].

    The score is intentionally state/chemistry oriented rather than template
    oriented. It rewards aromatic/hydrophobic recognition of exposed amyloid
    cores, charge complementarity at anchors, and polar H-bond compatibility.
    """
    p = _clean(paratope)
    e = _clean(epitope)
    if not p or not e:
        return 0.0
    pair_scores = []
    for ea in e:
        best = 0.0
        for pa in p:
            best = max(best, _aa_pair_score(pa, ea))
        pair_scores.append(best)
    base = sum(pair_scores) / len(pair_scores)
    # Encourage bounded paratope diversity: enough aromatic residues for amyloid
    # recognition, but not an all-hydrophobic sticky patch.
    aromatic_frac = sum(1 for a in p if a in AROMATIC) / len(p)
    hydro_frac = sum(1 for a in p if a in HYDROPHOBIC) / len(p)
    aromatic_bonus = 0.10 if 0.12 <= aromatic_frac <= 0.38 else -0.05
    hydro_penalty = max(0.0, hydro_frac - 0.65) * 0.35
    return float(max(0.0, min(1.0, base + aromatic_bonus - hydro_penalty)))


def score_candidate(candidate: Dict, spec: DesignSpec) -> Dict:
    """Attach state-specific scores to one candidate dict."""
    seq = _clean(candidate.get("cdr_h3") or candidate.get("sequence") or "")
    positive = complementarity_score(seq, spec.epitope_seq)
    neg_scores = [complementarity_score(seq, neg) for neg in spec.negative_epitopes]
    max_negative = max(neg_scores) if neg_scores else 0.0
    specificity_gap = positive - max_negative
    complexity = sequence_complexity(seq)
    immuno = immunogenicity_proxy(seq)
    # Conservative scalarization: specificity is primary; developability gates
    # keep sticky, low-complexity sequences from ranking highly.
    final = (
        0.50 * specificity_gap
        + 0.25 * positive
        + 0.15 * complexity
        + 0.10 * (1.0 - immuno)
    )
    failed = []
    if complexity < 0.55:
        failed.append("low_complexity")
    if immuno > 0.65:
        failed.append("high_immunogenicity_proxy")
    if specificity_gap < 0.05:
        failed.append("low_state_specificity_gap")
    if positive < 0.35:
        failed.append("weak_positive_state_score")
    return {
        **candidate,
        "sequence": seq,
        "cdr_h3": seq,
        "target": spec.target,
        "epitope_id": spec.epitope_id,
        "epitope_seq": spec.epitope_seq,
        "positive_state": spec.positive_state,
        "positive_score": round(positive, 4),
        "max_negative_score": round(max_negative, 4),
        "negative_scores": [round(v, 4) for v in neg_scores],
        "specificity_gap": round(specificity_gap, 4),
        "complexity": round(complexity, 4),
        "immunogenicity_proxy": round(immuno, 4),
        "state_specific_score": round(final, 4),
        "passes_filters": not failed,
        "filter_failures": ";".join(failed),
        "rationale": rationale_for_sequence(seq, spec),
    }


def rank_candidates(candidates: Iterable[Dict], spec: DesignSpec) -> List[Dict]:
    """Score and sort candidates by state-specific therapeutic score."""
    scored = [score_candidate(c, spec) for c in candidates]
    scored.sort(
        key=lambda r: (r["passes_filters"], r["state_specific_score"], r["specificity_gap"]),
        reverse=True,
    )
    for i, r in enumerate(scored, 1):
        r["rank"] = i
    return scored


def generate_paratope_candidates(
    spec: DesignSpec,
    n: int = 100,
    length: int = 13,
    seed: int = 42,
    scaffold_pool: Optional[Sequence[str]] = None,
) -> List[Dict]:
    """Generate CDR-H3-like paratope candidates from epitope chemistry.

    This is a deterministic heuristic generator for MVP validation. It is not a
    replacement for MPNN/BFN; it creates chemically plausible starting designs
    so the state-specific ranking pipeline can be exercised end-to-end.
    """
    rng = random.Random(seed)
    scaffold_pool = list(scaffold_pool or ["human_germline_carrier_A"])
    weights = _weights_from_epitope(spec.epitope_seq)
    candidates = []
    seen = set()
    attempts = 0
    while len(candidates) < n and attempts < n * 80:
        attempts += 1
        seq = _sample_sequence(rng, weights, length)
        seq = _place_anchors(seq, spec.epitope_seq, rng)
        if seq in seen:
            continue
        seen.add(seq)
        candidates.append({
            "candidate_id": f"ss_{len(candidates) + 1:04d}",
            "cdr_h3": seq,
            "scaffold": rng.choice(scaffold_pool),
            "generator": "state_specific_heuristic_v1",
        })
    return candidates


def rationale_for_sequence(seq: str, spec: DesignSpec) -> str:
    """Short human-readable mechanism label for reports."""
    seq = _clean(seq)
    aromatic = sum(1 for a in seq if a in AROMATIC)
    charged = sum(1 for a in seq if a in POSITIVE or a in NEGATIVE)
    hydro = sum(1 for a in seq if a in HYDROPHOBIC)
    labels = []
    if aromatic:
        labels.append(f"{aromatic} aromatic residues for hydrophobic/amyloid recognition")
    if charged:
        labels.append(f"{charged} charged residues for edge anchoring")
    if hydro / max(1, len(seq)) <= 0.65:
        labels.append("bounded hydrophobicity")
    if sequence_complexity(seq) >= 0.70:
        labels.append("diverse CDR composition")
    return "; ".join(labels) or f"state-specific paratope for {spec.epitope_id}"


def _clean(seq: str) -> str:
    return "".join(a for a in str(seq).upper() if a in AA)


def _aa_pair_score(pa: str, ea: str) -> float:
    score = 0.05
    if ea in HYDROPHOBIC and pa in HYDROPHOBIC:
        score += 0.35
    if ea in AROMATIC and pa in AROMATIC:
        score += 0.25
    if (ea in POSITIVE and pa in NEGATIVE) or (ea in NEGATIVE and pa in POSITIVE):
        score += 0.45
    if ea in POLAR and pa in POLAR:
        score += 0.22
    if ea in PROLINE_GLY and pa in set("SYG"):
        score += 0.12
    if ea == pa:
        score += 0.05
    if pa == "C":
        score -= 0.20
    return max(0.0, min(1.0, score))


def _weights_from_epitope(epitope: str) -> Dict[str, float]:
    epitope = _clean(epitope)
    hydrophobic_frac = sum(1 for a in epitope if a in HYDROPHOBIC) / max(1, len(epitope))
    charged_frac = sum(1 for a in epitope if a in POSITIVE or a in NEGATIVE) / max(1, len(epitope))
    weights = {a: 1.0 for a in AA}
    for a in "YWFS":
        weights[a] += 2.0 * hydrophobic_frac
    for a in "DEKRH":
        weights[a] += 1.4 * charged_frac
    for a in "NSTQG":
        weights[a] += 0.7
    weights["C"] = 0.05
    weights["P"] = 0.35
    return weights


def _sample_sequence(rng: random.Random, weights: Dict[str, float], length: int) -> str:
    aas = list(weights)
    probs = [weights[a] for a in aas]
    return "".join(rng.choices(aas, weights=probs, k=length))


def _place_anchors(seq: str, epitope: str, rng: random.Random) -> str:
    """Add minimal charge/aromatic anchors implied by the epitope."""
    chars = list(seq)
    if not chars:
        return seq
    if any(a in HYDROPHOBIC for a in epitope) and not any(a in AROMATIC for a in chars):
        chars[rng.randrange(len(chars))] = rng.choice("YW")
    if any(a in POSITIVE for a in epitope) and not any(a in NEGATIVE for a in chars):
        chars[rng.randrange(len(chars))] = rng.choice("DE")
    if any(a in NEGATIVE for a in epitope) and not any(a in POSITIVE for a in chars):
        chars[rng.randrange(len(chars))] = rng.choice("KRH")
    return "".join(chars)
