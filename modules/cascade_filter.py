#!/usr/bin/env python
"""Cascade filter for protein sequence design results.

Three-stage filtering + composite ranking:
  1. Hard thresholds: pLDDT / ipTM / PPL (or score)
  2. Sequence deduplication (keep best)
  3. Weighted composite score → ranked output
"""

import math
from typing import List, Dict, Optional, Tuple


# ── Default thresholds ──
DEFAULT_THRESHOLDS = {
    'plddt_min': -0.01,     # Effectively disabled (model heads output ~0 for non-Ab targets)
    'iptm_min': -0.01,      # Effectively disabled (model heads output ~0 for non-Ab targets)
    'ppl_max': 100.0,       # Maximum perplexity — adjust per use case
    'entropy_max': 2.5,     # Max mean entropy over design region (0-2.996)
    'score_max': 10.0,      # Maximum score (ProteinMPNN, lower is better)
    # Developability (negative-design + immunogenicity). Disabled by default
    # (set to large values); enable by passing a tighter threshold at call time.
    'off_target_iptm_max': 999.0,   # negative-design: max off-target ipTM (lower=better)
    'immunogenicity_max':   999.0,  # max proxy immunogenicity (0–1)
}

# ── Composite scoring weights ──
DEFAULT_WEIGHTS = {
    'iptm': 0.35,           # Interface/global fold confidence
    'plddt': 0.25,          # Per-residue confidence
    'ppl_inv': 0.10,        # Inverse perplexity → sequence quality
    'entropy_inv': 0.05,    # Inverse entropy → model certainty
    'recovery': 0.15,       # Native recovery rate (eval mode only)
    'complexity': 0.10,     # CDR amino-acid diversity (penalises poly-Thr etc.)
}


def disorder_aware_score(result: dict, base_weight: float = 0.35) -> float:
    """I-2: Downweight BFN ipTM contribution when disorder_head flags IDP regions.

    If disorder_mean > 0.5 over the CDR, the BFN confidence is less reliable
    (trained on folded proteins). Reduce ipTM weight proportionally and
    shift weight to pLDDT (per-residue, more robust for mixed order/disorder).

    Args:
        result: dict with iptm, plddt, disorder_mean keys
        base_weight: default ipTM weight (0.35)
    Returns:
        adjusted ipTM weight in [0.05, base_weight]
    """
    disorder = result.get('disorder_mean', 0)
    if disorder is None or disorder <= 0.3:
        return base_weight  # ordered — full confidence
    elif disorder >= 0.7:
        return 0.05  # strongly disordered — nearly zero ipTM contribution
    else:
        # Linear ramp: 0.3→base_weight, 0.7→0.05
        return base_weight - (disorder - 0.3) * (base_weight - 0.05) / 0.4


def _cdr_complexity(sequence: str) -> float:
    """Score CDR sequence diversity in [0, 1]. Higher = more diverse = better.

    Penalises:
      1. Homopolymeric runs (≥3 identical AAs)  → -0.3 per run
      2. Single-AA dominance (>40% of one type)  → proportional penalty
      3. Shannon entropy < 2.5 over 20-AA dist   → linear penalty below threshold

    Returns 1.0 for ideal diversity, 0.0 for poly-X.
    """
    if not sequence or len(sequence) < 3:
        return 1.0

    score = 1.0
    L = len(sequence)

    # 1. Homopolymeric runs: ≥3 consecutive identical residues
    run_len = 1
    for i in range(1, L):
        if sequence[i] == sequence[i-1]:
            run_len += 1
        else:
            if run_len >= 3:
                score -= 0.30 * (run_len - 2) / L
            run_len = 1
    if run_len >= 3:
        score -= 0.30 * (run_len - 2) / L

    # 2. Single-AA dominance
    from collections import Counter
    counts = Counter(sequence)
    max_freq = max(counts.values()) / L
    if max_freq > 0.40:
        score -= (max_freq - 0.40) * 2.0  # 0.40→0 penalty, 1.0→-1.2 penalty

    # 3. Shannon entropy of AA distribution (< 2.5 = low diversity)
    total = sum(counts.values())
    shannon = -sum((c/total) * math.log(c/total) for c in counts.values()) / math.log(20)
    if shannon < 0.83:  # 0.83 ≈ 2.5 in raw bits (2.5 / log(20))
        score -= (0.83 - shannon) * 1.5

    return max(0.0, min(1.0, score))


def apply_cascade(
    results: List[Dict],
    thresholds: Optional[Dict] = None,
    weights: Optional[Dict] = None,
) -> Tuple[List[Dict], str]:
    """Apply three-stage cascade filter and return (filtered_results, report).

    Each result dict must have at least 'sequence'. Expected fields:
        sequence  : str  — amino acid sequence
        ppl       : float — perplexity (BFN, lower is better)
        plddt     : float — mean pLDDT over design region [0, 1]
        iptm      : float — ipTM score [0, 1]
        pae       : float — mean PAE over design region
        recovery  : float — native recovery rate (optional, 0–1)
        score     : float — ProteinMPNN score (lower is better)

    Returns:
        filtered : list of result dicts sorted by composite_score descending
        report   : formatted text report
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    report_lines = []
    n_input = len(results)

    # ── Stage 1: Hard thresholds ──
    stage1_pass = []
    stage1_reasons = {
        'plddt': 0, 'iptm': 0, 'ppl': 0, 'entropy': 0, 'score': 0,
        'off_target': 0, 'immuno': 0,
    }

    for r in results:
        plddt = r.get('plddt')
        iptm = r.get('iptm')
        ppl = r.get('ppl')
        entropy = r.get('entropy')
        score = r.get('score')
        off_iptm = r.get('off_target_iptm')
        immuno = r.get('immunogenicity')

        if plddt is not None and plddt < th['plddt_min']:
            stage1_reasons['plddt'] += 1
            continue
        if iptm is not None and iptm < th['iptm_min']:
            stage1_reasons['iptm'] += 1
            continue
        if ppl is not None and ppl > th['ppl_max']:
            stage1_reasons['ppl'] += 1
            continue
        if entropy is not None and entropy > th['entropy_max']:
            stage1_reasons['entropy'] += 1
            continue
        if score is not None and score > th['score_max']:
            stage1_reasons['score'] += 1
            continue
        if off_iptm is not None and off_iptm > th.get('off_target_iptm_max', 999):
            stage1_reasons['off_target'] += 1
            continue
        if immuno is not None and immuno > th.get('immunogenicity_max', 999):
            stage1_reasons['immuno'] += 1
            continue
        stage1_pass.append(r)

    n_stage1 = len(stage1_pass)
    n_rejected = n_input - n_stage1

    report_lines.append("─" * 50)
    report_lines.append(f"📊 级联过滤报告")
    report_lines.append(f"  输入: {n_input} 条序列")
    if n_rejected > 0:
        report_lines.append(f"  第1级 (硬阈值): 拒绝 {n_rejected} 条")
        if stage1_reasons['plddt']:
            report_lines.append(f"    - pLDDT < {th['plddt_min']}: {stage1_reasons['plddt']} 条")
        if stage1_reasons['iptm']:
            report_lines.append(f"    - ipTM < {th['iptm_min']}: {stage1_reasons['iptm']} 条")
        if stage1_reasons['ppl']:
            report_lines.append(f"    - PPL > {th['ppl_max']}: {stage1_reasons['ppl']} 条")
        if stage1_reasons['entropy']:
            report_lines.append(f"    - entropy > {th['entropy_max']}: {stage1_reasons['entropy']} 条")
        if stage1_reasons['score']:
            report_lines.append(f"    - MPNN score > {th['score_max']}: {stage1_reasons['score']} 条")

    if not stage1_pass:
        report_lines.append(f"\n  ⚠ 所有序列被第1级过滤拒绝 — 放宽阈值后重试")
        return [], '\n'.join(report_lines)

    # ── Stage 2: Deduplication ──
    seen = {}
    for r in stage1_pass:
        seq = r['sequence']
        if seq in seen:
            # Keep the one with better PPL (or score for MPNN)
            existing = seen[seq]
            existing_metric = existing.get('ppl') or existing.get('score', float('inf'))
            current_metric = r.get('ppl') or r.get('score', float('inf'))
            if current_metric < existing_metric:
                seen[seq] = r
        else:
            seen[seq] = r

    stage2 = list(seen.values())
    n_dup = n_stage1 - len(stage2)
    if n_dup > 0:
        report_lines.append(f"  第2级 (去重): 移除 {n_dup} 条重复序列")
    report_lines.append(f"  保留: {len(stage2)} 条唯一序列")

    # ── Stage 3: Composite scoring ──
    # Normalize PPL → inverse and clip for scoring
    ppls = [r.get('ppl') for r in stage2 if r.get('ppl') is not None]
    max_ppl = max(ppls) if ppls else 1.0
    entropies = [r.get('entropy') for r in stage2 if r.get('entropy') is not None]
    max_ent = max(entropies) if entropies else 2.996

    for r in stage2:
        plddt = r.get('plddt', 0.0) or 0.0
        iptm = r.get('iptm', 0.0) or 0.0
        ppl = r.get('ppl')
        entropy = r.get('entropy')
        recovery = r.get('recovery', 0.0) or 0.0
        score_mpnn = r.get('score')
        quality = r.get('quality')

        # Use precomputed quality if available (combines PPL + entropy)
        if quality is not None:
            quality_score = quality
        else:
            # PPL contribution: 1/(PPL) normalized, or use MPNN score inversely
            if ppl is not None and ppl > 0:
                ppl_clipped = min(ppl, max_ppl)
                ppl_inv = 1.0 / max(ppl_clipped, 0.1)
            elif score_mpnn is not None:
                ppl_inv = 1.0 / max(score_mpnn, 0.01)
            else:
                ppl_inv = 0.0

            # Entropy contribution: lower entropy = more certain model = better
            if entropy is not None:
                entropy_norm = entropy / max(max_ent, 0.01)
                entropy_inv = max(0.0, 1.0 - min(entropy_norm, 1.0))
            else:
                entropy_inv = 0.0

            quality_score = w['ppl_inv'] * min(ppl_inv, 1.0) + w['entropy_inv'] * entropy_inv

        # CDR complexity bonus (penalises low-diversity sequences)
        seq = r.get('sequence', '')
        complexity = _cdr_complexity(seq) if seq else 1.0
        r['complexity'] = round(complexity, 4)

        composite = (
            w['iptm'] * iptm +
            w['plddt'] * plddt +
            quality_score +
            w['recovery'] * recovery +
            w.get('complexity', 0.10) * complexity
        )
        r['composite_score'] = composite
        if entropy is not None:
            r['entropy_inv'] = round(entropy_inv if not quality else entropy, 4)

    # Rank by composite score descending
    stage2.sort(key=lambda r: r['composite_score'], reverse=True)

    # Add rank
    for rank, r in enumerate(stage2, 1):
        r['rank'] = rank

    report_lines.append(f"  第3级 (综合评分):")
    report_lines.append(f"    权重 — ipTM={w['iptm']}  pLDDT={w['plddt']}  PPL={w['ppl_inv']}  ent={w['entropy_inv']}  recovery={w['recovery']}  complexity={w.get('complexity',0.10)}")
    report_lines.append("")
    report_lines.append(f"  Top-5 综合排名:")
    report_lines.append(f"  {'#':<4} {'CDR':<22} {'Comp':<8} {'ipTM':<7} {'PPL':<7} {'ent':<7} {'cmplx':<7}")
    report_lines.append(f"  {'-'*4} {'-'*22} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for r in stage2[:5]:
        seq = r['sequence']
        cs = r['composite_score']
        iptm_s = f"{r.get('iptm', 0) or 0:.3f}"
        ppl_s = f"{r.get('ppl', r.get('score', '-')):.2f}" if r.get('ppl') or r.get('score') else '-'
        ent_s = f"{r.get('entropy', 0):.3f}" if r.get('entropy') is not None else '-'
        cmplx_s = f"{r.get('complexity', 1.0):.3f}"
        report_lines.append(f"  {r['rank']:<4} {seq:<22} {cs:<8.3f} {iptm_s:<7} {ppl_s:<7} {ent_s:<7} {cmplx_s:<7}")

    report_lines.append("─" * 50)
    return stage2, '\n'.join(report_lines)


def format_filtered_fasta(filtered: List[Dict], pdb_name: str = 'design') -> Tuple[str, str]:
    """Build FASTA string from filtered results and identify best sequence.

    Returns:
        fasta_str : multi-entry FASTA with composite score in headers
        best_seq  : sequence with highest composite score
    """
    fasta_lines = []
    best_seq = ""

    for r in filtered:
        rank = r.get('rank', '?')
        cs = r.get('composite_score', 0)
        seq = r['sequence']
        plddt = r.get('plddt', 0) or 0
        iptm = r.get('iptm', 0) or 0
        ppl = r.get('ppl')
        score_mpnn = r.get('score')
        recovery = r.get('recovery')

        parts = [f"{pdb_name}_rank{rank}", f"composite={cs:.3f}", f"ipTM={iptm:.3f}", f"pLDDT={plddt:.2f}"]
        if ppl is not None:
            parts.append(f"PPL={ppl:.2f}")
        if score_mpnn is not None:
            parts.append(f"MPNN_score={score_mpnn:.2f}")
        if recovery is not None:
            parts.append(f"recovery={recovery*100:.0f}%")

        fasta_lines.append(f">{' '.join(parts)}\n{seq}")

        if rank == 1:
            best_seq = seq

    return '\n'.join(fasta_lines), best_seq


# ── AF2-aware cascade ──

DEFAULT_AF2_THRESHOLDS = {
    'plddt_min': 60.0,       # AF2 pLDDT (0-100 scale)
    'ptm_min': 0.4,          # AF2 pTM
    'iptm_min': 0.3,         # AF2 ipTM (multimer only)
    'max_pae_max': 15.0,     # Max PAE cutoff
}

DEFAULT_AF2_WEIGHTS = {
    'iptm': 0.35,            # Interface confidence (most important for binding)
    'ptm': 0.25,             # Global fold confidence
    'plddt': 0.25,           # Per-residue confidence
    'ppl_inv': 0.15,         # Sequence quality (BFN PPL)
}


def apply_cascade_af2(
    results: List[Dict],
    thresholds: Optional[Dict] = None,
    weights: Optional[Dict] = None,
) -> Tuple[List[Dict], str]:
    """AF2-aware cascade: filter + rank using ColabFold confidence scores.

    Each result dict may contain:
        sequence   : str  — amino acid sequence
        plddt      : float — AF2 mean pLDDT (0-100 scale, from scores JSON)
        ptm        : float — AF2 pTM (0-1)
        iptm       : float — AF2 ipTM (0-1, multimer only, may be None)
        max_pae    : float — AF2 max PAE (lower is better)
        ppl        : float — BFN perplexity (optional)
        pdb_path   : str  — AF2 output PDB path
        recovery   : float — native recovery rate (optional)
        score      : float — ProteinMPNN score (optional)

    Returns:
        filtered : list sorted by af2_composite_score descending
        report   : formatted text report
    """
    th = {**DEFAULT_AF2_THRESHOLDS, **(thresholds or {})}
    w = {**DEFAULT_AF2_WEIGHTS, **(weights or {})}

    report_lines = []
    n_input = len(results)

    # ── Stage 1: AF2 hard thresholds ──
    stage1_pass = []
    stage1_reasons = {'plddt': 0, 'ptm': 0, 'iptm': 0, 'max_pae': 0}

    for r in results:
        plddt = r.get('plddt', 0) or 0  # 0-100
        ptm = r.get('ptm', 0) or 0
        iptm = r.get('iptm')
        max_pae = r.get('max_pae', 999) or 999

        if plddt < th['plddt_min']:
            stage1_reasons['plddt'] += 1
            continue
        if ptm < th['ptm_min']:
            stage1_reasons['ptm'] += 1
            continue
        if iptm is not None and iptm < th['iptm_min']:
            stage1_reasons['iptm'] += 1
            continue
        if max_pae > th['max_pae_max']:
            stage1_reasons['max_pae'] += 1
            continue
        stage1_pass.append(r)

    n_stage1 = len(stage1_pass)
    n_rejected = n_input - n_stage1

    report_lines.append("─" * 50)
    report_lines.append(f"📊 AF2 级联过滤报告")
    report_lines.append(f"  输入: {n_input} 条序列")
    if n_rejected > 0:
        report_lines.append(f"  第1级 (AF2阈值): 拒绝 {n_rejected} 条")
        if stage1_reasons['plddt']:
            report_lines.append(f"    - pLDDT < {th['plddt_min']}: {stage1_reasons['plddt']} 条")
        if stage1_reasons['ptm']:
            report_lines.append(f"    - pTM < {th['ptm_min']}: {stage1_reasons['ptm']} 条")
        if stage1_reasons['iptm']:
            report_lines.append(f"    - ipTM < {th['iptm_min']}: {stage1_reasons['iptm']} 条")
        if stage1_reasons['max_pae']:
            report_lines.append(f"    - max PAE > {th['max_pae_max']}: {stage1_reasons['max_pae']} 条")

    if not stage1_pass:
        report_lines.append(f"\n  ⚠ 所有序列被AF2阈值拒绝 — 放宽阈值后重试")
        return [], '\n'.join(report_lines)

    # ── Stage 2: Deduplication ──
    seen = {}
    for r in stage1_pass:
        seq = r['sequence']
        if seq in seen:
            existing_ptm = seen[seq].get('ptm', 0) or 0
            current_ptm = r.get('ptm', 0) or 0
            if current_ptm > existing_ptm:
                seen[seq] = r
        else:
            seen[seq] = r

    stage2 = list(seen.values())
    n_dup = n_stage1 - len(stage2)
    if n_dup > 0:
        report_lines.append(f"  第2级 (去重): 移除 {n_dup} 条重复序列")
    report_lines.append(f"  保留: {len(stage2)} 条唯一序列")

    # ── Stage 3: AF2 composite scoring ──
    for r in stage2:
        plddt_norm = (r.get('plddt', 0) or 0) / 100.0  # 0-100 → 0-1
        ptm = r.get('ptm', 0) or 0
        iptm = r.get('iptm')
        ppl = r.get('ppl')
        recovery = r.get('recovery', 0) or 0

        ppl_inv = 1.0 / max(ppl, 0.1) if ppl else 0.0

        active_weights = {
            'iptm': w['iptm'] if iptm is not None else 0.0,
            'ptm': w['ptm'],
            'plddt': w['plddt'],
            'ppl_inv': w['ppl_inv'] if ppl else 0.0,
        }
        weight_sum = sum(active_weights.values()) or 1.0
        composite = (
            active_weights['iptm'] * (iptm or 0.0) +
            active_weights['ptm'] * ptm +
            active_weights['plddt'] * plddt_norm +
            active_weights['ppl_inv'] * min(ppl_inv, 1.0)
        ) / weight_sum
        r['af2_composite_score'] = composite

    stage2.sort(key=lambda r: r['af2_composite_score'], reverse=True)
    for rank, r in enumerate(stage2, 1):
        r['af2_rank'] = rank

    report_lines.append(f"  第3级 (AF2综合评分):")
    report_lines.append(f"    权重 — ipTM={w['iptm']}  pTM={w['ptm']}  pLDDT={w['plddt']}  PPL⁻¹={w['ppl_inv']}")
    report_lines.append("")
    report_lines.append(f"  🏆 AF2 Top-5:")
    report_lines.append(f"  {'排名':<4} {'序列':<22} {'综合分':<8} {'ipTM':<7} {'pTM':<7} {'pLDDT':<7} {'PPL':<7}")
    report_lines.append(f"  {'-'*4} {'-'*22} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for r in stage2[:5]:
        seq = r['sequence']
        cs = r['af2_composite_score']
        iptm_s = f"{r['iptm']:.3f}"[:7] if r.get('iptm') is not None else '-'
        ptm_s = f"{r.get('ptm', 0) or 0:.3f}"[:7]
        plddt_s = f"{(r.get('plddt', 0) or 0):.1f}"[:7]
        ppl_s = f"{r.get('ppl', '-'):.2f}"[:7] if r.get('ppl') else '-'
        report_lines.append(f"  {r['af2_rank']:<4} {seq:<22} {cs:<8.3f} {iptm_s:<7} {ptm_s:<7} {plddt_s:<7} {ppl_s:<7}")

    report_lines.append("─" * 50)
    return stage2, '\n'.join(report_lines)


# ── Misfolding disease cascade (IDP-aware) ──

MISFOLDING_CASCADE_THRESHOLDS = {
    'plddt_min': -0.01,     # Disabled: BFN pLDDT unreliable on IDPs
    'iptm_min': 0.25,        # Relaxed from 0.4
    'ppl_max': 200.0,         # Relaxed from 100
    'entropy_max': 2.8,       # Relaxed from 2.5
    'score_max': 10.0,
}

MISFOLDING_CASCADE_WEIGHTS = {
    'iptm': 0.45,            # Higher: interface quality is key
    'plddt': 0.15,           # Lower: unreliable on IDPs
    'ppl_inv': 0.15,
    'entropy_inv': 0.10,
    'recovery': 0.15,
}

MISFOLDING_AF2_THRESHOLDS = {
    'plddt_min': 50.0,       # Relaxed: IDP complexes often have lower AF2 pLDDT
    'ptm_min': 0.35,          # Relaxed
    'iptm_min': 0.25,         # Relaxed
    'max_pae_max': 20.0,      # Relaxed from 15
}

MISFOLDING_AF2_WEIGHTS = {
    'iptm': 0.50,            # Interface confidence dominates
    'ptm': 0.20,
    'plddt': 0.15,
    'ppl_inv': 0.15,
}


def apply_cascade_misfolding(results, use_af2=False):
    """Apply misfolding-disease-aware cascade with IDP-relaxed thresholds."""
    if use_af2:
        return apply_cascade_af2(
            results,
            thresholds=MISFOLDING_AF2_THRESHOLDS,
            weights=MISFOLDING_AF2_WEIGHTS,
        )
    return apply_cascade(
        results,
        thresholds=MISFOLDING_CASCADE_THRESHOLDS,
        weights=MISFOLDING_CASCADE_WEIGHTS,
    )


def format_idp_score_warning(target_name, idp_warning):
    """Format an IDP-aware score interpretation warning."""
    if not idp_warning:
        return ''
    return (
        '\n' + '=' * 40 + '\n'
        f'  IDP Target Warning: {target_name}\n'
        f'  {idp_warning}\n'
        + '=' * 40
    )


# ══════════════════════════════════════════════════════════════════
# Multi-Objective Closed-Loop Cascade (Rosetta + AF2 + sequence)
# ══════════════════════════════════════════════════════════════════

MULTI_OBJECTIVE_THRESHOLDS = {
    'plddt_min':    50.0,
    'ptm_min':      0.25,
    'iptm_min':     0.25,
    'max_pae_max':  20.0,
    'dG_max':       5.0,
    'ddG_max':      10.0,
    'ppl_max':      200.0,
    'entropy_max':  2.8,
}

MULTI_OBJECTIVE_WEIGHTS = {
    'iptm':      0.25,
    'ptm':       0.15,
    'plddt':     0.15,
    'max_pae':   0.05,
    'dG':        0.15,
    'ddG':       0.05,
    'relax_e':   0.05,
    'ppl_inv':   0.10,
    'entropy_inv': 0.05,
}


def apply_cascade_multi_objective(results, thresholds=None, weights=None):
    """Extended cascade with Rosetta physics dimensions (dG, ddG).

    Extends apply_cascade_af2() by adding Rosetta interface energy
    as an optimization dimension. Supports both Pareto and weighted
    composite ranking.

    Each result dict may contain:
        sequence, plddt (0-100), ptm [0,1], iptm [0,1], max_pae,
        dG (REU, negative = favorable), ddG (dG_gen - dG_ref),
        relax_e (post-relax energy), ppl, entropy

    Returns:
        filtered: list sorted by composite_score descending
        report: formatted text report
    """
    th = {**MULTI_OBJECTIVE_THRESHOLDS, **(thresholds or {})}
    w = {**MULTI_OBJECTIVE_WEIGHTS, **(weights or {})}

    report_lines = []
    n_input = len(results)

    # ── Stage 1: Hard thresholds ──
    stage1_pass = []
    stage1_reasons = {
        'plddt': 0, 'iptm': 0, 'ptm': 0, 'max_pae': 0,
        'dG': 0, 'ddG': 0, 'ppl': 0, 'entropy': 0,
    }

    for r in results:
        plddt = r.get('plddt')
        if plddt is not None and plddt < th['plddt_min']:
            stage1_reasons['plddt'] += 1; continue

        iptm = r.get('iptm')
        if iptm is not None and iptm < th['iptm_min']:
            stage1_reasons['iptm'] += 1; continue

        ptm = r.get('ptm')
        if ptm is not None and ptm < th['ptm_min']:
            stage1_reasons['ptm'] += 1; continue

        max_pae = r.get('max_pae')
        if max_pae is not None and max_pae > th['max_pae_max']:
            stage1_reasons['max_pae'] += 1; continue

        dG = r.get('dG')
        if dG is not None and dG > th['dG_max']:
            stage1_reasons['dG'] += 1; continue

        ddG = r.get('ddG')
        if ddG is not None and ddG > th['ddG_max']:
            stage1_reasons['ddG'] += 1; continue

        ppl = r.get('ppl')
        if ppl is not None and ppl > th['ppl_max']:
            stage1_reasons['ppl'] += 1; continue

        entropy = r.get('entropy')
        if entropy is not None and entropy > th['entropy_max']:
            stage1_reasons['entropy'] += 1; continue

        stage1_pass.append(r)

    n_s1 = len(stage1_pass)
    n_s1_rejected = n_input - n_s1

    # ── Stage 2: Deduplication ──
    seen = {}
    stage2_pass = []
    for r in stage1_pass:
        seq = r['sequence']
        ppl = r.get('ppl', 999)
        if seq not in seen:
            seen[seq] = r
            stage2_pass.append(r)
        else:
            existing = seen[seq]
            cur_ppl = existing.get('ppl', 999)
            if ppl < cur_ppl:
                stage2_pass.remove(existing)
                stage2_pass.append(r)
                seen[seq] = r

    n_s2 = len(stage2_pass)

    # ── Stage 3: Multi-objective weighted composite ──
    stage3_results = stage2_pass

    # Normalize helper
    def _norm(values, maximize=True):
        vals = [v for v in values if v is not None]
        if not vals:
            return [0.0] * len(values)
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin
        if rng < 1e-9:
            return [1.0] * len(values)
        result = []
        for v in values:
            if v is None:
                result.append(0.0)
            elif maximize:
                result.append((v - vmin) / rng)
            else:
                result.append((vmax - v) / rng)
        return result

    # Extract values for each active dimension
    dims = {}
    for key in ['iptm', 'ptm', 'plddt', 'dG', 'ddG', 'relax_e', 'max_pae', 'ppl', 'entropy']:
        vals = [r.get(key) for r in stage3_results]
        if any(v is not None for v in vals):
            dims[key] = vals

    # Normalize
    norm = {}
    for key, vals in dims.items():
        maximize = key not in ('dG', 'ddG', 'relax_e', 'max_pae', 'ppl', 'entropy')
        norm[key] = _norm(vals, maximize=maximize)

    # Compute composite
    for i, r in enumerate(stage3_results):
        score = 0.0
        for key, weight in w.items():
            if key == 'ppl_inv':
                ppl_val = r.get('ppl')
                if ppl_val is not None and ppl_val > 0:
                    inv = min(1.0 / ppl_val, 1.0)
                else:
                    inv = 0.0
                score += weight * inv
            elif key == 'entropy_inv':
                ent_val = r.get('entropy')
                if ent_val is not None and ent_val > 0:
                    inv = min(1.0 / ent_val, 1.0)
                else:
                    inv = 0.0
                score += weight * inv
            elif key in norm:
                score += weight * norm[key][i]
            elif key == 'recovery':
                rec = r.get('recovery') or 0
                score += weight * rec
        r['composite_score'] = round(score, 4)

    stage3_results.sort(key=lambda x: x.get('composite_score', 0), reverse=True)

    # ── Build report ──
    report_lines.append(f"📊 Multi-Objective Cascade Report")
    report_lines.append(f"  Input: {n_input} | Stage1 pass: {n_s1} | Stage2 unique: {n_s2} | Stage3 ranked: {len(stage3_results)}")

    if n_s1_rejected > 0:
        reasons_str = ', '.join(f'{k}={v}' for k, v in stage1_reasons.items() if v > 0)
        report_lines.append(f"  Stage1 rejected ({n_s1_rejected}): {reasons_str}")

    weight_str = ', '.join(f'{k}={v:.2f}' for k, v in w.items())
    report_lines.append(f"  Weights: {weight_str}")

    report_lines.append(f"\n  🏆 Top-5 Multi-Objective:")
    header = f"  {'Rank':<5} {'Sequence':<35} {'Composite':<10} {'ipTM':<8} {'dG':<8} {'PPL':<8}"
    report_lines.append(header)
    report_lines.append("  " + "-" * (len(header) - 2))

    for i, r in enumerate(stage3_results[:5]):
        seq = r['sequence'][:33]
        comp = r.get('composite_score', 0)
        iptm = r.get('iptm', 0) or 0
        dg = r.get('dG', 'N/A')
        ppl = r.get('ppl', 0) or 0
        dg_str = f"{dg:.1f}" if isinstance(dg, (int, float)) else str(dg)
        report_lines.append(
            f"  {i+1:<5} {seq:<35} {comp:<10.3f} {iptm:<8.3f} {dg_str:<8} {ppl:<8.1f}"
        )

    report_lines.append("─" * 50)
    return stage3_results, '\n'.join(report_lines)
