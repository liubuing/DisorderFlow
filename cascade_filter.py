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
}

# ── Composite scoring weights ──
DEFAULT_WEIGHTS = {
    'iptm': 0.35,           # Interface/global fold confidence
    'plddt': 0.25,          # Per-residue confidence
    'ppl_inv': 0.15,        # Inverse perplexity → sequence quality
    'entropy_inv': 0.10,    # Inverse entropy → model certainty
    'recovery': 0.15,       # Native recovery rate (eval mode only)
}


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
    stage1_reasons = {'plddt': 0, 'iptm': 0, 'ppl': 0, 'entropy': 0, 'score': 0}

    for r in results:
        plddt = r.get('plddt')
        iptm = r.get('iptm')
        ppl = r.get('ppl')
        entropy = r.get('entropy')
        score = r.get('score')

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

        composite = (
            w['iptm'] * iptm +
            w['plddt'] * plddt +
            quality_score +
            w['recovery'] * recovery
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
    report_lines.append(f"    权重 — ipTM={w['iptm']}  pLDDT={w['plddt']}  PPL⁻¹={w['ppl_inv']}  ent⁻¹={w['entropy_inv']}  recovery={w['recovery']}")
    report_lines.append("")
    report_lines.append(f"  🏆 Top-5 综合排名:")
    report_lines.append(f"  {'排名':<4} {'序列':<22} {'综合分':<8} {'ipTM':<7} {'PPL':<7} {'ent':<7} {'恢复率':<7}")
    report_lines.append(f"  {'-'*4} {'-'*22} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for r in stage2[:5]:
        seq = r['sequence']
        cs = r['composite_score']
        iptm_s = f"{r.get('iptm', 0) or 0:.3f}"
        ppl_s = f"{r.get('ppl', r.get('score', '-')):.2f}" if r.get('ppl') or r.get('score') else '-'
        ent_s = f"{r.get('entropy', 0):.3f}" if r.get('entropy') is not None else '-'
        rec_s = f"{(r.get('recovery', 0) or 0)*100:.0f}%" if r.get('recovery') is not None else '-'
        report_lines.append(f"  {r['rank']:<4} {seq:<22} {cs:<8.3f} {iptm_s:<7} {ppl_s:<7} {ent_s:<7} {rec_s:<7}")

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
        if iptm is None:
            iptm = ptm  # Fallback: use pTM as ipTM
        ppl = r.get('ppl')
        recovery = r.get('recovery', 0) or 0

        ppl_inv = 1.0 / max(ppl, 0.1) if ppl else 0.0

        composite = (
            w['iptm'] * iptm +
            w['ptm'] * ptm +
            w['plddt'] * plddt_norm +
            w['ppl_inv'] * min(ppl_inv, 1.0)
        )
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
        iptm_s = f"{r.get('iptm') or r.get('ptm', 0):.3f}"[:7]
        ptm_s = f"{r.get('ptm', 0) or 0:.3f}"[:7]
        plddt_s = f"{(r.get('plddt', 0) or 0):.1f}"[:7]
        ppl_s = f"{r.get('ppl', '-'):.2f}"[:7] if r.get('ppl') else '-'
        report_lines.append(f"  {r['af2_rank']:<4} {seq:<22} {cs:<8.3f} {iptm_s:<7} {ptm_s:<7} {plddt_s:<7} {ppl_s:<7}")

    report_lines.append("─" * 50)
    return stage2, '\n'.join(report_lines)
