#!/usr/bin/env python
"""IDP-Aware Antibody CDR Design — Stage 4 (design) + Stage 5 (ranking).

Core workflow:
  1. For each ordered segment identified by the disorder head, build an
     antibody-epitope complex and run BFN CDR design.
  2. Rank all designs by composite score combining BFN confidence metrics
     (pLDDT, ipTM, PAE) with sequence quality (PPL, entropy).

Differentiator vs standard design: only targets ORDERED segments of the
IDP, filtered by BFN's own disorder head. Outputs confidence self-assessment.
"""

import os
import re
import json
import time
import shutil
import tempfile
import subprocess
import numpy as np
import torch
from pathlib import Path

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'

# 3-letter to 1-letter amino acid code
AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def _load_bfn():
    """Lazy-load BFN model (cached across calls)."""
    import sys
    sys.path.insert(0, '.')
    from bfn_loader import load_bfn
    return load_bfn()


def _run_bfn_design_on_complex(complex_pdb, region_spec, num_samples=10,
                               stochastic=True, device='cuda', context_chains=None):
    """Run BFN protein design on a complex PDB with specified regions masked.

    Args:
        context_chains: list of chain IDs to include as visible context
            (e.g. ['B'] for epitope). Pass None or [] for FixBB mode.
    Returns the raw design results as (text, fasta_str, results_list).
    """
    import sys
    sys.path.insert(0, '.')
    from bfn_loader import run_bfn_design
    results_list = run_bfn_design(complex_pdb, region_spec, num_samples,
                                  stochastic, context_chains, device)
    return '', '', results_list


def design_cdrs_for_segment(model, config, scaffold_pdb, epitope_pdb,
                            scaffold_chain, epitope_chain='B',
                            cdr_spec=None, num_samples=10, stochastic=True,
                            device='cuda', context_chains=None):
    """Run BFN antibody CDR design for a single epitope segment.

    Args:
        model: loaded BFN model (unused, kept for API consistency)
        config: BFN config (unused, kept for API consistency)
        scaffold_pdb: path to antibody scaffold PDB
        epitope_pdb: path to epitope structure PDB
        scaffold_chain: chain ID of scaffold (required, no default)
        epitope_chain: chain ID to use for epitope in complex
        cdr_spec: region spec string, e.g. "B:26-33,51-58,97-113"
        num_samples: number of BFN design samples
        stochastic: use stochastic sampling
        device: torch device
        context_chains: list of chain IDs visible as context during design.
            None or [] = FixBB (only scaffold visible).
            e.g. [epitope_chain] = Complex mode (antigen visible).

    Returns:
        dict with keys: designs, segment_info, complex_pdb, mode
    """
    from antibody_epitope_complex import position_epitope

    if cdr_spec is None:
        cdr_spec = f"{scaffold_chain}:26-33,51-58,97-113"

    # Build complex
    complex_info = position_epitope(
        scaffold_pdb, epitope_pdb,
        scaffold_chain=scaffold_chain,
        epitope_chain=epitope_chain,
        distance=6.0,
    )

    # Determine context chains for Complex mode
    if context_chains is None:
        # Default: Complex mode (epitope visible)
        _ctx = [epitope_chain]
    elif context_chains == []:
        # Explicit FixBB mode
        _ctx = []
    else:
        _ctx = list(context_chains)

    design_mode = 'complex' if _ctx else 'fixbb'

    # Run BFN design
    text, fasta_str, results_list = _run_bfn_design_on_complex(
        complex_info['pdb_path'], cdr_spec,
        num_samples=num_samples, stochastic=stochastic,
        context_chains=_ctx, device=device,
    )

    return {
        'designs': results_list,
        'complex_pdb': complex_info['pdb_path'],
        'distance': complex_info['distance'],
        'design_log': text,
        'fasta': fasta_str,
        'mode': design_mode,
    }


# ── AF2 Multimer Validation ──

def _parse_cdr_ranges(cdr_spec):
    """Parse CDR region spec into ordered list of (start, end, length) tuples.

    Args:
        cdr_spec: e.g. "A:26-33,51-58,97-113"

    Returns:
        list of (start_1idx, end_1idx, length)
    """
    ranges = []
    for cid, spec in re.findall(r'([A-Za-z0-9]+):([0-9,\-\s]+)', cdr_spec):
        for seg in spec.split(','):
            seg = seg.strip()
            if not seg:
                continue
            if '-' in seg:
                a, b = seg.split('-')
                s, e = int(a.strip()), int(b.strip())
                ranges.append((s, e, e - s + 1))
            else:
                v = int(seg)
                ranges.append((v, v, 1))
    return ranges


def _extract_sequence_from_pdb(pdb_path, chain_id):
    """Extract amino acid sequence from a PDB file for a given chain.

    Returns:
        str: 1-letter amino acid sequence
    """
    seen = {}
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            cid = line[21:22].strip()
            if cid != chain_id:
                continue
            try:
                resid = int(line[22:26])
            except ValueError:
                continue
            if resid in seen:
                continue
            resname = line[17:20].strip()
            aa = AA3TO1.get(resname, 'X')
            seen[resid] = aa

    if not seen:
        return ''
    return ''.join(seen[rid] for rid in sorted(seen))


def graft_cdrs(scaffold_seq, designed_seq, cdr_ranges):
    """Graft designed CDR sequences back onto the scaffold backbone.

    The designed_seq is the concatenation of all CDR regions in order.
    CDR regions are replaced in the scaffold sequence.

    Args:
        scaffold_seq: full scaffold amino acid sequence (1-letter)
        designed_seq: concatenated CDR sequences from BFN design
        cdr_ranges: list of (start_1idx, end_1idx, length) tuples

    Returns:
        str: full antibody sequence with grafted CDRs
    """
    full_seq = list(scaffold_seq)
    pos = 0
    mutations = []
    for start, end, length in cdr_ranges:
        cdr_designed = designed_seq[pos:pos + length]
        pos += length
        for j, aa in enumerate(cdr_designed):
            idx = start - 1 + j  # 0-indexed
            if idx >= len(full_seq):
                full_seq.append(aa)
            else:
                if full_seq[idx] != aa:
                    mutations.append((idx + 1, full_seq[idx], aa))
                full_seq[idx] = aa
    return ''.join(full_seq), mutations


def run_af2_multimer(ab_seq, epi_seq, output_dir, label='design',
                     colabfold_exe='colabfold_batch', num_recycle=3,
                     timeout=1800):
    """Run AF2 multimer prediction for antibody-epitope complex.

    Args:
        ab_seq: full antibody sequence (1-letter)
        epi_seq: epitope sequence (1-letter)
        output_dir: output directory
        label: name prefix for output files
        colabfold_exe: path to colabfold_batch
        num_recycle: AF2 recycle count
        timeout: per-design timeout in seconds

    Returns:
        dict with: success, plddt, iptm, ptm, max_pae, interface_pae, pdb_path
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    fasta_content = f'>complex_{label}\n{ab_seq}:{epi_seq}\n'
    fasta_path = os.path.join(output_dir, f'{label}.fasta')
    with open(fasta_path, 'w') as f:
        f.write(fasta_content)

    cmd = [
        colabfold_exe,
        '--num-models', '1',
        '--num-recycle', str(num_recycle),
        '--model-type', 'alphafold2_multimer_v3',
        '--rank', 'iptm',
        fasta_path,
        output_dir,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': f'AF2 timeout ({timeout}s)'}
    except FileNotFoundError:
        return {'success': False, 'error': f'colabfold_batch not found: {colabfold_exe}'}

    if result.returncode != 0:
        return {
            'success': False,
            'error': f'AF2 exit {result.returncode}: {result.stderr[-300:]}',
        }

    return _parse_af2_multimer_output(output_dir, label, ab_seq, epi_seq)


def _parse_af2_multimer_output(output_dir, label, ab_seq, epi_seq):
    """Parse AF2 multimer output: find scores JSON and extract metrics."""
    out = Path(output_dir)
    complex_name = f'complex_{label}'

    # Find scores JSON
    scores_files = sorted(out.glob(f'{complex_name}*scores_rank_001*.json'))
    if not scores_files:
        # Try broader glob
        scores_files = sorted(out.glob('*scores_rank_001*.json'))

    if not scores_files:
        return {'success': False, 'error': 'No scores JSON found'}

    with open(scores_files[0]) as f:
        scores = json.load(f)

    # pLDDT
    plddt_array = scores.get('plddt', [0])
    mean_plddt = sum(plddt_array) / max(len(plddt_array), 1)

    ptm = scores.get('ptm', 0.0)
    iptm = scores.get('iptm', 0.0)
    max_pae = scores.get('max_pae', 999.0)

    # Compute interface PAE
    ab_len = len(ab_seq)
    pae = scores.get('pae', None)
    if pae is not None:
        # PAE is [N, N] where N = len(ab) + len(epi)
        pae_arr = np.array(pae)
        if pae_arr.shape[0] >= ab_len + 1:
            # Interface PAE: antibody-epitope cross-terms
            epi_len = pae_arr.shape[0] - ab_len
            if epi_len > 0:
                interface_pae = float(pae_arr[:ab_len, ab_len:].mean())
            else:
                interface_pae = None
        else:
            interface_pae = None
    else:
        interface_pae = None

    # Find PDB
    pdb_files = sorted(out.glob(f'{complex_name}*unrelaxed_rank_001*.pdb'))
    if not pdb_files:
        pdb_files = sorted(out.glob(f'{complex_name}*relaxed_rank_001*.pdb'))
    if not pdb_files:
        pdb_files = sorted(out.glob('*.pdb'))
    pdb_path = str(pdb_files[0]) if pdb_files else None

    return {
        'success': True,
        'plddt': mean_plddt / 100.0,
        'iptm': iptm if iptm is not None else ptm,
        'ptm': ptm,
        'max_pae': max_pae,
        'interface_pae': interface_pae if interface_pae is not None else max_pae,
        'pdb_path': pdb_path,
    }


def validate_designs_with_af2(designs, scaffold_pdb, scaffold_chain,
                               epi_seq, cdr_spec, output_dir,
                               colabfold_exe='colabfold_batch',
                               num_recycle=3, verbose=True,
                               use_jax=True):
    """Run AF2 multimer on each design and enrich with AF2 confidence scores.

    Args:
        designs: list of design dicts (from ranked list)
        scaffold_pdb: path to antibody scaffold PDB
        scaffold_chain: chain ID of scaffold
        epi_seq: epitope sequence
        cdr_spec: CDR region spec (e.g. "A:26-33,51-58,97-113")
        output_dir: base output directory
        colabfold_exe: path to colabfold_batch (CLI mode only)
        num_recycle: AF2 recycle count
        verbose: print progress
        use_jax: use JAX-native runner (default True); False = colabfold CLI

    Returns:
        list of design dicts enriched with af2_* fields
    """
    scaffold_seq = _extract_sequence_from_pdb(scaffold_pdb, scaffold_chain)
    cdr_ranges = _parse_cdr_ranges(cdr_spec)

    if designs:
        total_cdr_len = sum(r[2] for r in cdr_ranges)
        if len(designs[0].get('sequence', '')) != total_cdr_len:
            if verbose:
                print(f"  Warning: design seq length ({len(designs[0]['sequence'])}) "
                      f"!= CDR total ({total_cdr_len}), using actual lengths")

    if use_jax:
        from af2_jax_runner import validate_antibody_epitope

    af2_output_dir = os.path.join(output_dir, 'af2_validation')
    os.makedirs(af2_output_dir, exist_ok=True)

    enriched = []
    n = len(designs)

    for i, d in enumerate(designs):
        full_ab_seq, mutations = graft_cdrs(
            scaffold_seq, d['sequence'], cdr_ranges
        )

        if use_jax:
            if verbose:
                print(f"  AF2(JAX) [{i+1}/{n}] ", end='', flush=True)
            af2_result = validate_antibody_epitope(
                full_ab_seq, epi_seq, num_recycle=num_recycle, verbose=verbose)
        else:
            if verbose:
                print(f"  AF2(CLI) [{i+1}/{n}] ", end='', flush=True)
            label = f'design_{i+1:03d}'
            af2_dir = os.path.join(af2_output_dir, label)
            t0 = time.time()
            af2_result = run_af2_multimer(
                full_ab_seq, epi_seq, af2_dir, label=label,
                colabfold_exe=colabfold_exe, num_recycle=num_recycle,
            )
            if af2_result.get('success'):
                af2_result['plddt'] = af2_result.get('plddt', 0)
                af2_result['iptm'] = af2_result.get('iptm', 0)
                af2_result['interface_pae'] = af2_result.get('interface_pae', 99)

        d_enriched = dict(d)
        d_enriched['full_ab_seq'] = full_ab_seq
        d_enriched['full_ab_mutations'] = mutations
        d_enriched['af2_success'] = af2_result.get('success', False)

        if af2_result.get('success'):
            d_enriched['af2_plddt'] = af2_result.get('plddt')
            d_enriched['af2_iptm'] = af2_result.get('iptm')
            d_enriched['af2_ptm'] = af2_result.get('ptm')
            d_enriched['af2_interface_pae'] = af2_result.get('interface_pae')
            d_enriched['af2_pdb_path'] = af2_result.get('pdb_path')
        enriched.append(d_enriched)

    return enriched


# ── Composite Scoring ──

def composite_score(design, weights=None, use_af2=True):
    """Compute composite design quality score.

    When AF2 scores are available (use_af2=True), uses AF2-derived
    pLDDT/ipTM/interface_PAE instead of BFN's structure-conditioned values.
    This gives design-specific confidence that differentiates sequences.

    Without AF2, falls back to BFN confidence (identical for fixed-backbone
    designs) + PPL ranking.

    Components:
      - pLDDT (0-1, higher better): predicted local confidence
      - ipTM (0-1, higher better): predicted interface confidence
      - PPL (lower better): perplexity of designed sequence
      - Entropy (lower better): model certainty at design positions
      - PAE (lower better): predicted alignment error (interface PAE from AF2)

    Args:
        design: dict with 'plddt', 'iptm', 'ppl', 'entropy', 'pae' keys,
                optionally 'af2_plddt', 'af2_iptm', 'af2_interface_pae'
        weights: optional dict of per-component weights
        use_af2: if True and AF2 scores available, use AF2 confidence

    Returns:
        float: composite score (higher = better design)
    """
    has_af2 = use_af2 and design.get('af2_success')

    if weights is None:
        if has_af2:
            weights = {
                'iptm': 0.25,
                'plddt': 0.15,
                'ppl': 0.30,
                'entropy': 0.15,
                'pae': 0.15,
            }
        else:
            # FixBB mode: backbone confidence (pLDDT/ipTM/PAE) is constant
            # across all designs on the same scaffold (verified: std<1e-4).
            # Heavily weight PPL and entropy as the only differentiating signals.
            weights = {
                'plddt': 0.05,
                'iptm': 0.10,
                'ppl': 0.45,
                'entropy': 0.30,
                'pae': 0.10,
            }

    score = 0.0

    # Confidence (higher better)
    if has_af2:
        plddt = design.get('af2_plddt', 0.0) or 0.0
        iptm = design.get('af2_iptm', 0.0) or 0.0
    else:
        plddt = design.get('plddt', 0.0) or 0.0
        iptm = design.get('iptm', 0.0) or 0.0
    score += weights.get('plddt', 0.2) * plddt
    score += weights.get('iptm', 0.35) * iptm

    # Sequence quality (lower better → invert)
    ppl = design.get('ppl', 100.0) or 100.0
    entropy = design.get('entropy', 2.0) or 2.0

    if has_af2:
        pae = design.get('af2_interface_pae') or design.get('pae', 31.0) or 31.0
    else:
        pae = design.get('pae', 31.0) or 31.0

    ppl_score = max(0.0, 1.0 - ppl / 200.0)
    entropy_score = max(0.0, 1.0 - entropy / 3.0)
    pae_score = max(0.0, 1.0 - pae / 30.0)

    score += weights.get('ppl', 0.2) * ppl_score
    score += weights.get('entropy', 0.1) * entropy_score
    score += weights.get('pae', 0.15) * pae_score

    return score


def rank_designs(all_designs, use_af2=True):
    """Rank all designs across segments by composite score.

    Args:
        all_designs: list of dicts, each with segment info + 'designs' list
        use_af2: if True and AF2 scores present, use AF2-derived confidence

    Returns:
        list of dicts sorted by composite_score descending, each enriched with
        rank, quality_label, and segment info.
    """
    ranked = []
    for seg_result in all_designs:
        for i, d in enumerate(seg_result.get('designs', [])):
            cs = composite_score(d, use_af2=use_af2)
            entry = {
                'sequence': d.get('sequence', ''),
                'ppl': d.get('ppl'),
                'entropy': d.get('entropy'),
                'plddt': d.get('plddt'),
                'iptm': d.get('iptm'),
                'pae': d.get('pae'),
                'composite_score': cs,
                'sample_idx': i,
                'segment_start': seg_result.get('segment_start'),
                'segment_end': seg_result.get('segment_end'),
                'segment_mean_disorder': seg_result.get('segment_mean_disorder'),
                'epitope_mode': seg_result.get('epitope_mode'),
            }
            # Carry forward AF2 scores if present
            for key in ('af2_success', 'af2_plddt', 'af2_iptm', 'af2_ptm',
                        'af2_interface_pae', 'af2_pdb_path',
                        'full_ab_seq', 'full_ab_mutations'):
                if key in d:
                    entry[key] = d[key]
            ranked.append(entry)

    ranked.sort(key=lambda x: x['composite_score'], reverse=True)

    for rank, entry in enumerate(ranked):
        entry['rank'] = rank + 1
        cs = entry['composite_score']
        if cs >= 0.70:
            entry['quality_label'] = 'HIGH'
        elif cs >= 0.45:
            entry['quality_label'] = 'MEDIUM'
        else:
            entry['quality_label'] = 'LOW'

    return ranked


def format_design_report(ranked_designs, top_n=10):
    """Format a human-readable design ranking report."""
    if not ranked_designs:
        return "No designs generated."

    has_af2 = any(d.get('af2_success') for d in ranked_designs)

    if has_af2:
        header = (
            f"  {'Rank':<5s} {'Score':<8s} {'Quality':<8s} "
            f"{'AF2_ipTM':<10s} {'AF2_pLDDT':<11s} {'iPAE':<8s} "
            f"{'PPL':<8s} {'Segment':<20s} {'Seq'}"
        )
        sep = (
            f"  {'─'*5} {'─'*8} {'─'*8} "
            f"{'─'*10} {'─'*11} {'─'*8} "
            f"{'─'*8} {'─'*20} {'─'*30}"
        )
    else:
        header = (
            f"  {'Rank':<5s} {'Score':<8s} {'Quality':<8s} "
            f"{'pLDDT':<8s} {'ipTM':<8s} {'PPL':<8s} {'PAE':<8s} "
            f"{'Segment':<20s} {'Seq'}"
        )
        sep = (
            f"  {'─'*5} {'─'*8} {'─'*8} "
            f"{'─'*8} {'─'*8} {'─'*8} {'─'*8} "
            f"{'─'*20} {'─'*30}"
        )

    lines = [
        "=" * 80,
        f"  IDP-Aware Antibody CDR Design — Top {min(top_n, len(ranked_designs))} Ranked Designs",
        "=" * 80,
        "",
        header,
        sep,
    ]

    for entry in ranked_designs[:top_n]:
        seq = entry['sequence']
        if len(seq) > 30:
            seq = seq[:14] + '…' + seq[-14:]
        seg = f"{entry['segment_start']}-{entry['segment_end']}" if entry.get('segment_start') else 'N/A'

        if has_af2:
            lines.append(
                f"  {entry['rank']:<5d} "
                f"{entry['composite_score']:<8.3f} "
                f"{entry['quality_label']:<8s} "
                f"{entry.get('af2_iptm', 0) or 0:<10.3f} "
                f"{entry.get('af2_plddt', 0) or 0:<11.3f} "
                f"{entry.get('af2_interface_pae', 99) or 99:<8.1f} "
                f"{entry['ppl'] or 99:<8.1f} "
                f"{seg:<20s} "
                f"{seq}"
            )
        else:
            lines.append(
                f"  {entry['rank']:<5d} "
                f"{entry['composite_score']:<8.3f} "
                f"{entry['quality_label']:<8s} "
                f"{entry['plddt'] or 0:<8.3f} "
                f"{entry['iptm'] or 0:<8.3f} "
                f"{entry['ppl'] or 99:<8.2f} "
                f"{entry['pae'] or 99:<8.1f} "
                f"{seg:<20s} "
                f"{seq}"
            )

    lines.append("")
    lines.append("=" * 80)

    # Summary stats
    high = sum(1 for d in ranked_designs if d['quality_label'] == 'HIGH')
    med = sum(1 for d in ranked_designs if d['quality_label'] == 'MEDIUM')
    low = sum(1 for d in ranked_designs if d['quality_label'] == 'LOW')
    lines.append(f"  Total: {len(ranked_designs)} designs | "
                 f"HIGH: {high} | MEDIUM: {med} | LOW: {low}")

    if ranked_designs:
        best = ranked_designs[0]
        if best.get('af2_success'):
            af2_plddt = best.get('af2_plddt') or 0
            af2_iptm = best.get('af2_iptm') or 0
            af2_ipae = best.get('af2_interface_pae') or 0
            lines.append(f"  Best: composite={best['composite_score']:.3f} | "
                         f"AF2_pLDDT={af2_plddt:.3f} | "
                         f"AF2_ipTM={af2_iptm:.3f} | "
                         f"AF2_iPAE={af2_ipae:.1f}")
        else:
            lines.append(f"  Best: composite={best['composite_score']:.3f} | "
                         f"pLDDT={best['plddt'] or 0:.3f} | ipTM={best['iptm'] or 0:.3f}")
        lines.append(f"        {best['sequence']}")

    return '\n'.join(lines)


def run_idp_antibody_design(target_pdb, target_chain, scaffold_pdb,
                            scaffold_chain,
                            disorder_threshold=0.3,
                            num_segments=3, num_samples=10,
                            stochastic=True, output_dir=None,
                            device='cuda', verbose=True,
                            use_af2=False, af2_num_recycle=3,
                            use_af2_jax=True,
                            colabfold_exe='colabfold_batch',
                            context_chains=None):
    """Main orchestrator: disorder-aware antibody design against an IDP target.

    Full pipeline:
      Stage 1: Predict disorder on target protein
      Stage 2: Identify ordered segments
      Stage 3: Build epitope structures
      Stage 4: Run BFN CDR design for each segment (Complex mode by default)
      Stage 4b (optional): AF2 multimer validation for design-specific confidence
      Stage 5: Rank all designs by composite confidence score

    Args:
        target_pdb: path to target protein PDB (e.g., Abeta42 fibril)
        target_chain: chain ID to analyze
        scaffold_pdb: path to antibody scaffold PDB (e.g., 5IMK nanobody)
        scaffold_chain: chain ID of the scaffold (required, no default)
        disorder_threshold: max disorder for "ordered" classification
        num_segments: max number of ordered segments to target
        num_samples: BFN design samples per segment
        stochastic: use stochastic BFN sampling
        output_dir: directory for output files
        device: torch device
        verbose: print progress
        use_af2: if True, run AF2 multimer validation on each design
        af2_num_recycle: AF2 recycling steps
        use_af2_jax: if True, use JAX-native AF2 runner; False = colabfold CLI
        colabfold_exe: path to colabfold_batch executable
        context_chains: list of chain IDs visible as context during BFN design.
            None (default) = Complex mode: epitope chain visible.
            [] = FixBB mode: only scaffold visible.

    Returns:
        dict with keys:
            disorder_result, segments, segment_results, ranked_designs, report
    """
    import sys
    sys.path.insert(0, '.')
    sys.path.insert(0, 'modules')

    from idp_disorder_analysis import predict_disorder, find_ordered_segments, format_disorder_report
    from epitope_structure_builder import build_epitope_structure
    from antibody_epitope_complex import position_epitope

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='idp_design_')
    os.makedirs(output_dir, exist_ok=True)

    model, config = _load_bfn()

    # ── Stage 1: Disorder Analysis ──
    if verbose:
        print("[Stage 1] Predicting disorder scores...")
    t0 = time.time()

    disorder_result = predict_disorder(model, config, target_pdb, target_chain, device)
    if verbose:
        print(f"  {disorder_result['disorder_scores'].shape[0]} residues in {time.time()-t0:.1f}s")
        print(f"  Mean disorder: {disorder_result['disorder_scores'].mean():.4f}")

    # ── Stage 2: Find Ordered Segments ──
    if verbose:
        print("[Stage 2] Finding ordered segments...")

    segments = find_ordered_segments(
        disorder_result['disorder_scores'],
        disorder_result['residue_ids'],
        threshold=disorder_threshold,
    )

    if verbose:
        print(f"  Found {len(segments)} ordered segments")
        for i, seg in enumerate(segments[:num_segments]):
            print(f"  [{i+1}] {seg['start']}-{seg['end']} "
                  f"length={seg['length']} D_mean={seg['mean_disorder']:.4f}")

    if not segments:
        print("No ordered segments found — cannot proceed with design.")
        return {
            'disorder_result': disorder_result,
            'segments': [],
            'segment_results': [],
            'ranked_designs': [],
            'report': format_disorder_report(
                disorder_result['disorder_scores'],
                disorder_result['sequence'],
                disorder_result['residue_ids'],
                disorder_threshold,
            ),
        }

    # Limit to top N segments
    segments = segments[:num_segments]

    # ── Stage 3: Build Structures + Stage 4: Design ──
    segment_results = []

    for i, seg in enumerate(segments):
        if verbose:
            print(f"[Stage 3+4] Segment {i+1}/{len(segments)}: "
                  f"{seg['start']}-{seg['end']}...")

        # Build epitope structure (prefer extracted from real PDB)
        epi_info = build_epitope_structure(
            seg,
            target_seq=disorder_result['sequence'],
            target_pdb=target_pdb,
            chain_id=target_chain,
            output_dir=output_dir,
        )

        # Run BFN design
        seg_result = design_cdrs_for_segment(
            model, config, scaffold_pdb, epi_info['pdb_path'],
            scaffold_chain=scaffold_chain,
            epitope_chain='B',
            cdr_spec=f"{scaffold_chain}:26-33,51-58,97-113",
            num_samples=num_samples,
            stochastic=stochastic,
            device=device,
            context_chains=context_chains,
        )

        seg_result['segment_start'] = seg['start']
        seg_result['segment_end'] = seg['end']
        seg_result['segment_mean_disorder'] = seg['mean_disorder']
        seg_result['epitope_mode'] = epi_info['mode']
        seg_result['epitope_pdb'] = epi_info['pdb_path']
        segment_results.append(seg_result)

        if verbose:
            n_designs = len(seg_result.get('designs', []))
            print(f"  Generated {n_designs} designs")

    # ── Stage 5: AF2 Validation (optional) + Rank ──
    if use_af2 and segment_results:
        if verbose:
            print("[Stage 4b] AF2 Multimer validation...")
            print(f"  Validating {sum(len(r.get('designs', [])) for r in segment_results)} designs "
                  f"with AF2 multimer (recycle={af2_num_recycle})...")

        # Collect all designs across segments
        all_designs_flat = []
        for seg_result in segment_results:
            seg_start = seg_result.get('segment_start')
            seg_end = seg_result.get('segment_end')
            for d in seg_result.get('designs', []):
                d_copy = dict(d)
                d_copy['_seg_start'] = seg_start
                d_copy['_seg_end'] = seg_end
                all_designs_flat.append(d_copy)

        # Get epitope sequence for the first segment
        # (all segments share the same epitope protein, just different ordered windows)
        epi_seq = disorder_result['sequence']

        cdr_spec = f"{scaffold_chain}:26-33,51-58,97-113"
        validated = validate_designs_with_af2(
            all_designs_flat, scaffold_pdb, scaffold_chain,
            epi_seq, cdr_spec, output_dir,
            colabfold_exe=colabfold_exe, num_recycle=af2_num_recycle,
            verbose=verbose, use_jax=use_af2_jax,
        )

        # Rebuild segment_results with AF2-enriched designs
        for seg_result in segment_results:
            seg_start = seg_result.get('segment_start')
            seg_end = seg_result.get('segment_end')
            enriched_designs = [
                d for d in validated
                if d.get('_seg_start') == seg_start and d.get('_seg_end') == seg_end
            ]
            if enriched_designs:
                seg_result['designs'] = enriched_designs

    if verbose:
        print("[Stage 5] Ranking all designs...")

    ranked = rank_designs(segment_results, use_af2=use_af2)
    report = format_design_report(ranked)

    if verbose:
        print(report)

    # Save results
    result = {
        'disorder_result': disorder_result,
        'segments': segments,
        'segment_results': segment_results,
        'ranked_designs': ranked,
        'report': report,
        'output_dir': output_dir,
    }

    # Save JSON summary
    json_path = os.path.join(output_dir, 'idp_design_results.json')
    serializable = {
        'target_pdb': target_pdb,
        'scaffold_pdb': scaffold_pdb,
        'disorder_threshold': disorder_threshold,
        'num_segments': num_segments,
        'num_samples': num_samples,
        'ordered_segments': [{k: v for k, v in s.items() if k != 'residues'}
                             for s in segments],
        'ranked_designs': [
            {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
             for k, v in d.items()}
            for d in ranked[:20]
        ],
    }
    with open(json_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    if verbose:
        print(f"\nResults saved to: {json_path}")

    return result
