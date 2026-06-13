#!/usr/bin/env python
"""AQP4 Extracellular Water Channel Antibody Design Pipeline.

Targets the extracellular water channel of Aquaporin-4 (AQP4) for
nanobody CDR design using Bayesian Flow Networks.

Pipeline:
  Step 1: Load AQP4 structure (1RC2 = human AQP4 tetramer)
  Step 2: Epitope scoring (SASA + hydropathy + protrusion)
          with extracellular-loop priority
  Step 3: BFN disorder analysis 鈫?find ordered segments
  Step 4: Build epitope structures from ordered extracellular regions
  Step 5: BFN nanobody CDR design (Complex mode, 5IMK scaffold)
  Step 6: Cascade filtering & ranking
  Step 7: Output (JSON + FASTA + Markdown report)

Usage:
  python scripts/eval/aqp4_water_channel_design.py
  python scripts/eval/aqp4_water_channel_design.py --samples 30 --device cuda
  python scripts/eval/aqp4_water_channel_design.py --scaffold 5IMK --af2-validate
"""

import os, sys, json, time, argparse, textwrap, io
from pathlib import Path
import numpy as np

# Fix Windows console encoding for Unicode table characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 鈹€鈹€ Project path setup 鈹€鈹€
# __file__ is scripts/eval/aqp4_water_channel_design.py
# .parent = scripts/eval, .parent.parent = scripts, .parent.parent.parent = project root
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent  # disorderflow-main/
os.chdir(str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / 'modules'))

from target_design_helpers import (
    compute_residue_sasa,
    compute_hydrophilicity,
    compute_protrusion_index,
    score_epitope_residues,
    format_epitope_table,
    residues_to_region_spec,
    extract_seq_from_pdb,
    AA_3TO1,
    KYTE_DOOLITTLE,
    MAX_SASA,
)
from epitope_structure_builder import extract_epitope_pdb
from cascade_filter import apply_cascade

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'
DEFAULT_CDR = 'B:26-33,51-58,97-113'  # nanobody CDRs (5IMK scaffold chain B)

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# AQP4 Topology (human AQP4, UniProt P55087)
# Based on known structure: 6 TM helices, 3 extracellular loops (A, C, E)
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# Residue numbering for 1RC2 (crystal structure: residues 32-254 of full AQP4)
# In 1RC2 numbering starts at 1 (corresponds to Ser32 of full AQP4)
# TM1: 6-28, LoopA(extracellular): 55-68, TM2: 70-92,
# LoopB(intracellular): 93-109, TM3: 110-133,
# LoopC(extracellular): 134-141, TM4: 142-161,
# LoopD(intracellular): 162-171, TM5: 172-191,
# LoopE(extracellular): 192-194, TM6: 195-219

# Extracellular loop definitions for 1RC2 (1-indexed residue numbers)
AQP4_EXTRACELLULAR_LOOPS_1RC2 = {
    'Loop_A': (55, 68),   # Extracellular loop A
    'Loop_C': (134, 141), # Extracellular loop C (contains ar/R filter)
    'Loop_E': (192, 194), # Extracellular loop E (short)
}

# Broader extracellular regions (including flanks) for epitope analysis
AQP4_EXTRACELLULAR_BROAD = [
    (53, 72),    # Loop A + flanks
    (131, 145),  # Loop C + flanks
    (189, 198),  # Loop E + flanks
]

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Step 1: Load AQP4 structure
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def load_aqp4_structure(aqp4_pdb='data/aqp4/1RC2.pdb', chain='A'):
    """Load and validate AQP4 PDB structure. Download if needed."""
    if not os.path.exists(aqp4_pdb):
        print(f"[Step 1] AQP4 PDB not found at {aqp4_pdb}, downloading...")
        _fetch_aqp4()

    # Validate structure
    seq = extract_seq_from_pdb(aqp4_pdb, chain)
    if not seq:
        raise ValueError(f"No valid sequence found in {aqp4_pdb} chain {chain}")

    print(f"[Step 1] Loaded AQP4 structure: {aqp4_pdb}")
    print(f"  Chain {chain}: {len(seq)} residues")
    print(f"  Sequence (first 60): {seq[:60]}...")

    return aqp4_pdb, chain, seq


def _fetch_aqp4():
    """Download AQP4 structures from RCSB."""
    import requests
    os.makedirs('data/aqp4', exist_ok=True)

    for pdb_id in ['1RC2', '3GD8']:
        url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
        out_path = f'data/aqp4/{pdb_id}.pdb'
        if os.path.exists(out_path):
            continue
        print(f"  Downloading {pdb_id} from RCSB...")
        r = requests.get(url, timeout=60)
        if r.status_code == 200 and len(r.text) > 10000:
            with open(out_path, 'w') as f:
                f.write(r.text)
            print(f"  Saved {pdb_id}.pdb ({len(r.text)} bytes)")
        else:
            print(f"  Failed to download {pdb_id}")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Step 2: Extracellular Epitope Analysis
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def analyze_extracellular_epitopes(aqp4_pdb, chain, top_n=30):
    """Score AQP4 residues for extracellular epitope potential.

    Prioritizes residues in known extracellular loops using:
    - SASA (solvent accessible surface area)
    - Hydropathy (Kyte-Doolittle, positive = hydrophilic)
    - Protrusion index (structural exposure)
    - Extracellular loop membership bonus
    """
    print(f"\n[Step 2] Extracellular Epitope Analysis")
    print(f"  Known extracellular loops (1RC2 numbering):")
    for name, (s, e) in AQP4_EXTRACELLULAR_LOOPS_1RC2.items():
        print(f"    {name}: residues {s}-{e}")

    # Get full epitope scoring
    weights = {'sasa': 0.40, 'hydrophilicity': 0.35, 'protrusion': 0.25}
    epitope_data = score_epitope_residues(aqp4_pdb, chain, weights=weights)

    if not epitope_data:
        print("  ERROR: No epitope data computed")
        return [], {}

    # Extract sequence for display
    seq = extract_seq_from_pdb(aqp4_pdb, chain)

    # Build extracellular residue set
    extracellular_set = set()
    for s, e in AQP4_EXTRACELLULAR_BROAD:
        for r in range(s, e + 1):
            extracellular_set.add(r)

    # Annotate each residue
    for d in epitope_data:
        d['is_extracellular'] = d['resseq'] in extracellular_set
        loop_name = None
        for name, (s, e) in AQP4_EXTRACELLULAR_LOOPS_1RC2.items():
            if s <= d['resseq'] <= e:
                loop_name = name
                break
        d['extracellular_loop'] = loop_name

    # Separate extracellular vs intracellular
    ext_residues = [d for d in epitope_data if d['is_extracellular']]
    other_residues = [d for d in epitope_data if not d['is_extracellular']]

    print(f"\n  Total residues analyzed: {len(epitope_data)}")
    print(f"  Extracellular residues: {len(ext_residues)}")
    print(f"  Other residues: {len(other_residues)}")

    # Show top extracellular epitopes
    print(f"\n  {format_epitope_table(ext_residues[:top_n], top_n=top_n)}")

    # Build extracellular epitope region spec
    # Take top extracellular residues by combined score
    top_ext = sorted(ext_residues, key=lambda x: x['combined_score'], reverse=True)[:top_n]
    top_resseqs = [d['resseq'] for d in top_ext]
    region_spec = residues_to_region_spec(chain, top_resseqs)

    # Group by loop
    loop_groups = {}
    for d in ext_residues:
        loop = d.get('extracellular_loop') or 'other_extracellular'
        if loop not in loop_groups:
            loop_groups[loop] = []
        loop_groups[loop].append(d)

    print(f"\n  Epitope region spec: {region_spec}")

    # Build per-loop region specs for targeted design
    loop_regions = {}
    for loop_name, residues in loop_groups.items():
        if not residues:
            continue
        loop_resseqs = sorted(set(d['resseq'] for d in residues))
        loop_spec = residues_to_region_spec(chain, loop_resseqs)
        loop_regions[loop_name] = loop_spec
        print(f"  {loop_name} region: {loop_spec}")

    return epitope_data, {
        'ext_residues': ext_residues,
        'region_spec': region_spec,
        'loop_regions': loop_regions,
        'top_resseqs': top_resseqs,
        'loop_groups': loop_groups,
        'seq': seq,
    }


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Step 3: BFN Disorder Analysis
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def run_disorder_analysis(aqp4_pdb, chain, device, threshold=0.3):
    """Run BFN disorder head on AQP4 to confirm ordered extracellular regions."""
    print(f"\n[Step 3] BFN Disorder Analysis")

    try:
        from idp_disorder_analysis import predict_disorder, find_ordered_segments, format_disorder_report
        from bfn_loader import load_bfn

        model, config = load_bfn(device)

        result = predict_disorder(model, config, aqp4_pdb, chain, device)
        segments = find_ordered_segments(
            result['disorder_scores'], result['residue_ids'], threshold=threshold)
        report = format_disorder_report(
            result['disorder_scores'], result['sequence'],
            result['residue_ids'], threshold)

        print(report)

        # Check which ordered segments overlap with extracellular loops
        print(f"\n  Ordered segments overlapping extracellular loops:")
        for seg in segments:
            overlap = set(range(seg['start'], seg['end'] + 1))
            for name, (ls, le) in AQP4_EXTRACELLULAR_LOOPS_1RC2.items():
                loop_range = set(range(ls, le + 1))
                if overlap & loop_range:
                    shared = sorted(overlap & loop_range)
                    print(f"    [{seg['start']}-{seg['end']}] D={seg['mean_disorder']:.3f} "
                          f"鈫?{name} ({shared[0]}-{shared[-1]})")

        return result, segments, report

    except ImportError as e:
        print(f"  WARNING: Disorder analysis skipped (import error: {e})")
        return None, [], ""
    except RuntimeError as e:
        print(f"  WARNING: Disorder analysis skipped ({e})")
        return None, [], ""


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Step 4: Build Epitope Structures
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def build_epitope_structures(aqp4_pdb, chain, epitope_info, output_dir,
                             max_segments=3):
    """Extract epitope segments from AQP4 PDB as separate structures."""
    print(f"\n[Step 4] Building Epitope Structures")

    os.makedirs(output_dir, exist_ok=True)

    loop_groups = epitope_info['loop_groups']
    epitope_segments = []

    for loop_name, residues in loop_groups.items():
        if loop_name == 'other_extracellular':
            continue

        resseqs = sorted(set(d['resseq'] for d in residues))
        if not resseqs:
            continue

        # Use the full loop range for structure extraction
        loop_def = AQP4_EXTRACELLULAR_LOOPS_1RC2.get(loop_name)
        if loop_def:
            start, end = loop_def
        else:
            start, end = resseqs[0], resseqs[-1]

        try:
            epi_pdb = extract_epitope_pdb(
                aqp4_pdb, chain, (start, end),
                output_dir=output_dir, output_chain='B'
            )
            seq = extract_seq_from_pdb(epi_pdb, 'B')

            mean_score = np.mean([d['combined_score'] for d in residues if start <= d['resseq'] <= end])

            epitope_segments.append({
                'name': loop_name,
                'start': start, 'end': end,
                'pdb_path': epi_pdb,
                'sequence': seq,
                'n_residues': len(seq),
                'mean_epitope_score': float(mean_score),
                'top_residues': [d['resseq'] for d in residues[:5]],
            })

            print(f"  {loop_name}: {start}-{end} ({len(seq)} aa) 鈫?{epi_pdb}")
            print(f"    Mean epitope score: {mean_score:.3f}")
            print(f"    Sequence: {seq}")

        except Exception as e:
            print(f"  {loop_name}: EXTRACTION FAILED 鈥?{e}")

    # Sort by mean epitope score
    epitope_segments.sort(key=lambda x: x['mean_epitope_score'], reverse=True)

    return epitope_segments[:max_segments]


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Step 5: BFN Nanobody CDR Design
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def run_bfn_cdr_design(scaffold_pdb, scaffold_chain, epitope_segments,
                       cdr_spec, num_samples, stochastic, device):
    """Run BFN CDR design in Complex mode for each epitope segment.

    Complex mode: antigen (AQP4 epitope) is visible during design,
    providing interface context for better ipTM calibration.
    """
    print(f"\n[Step 5] BFN Nanobody CDR Design (Complex mode)")
    print(f"  Scaffold: {scaffold_pdb} (chain {scaffold_chain})")
    print(f"  CDRs: {cdr_spec}")
    print(f"  Samples per segment: {num_samples}")
    print(f"  Stochastic: {stochastic}")
    print(f"  Device: {device}")

    from bfn_loader import run_bfn_design
    from antibody_epitope_complex import position_epitope

    all_results = []

    for i, seg in enumerate(epitope_segments):
        print(f"\n  鈹€鈹€ Segment {i+1}/{len(epitope_segments)}: {seg['name']} "
              f"({seg['start']}-{seg['end']}, {seg['n_residues']}aa) 鈹€鈹€")

        try:
            # Position epitope facing scaffold CDRs
            complex_info = position_epitope(
                scaffold_pdb, seg['pdb_path'],
                scaffold_chain=scaffold_chain,
                epitope_chain='B',
                distance=6.0,
            )

            print(f"    Complex PDB: {complex_info['pdb_path']}")
            print(f"    Epitope distance: {complex_info['distance']:.1f}脜")

            # Run BFN design in Complex mode
            # context_chains=['B'] makes epitope chain B visible during design
            t0 = time.time()
            designs = run_bfn_design(
                complex_info['pdb_path'],
                cdr_spec,
                num_samples=num_samples,
                stochastic=stochastic,
                context_chains=['B'],  # Complex mode: epitope visible
                device=device,
            )
            elapsed = time.time() - t0

            if designs:
                # Enrich with segment info
                for d in designs:
                    d['epitope_segment'] = seg['name']
                    d['epitope_start'] = seg['start']
                    d['epitope_end'] = seg['end']
                    d['epitope_sequence'] = seg['sequence']
                    d['complex_pdb'] = complex_info['pdb_path']

                plddts = [d.get('plddt', 0) for d in designs]
                iptms = [d.get('iptm', 0) for d in designs]
                ppls = [d.get('ppl', 999) for d in designs]

                print(f"    Generated {len(designs)} designs in {elapsed:.0f}s")
                print(f"    pLDDT: {np.mean(plddts):.3f} 卤 {np.std(plddts):.3f} "
                      f"[{min(plddts):.3f}, {max(plddts):.3f}]")
                print(f"    ipTM:  {np.mean(iptms):.3f} 卤 {np.std(iptms):.3f} "
                      f"[{min(iptms):.3f}, {max(iptms):.3f}]")
                print(f"    PPL:   {np.mean(ppls):.1f} 卤 {np.std(ppls):.1f}")

                all_results.append({
                    'segment': seg,
                    'designs': designs,
                    'elapsed': elapsed,
                })
            else:
                print(f"    No designs generated")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    return all_results


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Step 6: Cascade Filtering & Ranking
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def rank_and_filter(all_design_results, cascade_config=None):
    """Apply cascade filter and rank all designs across segments."""
    print(f"\n[Step 6] Cascade Filtering & Ranking")

    if cascade_config is None:
        cascade_config = {
            'thresholds': {
                'plddt_min': 0.50,   # relaxed for AQP4 (membrane protein)
                'iptm_min': 0.30,
                'ppl_max': 150.0,
                'entropy_max': 2.8,
            },
            'weights': {
                'iptm': 0.35,
                'plddt': 0.25,
                'ppl_inv': 0.15,
                'entropy_inv': 0.10,
                'recovery': 0.15,
            },
        }

    # Collect all designs
    all_designs = []
    for seg_result in all_design_results:
        for d in seg_result['designs']:
            enriched = dict(d)
            enriched['_segment_name'] = seg_result['segment']['name']
            enriched['_epitope_seq'] = seg_result['segment']['sequence']
            all_designs.append(enriched)

    print(f"  Total designs before filter: {len(all_designs)}")

    if not all_designs:
        return [], [], "No designs to filter"

    # Apply cascade filter (3-stage: thresholds 鈫?dedup 鈫?composite score)
    from cascade_filter import apply_cascade

    ranked, cascade_report = apply_cascade(
        all_designs,
        thresholds=cascade_config['thresholds'],
        weights=cascade_config['weights'],
    )
    filtered = ranked  # apply_cascade already returns filtered + ranked

    print(cascade_report)

    if not ranked:
        print(f"  WARNING: No designs passed cascade filter, using raw ranking...")
        # Simple ranking by ipTM as fallback
        ranked = sorted(all_designs, key=lambda x: (x.get('iptm', 0) or 0), reverse=True)

    # Assign quality labels
    for r in ranked:
        cs = r.get('composite_score', 0)
        if cs >= 0.65:
            r['quality'] = 'HIGH'
        elif cs >= 0.45:
            r['quality'] = 'MEDIUM'
        else:
            r['quality'] = 'LOW'

    # Build report
    n_high = sum(1 for d in ranked if d['quality'] == 'HIGH')
    n_med = sum(1 for d in ranked if d['quality'] == 'MEDIUM')
    n_low = sum(1 for d in ranked if d['quality'] == 'LOW')

    report_lines = [
        "=" * 80,
        f"  AQP4 Extracellular Water Channel 鈥?Antibody Design Results",
        "=" * 80,
        f"  Total designs generated: {len(all_designs)}",
        f"  After filtering: {len(ranked)} unique sequences",
        f"  Quality: HIGH={n_high} MEDIUM={n_med} LOW={n_low}",
        "",
        f"  {'Rank':<5} {'Score':<8} {'Quality':<8} {'pLDDT':<8} {'ipTM':<8} "
        f"{'PPL':<8} {'Entropy':<8} {'Segment':<16} {'Sequence'}",
        f"  {'鈹€'*5} {'鈹€'*8} {'鈹€'*8} {'鈹€'*8} {'鈹€'*8} {'鈹€'*8} {'鈹€'*8} {'鈹€'*16} {'鈹€'*40}",
    ]

    for i, d in enumerate(ranked[:20]):
        seq = d.get('sequence', '')
        if len(seq) > 40:
            seq = seq[:18] + '鈥? + seq[-18:]
        seg = d.get('_segment_name', 'N/A')
        report_lines.append(
            f"  {i+1:<5} {d['composite_score']:<8.4f} {d['quality']:<8} "
            f"{d.get('plddt', 0) or 0:<8.3f} {d.get('iptm', 0) or 0:<8.3f} "
            f"{d.get('ppl', 0) or 0:<8.1f} {d.get('entropy', 0) or 0:<8.3f} "
            f"{seg:<16} {seq}"
        )

    report_lines.append("")
    report_lines.append("=" * 80)

    report = '\n'.join(report_lines)

    return ranked, filtered, report


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Step 7: Save Results
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def save_results(ranked, epitope_info, disorder_result, output_dir,
                 aqp4_pdb, scaffold_pdb, cdr_spec, config):
    """Save all results to JSON, FASTA, and Markdown report."""
    print(f"\n[Step 7] Saving Results 鈫?{output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # JSON summary
    summary = {
        'target': {
            'name': 'AQP4 (Aquaporin-4)',
            'type': 'Extracellular water channel',
            'pdb': str(aqp4_pdb),
            'chain': config.get('target_chain', 'A'),
            'extracellular_loops': {
                name: list(range(s, e+1))
                for name, (s, e) in AQP4_EXTRACELLULAR_LOOPS_1RC2.items()
            },
        },
        'scaffold': {
            'pdb': str(scaffold_pdb),
            'chain': config.get('scaffold_chain', 'B'),
            'type': 'nanobody (VHH)',
        },
        'design_config': {
            'cdr_spec': cdr_spec,
            'num_samples': config.get('num_samples', 10),
            'stochastic': config.get('stochastic', True),
            'mode': 'complex',
            'device': config.get('device', 'cuda'),
        },
        'epitope_analysis': {
            'total_residues_analyzed': len(epitope_info.get('ext_residues', [])),
            'extracellular_residues': len(epitope_info.get('ext_residues', [])),
            'top_epitope_region': epitope_info.get('region_spec', ''),
        },
        'filtering': {
            'total_generated': config.get('total_generated', 0),
            'unique_designs': len(ranked),
            'n_high': sum(1 for d in ranked if d.get('quality') == 'HIGH'),
            'n_medium': sum(1 for d in ranked if d.get('quality') == 'MEDIUM'),
            'n_low': sum(1 for d in ranked if d.get('quality') == 'LOW'),
        },
        'top_designs': [],
    }

    for i, d in enumerate(ranked[:20]):
        entry = {
            'rank': i + 1,
            'composite_score': d.get('composite_score'),
            'quality': d.get('quality'),
            'sequence': d.get('sequence'),
            'plddt': d.get('plddt'),
            'iptm': d.get('iptm'),
            'pae': d.get('pae'),
            'ppl': d.get('ppl'),
            'entropy': d.get('entropy'),
            'disorder_score': d.get('disorder_score'),
            'epitope_segment': d.get('_segment_name'),
            'cdr_length': len(d.get('sequence', '')),
        }
        summary['top_designs'].append(entry)

    json_path = os.path.join(output_dir, 'aqp4_water_channel_design_results.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  JSON: {json_path}")

    # FASTA output
    fasta_path = os.path.join(output_dir, 'aqp4_designed_cdrs.fasta')
    with open(fasta_path, 'w') as f:
        for i, d in enumerate(ranked[:50]):
            seq = d.get('sequence', '')
            seg = d.get('_segment_name', 'unknown')
            score = d.get('composite_score', 0)
            quality = d.get('quality', '?')
            f.write(f">design_{i+1:03d} | segment={seg} | composite={score:.4f} "
                    f"| quality={quality} | pLDDT={d.get('plddt',0):.3f} "
                    f"| ipTM={d.get('iptm',0):.3f} | PPL={d.get('ppl',0):.1f}\n")
            f.write(f"{seq}\n")
    print(f"  FASTA: {fasta_path} ({min(50, len(ranked))} sequences)")

    # Markdown report
    md_path = os.path.join(output_dir, 'aqp4_design_report.md')
    _write_markdown_report(md_path, summary, ranked, epitope_info)
    print(f"  Markdown: {md_path}")

    return json_path, fasta_path, md_path


def _write_markdown_report(md_path, summary, ranked, epitope_info):
    """Generate comprehensive Markdown design report."""
    t = summary['target']
    s = summary['scaffold']
    dc = summary['design_config']
    f = summary['filtering']

    lines = [
        f"# AQP4 Extracellular Water Channel 鈥?Antibody Design Report",
        f"",
        f"**Target**: {t['name']} ({t['type']})  ",
        f"**PDB**: {t['pdb']} (chain {t['chain']})  ",
        f"**Scaffold**: {s['pdb']} (chain {s['chain']}, {s['type']})  ",
        f"**Mode**: {dc['mode']} | **Samples/segment**: {dc['num_samples']} | **Device**: {dc['device']}",
        f"",
        f"---",
        f"",
        f"## Epitope Analysis",
        f"",
        f"| Loop | Residue Range | Key Residues |",
        f"|------|--------------|--------------|",
    ]

    for name, (start, end) in AQP4_EXTRACELLULAR_LOOPS_1RC2.items():
        key_res = []
        for d in epitope_info.get('ext_residues', []):
            if start <= d['resseq'] <= end:
                key_res.append(f"{d['resname']}{d['resseq']}({d['combined_score']:.3f})")
        lines.append(f"| {name} | {start}-{end} | {', '.join(key_res[:5])} |")

    lines += [
        f"",
        f"**Top epitope region**: `{epitope_info.get('region_spec', 'N/A')}`",
        f"",
        f"---",
        f"",
        f"## Design Results",
        f"",
        f"- **Total generated**: {f['total_generated']}",
        f"- **Unique designs**: {f['unique_designs']}",
        f"- **HIGH quality**: {f['n_high']} | **MEDIUM**: {f['n_medium']} | **LOW**: {f['n_low']}",
        f"",
        f"### Top 20 Designs",
        f"",
        f"| Rank | Score | Quality | pLDDT | ipTM | PPL | Entropy | Segment | Sequence |",
        f"|------|-------|---------|-------|------|-----|---------|---------|----------|",
    ]

    for d in ranked[:20]:
        seq = d.get('sequence', '')
        if len(seq) > 30:
            seq = seq[:13] + '鈥? + seq[-13:]
        seq = seq.replace('|', '\\|')

        lines.append(
            f"| {d.get('rank', '?')} | {d.get('composite_score', 0):.4f} | "
            f"{d.get('quality', '?')} | {d.get('plddt', 0) or 0:.3f} | "
            f"{d.get('iptm', 0) or 0:.3f} | {d.get('ppl', 0) or 0:.1f} | "
            f"{d.get('entropy', 0) or 0:.3f} | {d.get('_segment_name', 'N/A')} | "
            f"`{seq}` |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Configuration",
        f"",
        f"```yaml",
        f"cdr_spec: \"{dc['cdr_spec']}\"",
        f"mode: {dc['mode']}",
        f"num_samples: {dc['num_samples']}",
        f"stochastic: {dc['stochastic']}",
        f"device: {dc['device']}",
        f"```",
        f"",
        f"---",
        f"*Generated by AQP4 Water Channel Design Pipeline*  ",
        f"*{time.strftime('%Y-%m-%d %H:%M:%S')}*",
    ]

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Main Pipeline
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def main():
    parser = argparse.ArgumentParser(
        description='AQP4 Extracellular Water Channel Antibody Design',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
        Examples:
          python scripts/eval/aqp4_water_channel_design.py
          python scripts/eval/aqp4_water_channel_design.py --samples 30 --device cuda
          python scripts/eval/aqp4_water_channel_design.py --scaffold 5IMK --num-segments 2
        '''))

    # Target
    parser.add_argument('--aqp4-pdb', default='data/aqp4/1RC2.pdb',
                       help='Path to AQP4 PDB file')
    parser.add_argument('--target-chain', default='A',
                       help='AQP4 chain ID')

    # Scaffold
    parser.add_argument('--scaffold', default='data/misfolding_targets/5IMK.pdb',
                       help='Path to nanobody scaffold PDB')
    parser.add_argument('--scaffold-chain', default='B',
                       help='Scaffold chain ID')

    # CDR spec
    parser.add_argument('--cdr', default=DEFAULT_CDR,
                       help='CDR region specification')

    # Design params
    parser.add_argument('--samples', type=int, default=20,
                       help='BFN design samples per epitope segment')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic sampling (default: stochastic)')
    parser.add_argument('--num-segments', type=int, default=3,
                       help='Max epitope segments to design against')

    # Modes
    parser.add_argument('--skip-disorder', action='store_true',
                       help='Skip BFN disorder analysis')
    parser.add_argument('--fixbb', action='store_true',
                       help='Use FixBB mode instead of Complex mode')
    parser.add_argument('--epitope-only', action='store_true',
                       help='Run epitope analysis only (no design)')

    # Output
    parser.add_argument('--output-dir', default=None,
                       help='Output directory (default: design_results/aqp4_water_channel_TIMESTAMP)')
    parser.add_argument('--device', default=None,
                       help='Torch device (auto-detect: cuda > xpu > cpu)')

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        try:
            import torch
            if torch.cuda.is_available():
                args.device = 'cuda'
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                args.device = 'xpu'
            else:
                args.device = 'cpu'
        except ImportError:
            args.device = 'cpu'

    # Output directory
    if args.output_dir is None:
        args.output_dir = f'design_results/aqp4_water_channel_{int(time.time())}'
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("  AQP4 Extracellular Water Channel 鈥?Antibody Design Pipeline")
    print("  Bayesian Flow Network (BFN) + Epitope Analysis")
    print("=" * 80)
    print(f"  Target:      {args.aqp4_pdb} (chain {args.target_chain})")
    print(f"  Scaffold:    {args.scaffold} (chain {args.scaffold_chain})")
    print(f"  CDRs:        {args.cdr}")
    print(f"  Samples:     {args.samples}/segment")
    print(f"  Mode:        {'FixBB' if args.fixbb else 'Complex (antigen visible)'}")
    print(f"  Device:      {args.device}")
    print(f"  Output:      {args.output_dir}")
    print()

    # 鈹€鈹€ Step 1: Load AQP4 鈹€鈹€
    aqp4_pdb, aqp4_chain, aqp4_seq = load_aqp4_structure(args.aqp4_pdb, args.target_chain)

    # 鈹€鈹€ Step 2: Epitope Analysis 鈹€鈹€
    epitope_data, epitope_info = analyze_extracellular_epitopes(
        aqp4_pdb, aqp4_chain, top_n=30)

    if args.epitope_only:
        # Save epitope analysis only
        epitope_json = os.path.join(args.output_dir, 'aqp4_epitope_analysis.json')
        with open(epitope_json, 'w') as f:
            json.dump({
                'target': {'pdb': aqp4_pdb, 'chain': aqp4_chain},
                'extracellular_loops': {
                    name: {'start': s, 'end': e}
                    for name, (s, e) in AQP4_EXTRACELLULAR_LOOPS_1RC2.items()
                },
                'top_epitopes': [
                    {'rank': int(d['rank']), 'resseq': int(d['resseq']), 'resname': str(d['resname']),
                     'sasa': float(d['sasa']), 'rel_sasa': float(d['rel_sasa']),
                     'hydrophilicity': float(d['hydrophilicity']), 'protrusion': float(d['protrusion']),
                     'combined_score': float(d['combined_score']),
                     'extracellular_loop': d.get('extracellular_loop')}
                    for d in epitope_info['ext_residues'][:30]
                ],
                'region_spec': epitope_info['region_spec'],
                'loop_specific_regions': epitope_info.get('loop_regions', {}),
            }, f, indent=2)
        print(f"\n[Epitope analysis saved to: {epitope_json}]")
        print("Done (epitope-only mode).")
        return

    # 鈹€鈹€ Step 3: Disorder Analysis 鈹€鈹€
    disorder_result = None
    disorder_segments = []
    if not args.skip_disorder:
        disorder_result, disorder_segments, disorder_report = run_disorder_analysis(
            aqp4_pdb, aqp4_chain, args.device)

    # 鈹€鈹€ Step 4: Build Epitope Structures 鈹€鈹€
    epitope_segments = build_epitope_structures(
        aqp4_pdb, aqp4_chain, epitope_info, args.output_dir,
        max_segments=args.num_segments)

    if not epitope_segments:
        print("\nERROR: No epitope segments built. Check AQP4 PDB structure.")
        return

    # 鈹€鈹€ Step 5: BFN CDR Design 鈹€鈹€
    ctx = [] if args.fixbb else None  # None = Complex mode (default)
    all_design_results = run_bfn_cdr_design(
        args.scaffold, args.scaffold_chain,
        epitope_segments, args.cdr,
        num_samples=args.samples,
        stochastic=not args.deterministic,
        device=args.device,
    )

    if not all_design_results:
        print("\nERROR: No designs generated.")
        return

    # 鈹€鈹€ Step 6: Rank & Filter 鈹€鈹€
    ranked, filtered, report = rank_and_filter(all_design_results)
    print(f"\n{report}")

    # 鈹€鈹€ Step 7: Save Results 鈹€鈹€
    config = {
        'target_chain': aqp4_chain,
        'scaffold_chain': args.scaffold_chain,
        'num_samples': args.samples,
        'stochastic': not args.deterministic,
        'device': args.device,
        'total_generated': sum(len(r['designs']) for r in all_design_results),
    }

    json_path, fasta_path, md_path = save_results(
        ranked, epitope_info, disorder_result,
        args.output_dir, aqp4_pdb, args.scaffold, args.cdr, config)

    # 鈹€鈹€ Summary 鈹€鈹€
    print(f"\n{'='*80}")
    print(f"  AQP4 WATER CHANNEL ANTIBODY DESIGN 鈥?COMPLETE")
    print(f"{'='*80}")
    print(f"  Output: {args.output_dir}")
    print(f"  Files:")
    print(f"    {json_path}")
    print(f"    {fasta_path}")
    print(f"    {md_path}")

    if ranked:
        best = ranked[0]
        print(f"\n  Best Design:")
        print(f"    Score: {best['composite_score']:.4f} [{best['quality']}]")
        print(f"    pLDDT: {best.get('plddt', 0):.4f}  ipTM: {best.get('iptm', 0):.4f}")
        print(f"    PPL: {best.get('ppl', 0):.1f}  Segment: {best.get('_segment_name', '?')}")
        print(f"    Sequence: {best.get('sequence', '')}")

    print(f"{'='*80}")


if __name__ == '__main__':
    main()
