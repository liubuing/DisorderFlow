#!/usr/bin/env python
"""Run 8 Alzheimer's proteins through complete BFN workflow using actual PDB files.

Pipeline per protein:
  1. Load PDB from C:/biological/protein/
  2. BFN confidence evaluation (pLDDT/ipTM/PAE)
  3. BFN sequence design (8 samples)
  4. Cascade filter ranking
  5. Composite score report
"""
import sys, os, json, time, traceback

os.chdir(r'C:\biological\AntibodyDesignBFN-main\AntibodyDesignBFN-main')
sys.path.insert(0, '.')

print("=" * 70)
print("  8 Alzheimer's Proteins — Complete BFN Workflow Validation")
print("=" * 70)

# ── Imports ──────────────────────────────────────────────
from app import (
    load_bfn, run_bfn_protein, run_bfn_confidence_evaluation,
    CONFIDENCE_DEFAULTS,
)
from cascade_filter import apply_cascade

# ── Load model ───────────────────────────────────────────
print("\n[1/4] Loading BFN model...")
model, config = load_bfn()
print(f"  Model: {type(model).__name__}")
print(f"  Checkpoint: {config.get('_checkpoint_path', 'default')}")

# ── Protein definitions ──────────────────────────────────
PDB_DIR = r'C:\biological\protein'

# Each protein: name, pdb_filename, chain, design_region
# Region chosen to be 30-60aa within model's training range (50-250)
PROTEINS = [
    {'name': 'ENO1',   'pdb': 'ENO1.pdb',   'chain': 'A', 'region': 'A:10-55'},
    {'name': 'ENO2',   'pdb': 'ENO2.pdb',   'chain': 'A', 'region': 'A:10-55'},
    {'name': 'FABP3',  'pdb': 'FABP3.pdb',  'chain': 'A', 'region': 'A:5-50'},
    {'name': 'GLOD4',  'pdb': 'GLOD4.pdb',  'chain': 'A', 'region': 'A:5-50'},
    {'name': 'MAPT',   'pdb': 'MAPT.pdb',   'chain': 'A', 'region': 'A:10-55'},
    {'name': 'MIF',    'pdb': 'MIF.pdb',    'chain': 'A', 'region': 'A:10-55'},
    {'name': 'NRGN',   'pdb': 'NRGN.pdb',   'chain': 'A', 'region': 'A:5-50'},
    {'name': 'TMSB10','pdb': 'TMSB10.pdb',  'chain': 'A', 'region': 'A:2-42'},
]

summary = {}
all_passed = 0
all_failed = 0
results_for_report = []

print("\n[2/4] Running workflow on each protein...")

for i, prot in enumerate(PROTEINS):
    name = prot['name']
    pdb_path = os.path.join(PDB_DIR, prot['pdb'])
    region = prot['region']

    print(f"\n{'─'*60}")
    print(f"  [{i+1}/8] {name} — Region: {region}")
    print(f"  PDB: {pdb_path}")
    print(f"{'─'*60}")

    if not os.path.exists(pdb_path):
        print(f"  SKIP: PDB not found")
        summary[name] = {'status': 'SKIP (PDB not found)'}
        all_failed += 1
        continue

    entry = {'name': name, 'pdb': prot['pdb'], 'region': region}
    start_time = time.time()

    try:
        # Step A: Confidence evaluation
        print(f"  [A] Confidence evaluation...")
        conf_report, conf_df = run_bfn_confidence_evaluation(pdb_path, region)

        # Parse confidence values
        plddt_mean = None
        iptm_val = None
        pae_mean = None
        for line in conf_report.split('\n'):
            line_clean = line.strip()
            if 'pLDDT (region mean)' in line_clean:
                try:
                    parts = line_clean.split(':')
                    if len(parts) >= 2:
                        plddt_mean = float(parts[1].strip().split()[0])
                except: pass
            if 'ipTM (global)' in line_clean:
                try:
                    parts = line_clean.split(':')
                    if len(parts) >= 2:
                        iptm_val = float(parts[1].strip().split()[0])
                except: pass
            if 'PAE (region mean)' in line_clean:
                try:
                    parts = line_clean.split(':')
                    if len(parts) >= 2:
                        pae_mean = float(parts[1].strip().split()[0])
                except: pass

        print(f"      pLDDT={plddt_mean:.3f}" if plddt_mean else "      pLDDT=N/A",
              f"ipTM={iptm_val:.3f}" if iptm_val else "ipTM=N/A",
              f"PAE={pae_mean:.2f}" if pae_mean else "PAE=N/A")

        entry['pre_confidence'] = {
            'plddt': plddt_mean,
            'iptm': iptm_val,
            'pae': pae_mean,
        }

        # Step B: BFN sequence design
        print(f"  [B] BFN sequence design (8 samples)...")
        design_text, fasta_str, results_list = run_bfn_protein(
            pdb_path, region,
            num_samples=8,
            stochastic=True,
            eval_mode=False,
        )

        n_designs = len(results_list) if results_list else 0
        print(f"      Generated {n_designs} designs")

        if n_designs == 0:
            print(f"  WARNING: No designs generated")
            entry['status'] = 'FAIL (no designs)'
            entry['designs'] = 0
            entry['filter_passed'] = 0
            summary[name] = entry
            all_failed += 1
            continue

        # Step C: Cascade filter
        print(f"  [C] Cascade filter...")
        filtered, filter_report = apply_cascade(results_list)
        n_passed = len(filtered)
        print(f"      {n_passed}/{n_designs} designs passed filter")

        entry['designs'] = n_designs
        entry['filter_passed'] = n_passed
        entry['all_results'] = []

        # Collect all design results
        for r in results_list:
            entry['all_results'].append({
                'sequence': r.get('sequence', '')[:40] + ('...' if len(r.get('sequence', '')) > 40 else ''),
                'ppl': r.get('ppl'),
                'entropy': r.get('entropy'),
                'plddt': r.get('plddt'),
                'iptm': r.get('iptm'),
                'pae': r.get('pae'),
                'recovery': r.get('recovery'),
                'composite_score': r.get('composite_score'),
            })

        if filtered:
            top = filtered[0]
            entry['top_result'] = {
                'sequence': top.get('sequence', ''),
                'plddt': top.get('plddt'),
                'iptm': top.get('iptm'),
                'pae': top.get('pae'),
                'ppl': top.get('ppl'),
                'entropy': top.get('entropy'),
                'recovery': top.get('recovery'),
                'composite_score': top.get('composite_score'),
            }
            cs = top.get('composite_score')
            cs_str = f"{cs:.3f}" if isinstance(cs, (int, float)) else str(cs)
            print(f"      Top: pLDDT={top.get('plddt', 0):.3f} ipTM={top.get('iptm', 0):.3f} "
                  f"PPL={top.get('ppl', 0):.1f} Composite={cs_str}")
            entry['status'] = 'PASS'
            all_passed += 1
        else:
            entry['status'] = 'FAIL (all designs filtered out)'
            all_failed += 1

        elapsed = time.time() - start_time
        print(f"  Time: {elapsed:.1f}s")

        summary[name] = entry

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        entry['status'] = f'ERROR: {str(e)[:120]}'
        entry['error'] = traceback.format_exc()
        summary[name] = entry
        all_failed += 1

# ── Final Report ─────────────────────────────────────────
print("\n" + "=" * 70)
print("  WORKFLOW VALIDATION SUMMARY")
print("=" * 70)
print(f"  Total: {len(PROTEINS)} | Passed: {all_passed} | Failed: {all_failed}")
print()

# Header
hdr = f"  {'Protein':<10} {'Status':<6} {'Designs':<8} {'Filter':<7} {'pLDDT':<8} {'ipTM':<8} {'PPL':<8} {'Composite':<10}"
print(hdr)
print(f"  {'-'*(len(hdr)-2)}")

for prot in PROTEINS:
    name = prot['name']
    s = summary.get(name, {})
    top = s.get('top_result', {}) or {}
    pre = s.get('pre_confidence', {}) or {}

    status = 'OK' if s.get('status') == 'PASS' else 'FAIL'
    designs = str(s.get('designs', '?'))
    fpass = str(s.get('filter_passed', '?'))

    plddt_s = f"{top.get('plddt', 0):.3f}" if isinstance(top.get('plddt'), float) else 'N/A'
    iptm_s = f"{top.get('iptm', 0):.3f}" if isinstance(top.get('iptm'), float) else 'N/A'
    ppl_s = f"{top.get('ppl', 0):.1f}" if isinstance(top.get('ppl'), float) else 'N/A'
    comp_s = f"{top.get('composite_score', 0):.3f}" if isinstance(top.get('composite_score'), float) else 'N/A'

    print(f"  {name:<10} {status:<6} {designs:<8} {fpass:<7} {plddt_s:<8} {iptm_s:<8} {ppl_s:<8} {comp_s:<10}")

# ── Detailed report ──────────────────────────────────────
print("\n[3/4] Generating detailed report...")

# Per-protein detailed results
print("\n  Per-protein details:")
print(f"  {'Protein':<10} {'Pre pLDDT':<11} {'Pre ipTM':<9} {'Top pLDDT':<11} {'Top ipTM':<9} {'Top PPL':<9} {'Top Entropy':<12}")
print(f"  {'-'*70}")
for prot in PROTEINS:
    name = prot['name']
    s = summary.get(name, {})
    pre = s.get('pre_confidence', {}) or {}
    top = s.get('top_result', {}) or {}

    pre_p = f"{pre.get('plddt', 0):.3f}" if isinstance(pre.get('plddt'), float) else 'N/A'
    pre_i = f"{pre.get('iptm', 0):.3f}" if isinstance(pre.get('iptm'), float) else 'N/A'
    top_p = f"{top.get('plddt', 0):.3f}" if isinstance(top.get('plddt'), float) else 'N/A'
    top_i = f"{top.get('iptm', 0):.3f}" if isinstance(top.get('iptm'), float) else 'N/A'
    top_ppl = f"{top.get('ppl', 0):.1f}" if isinstance(top.get('ppl'), float) else 'N/A'
    top_ent = f"{top.get('entropy', 0):.3f}" if isinstance(top.get('entropy'), float) else 'N/A'

    print(f"  {name:<10} {pre_p:<11} {pre_i:<9} {top_p:<11} {top_i:<9} {top_ppl:<9} {top_ent:<12}")

# ── Save reports ─────────────────────────────────────────
report_data = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'model_checkpoint': config.get('_checkpoint_path', 'unknown'),
    'total': len(PROTEINS),
    'passed': all_passed,
    'failed': all_failed,
    'results': summary,
}

# JSON report
json_path = 'workflow_8proteins_report.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
print(f"\n[4/4] Reports saved:")
print(f"  JSON: {json_path}")

# Markdown summary report
md_path = 'workflow_8proteins_report.md'
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(f"# 8 Alzheimer's Proteins — BFN Workflow Validation Report\n\n")
    f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"**Model**: V6 Phase 2 ({config.get('_checkpoint_path', 'unknown')})\n\n")
    f.write(f"## Summary\n\n")
    f.write(f"| Metric | Value |\n")
    f.write(f"|--------|-------|\n")
    f.write(f"| Total proteins | {len(PROTEINS)} |\n")
    f.write(f"| Passed cascade | {all_passed} |\n")
    f.write(f"| Failed | {all_failed} |\n\n")

    f.write(f"## Results\n\n")
    f.write(f"| Protein | Status | Pre pLDDT | Pre ipTM | Top pLDDT | Top ipTM | Top PPL | Top Entropy | Composite |\n")
    f.write(f"|---------|--------|-----------|----------|-----------|----------|---------|-------------|-----------|\n")

    for prot in PROTEINS:
        name = prot['name']
        s = summary.get(name, {})
        pre = s.get('pre_confidence', {}) or {}
        top = s.get('top_result', {}) or {}
        status = 'OK' if s.get('status') == 'PASS' else 'FAIL'

        pre_p = f"{pre.get('plddt', 0):.3f}" if isinstance(pre.get('plddt'), float) else 'N/A'
        pre_i = f"{pre.get('iptm', 0):.3f}" if isinstance(pre.get('iptm'), float) else 'N/A'
        top_p = f"{top.get('plddt', 0):.3f}" if isinstance(top.get('plddt'), float) else 'N/A'
        top_i = f"{top.get('iptm', 0):.3f}" if isinstance(top.get('iptm'), float) else 'N/A'
        top_ppl = f"{top.get('ppl', 0):.1f}" if isinstance(top.get('ppl'), float) else 'N/A'
        top_ent = f"{top.get('entropy', 0):.3f}" if isinstance(top.get('entropy'), float) else 'N/A'
        comp = f"{top.get('composite_score', 0):.3f}" if isinstance(top.get('composite_score'), float) else 'N/A'

        f.write(f"| {name} | {status} | {pre_p} | {pre_i} | {top_p} | {top_i} | {top_ppl} | {top_ent} | {comp} |\n")

    f.write(f"\n## Notes\n\n")
    f.write(f"- MAPT (Tau) is an intrinsically disordered protein (IDP) — BFN confidence heads are known to be unreliable on IDPs\n")
    f.write(f"- Model trained on L=50-250; TMSB10 (44 aa) is below the training range\n")
    f.write(f"- Cascade thresholds: pLDDT≥0.6, ipTM≥0.4, PPL≤100, entropy≤2.5\n")

print(f"  Markdown: {md_path}")

print("\n" + "=" * 70)
print("  DONE — All 8 proteins processed!")
print("=" * 70)
