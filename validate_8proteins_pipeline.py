#!/usr/bin/env python
"""Phase 5: 8 Alzheimer's proteins through full unified pipeline validation.

Pipeline per protein:
  1. Generate helical PDB from sequence
  2. Confidence evaluation (pLDDT/ipTM/PAE)
  3. BFN sequence design (8 samples, stochastic)
  4. Cascade filter ranking
  5. Composite score report

Proteins: FABP3, ENO2, ENO1, MAPT, NRGN, MIF, TMSB10, GLOD4
"""
import sys, os, json, time, tempfile, traceback

SEQUENCES = {
    'FABP3':  'MVDAFLGTWKLVDSKNFDDYMKSLGVGFATRQVASMTKPTTIIEKNGDILTLKTHSTFKNTEISFKLGVEFDETTADDRKVKSIVTLDGGKLVHLQKWDGQET',
    'ENO2':   'MSIEKIWAREILDSRGNPTVEVDLYTAKGLFRAAVPSGASTGIYEALELRDGDKGRYLGKGVLKAVENINNTLGPALLQKKLSVVDQEKVDKFMIELDG',
    'ENO1':   'MSILKIHAREIFDSRGNPTVEVDLFTSKGLFRAAVPSGASTGIYEALELRDNDKTRYMGKGVSRAVEHINKTIAPALVSKKLNVTEQEKIDKLMIEMDG',
    'MAPT':   'MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKESPLQTPTEDGSEEPGSETSDAKSTPTAEDVTAPLVDEGAPGKQAAAQPHTEIPEG',
    'NRGN':   'MDCCTENACSKPDDDILDIPLDDPGANAAAAKIQASFRGHMARKKIKSGECGRKGPGPGGPGGAGGARGGAGGGPSGD',
    'MIF':    'MPMFIVNTNVPRASVPDGFLSELTQQLAQATGKPPQYIAVHVVPDQLMAFGGSSEPCALCSLHSIGKIGGAQNRSYSKLLCGLLAERLRISPDRVYINYY',
    'TMSB10': 'MADKPDMGEIASFDKAKLKKTETQEKNTLPTKETIEQEKRSEIS',
    'GLOD4':  'MAAVQALEVLKEQGLVQLRAQGTGTSNGLTAQKHYLLGNVLKPNKGSGVQGWRVGSVFHQDPENPSLFLGQGGQCVSLWGRDVPGSPAAGALQAPGDL',
}

def generate_pdb_from_sequence(sequence):
    """Generate an alpha-helical PDB from a sequence."""
    import math
    lines = ['REMARK    Generated helical structure for validation']
    # Alpha-helix params: 1.5A rise/residue, 100 deg rotation, ~3.6 res/turn
    rise = 1.5
    rotation = math.radians(100)
    radius = 2.3
    atom_serial = 0
    for i, aa in enumerate(sequence):
        angle = i * rotation
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = i * rise
        atom_serial += 1
        lines.append(f"ATOM  {atom_serial:5d}  N   {aa:3s} A{1+i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           N")
    atom_serial += 1
    lines.append(f"TER   {atom_serial:5d}      A{len(sequence):4d}")
    lines.append('END')
    return '\n'.join(lines)


def main():
    print("=" * 70)
    print("  PHASE 5: 8 Alzheimer's Proteins — Full Pipeline Validation")
    print("=" * 70)

    # Import app components
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from app import (
        load_bfn, run_bfn_protein, run_bfn_confidence_evaluation,
        build_results_dataframe, CONFIDENCE_DEFAULTS, confidence_quality_label,
    )
    from cascade_filter import apply_cascade

    model, config = load_bfn()
    print(f"Model loaded: {type(model).__name__}")
    print(f"Checkpoint: {config.get('_checkpoint_path', 'default')}")

    summary = {}
    all_passed = 0
    all_failed = 0

    for name, seq in SEQUENCES.items():
        print(f"\n{'─'*60}")
        print(f"  {name} ({len(seq)} aa)")
        print(f"{'─'*60}")

        entry = {'name': name, 'length': len(seq)}
        tmp_pdb = None

        try:
            # Step 1: Generate PDB (use first 50 aa for speed)
            pdb_text = generate_pdb_from_sequence(seq[:60])
            tmp_pdb = tempfile.NamedTemporaryFile(
                mode='w', suffix='.pdb', delete=False, prefix=f'v8_{name}_')
            tmp_pdb.write(pdb_text)
            tmp_pdb.close()
            pdb_path = tmp_pdb.name

            # Step 2: Confidence evaluation
            conf_report, conf_df = run_bfn_confidence_evaluation(pdb_path, 'A:10-40')
            print(f"  Confidence: {conf_report.split(chr(10))[3] if len(conf_report.split(chr(10))) > 3 else 'N/A'}")

            # Extract confidence values from report
            plddt_mean = None
            iptm_val = None
            for line in conf_report.split('\n'):
                if 'pLDDT (region mean)' in line and ':' in line:
                    try: plddt_mean = float(line.split(':')[1].strip().split()[0])
                    except: pass
                if 'ipTM (global)' in line and ':' in line:
                    try: iptm_val = float(line.split(':')[1].strip().split()[0])
                    except: pass
            entry['confidence'] = {'plddt': plddt_mean, 'iptm': iptm_val}

            # Step 3: BFN Design (8 samples, stochastic)
            design_text, fasta_str, results_list = run_bfn_protein(
                pdb_path, 'A:10-40', num_samples=8, stochastic=True, eval_mode=False)

            n_designs = len(results_list)
            print(f"  Designs generated: {n_designs}")

            # Step 4: Cascade filter
            filtered, filter_report = apply_cascade(results_list)
            n_passed = len(filtered)
            print(f"  Cascade filter: {n_passed}/{n_designs} passed")

            entry['designs'] = n_designs
            entry['filter_passed'] = n_passed

            if filtered:
                top = filtered[0]
                entry['top_result'] = {
                    'sequence': top.get('sequence', '')[:40],
                    'plddt': top.get('plddt'),
                    'iptm': top.get('iptm'),
                    'pae': top.get('pae'),
                    'ppl': top.get('ppl'),
                    'composite_score': top.get('composite_score'),
                }
                print(f"  Top: pLDDT={top.get('plddt','?'):.3f} ipTM={top.get('iptm','?'):.3f} "
                      f"Composite={top.get('composite_score','?'):.3f}" if isinstance(top.get('composite_score'), float) else f"  Top: pLDDT={top.get('plddt','?'):.3f} ipTM={top.get('iptm','?'):.3f}")
                entry['status'] = 'PASS'
                all_passed += 1
            else:
                entry['status'] = 'FAIL (no designs passed filter)'
                all_failed += 1

            summary[name] = entry

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            entry['status'] = f'ERROR: {str(e)[:100]}'
            entry['error'] = traceback.format_exc()
            summary[name] = entry
            all_failed += 1
        finally:
            if tmp_pdb and os.path.exists(tmp_pdb.name):
                try: os.unlink(tmp_pdb.name)
                except: pass

    # Final report
    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Total: {len(SEQUENCES)} | Passed: {all_passed} | Failed: {all_failed}")
    print()
    header = f"  {'Protein':<10} {'Len':<5} {'Status':<6} {'Designs':<8} {'Filter':<7} {'Top pLDDT':<10} {'Top ipTM':<10} {'Composite':<10}"
    print(header)
    print(f"  {'-'*(len(header)-2)}")
    for name, s in summary.items():
        top = s.get('top_result', {})
        plddt_s = f"{top.get('plddt', 0):.3f}" if isinstance(top.get('plddt'), float) else 'N/A'
        iptm_s = f"{top.get('iptm', 0):.3f}" if isinstance(top.get('iptm'), float) else 'N/A'
        comp_s = f"{top.get('composite_score', 0):.3f}" if isinstance(top.get('composite_score'), float) else 'N/A'
        status = 'OK' if s.get('status') == 'PASS' else 'FAIL'
        print(f"  {name:<10} {s.get('length','?'):<5} {status:<6} {s.get('designs','?'):<8} {s.get('filter_passed','?'):<7} {plddt_s:<10} {iptm_s:<10} {comp_s:<10}")

    # Save detailed results
    report_path = 'phase5_validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nDetailed report saved to {report_path}")

    return all_failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
