#!/usr/bin/env python
"""Validation: BFN confidence recycling + cascade filter on 8 target proteins.

Proteins: FABP3, ENO2, ENO1, MAPT, NRGN, MIF, TMSB10, GLOD4
Checks:  1. Recycling does not crash
         2. Confidence scores (pLDDT/ipTM/PAE) are produced
         3. Cascade filter runs without errors
         4. Recycling improves confidence vs no-recycling baseline
"""
import sys, os, torch, time, json

# UniProt canonical sequences (first 100 aa for speed, full for small proteins)
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

CONFIG_PATH = 'configs/demo_design.yml'
DEVICE = 'cpu'

def run_bfn_design(seq, name, num_samples=3):
    """Run BFN on a 20aa region of the protein with and without recycling."""
    from app import generate_pdb_from_sequence, load_bfn, AA_LETTERS, AA_NAMES
    from antibodydesignbfn.datasets.protein import preprocess_protein_structure
    from antibodydesignbfn.utils.train import recursive_to
    from antibodydesignbfn.utils.misc import seed_all
    from antibodydesignbfn.utils.data import PaddingCollate
    from antibodydesignbfn.utils.transforms import get_transform

    # Generate PDB from first 50 residues of sequence
    short_seq = seq[:50]
    pdb_text = generate_pdb_from_sequence(short_seq)
    if pdb_text is None:
        return {'error': 'PDB generation failed'}

    tmp = f'val_{name}.pdb'
    with open(tmp, 'w') as f:
        f.write(pdb_text)

    try:
        model, config = load_bfn()
        seed_all(42)
        structure = preprocess_protein_structure(tmp, chain_ids=['A'])
        if structure is None:
            os.unlink(tmp)
            return {'error': 'Structure parse failed'}

        transform = get_transform([
            {'type': 'mask_region', 'regions': {'A': list(range(10, 30))}},
            {'type': 'merge_protein'},
            {'type': 'patch_protein'},
        ])
        data = transform(structure)
        batch = recursive_to(PaddingCollate()([data]), DEVICE)
        gen_mask = batch['generate_flag'][0].bool()

        # Test without recycling (baseline)
        sample_opt_no = {'deterministic': False, 'num_recycles': 1}
        sample_opt_yes = {'deterministic': False, 'num_recycles': 3}

        results = {}
        for label, sopt in [('recycle=1', sample_opt_no), ('recycle=3', sample_opt_yes)]:
            t0 = time.time()
            sample_results = []
            for i in range(num_samples):
                with torch.no_grad():
                    traj = model.sample(batch, sample_opt=sopt)
                pred_aa = traj[0][2][0][gen_mask]
                seq_des = ''.join(AA_LETTERS[a] if a < 20 else 'X' for a in pred_aa.cpu())
                plddt = traj['plddt'][0][gen_mask].mean().item()
                iptm = traj['iptm'][0].item()
                pae = traj['pae'][0][gen_mask][:, gen_mask].mean().item()
                logits = traj['pred_logits'][0][gen_mask]
                lp = torch.log_softmax(logits[..., :20], dim=-1)
                nll = -lp[range(len(pred_aa)), pred_aa].mean()
                ppl = torch.exp(nll).item()
                sample_results.append({
                    'sequence': seq_des, 'ppl': ppl,
                    'plddt': plddt, 'iptm': iptm, 'pae': pae
                })
            elapsed = time.time() - t0
            avg_plddt = sum(r['plddt'] for r in sample_results) / len(sample_results)
            avg_iptm = sum(r['iptm'] for r in sample_results) / len(sample_results)
            avg_ppl = sum(r['ppl'] for r in sample_results) / len(sample_results)
            results[label] = {
                'avg_plddt': avg_plddt, 'avg_iptm': avg_iptm,
                'avg_ppl': avg_ppl, 'time': elapsed, 'samples': sample_results
            }

        os.unlink(tmp)
        return results

    except Exception as e:
        import traceback
        if os.path.exists(tmp):
            os.unlink(tmp)
        return {'error': str(e), 'trace': traceback.format_exc()}


if __name__ == '__main__':
    print("=" * 70)
    print("  VALIDATION: BFN Recycling + Cascade Filter")
    print("  Target proteins: FABP3, ENO2, ENO1, MAPT, NRGN, MIF, TMSB10, GLOD4")
    print("=" * 70)

    from cascade_filter import apply_cascade

    summary = {}
    passed = 0
    failed = 0

    for name, seq in SEQUENCES.items():
        print(f"\n--- {name} ({len(seq)} aa) ---")
        result = run_bfn_design(seq, name, num_samples=3)

        if 'error' in result:
            print(f"  ❌ FAILED: {result['error']}")
            failed += 1
            summary[name] = result
            continue

        r1 = result['recycle=1']
        r3 = result['recycle=3']
        delta_iptm = r3['avg_iptm'] - r1['avg_iptm']
        delta_plddt = r3['avg_plddt'] - r1['avg_plddt']
        delta_ppl = r3['avg_ppl'] - r1['avg_ppl']

        print(f"  recycle=1 → avg pLDDT={r1['avg_plddt']:.3f}  ipTM={r1['avg_iptm']:.3f}  PPL={r1['avg_ppl']:.2f}  time={r1['time']:.0f}s")
        print(f"  recycle=3 → avg pLDDT={r3['avg_plddt']:.3f}  ipTM={r3['avg_iptm']:.3f}  PPL={r3['avg_ppl']:.2f}  time={r3['time']:.0f}s")
        print(f"  Δ ipTM={delta_iptm:+.3f}  Δ pLDDT={delta_plddt:+.3f}  Δ PPL={delta_ppl:+.2f}")

        # Test cascade filter
        filtered, report = apply_cascade(r3['samples'])
        n_pass = len(filtered)
        print(f"  Cascade filter: {n_pass}/{len(r3['samples'])} passed")

        summary[name] = {
            'recycle1': {'plddt': r1['avg_plddt'], 'iptm': r1['avg_iptm'], 'ppl': r1['avg_ppl'], 'time': r1['time']},
            'recycle3': {'plddt': r3['avg_plddt'], 'iptm': r3['avg_iptm'], 'ppl': r3['avg_ppl'], 'time': r3['time']},
            'delta_iptm': delta_iptm, 'delta_plddt': delta_plddt, 'delta_ppl': delta_ppl,
            'cascade_passed': n_pass,
        }
        passed += 1

    # Final summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Passed: {passed}/{len(SEQUENCES)}  Failed: {failed}/{len(SEQUENCES)}")
    print()
    print(f"  {'Protein':<10} {'ipTM(×1)':<10} {'ipTM(×3)':<10} {'ΔipTM':<8} {'pLDDT(×3)':<10} {'Filter':<8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")
    for name, s in summary.items():
        if 'recycle3' in s:
            r1 = s['recycle1']; r3 = s['recycle3']
            print(f"  {name:<10} {r1['iptm']:<10.3f} {r3['iptm']:<10.3f} {s['delta_iptm']:<+8.3f} {r3['plddt']:<10.3f} {s['cascade_passed']}/3")

    print()
    # Check for regression
    positives = sum(1 for s in summary.values() if s.get('delta_iptm', 0) > 0)
    negatives = sum(1 for s in summary.values() if s.get('delta_iptm', 0) <= 0)
    print(f"  ipTM improvement: {positives}/{passed} proteins (Δ > 0)")
    print(f"  No crashes or errors: {'YES' if failed == 0 else 'NO (' + str(failed) + ' failed)'}")
    print("=" * 70)

    # Save detailed results
    with open('validation_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nDetailed results saved to validation_results.json")
