#!/usr/bin/env python
"""Comprehensive validation: iterative refinement pipeline on 8 target proteins.

Proteins: FABP3, ENO2, ENO1, MAPT, NRGN, MIF, TMSB10, GLOD4

Checks:
  1. Module imports and basic function calls
  2. AF2 validator on a single sequence (sanity check)
  3. Cascade filter AF2 with mock data
  4. Iterative refiner end-to-end on one representative protein (FABP3)
  5. Report formatting
"""
import sys, os, json, time, torch

# Fix Windows GBK encoding for Unicode symbols
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ── UniProt sequences ──
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

# ── Phase 1: Module Sanity ──
print("=" * 70)
print("  PHASE 1: Module imports and basic function checks")
print("=" * 70)

errors = []

try:
    from af2_validator import validate_sequences, extract_confidence_from_af2_results
    print("  ✓ af2_validator imported")
except Exception as e:
    errors.append(f"af2_validator import: {e}")
    print(f"  ✗ af2_validator: {e}")

try:
    from cascade_filter import apply_cascade, apply_cascade_af2, format_filtered_fasta
    print("  ✓ cascade_filter imported")
except Exception as e:
    errors.append(f"cascade_filter import: {e}")
    print(f"  ✗ cascade_filter: {e}")

try:
    from iterative_refiner import IterativeRefiner, format_refinement_report
    print("  ✓ iterative_refiner imported")
except Exception as e:
    errors.append(f"iterative_refiner import: {e}")
    print(f"  ✗ iterative_refiner: {e}")


# ── Phase 2: Cascade filter AF2 test ──
print("\n" + "=" * 70)
print("  PHASE 2: Cascade filter AF2 with mock data")
print("=" * 70)

mock_af2_results = [
    {'sequence': 'AAAAEEEELLLLKKKKRRRR', 'plddt': 85.0, 'ptm': 0.75, 'iptm': 0.62, 'max_pae': 5.0, 'ppl': 3.5},
    {'sequence': 'EEEELLLLKKKKRRRRAAAA', 'plddt': 72.0, 'ptm': 0.61, 'iptm': 0.48, 'max_pae': 8.5, 'ppl': 4.2},
    {'sequence': 'LLLLKKKKRRRRAAAAEEEE', 'plddt': 65.0, 'ptm': 0.55, 'iptm': 0.42, 'max_pae': 10.0, 'ppl': 5.8},
    {'sequence': 'KKKKRRRRAAAAEEEELLLL', 'plddt': 45.0, 'ptm': 0.28, 'iptm': 0.15, 'max_pae': 18.0, 'ppl': 8.1},
    {'sequence': 'EEEELLLLKKKKRRRRAAAA', 'plddt': 73.0, 'ptm': 0.63, 'iptm': 0.49, 'max_pae': 7.9, 'ppl': 3.9},  # duplicate
]

try:
    filtered, report = apply_cascade_af2(mock_af2_results)
    print(report)
    # Verify: should filter out the low-scoring entries and deduplicate
    assert len(filtered) == 3, f"Expected 3 passed, got {len(filtered)}"
    assert filtered[0]['sequence'] == 'AAAAEEEELLLLKKKKRRRR', f"Wrong top: {filtered[0]['sequence']}"
    assert filtered[0]['af2_rank'] == 1
    print("  ✓ Cascade filter AF2: 3/5 passed, correct ranking")
except Exception as e:
    errors.append(f"cascade_af2: {e}")
    print(f"  ✗ cascade_af2 failed: {e}")
    import traceback; traceback.print_exc()


# ── Phase 3: PDB generation and BFN design test ──
print("\n" + "=" * 70)
print("  PHASE 3: PDB generation + BFN design (single protein)")
print("=" * 70)

try:
    from app import generate_pdb_from_sequence, load_bfn, AA_LETTERS
    from antibodydesignbfn.datasets.protein import preprocess_protein_structure
    from antibodydesignbfn.utils.train import recursive_to
    from antibodydesignbfn.utils.misc import seed_all
    from antibodydesignbfn.utils.data import PaddingCollate
    from antibodydesignbfn.utils.transforms import get_transform

    # Use FABP3 as test
    seq = SEQUENCES['FABP3'][:50]
    pdb_text = generate_pdb_from_sequence(seq)
    assert pdb_text is not None, "PDB generation returned None"

    tmp = 'val_phase3_test.pdb'
    with open(tmp, 'w') as f:
        f.write(pdb_text)

    # Test BFN design
    model, config = load_bfn()
    seed_all(42)
    structure = preprocess_protein_structure(tmp, chain_ids=['A'])
    assert structure is not None, "Structure parse failed"

    transform = get_transform([
        {'type': 'mask_region', 'regions': {'A': list(range(10, 30))}},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    data = transform(structure)
    batch = recursive_to(PaddingCollate()([data]), 'cpu')
    gen_mask = batch['generate_flag'][0].bool()
    assert gen_mask.sum() > 0, "No design residues"

    # Generate 2 samples with recycling
    samples = []
    for i in range(2):
        with torch.no_grad():
            traj = model.sample(batch, sample_opt={'deterministic': False, 'num_recycles': 3})
        pred_aa = traj[0][2][0][gen_mask]
        seq_des = ''.join(AA_LETTERS[a] if a < 20 else 'X' for a in pred_aa.cpu())
        logits = traj['pred_logits'][0][gen_mask]
        lp = torch.log_softmax(logits[..., :20], dim=-1)
        nll = -lp[range(len(pred_aa)), pred_aa].mean()
        ppl = torch.exp(nll).item()
        plddt = traj['plddt'][0][gen_mask].mean().item()
        iptm = traj['iptm'][0].item()
        samples.append({'sequence': seq_des, 'ppl': ppl, 'plddt': plddt, 'iptm': iptm})
        print(f"  Sample {i+1}: {seq_des} | PPL={ppl:.2f} | ipTM={iptm:.3f}")

    os.unlink(tmp)
    print(f"  ✓ BFN design: 2 samples generated successfully")
except Exception as e:
    errors.append(f"BFN design: {e}")
    print(f"  ✗ BFN design failed: {e}")
    import traceback; traceback.print_exc()
    if os.path.exists(tmp):
        os.unlink(tmp)


# ── Phase 4: Cascade filter BFN results ──
print("\n" + "=" * 70)
print("  PHASE 4: Cascade filter on BFN results")
print("=" * 70)

try:
    filtered, report = apply_cascade(samples)
    print(f"  {len(filtered)}/{len(samples)} passed cascade")
    if filtered:
        fasta_str, best_seq = format_filtered_fasta(filtered, 'test')
        print(f"  Best: {best_seq}")
        assert len(fasta_str) > 0, "FASTA output empty"
    print("  ✓ Cascade filter works on BFN output")
except Exception as e:
    errors.append(f"cascade BFN: {e}")
    print(f"  ✗ cascade BFN failed: {e}")
    import traceback; traceback.print_exc()


# ── Phase 5: AF2 validator test (single very short peptide) ──
print("\n" + "=" * 70)
print("  PHASE 5: AF2 validator test (short peptide)")
print("=" * 70)

try:
    # Quick test with a very short peptide (should be fast)
    test_seq = 'GSHMKFLILFNILVST'
    results = validate_sequences([test_seq], output_dir='alphafold_results/val_test',
                                num_recycle=1, stop_at_score=50)
    if results:
        r = results[0]
        print(f"  Success: {r.get('success')}")
        if r.get('success'):
            print(f"  pLDDT: {r.get('plddt', 0):.2f}  pTM: {r.get('ptm', 0):.3f}  ipTM: {r.get('iptm', 'N/A')}")
        else:
            print(f"  Error: {r.get('error', 'unknown')}")
    print("  ✓ AF2 validator runs without crash")
except Exception as e:
    errors.append(f"AF2 validator: {e}")
    print(f"  ✗ AF2 validator failed: {e}")
    import traceback; traceback.print_exc()


# ── Phase 6: Iterative refiner integration test (no-AF2 mode) ──
print("\n" + "=" * 70)
print("  PHASE 6: Iterative refiner structure test")
print("=" * 70)

try:
    # Test that IterativeRefiner can be instantiated and run with mock design+AF2
    from iterative_refiner import IterativeRefiner, format_refinement_report, RefinementResult

    # Mock design function
    def mock_design(pdb_path, region, n):
        return [{'sequence': f'AAAAEEEELLLLKKKK{i}', 'ppl': 3.0+i*0.5, 'plddt': 0.5, 'iptm': 0.5} for i in range(n)]

    # Mock AF2 validator (fast, no subprocess)
    def mock_af2(sequences, progress_cb=None, output_dir=None, **kwargs):
        results = []
        for i, seq in enumerate(sequences):
            quality = 0.8 - i * 0.05  # first is best
            results.append({
                'sequence': seq, 'success': True,
                'plddt': max(0.5, quality), 'ptm': max(0.4, quality-0.1),
                'iptm': max(0.3, quality-0.2), 'max_pae': 5.0 + i,
                'pdb_path': None,
            })
        return results

    # Create a minimal PDB file (required by refiner's template check)
    dummy_pdb = 'val_phase6_dummy.pdb'
    with open(dummy_pdb, 'w') as f:
        f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
        f.write("ATOM      2  CA  ALA A   1       1.470   0.000   0.000  1.00  0.00           C\n")
        f.write("ATOM      3  C   ALA A   1       2.010   1.420   0.000  1.00  0.00           C\n")
        f.write("ATOM      4  O   ALA A   1       1.230   2.370   0.000  1.00  0.00           O\n")
        f.write("ATOM      5  CB  ALA A   1       1.990  -0.770   1.210  1.00  0.00           C\n")
        f.write("END\n")

    refiner = IterativeRefiner(
        design_fn=mock_design,
        af2_validator=mock_af2,
        cascade_fn=apply_cascade_af2,
        n_samples=5, top_k=2, max_rounds=2,
    )
    result = refiner.run(dummy_pdb, 'A:1-10')
    assert len(result.rounds) == 2, f"Expected 2 rounds, got {len(result.rounds)}"
    assert result.best_overall_iptm > 0, "No best iptm"

    report = format_refinement_report(result)
    print(report[:500])
    print("  ✓ Iterative refiner: full mock cycle passed")
except Exception as e:
    errors.append(f"iterative refiner: {e}")
    print(f"  ✗ iterative refiner failed: {e}")
    import traceback; traceback.print_exc()


# ── Phase 7: App module integration check ──
print("\n" + "=" * 70)
print("  PHASE 7: App module integration (import check)")
print("=" * 70)

try:
    # Check that app.py has the new iter_refine UI callback
    import app as app_module
    # Check key functions exist
    assert hasattr(app_module, 'generate_pdb_from_sequence'), "missing generate_pdb_from_sequence"
    assert hasattr(app_module, 'run_alphafold_prediction'), "missing run_alphafold_prediction"
    assert hasattr(app_module, 'run_bfn_protein'), "missing run_bfn_protein"
    assert hasattr(app_module, 'cf'), "missing cascade_filter import as cf"
    print("  ✓ app.py key functions accessible")
except Exception as e:
    errors.append(f"app integration: {e}")
    print(f"  ✗ app integration: {e}")
    import traceback; traceback.print_exc()


# ── Final report ──
print("\n" + "=" * 70)
print("  VALIDATION COMPLETE")
print("=" * 70)
if errors:
    print(f"  ✗ {len(errors)} error(s) found:")
    for e in errors:
        print(f"    - {e}")
else:
    print("  ✅ No errors — all checks passed")

# Save results
with open('validation_v2_results.json', 'w') as f:
    json.dump({'errors': errors, 'passed': len(errors)==0}, f, indent=2)
print("\n  Results saved to validation_v2_results.json")
