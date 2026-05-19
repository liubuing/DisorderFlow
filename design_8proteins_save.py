#!/usr/bin/env python
"""Design all 8 target proteins with BFN and save results."""
import sys, os, json, time, torch

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from app import generate_pdb_from_sequence, load_bfn, AA_LETTERS
from antibodydesignbfn.datasets.protein import preprocess_protein_structure
from antibodydesignbfn.utils.train import recursive_to
from antibodydesignbfn.utils.misc import seed_all
from antibodydesignbfn.utils.data import PaddingCollate
from antibodydesignbfn.utils.transforms import get_transform
from cascade_filter import apply_cascade, format_filtered_fasta

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

N_SAMPLES = 5
OUT_DIR = 'design_8proteins_results'
os.makedirs(OUT_DIR, exist_ok=True)

model, config = load_bfn()
all_results = {}
summary_lines = []
summary_lines.append("蛋白质\t原始长度\t设计区域长度\t样本数\t最佳PPL\t平均PPL\t最佳ent\t最佳Q\t最佳ipTM")
summary_lines.append("-" * 100)

for name, seq in SEQUENCES.items():
    print(f"\n{'='*60}")
    print(f"  设计: {name} ({len(seq)} aa)")
    print(f"{'='*60}")

    # Use FULL protein sequence with masked design window (central ~50aa)
    # This provides context residues (N/C termini) for context-only ipTM pooling
    full_seq = seq
    if len(seq) > 50:
        design_start = max(0, (len(seq) - 50) // 2)
        design_end = design_start + 50
        design_indices = list(range(design_start, design_end))
        design_seq = seq[design_start:design_end]
    else:
        design_indices = list(range(len(seq)))
        design_seq = seq

    # Generate PDB from FULL sequence (context residues provide structural features)
    pdb_path = os.path.join(OUT_DIR, f'{name}_template.pdb')
    pdb_text = generate_pdb_from_sequence(full_seq)
    with open(pdb_path, 'w') as f:
        f.write(pdb_text)

    # Load and preprocess (unique seed per protein for distinct designs)
    seed_all(hash(name) % 10000)
    structure = preprocess_protein_structure(pdb_path, chain_ids=['A'])
    transform = get_transform([
        {'type': 'mask_region', 'regions': {'A': design_indices}},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    data = transform(structure)
    batch = recursive_to(PaddingCollate()([data]), 'cpu')
    gen_mask = batch['generate_flag'][0].bool()
    n_context = (~gen_mask).sum().item()

    print(f"  Template: {len(full_seq)}aa total, {len(design_indices)}aa design window ({n_context}aa context)")
    # Generate samples
    samples = []
    sample_opt = {'deterministic': False, 'num_recycles': 3}
    for i in range(N_SAMPLES):
        t0 = time.time()
        with torch.no_grad():
            traj = model.sample(batch, sample_opt=sample_opt)
        pred_aa = traj[0][2][0][gen_mask]
        seq_des = ''.join(AA_LETTERS[a] if a < 20 else 'X' for a in pred_aa.cpu())
        logits = traj['pred_logits'][0][gen_mask]
        lp = torch.log_softmax(logits[..., :20], dim=-1)
        nll = -lp[range(len(pred_aa)), pred_aa].mean()
        ppl = torch.exp(nll).item()
        entropy = traj['pred_entropy'][0][gen_mask].mean().item()
        plddt = traj['plddt'][0][gen_mask].mean().item()
        iptm = traj['iptm'][0].item()
        # Combined quality score: PPL dominates (sequence-varying), entropy as baseline (context-dependent)
        # Entropy is constant within protein (depends on context, like ipTM)
        # PPL varies per-sequence → primary quality discriminator
        ppl_score = 1.0 / max(ppl, 0.1)
        ent_score = 1.0 - min(entropy / 2.996, 1.0)
        quality = 0.7 * ppl_score + 0.3 * ent_score  # PPL-weighted quality proxy
        samples.append({
            'sequence': seq_des,
            'ppl': round(ppl, 2),
            'entropy': round(entropy, 4),
            'quality': round(quality, 4),
            'plddt': round(plddt, 4),
            'iptm': round(iptm, 4),
        })
        elapsed = time.time() - t0
        print(f"  Sample {i+1}: {seq_des} | PPL={ppl:.2f} | ent={entropy:.4f} | Q={quality:.3f} | ipTM={iptm:.4f} | {elapsed:.1f}s")

    # Cascade filter
    filtered, report = apply_cascade(samples)
    fasta_str, best_seq = format_filtered_fasta(filtered, name)

    # Save FASTA
    fasta_path = os.path.join(OUT_DIR, f'{name}_designs.fasta')
    with open(fasta_path, 'w', encoding='utf-8') as f:
        f.write(fasta_str)

    # Save JSON
    json_path = os.path.join(OUT_DIR, f'{name}_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'protein': name, 'native_length': len(seq),
                   'design_region': f'A:1-{len(design_seq)}',
                   'samples': samples, 'filtered': filtered,
                   'best_sequence': best_seq,
                   'filter_report': report}, f, indent=2, ensure_ascii=False)

    # Summary
    ppls = [s['ppl'] for s in samples]
    ents = [s.get('entropy', 0) for s in samples]
    quals = [s.get('quality', 0) for s in samples]
    best = filtered[0] if filtered else samples[0]
    summary_lines.append(f"{name}\t{len(seq)}\t{len(design_seq)}\t{N_SAMPLES}\t"
                         f"{min(ppls):.2f}\t{sum(ppls)/len(ppls):.2f}\t"
                         f"{min(ents):.4f}\t{max(quals):.4f}\t"
                         f"{best.get('iptm', 0):.4f}")

    all_results[name] = {'samples': samples, 'filtered': filtered, 'best_sequence': best_seq}
    print(f"  Best: {best_seq} | PPL={min(ppls):.2f}")

# Save summary
with open(os.path.join(OUT_DIR, 'summary.tsv'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))
with open(os.path.join(OUT_DIR, 'all_results.json'), 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"  全部完成! 结果保存至: {OUT_DIR}/")
print(f"  - summary.tsv (汇总表)")
print(f"  - all_results.json (完整JSON)")
print(f"  - *_designs.fasta (各蛋白质设计FASTA)")
print(f"  - *_results.json (各蛋白质详细结果)")
print(f"{'='*60}")
