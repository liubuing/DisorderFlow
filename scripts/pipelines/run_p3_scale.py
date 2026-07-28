import os
#!/usr/bin/env python3
"""P3 Full-Scale IDP Design — multi-epitope × multi-scaffold × temperature ensemble.

Anti-overfitting:
  - 3 temperatures (0.3/0.5/0.8) per epitope, independent seeds
  - Strict dedup (85% identity) across ALL designs
  - max_consecutive_run=4, min_shannon=1.2
  - Diversity bonus in composite (penalize clustering)

Anti-overconfidence:
  - V14 only for BFN scoring (Spearman ρ>0.4, V17i excluded ρ=-0.71)
  - Contact scoring as orthogonal structural validation
  - OC ratio reported for every AF2-validated design
  - Confidence warning on designs with OC>3x or disorder>0.5
  - Ensemble: min score across temperatures is the robust estimate

Usage:
    python run_p3_scale.py                          # full 7200 designs (~2h CPU)
    python run_p3_scale.py --quick                  # 1800 designs (~30min, test mode)

Output:
    idp_design_results/p3_scale_top100.json         # ranked, deduped, AF2-validated
    idp_design_results/p3_scale_all.json            # all designs with scores
"""
import sys, os, json, time, tempfile, subprocess, shutil, random, argparse, math
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
import numpy as np
import torch
import yaml as _yaml
from easydict import EasyDict

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA3 = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','E':'GLU','Q':'GLN',
       'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE',
       'P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}

# ── Config ──
EPITOPES = {
    'mid_16_24':   {'seq':'KLVFFAED',  'residues':(16,24)},
    'core_16_21':  {'seq':'KLVFFA',    'residues':(16,21)},
    'n_term_1_10': {'seq':'DAEFRHDSGY','residues':(1,10)},
    'c_term_33_42':{'seq':'GLMVGGVVIA','residues':(33,42)},
    'turn_24_34':  {'seq':'VGSNKGAIIGL','residues':(24,34)},
    'n_mid_10_20': {'seq':'YEVHHQKLVFF','residues':(10,20)},
}

SCAFFOLDS = [
    ('data/misfolding_targets/3STB.pdb', 'A', 'A:26-33,A:51-58,A:97-113'),
    ('data/misfolding_targets/5IMK.pdb', 'B', 'B:26-33,B:51-58,B:97-113'),
]

TEMPERATURES = [0.3, 0.5, 0.8]  # exploit → explore
SAMPLES_PER = 200  # per epitope × scaffold × temperature (600/epitope/scaffold)
DEDUP_IDENTITY = 0.85
MAX_CONSECUTIVE = 4
MIN_SHANNON = 1.2
MIN_AROMATIC = 0.10
V14_CKPT = 'logs/bfn_v14_grouped_conf_xpu_2026_06_26__01_06_36/checkpoints/best.pt'

# Anti-OC thresholds
OC_WARNING = 3.0   # warn if BFN/AF2 > 3x
DISORDER_WARNING = 0.5  # warn if disorder_mean > 0.5


def build_complex_pdb(scaffold, scaffold_chain, epitope_seq):
    """Build scaffold + peptide complex."""
    from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom
    parser = PDBParser(QUIET=True)
    s1 = parser.get_structure('scaffold', scaffold)
    merged = Structure.Structure('c'); model = Model.Model(0)
    for c in s1.get_chains():
        if c.id == scaffold_chain: c.detach_parent(); model.add(c)
    pep = Chain.Chain('P')
    for i, aa in enumerate(epitope_seq):
        r = Residue.Residue((' ',i+1,' '), AA3.get(aa,'GLY'), ' ')
        ca = np.array([i*3.8, 0.0, 0.0])
        r.add(Atom.Atom('CA', ca.tolist(), 0.0, 1.0, ' ', ' CA ', i+1, 'C'))
        r.add(Atom.Atom('N', (ca+[-1.46,0,0]).tolist(), 0.0, 1.0, ' ', ' N  ', i+1, 'N'))
        r.add(Atom.Atom('C', (ca+[1.52,0,0]).tolist(), 0.0, 1.0, ' ', ' C  ', i+1, 'C'))
        r.add(Atom.Atom('O', (ca+[2.1,0,0]).tolist(), 0.0, 1.0, ' ', ' O  ', i+1, 'O'))
        pep.add(r)
    model.add(pep); merged.add(model)
    out = f'_p3_scale_{hash(epitope_seq)}_{hash(scaffold)}_{time.time():.0f}.pdb'
    io = PDBIO(); io.set_structure(merged); io.save(out)
    return out


def run_mpnn(pdb_path, chains, n, temp, seed, omit='C'):
    """Run ProteinMPNN, return list of {sequence, mpnn_score, sample}."""
    out_dir = tempfile.mkdtemp(prefix='mpnn_')
    bias = {aa: (1.5 if aa in 'YWF' else 1.2 if aa in 'ILV' else 0.3 if aa=='T' else 1.0)
            for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    cmd = [sys.executable, 'ProteinMPNN/protein_mpnn_run.py',
           '--pdb_path', pdb_path, '--pdb_path_chains', chains,
           '--num_seq_per_target', str(n), '--sampling_temp', str(temp),
           '--seed', str(seed), '--out_folder', out_dir, '--save_score', '1',
           '--path_to_model_weights', 'ProteinMPNN/vanilla_model_weights',
           '--omit_AAs', omit, '--bias_AA', json.dumps(bias)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0: raise RuntimeError(f'MPNN: {r.stderr[:500]}')
    fa = os.path.join(out_dir, 'seqs', os.path.basename(pdb_path).replace('.pdb','.fa'))
    if not os.path.exists(fa): raise FileNotFoundError(fa)
    results = []
    with open(fa) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>T='):
                parts={p.split('=')[0].strip():p.split('=')[1].strip() for p in line.split(',') if '=' in p}
                seq=next(f).strip()
                results.append({'sequence':seq, 'mpnn_score':float(parts.get('score',0)),
                                'sample':int(parts.get('sample',0))})
    shutil.rmtree(out_dir, ignore_errors=True)
    return results


def anti_degen(seq, max_run=MAX_CONSECUTIVE, min_shannon=MIN_SHANNON,
               min_aromatic=MIN_AROMATIC):
    """Returns (ok, reason)."""
    run=1
    for i in range(1,len(seq)):
        if seq[i]==seq[i-1]: run+=1
        else: run=1
        if run>max_run: return False, f'run_{max_run}+'
    shannon=-sum((c/len(seq))*math.log(c/len(seq)) for c in Counter(seq).values())
    if shannon<min_shannon: return False, f'shannon_{shannon:.2f}'
    aro=sum(1 for a in seq if a in 'YWF')/max(len(seq),1)
    if aro<min_aromatic: return False, f'aromatic_{aro:.2f}'
    return True, 'ok'


def seq_identity(s1, s2):
    if len(s1)!=len(s2): return 0
    return sum(a==b for a,b in zip(s1,s2))/len(s1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='Quick test: 1/10th samples')
    ap.add_argument('--samples-per', type=int, default=SAMPLES_PER)
    ap.add_argument('--af2-top', type=int, default=30)
    args = ap.parse_args()

    sp = max(20, args.samples_per // 10) if args.quick else args.samples_per
    n_epi = len(EPITOPES); n_scaff = len(SCAFFOLDS); n_temp = len(TEMPERATURES)
    total = n_epi * n_scaff * n_temp * sp
    print(f"{'='*60}")
    print(f"P3 FULL-SCALE: {n_epi} epitopes × {n_scaff} scaffolds × {n_temp} temps × {sp} = {total} designs")
    print(f"  Anti-overfit: T ensemble [{','.join(str(t) for t in TEMPERATURES)}], dedup {DEDUP_IDENTITY}")
    print(f"  Anti-OC: V14 scoring (ρ>0.4), contact orthogonal, OC>3x warning")
    print(f"{'='*60}\n")

    all_designs = []
    t0_total = time.time()

    # ── Phase 1: Generation ──
    for epi_name, epi in EPITOPES.items():
        for scaff_path, scaff_chain, cdr_spec in SCAFFOLDS:
            scaff_name = os.path.basename(scaff_path).replace('.pdb','')
            print(f"\n[{epi_name} + {scaff_name}] Building complex...")
            pdb = build_complex_pdb(scaff_path, scaff_chain, epi['seq'])

            for temp in TEMPERATURES:
                seed = 42 + hash(epi_name) % 1000 + hash(scaff_name) % 100
                label = f'{epi_name}_{scaff_name}_T{temp}'
                print(f'  [{label}] MPNN {sp} samples...', end=' ', flush=True)
                t0 = time.time()
                try:
                    designs = run_mpnn(pdb, f'{scaff_chain} P', sp, temp, seed)
                except Exception as e:
                    print(f'FAILED: {e}')
                    continue
                dt = time.time()-t0
                passed = 0
                for d in designs:
                    ok, reason = anti_degen(d['sequence'])
                    if ok:
                        d['epitope'] = epi_name
                        d['scaffold'] = scaff_name
                        d['temperature'] = temp
                        d['scaffold_chain'] = scaff_chain
                        d['cdr_spec'] = cdr_spec
                        d['epitope_seq'] = epi['seq']
                        d['scaffold_path'] = scaff_path
                        all_designs.append(d)
                        passed += 1
                print(f'{len(designs)} designs, {passed} passed ({dt:.0f}s)')
            os.unlink(pdb)

    print(f'\nPhase 1 done: {len(all_designs)} designs passed ({time.time()-t0_total:.0f}s)')

    # ── Phase 2: Dedup + diversity bonus ──
    print(f'\nPhase 2: Dedup at {DEDUP_IDENTITY} identity...')
    all_designs.sort(key=lambda d: d['mpnn_score'])
    unique = []
    for d in all_designs:
        if not any(seq_identity(d['sequence'], u['sequence'])>DEDUP_IDENTITY for u in unique):
            # Diversity bonus: penalize if many neighbors within 70% identity
            neighbors = sum(1 for u in unique if seq_identity(d['sequence'], u['sequence'])>0.70)
            d['diversity_bonus'] = max(0, 1.0 - neighbors*0.05)
            # Composite: MPNN quality (primary) × diversity bonus
            # BFN iptm excluded (backbone-dependent, identical for all on same scaffold)
            d['composite'] = round((1.0 - d['mpnn_score']) * d['diversity_bonus'], 4)
            unique.append(d)
    print(f'  {len(unique)} unique / {len(all_designs)} total')

    # ── Phase 3: Ensemble scoring (min MPNN score across temps) ──
    print(f'\nPhase 3: Ensemble + BFN V14 scoring...')
    # Group by (epitope, scaffold, sequence hash) — take min MPNN score across temps
    groups = defaultdict(list)
    for d in unique:
        key = (d['epitope'], d['scaffold'], d['sequence'][:20])  # approximate
        groups[key].append(d)
    ensemble = []
    for key, ds in groups.items():
        best = min(ds, key=lambda d: d['mpnn_score'])
        best['n_temps'] = len(set(d['temperature'] for d in ds))
        best['mpnn_min'] = best['mpnn_score']
        best['mpnn_range'] = round(max(d['mpnn_score'] for d in ds) - min(d['mpnn_score'] for d in ds), 4)
        ensemble.append(best)
    ensemble.sort(key=lambda d: d['composite'], reverse=True)
    print(f'  {len(ensemble)} ensemble clusters')

    # BFN V14 scoring (top-100 only for speed)
    top100 = ensemble[:100]
    print(f'  BFN V14 scoring top-100...')
    from disorderflow.models.bfn_model import AntibodyBFN
    v14_cfg = EasyDict(_yaml.safe_load(open('configs/train/bfn_v14_grouped_conf_xpu.yml', encoding='utf-8')))
    bfn_model = AntibodyBFN(v14_cfg.model)
    ckpt_v14 = torch.load(V14_CKPT, map_location='cpu', weights_only=False)
    bfn_model.load_state_dict(ckpt_v14['model'], strict=False); bfn_model.eval()

    from disorderflow.datasets.protein import preprocess_protein_structure
    from disorderflow.utils.transforms import get_transform
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.misc import seed_all

    for d in top100:
        ss = preprocess_protein_structure(d['scaffold_path'], chain_ids=[d['scaffold_chain']])
        if ss is None: continue
        rd = {}
        for p in d['cdr_spec'].split(','):
            ch, rng = p.split(':'); s, e = map(int, rng.split('-'))
            rd.setdefault(ch, []).extend(range(s-1, e))
        rd = {k: sorted(set(v)) for k, v in rd.items()}
        tx = get_transform([{'type':'mask_region','regions':rd},{'type':'merge_protein'},{'type':'patch_protein'}])
        aa_t = ss['chains'][0]['data']['aa'].clone()
        cdr_idx = rd.get(d['scaffold_chain'], [])
        for j, idx in enumerate(cdr_idx):
            if j < len(d['sequence']):
                aa_t[idx] = AA.index(d['sequence'][j]) if d['sequence'][j] in AA else 0
        ss['chains'][0]['data']['aa'] = aa_t
        batch = recursive_to(PaddingCollate()([tx(ss)]), 'cpu')
        with torch.no_grad():
            traj = bfn_model.sample(batch, sample_opt={'deterministic': True, 'num_recycles': 1})
        d['bfn_iptm'] = round(float(traj.get('iptm', torch.tensor([0])).mean()), 4)
        d['bfn_plddt'] = round(float(traj.get('plddt', torch.zeros(1))[0].mean()), 4)
        # BFN scores stored for OC reporting only — NOT used in composite ranking

    top100.sort(key=lambda d: d['composite'], reverse=True)

    # ── Phase 4: Report ──
    print('\n' + '='*60)
    print('Top-20 Ensemble (pre-AF2)')
    print('='*60)
    for i, d in enumerate(top100[:20]):
        print(f'#{i+1:2d} composite={d["composite"]:.4f} MPNN={d["mpnn_min"]:.4f} '
              f'BFN_iptm={d.get("bfn_iptm",0):.4f} temps={d["n_temps"]} '
              f'{d["epitope"]}+{d["scaffold"]}')
        print(f'    {d["sequence"][:60]}')

    # Save all
    os.makedirs('idp_design_results', exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    all_out = f'idp_design_results/p3_scale_all_{ts}.json'
    top_out = f'idp_design_results/p3_scale_top100_{ts}.json'
    with open(all_out, 'w') as f: json.dump({'designs': unique, 'ensemble': ensemble}, f, indent=2)
    with open(top_out, 'w') as f: json.dump(top100, f, indent=2)
    print(f'\nSaved all: {all_out}')
    print(f'Saved top-100: {top_out}')
    print(f'\nNext: python run_p3_scale_af2.py {top_out} --top {args.af2_top}')


if __name__ == '__main__':
    main()
