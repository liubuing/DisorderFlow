import os
#!/usr/bin/env python3
"""P3: IDP-aware antibody design — MPNN generation + contact scoring + amyloid prior.

IDP_DESIGN_SCHEME_V2 §P3:
  MPNN generates CDR candidates → contact scoring filters → BFN V14 validates.
  Amyloid-aware AA prior biases toward Tyr/Trp/Phe for aggregation core epitopes.
  Anti-degeneration: rejects CDRs with ≥6 consecutive identical residues.
  Template seeding: 4HIX/5CSZ native CDRs included as positive controls.

Usage:
    python run_p3_idp_design.py --epitope mid_16_24 --samples 100 --top 20

Output: idp_design_results/p3_top20_<timestamp>.json
"""
import sys, os, json, time, tempfile, subprocess, shutil, random, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
import numpy as np

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA3 = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','E':'GLU','Q':'GLN',
       'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE',
       'P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}

# ── P1 epitope config ──
EPITOPES = {
    'mid_16_24':  {'seq': 'KLVFFAED', 'residues': (16, 24), 'scaffold_chain': 'A'},
    'core_16_21': {'seq': 'KLVFFA',   'residues': (16, 21), 'scaffold_chain': 'A'},
    'n_term_1_10':{'seq': 'DAEFRHDSGY','residues':(1, 10),  'scaffold_chain': 'A'},
}

# ── Known anti-Aβ CDR templates (positive controls) ──
TEMPLATES = {
    '4HIX_solanezumab': {
        'cdr': 'GFTFSNYRSGGGYCVRYDHY',  # VH H1+H2+H3 (Chothia)
        'antibody': 'solanezumab (3D6)', 'target': 'KLVFFAED',
    },
    '5CSZ_gantenerumab': {
        'cdr': 'GFTFSSYAISGSGGSTYYADSVKG',  # VH H1+H2 (partial)
        'antibody': 'gantenerumab', 'target': 'DAEFRHDSGY',
    },
}

# ── Amyloid-aware AA prior (§P3-2) ──
# For aggregation core (KLVFFAED): aromatic (Y/W/F) + hydrophobic (I/L/V)
AMYLOID_AA_BIAS = {'Y': 1.5, 'W': 1.5, 'F': 1.5, 'I': 1.2, 'L': 1.2, 'V': 1.2,
                    'R': 1.1, 'K': 1.1,  # charged for solubility balance
                    'T': 0.3, 'S': 0.5, 'G': 0.8, 'A': 0.8, 'P': 0.5}  # penalty
AMYLOID_OMIT_AA = ''  # 'C' would disallow disulfide complications

SCAFFOLD = 'data/misfolding_targets/3STB.pdb'
SCAFFOLD_CHAIN = 'A'
CDR_SPEC = 'A:26-33,A:51-58,A:97-113'  # Chothia VHH CDRs
V14_CKPT = 'logs/bfn_v14_grouped_conf_xpu_2026_06_26__01_06_36/checkpoints/best.pt'


def run_mpnn(pdb_path, chains, num_samples, temperature=0.5, seed=42, omit_aas='',
             bias_aas=None):
    """Run ProteinMPNN with optional AA bias."""
    out_dir = tempfile.mkdtemp(prefix='mpnn_')
    cmd = [
        sys.executable, 'ProteinMPNN/protein_mpnn_run.py',
        '--pdb_path', pdb_path, '--pdb_path_chains', chains,
        '--num_seq_per_target', str(num_samples),
        '--sampling_temp', str(temperature),
        '--seed', str(seed),
        '--out_folder', out_dir, '--save_score', '1',
        '--path_to_model_weights', 'ProteinMPNN/vanilla_model_weights',
    ]
    if omit_aas:
        cmd.extend(['--omit_AAs', omit_aas])
    if bias_aas:
        cmd.extend(['--bias_AA', json.dumps(bias_aas)])

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f'ProteinMPNN failed:\n{r.stderr[:2000]}')
    fa = os.path.join(out_dir, 'seqs',
                      os.path.basename(pdb_path).replace('.pdb', '.fa'))
    if not os.path.exists(fa):
        raise FileNotFoundError(f'MPNN output not found: {fa}')
    results = []
    with open(fa) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>T='):
                parts = {p.split('=')[0].strip(): p.split('=')[1].strip()
                         for p in line.split(',') if '=' in p}
                seq = next(f).strip()
                results.append({'sequence': seq,
                                'mpnn_score': float(parts.get('score', 0)),
                                'sample': int(parts.get('sample', 0))})
    shutil.rmtree(out_dir, ignore_errors=True)
    return results


def build_complex_pdb(scaffold_pdb, scaffold_chain, epitope_seq, epitope_chain='P'):
    """Build scaffold + peptide complex PDB for ProteinMPNN."""
    from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom
    parser = PDBParser(QUIET=True)
    s1 = parser.get_structure('scaffold', scaffold_pdb)

    # Build peptide as poly-alanine trace
    merged = Structure.Structure('complex')
    model = Model.Model(0)

    # Add scaffold chain
    for c in s1.get_chains():
        if c.id == scaffold_chain:
            c.detach_parent()
            model.add(c)

    # Add peptide chain
    pep_chain = Chain.Chain(epitope_chain)
    for i, aa_char in enumerate(epitope_seq):
        resname = AA3.get(aa_char, 'GLY')
        res = Residue.Residue((' ', i+1, ' '), resname, ' ')
        ca_pos = [i * 3.8, 0.0, 0.0]  # extended conformation
        res.add(Atom.Atom('CA', ca_pos, 0.0, 1.0, ' ', ' CA ', i+1, 'C'))
        res.add(Atom.Atom('N', [ca_pos[0]-1.46, ca_pos[1], ca_pos[2]], 0.0, 1.0, ' ', ' N  ', i+1, 'N'))
        res.add(Atom.Atom('C', [ca_pos[0]+1.52, ca_pos[1], ca_pos[2]], 0.0, 1.0, ' ', ' C  ', i+1, 'C'))
        res.add(Atom.Atom('O', [ca_pos[0]+2.1, ca_pos[1], ca_pos[2]], 0.0, 1.0, ' ', ' O  ', i+1, 'O'))
        pep_chain.add(res)
    model.add(pep_chain)
    merged.add(model)
    out = tempfile.mktemp(suffix='.pdb')
    io = PDBIO(); io.set_structure(merged); io.save(out)
    return out


def graft_and_score(cdr_seq, scaffold_pdb, scaffold_chain, cdr_spec,
                    epitope_seq, epitope_chain='P'):
    """Graft CDR into scaffold, build complex PDB, score with contacts."""
    from Bio.PDB import PDBParser, PDBIO
    from modules.idp_validation_triplet import interface_contacts

    parser = PDBParser(QUIET=True)
    s = parser.get_structure('s', scaffold_pdb)

    # Parse CDR regions
    regions = []
    for part in cdr_spec.split(','):
        ch, rng = part.split(':')
        start, end = map(int, rng.split('-'))
        if ch == scaffold_chain:
            regions.append((start, end))

    # Graft CDR residues with approximate Cβ reconstruction
    import numpy as np
    # Standard bond geometry: N-CA-CB angle ~110°, CA-CB ~1.52A
    aa3_rev = {v: k for k, v in AA3.items()}
    cdr_regions_flat = []
    for start, end in regions:
        for j in range(start, end + 1):
            if any(r[0] <= j <= r[1] for r in regions):
                cdr_regions_flat.append(j)

    pos = 0
    for j in cdr_regions_flat:
        if pos < len(cdr_seq):
            aa_char = cdr_seq[pos]
            resname = AA3.get(aa_char, 'GLY')
            try:
                res = s[0][scaffold_chain][j]
                res.resname = resname
                # Reconstruct approximate Cβ from N, CA, C
                if 'N' in res and 'CA' in res and 'C' in res:
                    n_pos = res['N'].get_vector().get_array()
                    ca_pos = res['CA'].get_vector().get_array()
                    c_pos = res['C'].get_vector().get_array()
                    # Cβ direction: (N-CA)/|N-CA| rotated ~120° around CA-C bond
                    n_ca = n_pos - ca_pos
                    ca_c = c_pos - ca_pos
                    # Bisector method: Cβ = CA - 0.58*(N-CA) + 0.57*(C-CA) + perpendicular
                    bisector = -(n_ca + ca_c)
                    bisector = bisector / (np.linalg.norm(bisector) + 1e-8)
                    # Perpendicular component
                    perp = np.cross(ca_c, n_ca)
                    perp = perp / (np.linalg.norm(perp) + 1e-8)
                    cb_pos = ca_pos + 1.52 * bisector  # approximate
                    if 'CB' in res:
                        res['CB'].set_coord(cb_pos.tolist())
                    else:
                        from Bio.PDB import Atom
                        res.add(Atom.Atom('CB', cb_pos.tolist(), 0.0, 1.0, ' ', ' CB ', j, 'C'))
            except (KeyError, ValueError, np.linalg.LinAlgError):
                pass
            pos += 1

    # Build peptide chain — position near CDR loops
    # Get CDR CA positions from scaffold to place peptide nearby
    cdr_ca_coords = []
    for res in s[0][scaffold_chain]:
        ri = res.id[1]
        if (26 <= ri <= 32) or (52 <= ri <= 56) or (95 <= ri <= 102):
            if 'CA' in res:
                cdr_ca_coords.append(res['CA'].get_coord())
    import numpy as np
    cdr_center = np.array(cdr_ca_coords).mean(axis=0) if cdr_ca_coords else np.array([0, 0, 0])
    # Place peptide ~15A from CDR center (close enough for contact checking)
    pep_start = cdr_center + np.array([15.0, 5.0, 0.0])

    from Bio.PDB import Model, Chain, Residue, Atom
    for model in s:
        pep_chain = Chain.Chain(epitope_chain)
        for i, aa_char in enumerate(epitope_seq):
            resname = AA3.get(aa_char, 'GLY')
            res = Residue.Residue((' ', i+1, ' '), resname, ' ')
            ca_pos = pep_start + np.array([i * 3.8, 0.0, 0.0])
            res.add(Atom.Atom('CA', ca_pos.tolist(), 0.0, 1.0, ' ', ' CA ', i+1, 'C'))
            res.add(Atom.Atom('N', (ca_pos + [-1.46, 0, 0]).tolist(), 0.0, 1.0, ' ', ' N  ', i+1, 'N'))
            res.add(Atom.Atom('C', (ca_pos + [1.52, 0, 0]).tolist(), 0.0, 1.0, ' ', ' C  ', i+1, 'C'))
            res.add(Atom.Atom('O', (ca_pos + [2.1, 0, 0]).tolist(), 0.0, 1.0, ' ', ' O  ', i+1, 'O'))
            pep_chain.add(res)
        model.add(pep_chain)

    # Write temporary complex PDB
    tmp = tempfile.mktemp(suffix='_complex.pdb')
    io = PDBIO(); io.set_structure(s); io.save(tmp)

    # Score with contacts
    h1, h2, h3 = (26, 32), (52, 56), (95, 102)  # Chothia VH
    cs = interface_contacts(tmp, scaffold_chain, epitope_chain,
                            [(h1[0], h1[1]), (h2[0], h2[1]), (h3[0], h3[1])])

    os.unlink(tmp)
    return cs


def anti_degen_filter(cdr_seq, max_run=6, min_shannon=1.0, min_aromatic=0.08):
    """§P3-2: reject low-complexity CDRs."""
    import math
    # Consecutive run check
    run = 1
    for i in range(1, len(cdr_seq)):
        if cdr_seq[i] == cdr_seq[i-1]:
            run += 1
            if run > max_run:
                return False, f'run_{max_run}+_{cdr_seq[i]}'
        else:
            run = 1
    # Shannon entropy
    from collections import Counter
    counts = Counter(cdr_seq)
    total = sum(counts.values())
    shannon = -sum((c/total) * math.log(c/total) for c in counts.values())
    if shannon < min_shannon:
        return False, f'low_shannon_{shannon:.2f}'
    # Aromatic fraction check
    aromatic = sum(1 for a in cdr_seq if a in 'YWF')
    if aromatic / max(len(cdr_seq), 1) < min_aromatic:
        return False, f'low_aromatic_{aromatic/max(len(cdr_seq),1):.2f}'
    return True, 'ok'


def main():
    ap = argparse.ArgumentParser(description='P3: IDP-aware CDR design')
    ap.add_argument('--epitope', default='mid_16_24', choices=list(EPITOPES.keys()))
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--temperature', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    epi = EPITOPES[args.epitope]
    epitope_seq = epi['seq']
    print(f"{'='*60}")
    print(f"P3: IDP-Aware CDR Design")
    print(f"  Epitope: {args.epitope} ({epitope_seq})")
    print(f"  Scaffold: {SCAFFOLD} chain {SCAFFOLD_CHAIN}")
    print(f"  Samples: {args.samples}, Top: {args.top}")
    print(f"{'='*60}\n")

    # ── Step 1: MPNN generation ──
    print("[1/4] Building scaffold + peptide complex...")
    # Use a short relative path to avoid ProteinMPNN's Windows path bug
    # (os.path.basename on 'C:\\...' produces the full path)
    complex_pdb = '_p3_complex.pdb'
    # Build complex in-place
    from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom
    import numpy as np
    parser = PDBParser(QUIET=True)
    s1 = parser.get_structure('scaffold', SCAFFOLD)
    merged = Structure.Structure('complex')
    model = Model.Model(0)
    for c in s1.get_chains():
        if c.id == SCAFFOLD_CHAIN:
            c.detach_parent()
            model.add(c)
    pep_chain = Chain.Chain('P')
    for i, aa_char in enumerate(epitope_seq):
        resname = AA3.get(aa_char, 'GLY')
        res = Residue.Residue((' ', i+1, ' '), resname, ' ')
        ca_pos = np.array([i * 3.8, 0.0, 0.0])
        res.add(Atom.Atom('CA', ca_pos.tolist(), 0.0, 1.0, ' ', ' CA ', i+1, 'C'))
        res.add(Atom.Atom('N', (ca_pos + [-1.46, 0, 0]).tolist(), 0.0, 1.0, ' ', ' N  ', i+1, 'N'))
        res.add(Atom.Atom('C', (ca_pos + [1.52, 0, 0]).tolist(), 0.0, 1.0, ' ', ' C  ', i+1, 'C'))
        res.add(Atom.Atom('O', (ca_pos + [2.1, 0, 0]).tolist(), 0.0, 1.0, ' ', ' O  ', i+1, 'O'))
        pep_chain.add(res)
    model.add(pep_chain)
    merged.add(model)
    io = PDBIO(); io.set_structure(merged); io.save(complex_pdb)
    print(f"  Complex: {complex_pdb}")

    print(f"[2/4] Running ProteinMPNN ({args.samples} designs, T={args.temperature})...")
    t0 = time.time()
    # Convert bias dict to MPNN JSON format
    import json as _json
    bias_dict = {AA: AMYLOID_AA_BIAS.get(AA, 1.0) for AA in 'ACDEFGHIKLMNPQRSTVWY'}
    bias_json = _json.dumps(bias_dict)
    omit = 'C'  # disallow cysteine (disulfide complications)
    try:
        designs = run_mpnn(complex_pdb, f'{SCAFFOLD_CHAIN} P', args.samples,
                           args.temperature, args.seed, omit_aas=omit,
                           bias_aas=bias_json)
    except Exception as e:
        print(f"  MPNN failed: {e}")
        designs = []
    print(f"  {len(designs)} designs ({time.time()-t0:.0f}s)")

    # ── Step 2: Contact scoring ──
    print(f"[3/4] Contact scoring + anti-degeneration filter...")
    t0 = time.time()
    scored = []
    n_filtered = 0
    for d in designs:
        # Anti-degeneration
        ok, reason = anti_degen_filter(d['sequence'])
        if not ok:
            n_filtered += 1
            continue
        # Contact score
        cs = graft_and_score(d['sequence'], SCAFFOLD, SCAFFOLD_CHAIN,
                             CDR_SPEC, epitope_seq)
        d['contacts'] = cs['contacts']
        d['density'] = cs['density']
        d['mean_distance'] = cs['mean_distance']
        # Composite: contact density + MPNN score (lower=better, invert)
        mpnn_quality = 1.0 / (1.0 + d.get('mpnn_score', 1.0))  # [0.33, 1.0]
        d['composite'] = round(d['density'] * max(0, 1.0 - cs['mean_distance']/15.0) + 0.5 * mpnn_quality, 4)
        scored.append(d)
    print(f"  {len(scored)} passed ({n_filtered} filtered, {time.time()-t0:.0f}s)")

    # ── Step 3: Add template baselines ──
    print(f"[4/4] Adding known antibody templates...")
    for name, t in TEMPLATES.items():
        if t['target'] == epitope_seq or args.epitope in ('mid_16_24', 'core_16_21'):
            cs = graft_and_score(t['cdr'], SCAFFOLD, SCAFFOLD_CHAIN,
                                 CDR_SPEC, epitope_seq)
            t['contacts'] = cs['contacts']
            t['density'] = cs['density']
            t['mean_distance'] = cs['mean_distance']
            t['composite'] = round(t['density'] * max(0, 1.0 - cs['mean_distance']/15.0), 4)
            t['sequence'] = t['cdr']
            t['mpnn_score'] = 0
            t['sample'] = -1
            scored.append(t)
            print(f"  {name}: contacts={t['contacts']} density={t['density']:.3f} composite={t['composite']:.4f}")

    # Sort by composite (density * distance_penalty)
    scored.sort(key=lambda d: d['composite'], reverse=True)
    top = scored[:args.top]

    # ── BFN V14 scoring ──
    print(f"\n[+] BFN V14 confidence scoring (Spearman ρ>0.4)...")
    import torch
    import yaml as _yaml
    # Use AntibodyBFN full wrapper (handles pair_feat encoding)
    from disorderflow.models.bfn_model import AntibodyBFN
    from easydict import EasyDict
    v14_cfg = EasyDict(_yaml.safe_load(open('configs/train/bfn_v14_grouped_conf_xpu.yml', encoding='utf-8')))
    bfn_model = AntibodyBFN(v14_cfg.model)
    ckpt = torch.load(V14_CKPT, map_location='cpu', weights_only=False)
    bfn_model.load_state_dict(ckpt['model'], strict=False)
    bfn_model.eval()

    from disorderflow.datasets.protein import preprocess_protein_structure
    from disorderflow.utils.transforms import get_transform
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.misc import seed_all

    # Parse CDR regions for BFN
    def _parse_regions(spec):
        r = {}
        for p in spec.split(','):
            ch, rg = p.split(':'); s, e = map(int, rg.split('-'))
            r.setdefault(ch, []).extend(range(s-1, e))
        return {k: sorted(set(v)) for k, v in r.items()}
    rd = _parse_regions(CDR_SPEC)
    bfn_tx = get_transform([{'type': 'mask_region', 'regions': rd},
                            {'type': 'merge_protein'}, {'type': 'patch_protein'}])

    for d in top:
        # Graft CDR into scaffold for BFN scoring
        scaffold_struct = preprocess_protein_structure(SCAFFOLD, chain_ids=[SCAFFOLD_CHAIN])
        if scaffold_struct is None: continue
        cdr_seq = d.get('sequence', d.get('cdr', ''))
        # Graft CDR sequence into scaffold aa
        aa_tensor = scaffold_struct['chains'][0]['data']['aa'].clone()
        pos = 0
        cdr_indices = rd.get(SCAFFOLD_CHAIN, [])
        for idx in cdr_indices:
            if pos < len(cdr_seq):
                aa_tensor[idx] = AA.index(cdr_seq[pos]) if cdr_seq[pos] in AA else 0
                pos += 1
        scaffold_struct['chains'][0]['data']['aa'] = aa_tensor
        batch = recursive_to(PaddingCollate()([bfn_tx(scaffold_struct)]), 'cpu')
        with torch.no_grad():
            traj = bfn_model.sample(batch, sample_opt={'deterministic': True, 'num_recycles': 1})
        iptm = float(traj.get('iptm', torch.tensor([0])).mean()) if 'iptm' in traj else 0
        plddt = float(traj.get('plddt', torch.zeros(1))[0].mean()) if 'plddt' in traj else 0
        disorder = float(traj.get('plddt', torch.tensor([0])).mean()) if 'plddt' in traj else 0  # no disorder in V14
        d['bfn_iptm'] = round(iptm, 4)
        d['bfn_plddt'] = round(plddt, 4)
        d['bfn_disorder'] = round(disorder, 4)
    print(f"  Scored {len(top)} designs with BFN V14")
    # ── Report ──
    print(f"\n{'='*60}")
    print(f"Top-{args.top} (contact + BFN V14)")
    print(f"{'='*60}")
    for i, d in enumerate(top):
        tag = '[TEMPLATE]' if d.get('sample', 0) is not None and d.get('sample', -1) < 0 else ''
        print(f"#{i+1:2d} {tag} contacts={d['contacts']:2d} bfn_iptm={d.get('bfn_iptm',0):.4f} "
              f"bfn_plddt={d.get('bfn_plddt',0):.3f} composite={d['composite']:.4f}")
        cdr = d.get('sequence', d.get('cdr', ''))
        print(f"    CDR: {cdr[:70]}")

    best_bfn = max((d for d in top if d.get('sample', 0) >= 0),
                   key=lambda d: d.get('composite', 0), default=None)
    best_tmpl = max((d for d in top if d.get('sample', 0) < 0),
                    key=lambda d: d.get('composite', 0), default=None)

    if best_bfn and best_tmpl:
        delta = best_tmpl['composite'] - best_bfn['composite']
        print(f"\nTemplate baseline: {best_tmpl['composite']:.4f} (known anti-Aβ)")
        print(f"Best MPNN design:  {best_bfn['composite']:.4f}")
        print(f"Gap (template − MPNN): {delta:+.4f}")
        if delta > 0:
            print("  → Known antibody still ahead. More sampling or MPNN temp tuning needed.")
        else:
            print("  → MPNN design surpasses template! 🎉")

    # Save
    os.makedirs('idp_design_results', exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out = f'idp_design_results/p3_{args.epitope}_top{args.top}_{ts}.json'
    with open(out, 'w') as f:
        json.dump({'top': top, 'all': scored, 'config': vars(args)}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
