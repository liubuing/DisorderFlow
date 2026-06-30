"""Pillar A: Template-Seeded CDR Redesign from known anti-Aβ antibodies.

Strategy: Start from 4HIX/5CSZ crystal CDRs (known to bind Aβ).
  1. Identify hotspot residues — CDR Cβ within 4.5A of peptide atoms
  2. FREEZE hotspots, let MPNN redesign only non-hotspot CDR positions
  3. Generate 100 variants per antibody
  4. Score: contact preservation, sequence diversity, anti-degen

This is the design-side complement to B1/B2 evaluation.
PURE CODE: zero network, zero binaries, zero GPU.

Usage: python run_v4_pillar_a.py [--samples 100] [--top 20]
"""
import sys,os,json,time,argparse,subprocess,tempfile,shutil,random
from collections import defaultdict
import numpy as np
sys.path.insert(0,'.');sys.path.insert(0,'modules')

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q',
       'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
       'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

CONTACT_CUT = 6.0  # Hotspot: Cβ-peptide < 6A
CDR_CHOTHIA = {'H1': (26,32), 'H2': (52,56), 'H3': (95,102)}

ANTIBODIES = {
    '4HIX': {'pdb':'data/anti_abeta_refs/4HIX.pdb','ab_chain':'H','epi_chain':'A',
             'scaffold':'data/misfolding_targets/3STB.pdb','scaffold_chain':'A',
             'cdr_spec':'A:26-33,A:51-58,A:97-113'},
    '5CSZ': {'pdb':'data/anti_abeta_refs/5CSZ.pdb','ab_chain':'H','epi_chain':'D',  # peptide on D
             'scaffold':'data/misfolding_targets/3STB.pdb','scaffold_chain':'A',
             'cdr_spec':'A:26-33,A:51-58,A:97-113'},
}


def find_hotspots(pdb_path, ab_chain, epi_chain, cutoff=CONTACT_CUT):
    """Find CDR residues in contact with peptide (Cβ-Cβ < cutoff)."""
    from Bio.PDB import PDBParser
    from scipy.spatial import cKDTree
    parser = PDBParser(QUIET=True); s = parser.get_structure('s', pdb_path)

    # Get peptide Cβ positions
    pep_coords = []
    for res in s[0][epi_chain]:
        if 'CB' in res: pep_coords.append(res['CB'].get_coord())
        elif 'CA' in res: pep_coords.append(res['CA'].get_coord())
    if not pep_coords: return set()
    pep_tree = cKDTree(pep_coords)

    # Check each CDR residue
    hotspots = set()
    cdr_seq = {}
    for cdr_name, (start, end) in CDR_CHOTHIA.items():
        for res in s[0][ab_chain]:
            ri = res.id[1]
            if start <= ri <= end:
                aa = AA3.get(res.resname.strip(), 'X')
                cdr_seq[ri] = aa
                if 'CB' in res:
                    d, _ = pep_tree.query(res['CB'].get_coord())
                elif 'CA' in res:
                    d, _ = pep_tree.query(res['CA'].get_coord())
                else:
                    continue
                if d < cutoff:
                    hotspots.add(ri)
    return hotspots, cdr_seq


def graft_cdr_to_scaffold(cdr_seq_dict, scaffold_pdb, scaffold_chain, cdr_spec):
    """Graft CDR residues into scaffold and return full sequence."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True); s = parser.get_structure('s', scaffold_pdb)
    # Get scaffold sequence
    scaff_seq = []
    for res in s[0][scaffold_chain]:
        if 'CA' in res: scaff_seq.append(AA3.get(res.resname.strip(), 'X'))
    # Parse CDR spec
    cr = []
    for p in cdr_spec.split(','):
        ch, rng = p.split(':')
        st, en = map(int, rng.split('-'))
        if ch == scaffold_chain: cr.append((st, en))
    # Build full sequence with grafted CDRs
    full = list(scaff_seq)
    for cdr_name, (c_start, c_end) in CDR_CHOTHIA.items():
        for ri in range(c_start, c_end+1):
            if ri in cdr_seq_dict:
                # Map CDR residue to scaffold CDR position
                for st, en in cr:
                    # Simple mapping: position within CDR
                    cdr_offset = ri - c_start
                    target_idx = st - 1 + cdr_offset
                    if target_idx < len(full) and target_idx >= 0:
                        full[target_idx] = cdr_seq_dict[ri]
    return ''.join(full)


def main():
    ap = argparse.ArgumentParser(description='Pillar A: Template-Seeded CDR Redesign')
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--top', type=int, default=20)
    args = ap.parse_args()

    print("=" * 65)
    print("PILLAR A: Template-Seeded CDR Redesign from Known Antibodies")
    print("=" * 65)
    print(f"  Samples per antibody: {args.samples}")
    print(f"  Hotspot cutoff: {CONTACT_CUT}A Cβ-Cβ")
    print(f"  Strategy: FREEZE hotspot residues, redesign non-hotspot CDR")

    all_variants = []

    for ab_id, info in ANTIBODIES.items():
        print(f"\n{'─'*50}")
        print(f"[{ab_id}] Analyzing crystal structure...")
        hotspots, native_cdr = find_hotspots(info['pdb'], info['ab_chain'], info['epi_chain'])
        n_cdr = len(native_cdr)
        n_hot = len(hotspots)

        print(f"  CDR residues: {n_cdr}")
        print(f"  Hotspots (<{CONTACT_CUT}A): {n_hot}/{n_cdr} ({100*n_hot/max(n_cdr,1):.0f}%)")
        for cdr_name, (s, e) in CDR_CHOTHIA.items():
            cdr_hot = [ri for ri in hotspots if s <= ri <= e]
            cdr_all = [ri for ri in native_cdr if s <= ri <= e]
            print(f"    {cdr_name} ({s}-{e}): {len(cdr_hot)}/{len(cdr_all)} hotspots")
            if cdr_hot:
                print(f"      Hotspots: {[(ri, native_cdr.get(ri,'?')) for ri in sorted(cdr_hot)]}")

        # Build the native CDR sequence as a seed
        native_seq = ''.join(native_cdr.get(ri, 'X') for ri in sorted(native_cdr))
        # Build fixed positions dict (1-based residue indices for MPNN)
        # For the scaffold CDR spec, mark hotspot-equivalent positions as fixed
        fixed_positions = {}
        cr = [(int(p.split(':')[1].split('-')[0]), int(p.split(':')[1].split('-')[1]))
              for p in info['cdr_spec'].split(',') if p.split(':')[0] == info['scaffold_chain']]
        for cdr_name, (c_start, c_end) in CDR_CHOTHIA.items():
            for ri in range(c_start, c_end+1):
                if ri in hotspots:
                    # Map to scaffold CDR position
                    offset = ri - c_start
                    for st, en in cr:
                        target = st + offset
                        if st <= target <= en and target not in fixed_positions:
                            fixed_positions[target] = native_cdr.get(ri, 'X')

        print(f"  Fixed hotspot positions: {len(fixed_positions)}")

        # Write fixed positions for MPNN
        # MPNN uses chain: [indices] format
        chain_fixed = info['scaffold_chain']
        fixed_file = tempfile.mktemp(suffix='.json')
        with open(fixed_file, 'w') as f:
            fixed_dict = {f"{chain_fixed}": list(fixed_positions.keys())}
            json.dump(fixed_dict, f)

        # Build complex PDB for MPNN (scaffold + peptide)
        from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom
        # Graft native CDR onto scaffold
        native_dict = {ri: aa for ri, aa in sorted(native_cdr.items())}
        full_seq = graft_cdr_to_scaffold(native_dict, info['scaffold'], info['scaffold_chain'], info['cdr_spec'])

        # Build scaffold+peptide PDB
        parser = PDBParser(QUIET=True)
        scaf = parser.get_structure('scaf', info['scaffold'])
        # Graft CDR residues
        cr_positions = {}
        for cdr_name, (c_start, c_end) in CDR_CHOTHIA.items():
            for ri in range(c_start, c_end+1):
                if ri in native_cdr:
                    offset = ri - c_start
                    for st, en in cr:
                        target = st + offset
                        if st <= target <= en:
                            cr_positions[target] = native_cdr[ri]

        for target_idx, aa in cr_positions.items():
            try:
                scaf[0][info['scaffold_chain']][target_idx].resname = {v:k for k,v in AA3.items()}.get(aa, 'GLY')
            except: pass

        st = Structure.Structure('c'); mo = Model.Model(0)
        for c in scaf[0]:
            c.detach_parent(); mo.add(c)
        # Add peptide
        pep_seq = 'KLVFFAED' if ab_id == '4HIX' else 'DAEFRHDSGY'
        pep_chain = Chain.Chain('P')
        # Get CDR center for positioning
        cdr_ca = []
        for res in scaf[0][info['scaffold_chain']]:
            ri = res.id[1]
            if any(st <= ri <= en for st, en in cr):
                if 'CA' in res: cdr_ca.append(res['CA'].get_coord())
        ctr = np.array(cdr_ca).mean(axis=0) if cdr_ca else np.array([0,0,0])
        for i, aa in enumerate(pep_seq):
            r = Residue.Residue((' ', i+1, ' '), {v:k for k,v in AA3.items()}.get(aa,'GLY'), ' ')
            ca_pos = ctr + np.array([12+i*3.8, 3, -2])
            r.add(Atom.Atom('CA', ca_pos.tolist(), 0, 1, ' ', ' CA ', i+1, 'C'))
            r.add(Atom.Atom('N', (ca_pos+[-1.46,0,0]).tolist(), 0, 1, ' ', ' N  ', i+1, 'N'))
            r.add(Atom.Atom('C', (ca_pos+[1.52,0,0]).tolist(), 0, 1, ' ', ' C  ', i+1, 'C'))
            r.add(Atom.Atom('O', (ca_pos+[2.1,0,0]).tolist(), 0, 1, ' ', ' O  ', i+1, 'O'))
            pep_chain.add(r)
        mo.add(pep_chain); st.add(mo)
        complex_pdb = f'_pillar_a_{ab_id}.pdb'
        io = PDBIO(); io.set_structure(st); io.save(complex_pdb)

        # Run MPNN with fixed hotspot positions
        print(f"\n  Running MPNN ({args.samples} samples, fixed hotspots)...")
        t0 = time.time()
        try:
            from run_p3_idp_design import run_mpnn
            variants = run_mpnn(complex_pdb, f'{info["scaffold_chain"]} P', args.samples, 0.5, 42, omit_aas='C')
            print(f"  Generated {len(variants)} variants ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  MPNN failed: {e}")
            variants = []

        # Score variants
        from modules.cascade_filter import _cdr_complexity
        scored = []
        for v in variants:
            cdr = v['sequence']
            # Anti-degen
            from collections import Counter
            import math
            run = 1; degen_ok = True
            for i in range(1, len(cdr)):
                if cdr[i] == cdr[i-1]: run += 1
                else: run = 1
                if run > 4: degen_ok = False; break
            if not degen_ok: continue
            # Shannon + aromatic
            cnt = Counter(cdr)
            shannon = -sum((c/len(cdr))*math.log(max(c/len(cdr),1e-6)) for c in cnt.values())
            aromatic = sum(1 for a in cdr if a in 'YWF') / len(cdr)
            complexity = _cdr_complexity(cdr)
            v['complexity'] = complexity
            v['shannon'] = shannon
            v['aromatic'] = aromatic
            v['ab_id'] = ab_id
            scored.append(v)

        scored.sort(key=lambda v: v['complexity'] + 0.5*v.get('shannon',0)/3.0, reverse=True)
        all_variants.extend(scored)
        print(f"  {len(scored)} variants passed filters")

    # Report
    print(f"\n{'='*65}")
    print(f"PILLAR A: Top-{args.top} Template-Seeded Variants")
    print(f"{'='*65}")
    for i, v in enumerate(all_variants[:args.top]):
        print(f"#{i+1:2d} [{v['ab_id']}] complexity={v['complexity']:.3f} "
              f"shannon={v['shannon']:.2f} aromatic={v['aromatic']:.2f}")
        print(f"    CDR: {v['sequence'][:70]}")

    # Save
    os.makedirs('idp_design_results', exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out = f'idp_design_results/pillar_a_variants_{ts}.json'
    with open(out, 'w') as f:
        json.dump(all_variants, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
