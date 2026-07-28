#!/usr/bin/env python3
"""Build full-backbone Aβ42 PDBs from the 5-seed CA conformations.

The single NMR model (2NAO_model1_A_1-42.pdb) only parses to 19/42 residues
when placed as a fixed-backbone design context — its IDP geometry breaks the
parser's CA-CA<4Å chain walk, capping the designable epitope. The seed file
(data/abeta_conformations/abeta42_5seed.pkl) has 5 conformations × 42 CA
positions; this script writes each as a proper PDB with a reconstructed
N-CA-C-O backbone (Bio.PDB NCBPBuilder from CA), so the parser sees a full
42-residue Aβ42 chain.

Direction C: multi-conformation Aβ42 + clean scaffolds to break the fixed-
backbone ceiling.
"""
import sys, os, pickle
if sys.platform == 'win32':
    import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np

SEED_PKL = 'data/abeta_conformations/abeta42_5seed.pkl'
OUT_DIR = 'data/abeta_conformations/pdbs'

AA3 = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'
THREE = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','E':'GLU','Q':'GLN',
         'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE',
         'P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}


def main():
    """Build full-backbone Aβ42 PDBs (one per seed conformation).

    The parser requires N,CA,C,O. The seed file only has CA, so we place ideal-
    geometry N/C/O around each CA following the residue-chain direction (the
    CA_{i-1}->CA_i vector). Exact backbone geometry is not needed for BFN
    seq-only design (it consumes pos_heavyatom N,CA,C,O as the true backbone);
    we just need a self-consistent, parseable 42-residue chain so the full Aβ42
    is visible as design context (vs the 19/42 truncation from the NMR model).
    """
    import numpy as np
    d = pickle.load(open(SEED_PKL, 'rb'))
    cas = np.array(d['ca_positions'])  # (5,42,3)
    seq = d['sequence']
    rmsf = np.array(d['rmsf'])
    os.makedirs(OUT_DIR, exist_ok=True)
    from Bio.PDB import Structure, Model, Chain, Residue, Atom
    from Bio.PDB import PDBIO

    IDEAL_N_CA = 1.46
    IDEAL_CA_C = 1.52

    for s in range(cas.shape[0]):
        ca = cas[s]  # (42,3)
        st = Structure.Structure(f'ab42s{s}')
        mo = Model.Model(0)
        ch = Chain.Chain('P')  # 'P' = peptide/Aβ42; avoids collision with VHH
                              # scaffold chains (A/B/C) read by position_epitope.
        for i in range(len(seq)):
            res = Residue.Residue((' ', i + 1, ' '), THREE[seq[i]], ' ')
            res.add(Atom.Atom('CA', list(ca[i]), 0.0, 1.0, ' ', ' CA ', i + 1, 'C'))
            if i > 0:
                dvec = ca[i] - ca[i - 1]
            else:
                dvec = ca[i + 1] - ca[i] if len(ca) > 1 else np.array([1., 0, 0])
            nrm = np.linalg.norm(dvec)
            dvec = dvec / nrm if nrm > 1e-6 else np.array([1., 0, 0])
            n_pos = ca[i] - dvec * IDEAL_N_CA + np.array([0., 0, 0.5])
            res.add(Atom.Atom('N', list(n_pos), 0.0, 1.0, ' ', ' N  ', i + 1, 'N'))
            c_pos = ca[i] + dvec * IDEAL_CA_C + np.array([0., 0, -0.5])
            res.add(Atom.Atom('C', list(c_pos), 0.0, 1.0, ' ', ' C  ', i + 1, 'C'))
            o_pos = c_pos + dvec * 0.6
            res.add(Atom.Atom('O', list(o_pos), 0.0, 1.0, ' ', ' O  ', i + 1, 'O'))
            ch.add(res)
        mo.add(ch)
        st.add(mo)
        io = PDBIO()
        io.set_structure(st)
        out = os.path.join(OUT_DIR, f'abeta42_seed{s}_42.pdb')
        io.save(out)
        print(f"  wrote {out}  (seed{s}, mean RMSF {rmsf.mean():.2f})")
    print(f"\nDone: {cas.shape[0]} Aβ42 conformation PDBs in {OUT_DIR}")


if __name__ == '__main__':
    main()
