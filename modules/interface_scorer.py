#!/usr/bin/env python3
"""V3 L3: Interface quality scorer on docked poses (BSA + shape + electrostatics).

Scoring terms (all computable on given coordinates, no refolding):
  E_contact:  Cb-Cb pairs <=8A (geometry)
  E_bsa:      Buried Surface Area between chains (geometry)
  E_elec:     Electrostatic complementarity (all-pairs, distance-weighted)
  E_hydro:    Hydrophobic matching (all-pairs, distance-weighted)
  E_shape:    Contact distance uniformity
  E_composite: Weighted sum (0.25*contact + 0.25*bsa + 0.20*elec + 0.15*hydro + 0.15*shape)

This is the POSE SCORER — L2 (FFT) generates poses, L3 scores them.
V3_ROOT: NO AF2 refolding. Physics on given coordinates only.

Usage:
    from modules.interface_scorer import score_pose, three_way_vote
    result = score_pose('docked_complex.pdb', rec_chain='H', lig_chain='P')
"""
import sys, os
import numpy as np
from scipy.spatial import cKDTree

CHARGE = {'A':0,'R':1,'N':0,'D':-1,'C':0,'E':-1,'Q':0,'G':0,'H':0,'I':0,
          'L':0,'K':1,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}
HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'E':-3.5,'Q':-3.5,'G':-0.4,
         'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
         'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CONTACT_CUT = 8.0; ELEC_CUT = 12.0; HYDRO_CUT = 10.0


def _get_atoms(pdb_path, chain):
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True); s = parser.get_structure('s', pdb_path)
    coords, resis = [], []
    for model in s:
        for c in model:
            if c.id.strip() != chain: continue
            for res in c:
                aa = res.resname.strip()
                aa1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E',
                       'GLN':'Q','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K',
                       'MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
                       'TYR':'Y','VAL':'V'}.get(aa,'X')
                if 'CB' in res: coords.append(res['CB'].get_coord())
                elif 'CA' in res: coords.append(res['CA'].get_coord())
                else: continue
                resis.append(aa1)
    return np.array(coords), resis


def _bsa(pdb_path, c1, c2):
    from Bio.PDB import PDBParser; from Bio.PDB.SASA import ShrakeRupley
    parser = PDBParser(QUIET=True); s = parser.get_structure('s', pdb_path)
    sr = ShrakeRupley()
    sr.compute(s, level='A')
    all_atoms = list(s.get_atoms())
    sasa_complex = sum(a.sasa for a in all_atoms)
    # Chain1 isolated
    from Bio.PDB import Structure, Model
    iso = Structure.Structure('iso'); iso_m = Model.Model(0)
    for c in s[0]: 
        if c.id.strip()==c1: iso_m.add(c.copy())
    sr.compute(iso_m, level='A')
    sasa1 = sum(a.sasa for a in iso_m.get_atoms())
    # Chain2 isolated
    iso2 = Structure.Structure('iso2'); iso2_m = Model.Model(0)
    for c in s[0]:
        if c.id.strip()==c2: iso2_m.add(c.copy())
    sr.compute(iso2_m, level='A')
    sasa2 = sum(a.sasa for a in iso2_m.get_atoms())
    return max(0.0, (sasa1+sasa2-sasa_complex)/2.0)


def score_pose(pdb_path, rec_chain='H', lig_chain='P', cdr_ranges=None):
    rec_coords, rec_res = _get_atoms(pdb_path, rec_chain)
    lig_coords, lig_res = _get_atoms(pdb_path, lig_chain)
    if len(rec_coords)==0 or len(lig_coords)==0:
        return {'error':'Empty chain','composite':0}
    if cdr_ranges:
        mask = np.zeros(len(rec_coords), dtype=bool)
        for s,e in cdr_ranges: mask[s-1:e] = True
        rec_coords = rec_coords[mask]; rec_res = [rec_res[i] for i in range(len(mask)) if mask[i]]
    n_rec = len(rec_coords)
    # E_contact
    tree = cKDTree(lig_coords)
    dists, _ = tree.query(rec_coords, distance_upper_bound=CONTACT_CUT)
    contacts = int((dists < CONTACT_CUT).sum())
    e_contact = contacts/max(n_rec,1)
    # E_bsa
    bsa_val = _bsa(pdb_path, rec_chain, lig_chain)
    e_bsa = min(1.0, bsa_val/500.0)
    # All-pairs distance matrix
    all_d = np.linalg.norm(rec_coords[:,None,:]-lig_coords[None,:,:], axis=-1)
    # E_elec
    ew = np.maximum(0, 1.0-all_d/ELEC_CUT)
    e_elec = 0.0; ne = 0
    for i in range(len(rec_coords)):
        for j in range(len(lig_coords)):
            w = ew[i,j]
            if w<0.01: continue
            rc = CHARGE.get(rec_res[i],0); lc = CHARGE.get(lig_res[j],0)
            if rc!=0 and lc!=0: e_elec += w*(-rc*lc); ne+=1
    e_elec_n = e_elec/max(ne,1)
    # E_hydro
    hw = np.maximum(0, 1.0-all_d/HYDRO_CUT)
    e_hydro = 0.0; nh = 0
    for i in range(len(rec_coords)):
        for j in range(len(lig_coords)):
            w = hw[i,j]
            if w<0.01: continue
            rh = HYDRO.get(rec_res[i],0); lh = HYDRO.get(lig_res[j],0)
            if rh==0 and lh==0: continue
            nh += 1
            if rh>0 and lh>0: e_hydro += w*min(rh,lh)/4.5
            elif rh<0 and lh<0: e_hydro += w*0.3
    e_hydro_n = e_hydro/max(nh,1)
    # E_shape
    cd = dists[dists<CONTACT_CUT]
    e_shape = 1.0-cd.std()/CONTACT_CUT if len(cd)>1 else 0.0
        # Composite — zero if contacts < 5 (invalid dock)
    if contacts < 5:
        return {'contacts':contacts,'e_contact':round(e_contact,4),'bsa':round(bsa_val,1),
                'e_bsa':round(e_bsa,4),'e_elec':round(e_elec_n,4),'e_hydro':round(e_hydro_n,4),
                'e_shape':round(e_shape,4),'composite':0.0,'invalid':True}
    comp = round(0.25*e_contact+0.25*e_bsa+0.20*max(0,e_elec_n)+0.15*max(0,e_hydro_n)+0.15*e_shape,4)
    return {'contacts':contacts,'e_contact':round(e_contact,4),'bsa':round(bsa_val,1),
            'e_bsa':round(e_bsa,4),'e_elec':round(e_elec_n,4),'e_hydro':round(e_hydro_n,4),
            'e_shape':round(e_shape,4),'composite':comp,'invalid':False}


def three_way_vote(physical_score, inverse_fold_score=None, single_chain_plddt=None):
    e1_ok = physical_score > 0.2
    e2_ok = inverse_fold_score is not None and inverse_fold_score > 0.5
    e3_ok = single_chain_plddt is not None and single_chain_plddt > 0.4
    passed = e1_ok and (e2_ok or e3_ok)
    reason = []
    if e1_ok: reason.append('E1')
    if e2_ok: reason.append('E2')
    if e3_ok: reason.append('E3')
    if not passed and not e1_ok: reason.append('E1_FAIL')
    return {'pass':passed,'reason':'+'.join(reason) if reason else 'NONE'}


if __name__ == '__main__':
    print("V3 L3 Interface Scorer — 5CSZ sanity")
    pdb = 'data/anti_abeta_refs/5CSZ.pdb'
    if os.path.exists(pdb):
        r = score_pose(pdb, rec_chain='H', lig_chain='D', cdr_ranges=[(26,32),(52,56),(95,102)])
        for k,v in r.items(): print(f'  {k}: {v}')
