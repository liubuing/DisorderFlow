#!/usr/bin/env python3
"""V11 Step 0+1: Fold-then-Dock — break CDR-side backbone domination.

V10 P1 failed: graft only changes residue names, CDR backbone CA RMSD=0.0000
across all designs → HDOCK scores identical (backbone-dominated).
V11 fix: AF2 single-chain fold each design → take REAL CDR loop backbone
(N/CA/C/O coords that differ per sequence) → graft ONLY the CDR loop segment
back into scaffold (framework stays native scaffold coords) → HDOCK.

Run in WSL (uses af2_jax_runner). Usage:
    wsl python run_v11_fold_graft.py
"""
import sys, os, json, time, random, subprocess, tempfile, shutil
import numpy as np
os.chdir("/mnt/c/biological/DisorderFlow")
sys.path.insert(0, "/mnt/c/biological/DisorderFlow")
sys.path.insert(0, "/mnt/c/biological/DisorderFlow/modules")
from af2_jax_runner import run_multimer_prediction
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue
from Bio.PDB.vectors import Vector

HDOCK_DIR = "HDOCKlite-v1.1"
AA3 = {"A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","E":"GLU","Q":"GLN",
       "G":"GLY","H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE",
       "P":"PRO","S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL"}
AA1 = {v:k for k,v in AA3.items()}

# 4HIX/5CSZ native CDR (from idp_benchmark_known_abs / run_v4_fold_only)
NATIVES = [
    {"id":"4HIX","scaffold":"data/anti_abeta_refs/4HIX.pdb","scaffold_chain":"H",
     "cdr_spec":"H:26-33,H:51-58,H:95-102","cdr_seq":"GFTFSNYRSGGGYCVRYDHY",
     "ligand":"data/anti_abeta_refs/4HIX.pdb","ligand_chain":"A"},  # A=6aa peptide
    {"id":"5CSZ","scaffold":"data/anti_abeta_refs/5CSZ.pdb","scaffold_chain":"H",
     "cdr_spec":"H:26-33,H:51-58,H:95-102","cdr_seq":"GFTFSSYAISGSGGSTYYADSVKG",
     "ligand":"data/anti_abeta_refs/5CSZ.pdb","ligand_chain":"D"},  # D=11aa peptide
]
N_SCRAM = 3
NATIVE_SEEDS = 3  # fold runs per design (fold stochasticity)

def parse_cdr_spec(spec):
    """H:26-33,H:51-58,H:95-102 -> [(26,33),(51,58),(95,102)]"""
    out=[]
    for p in spec.split(","):
        rng=p.split(":")[1].split("-")
        out.append((int(rng[0]),int(rng[1])))
    return out

def get_scaffold_seq(pdb, chain):
    s=PDBParser(QUIET=True).get_structure("s",pdb)
    seq=[]
    for r in s[0][chain]:
        if r.id[0]==" " and "CA" in r:
            seq.append(AA1.get(r.resname.strip(),"X"))
    return "".join(seq), s

def graft_cdr_into_scaffold_seq(scaff_seq, cdr_spec, cdr_seq):
    """Replace CDR positions in scaffold seq with designed CDR seq."""
    full=list(scaff_seq); pos=0
    for st,en in parse_cdr_spec(cdr_spec):
        for j in range(st-1,en):
            if pos<len(cdr_seq) and j<len(full):
                full[j]=cdr_seq[pos]; pos+=1
    return "".join(full)

def fold_sequence(seq):
    """AF2 single-chain fold, return (atom_positions [L,4,3] N/CA/C/O, plddt) or None."""
    res=run_multimer_prediction(seq, "", num_recycle=1, return_structure=True)
    if "final_atom_positions" not in res: return None,None
    pos=np.array(res["final_atom_positions"])  # [L, atoms, 3]
    # atoms order: typically N, CA, C, O (first 4)
    return pos[:,:4,:], float(res.get("plddt",0))

def build_fold_graft_receptor(scaffold_pdb, scaffold_chain, cdr_spec, fold_pos, cdr_seq):
    """Build receptor PDB: scaffold framework + fold-out CDR loop backbone.

    Framework residues: keep original scaffold coords.
    CDR residues: replace N/CA/C/O with fold-out coords, set resname by cdr_seq.
    Sidechains: omitted (HDOCK builds its own / uses backbone).
    """
    s=PDBParser(QUIET=True).get_structure("s",scaffold_pdb)
    scaff_chain=s[0][scaffold_chain]
    cdr_ranges=parse_cdr_spec(cdr_spec)
    cdr_pos_set=set()
    for st,en in cdr_ranges:
        for j in range(st,en): cdr_pos_set.add(j)  # 0-based

    # New structure
    ns=Structure.Structure("r"); m=Model.Model(0); ns.add(m)
    nc=Chain.Chain(scaffold_chain); m.add(nc)

    seq_idx=0  # walks cdr_seq
    fold_idx=0  # walks fold_pos (0-based residue index = scaffold residue index)
    for res in scaff_chain:
        if res.id[0]!=" ": continue
        resseq=res.id[1]
        ridx=resseq-1  # 0-based
        # copy this residue
        new_res=Residue.Residue(res.id, res.resname, res.segid)
        if ridx in cdr_pos_set:
            # CDR residue: use fold coords, set resname from cdr_seq
            if seq_idx<len(cdr_seq):
                aa=cdr_seq[seq_idx]; new_res.resname=AA3.get(aa,"GLY")
                seq_idx+=1
            atom_names=["N","CA","C","O"]
            for ai,an in enumerate(atom_names):
                if fold_idx<len(fold_pos):
                    from Bio.PDB.Atom import Atom
                    coord=fold_pos[fold_idx,ai,:]
                    atom=Atom(an, coord, 0, 1, " ", an, 1, element=an[0])
                    new_res.add(atom)
            fold_idx+=1
        else:
            # framework: copy original atoms
            for atom in res:
                new_res.add(atom.copy())
        nc.add(new_res)
    return ns

def run_hdock(receptor_pdb, ligand_pdb, out_name):
    """Run HDOCK, return top1 score (min). Reuse V10 logic."""
    cmd=(f'cd {HDOCK_DIR} && LD_LIBRARY_PATH=. ./hdock {receptor_pdb} {ligand_pdb} '
         f'-out {out_name} 2>&1')
    try:
        subprocess.run(["wsl","bash","-c",cmd],capture_output=True,text=True,timeout=300)
    except subprocess.TimeoutExpired:
        return None
    out_path=os.path.join(HDOCK_DIR,out_name)
    if not os.path.exists(out_path): return None
    scores=[]
    for line in open(out_path):
        parts=line.strip().split()
        if len(parts)==9:
            try: scores.append(float(parts[6]))
            except ValueError: continue
    if not scores: return None
    return {"top1":min(scores),"mean":float(np.mean(scores)),"n":len(scores)}

def extract_ligand_pdb(src_pdb, lig_chain, out_pdb):
    """Extract ligand (peptide) chain to its own PDB for HDOCK."""
    s=PDBParser(QUIET=True).get_structure("s",src_pdb)
    io=PDBIO(); io.set_structure(s[0][lig_chain]); io.save(out_pdb)

def cd_loop_rmsd(pos_a, pos_b):
    """RMSD of CA coords between two fold outputs (sanity check signal exists)."""
    n=min(len(pos_a),len(pos_b))
    a=pos_a[:n,1,:]; b=pos_b[:n,1,:]  # CA
    return float(np.sqrt(((a-b)**2).sum(-1).mean()))

print("="*60); print("V11 Fold-then-Dock: native vs scrambled gate"); print("="*60)

results={"probe":"V11 fold-then-dock gate","results":{}}
all_scores={"native":[],"scrambled":[]}

for nat in NATIVES:
    ab=nat["id"]
    print(f"\n--- {ab} ---")
    scaff_seq, _ = get_scaffold_seq(nat["scaffold"], nat["scaffold_chain"])
    # ligand
    lig_pdb=f"/tmp/v11_{ab}_ligand.pdb"
    extract_ligand_pdb(nat["ligand"], nat["ligand_chain"], lig_pdb)

    # native + scrambled CDRs
    cands=[("native", nat["cdr_seq"])]
    for si in range(N_SCRAM):
        s=list(nat["cdr_seq"]); random.shuffle(s)
        cands.append((f"scram_{si}", "".join(s)))

    ab_scores={"native":[],"scrambled":[]}
    for label, cdr in cands:
        full=graft_cdr_into_scaffold_seq(scaff_seq, nat["cdr_spec"], cdr)
        # fold
        print(f"  fold {label} ({len(full)}aa)...", end=" ", flush=True)
        t1=time.time()
        fold_pos, plddt = fold_sequence(full)
        if fold_pos is None:
            print("FAIL"); continue
        print(f"plddt={plddt:.3f} ({time.time()-t1:.0f}s)")
        # build receptor: scaffold framework + fold CDR loop
        rec_struct=build_fold_graft_receptor(nat["scaffold"], nat["scaffold_chain"],
                                              nat["cdr_spec"], fold_pos, cdr)
        rec_pdb=f"/tmp/v11_{ab}_{label}.pdb"
        io=PDBIO(); io.set_structure(rec_struct); io.save(rec_pdb)
        # HDOCK
        out_name=f"v11_{ab}_{label}.out"
        sc=run_hdock(rec_pdb, lig_pdb, out_name)
        if sc is None: print(f"    HDOCK fail"); continue
        print(f"    hdock top1={sc['top1']:.2f} mean={sc['mean']:.2f}")
        bucket="native" if label=="native" else "scrambled"
        ab_scores[bucket].append(sc["top1"])
        all_scores[bucket].append(sc["top1"])
    results["results"][ab]=ab_scores

# Stats: Mann-Whitney native vs scrambled
from scipy.stats import mannwhitneyu
nat_scores=all_scores["native"]; scr_scores=all_scores["scrambled"]
print("\n=== Gate ===")
print(f"native   n={len(nat_scores)} scores={[round(x,1) for x in nat_scores]}")
print(f"scrambled n={len(scr_scores)} scores={[round(x,1) for x in scr_scores]}")
if len(nat_scores)>=1 and len(scr_scores)>=1:
    sep=np.mean(nat_scores)-np.mean(scr_scores)
    try:
        u,p=mannwhitneyu(nat_scores, scr_scores, alternative="less")  # native top1 smaller=better
        gate = p<0.05
    except Exception:
        p=1.0; gate=False
    print(f"mean native={np.mean(nat_scores):.2f} scram={np.mean(scr_scores):.2f} sep={sep:+.2f}")
    print(f"Mann-Whitney p={p:.2e}  GATE={'PASS' if gate else 'FAIL'}")
    results["gate"]=gate; results["p"]=float(p); results["sep"]=float(sep)
else:
    results["gate"]=False; print("INSUFFICIENT DATA")

ts=time.strftime("%Y%m%d_%H%M%S")
out=f"idp_benchmark_results/v11_fold_dock_gate_{ts}.json"
os.makedirs("idp_benchmark_results",exist_ok=True)
json.dump(results, open(out,"w"), indent=2)
print(f"\nSaved {out}")
