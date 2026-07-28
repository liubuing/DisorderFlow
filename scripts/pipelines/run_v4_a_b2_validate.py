"""B2 fold-then-score validation of Pillar A template-seeded variants.

Tests 5 top A variants + 2 native seed CDRs (4HIX,5CSZ).
AF2 single-chain fold each, extract CDR coords, ESM-IF score.
Key question: do native seeds score HIGHER than variants?
(Indirect evidence that binding signal is preserved.)

Usage (WSL): python run_v4_a_b2_validate.py
"""
import sys,os,json,time,random
os.chdir("/mnt/c/biological/DisorderFlow")
sys.path.insert(0,"/mnt/c/biological/DisorderFlow")
sys.path.insert(0,"/mnt/c/biological/DisorderFlow/modules")
import numpy as np
from af2_jax_runner import run_multimer_prediction
from Bio.PDB import PDBParser
AA3={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

# Load A variants
with open("idp_design_results/pillar_a_variants_20260701_005208.json") as f:
    variants=json.load(f)

# Top 5 variants + 2 native seeds
designs=[]
for v in variants[:3]:  # top 3 from 4HIX
    designs.append({"sequence":v["sequence"],"label":f"A_4HIX_var","epitope":"variant"})
for v in variants[37:40]:  # top 3 from 5CSZ
    designs.append({"sequence":v["sequence"],"label":f"A_5CSZ_var","epitope":"variant"})

# Native seeds
NATIVES=[
    {"sequence":"GFTFSNYRSGGGYCVRYDHY","label":"4HIX_native","epitope":"4HIX_native","scaffold":"4HIX","cdr_spec":"A:26-33,A:51-58,A:97-113","scaffold_chain":"A","scaffold_path":"data/anti_abeta_refs/4HIX.pdb"},
    {"sequence":"GFTFSSYAISGSGGSTYYADSVKG","label":"5CSZ_native","epitope":"5CSZ_native","scaffold":"5CSZ","cdr_spec":"A:26-33,A:51-58,A:97-113","scaffold_chain":"A","scaffold_path":"data/anti_abeta_refs/5CSZ.pdb"},
]
# Add scrambled versions as negative controls
for nat in NATIVES[:]:
    s=list(nat["sequence"]);random.shuffle(s)
    NATIVES.append({**nat,"sequence":"".join(s),"label":nat["label"]+"_scram","epitope":nat["epitope"]+"_scram"})

all_d=NATIVES+designs
print(f"B2 validate Pillar A: {len(all_d)} designs (2 native + 2 scram + 6 variants)")
print("="*60)

# Step 1: AF2 fold each
print("Step 1: AF2 single-chain folding...")
for i,d in enumerate(all_d):
    sp=d.get("scaffold_path","") or "data/misfolding_targets/3STB.pdb"
    if os.path.exists(sp):
        s=PDBParser(QUIET=True).get_structure("s",sp)
        ch=d.get("scaffold_chain","A")
        cs=d.get("cdr_spec","A:26-33,A:51-58,A:97-113")
        cr=[(int(p.split(":")[1].split("-")[0]),int(p.split(":")[1].split("-")[1]))for p in cs.split(",")]
        scaff_seq=[AA3.get(res.resname.strip(),"X")for res in s[0][ch]if"CA"in res]
        full=list(scaff_seq);cdr=d["sequence"];pos=0
        for st,en in cr:
            for j in range(st-1,en):
                if pos<len(cdr)and j<len(full):full[j]=cdr[pos];pos+=1
        full_str="".join(full)
    else:full_str=d.get("sequence","")

    print(f"  [{i+1}/{len(all_d)}] {d['label']:20s} {len(full_str)}aa...",end=" ",flush=True)
    t1=time.time()
    try:
        res=run_multimer_prediction(full_str,"",num_recycle=1,return_structure=True)
    except Exception as e:print(f"FAIL {e}");continue
    if"final_atom_positions"not in res:print("no coords");continue
    pos=np.array(res["final_atom_positions"])
    # Extract CDR coords
    cs2=d.get("cdr_spec","A:26-33,A:51-58,A:97-113")
    cr2=[(int(p.split(":")[1].split("-")[0]),int(p.split(":")[1].split("-")[1]))for p in cs2.split(",")]
    crd=[];cdr_seq=[]
    for st,en in cr2:
        for j in range(st-1,min(en,len(pos))):
            crd.append([pos[j,0,:].tolist(),pos[j,1,:].tolist(),pos[j,2,:].tolist()])
            if j<len(full_str):cdr_seq.append(full_str[j])
    d["v4_cdr_coords"]=crd;d["v4_cdr_seq"]="".join(cdr_seq)
    d["v4_plddt"]=float(res.get("plddt",0));d["v4_n_coords"]=len(crd)
    print(f"plddt={d['v4_plddt']:.3f} n={len(crd)} ({time.time()-t1:.0f}s)")

# Step 2: ESM-IF scoring
print("\nStep 2: ESM-IF scoring on folded CDR coordinates...")
import esm,esm.inverse_folding.util as util
model,alphabet=esm.pretrained.esm_if1_gvp4_t16_142M_UR50();model.eval()

results=[]
for d in all_d:
    if d.get("v4_n_coords",0)<3:continue
    crd=np.array(d["v4_cdr_coords"],dtype=np.float32)
    seq=d["v4_cdr_seq"]
    ll_full,ll_coord=util.score_sequence(model,alphabet,crd,seq)
    d["b2_ll"]=float(ll_coord)
    results.append(d)
    print(f"  {d['label']:25s} LL={ll_coord:.4f} plddt={d.get('v4_plddt',0):.3f}")

# Report
nat=[d["b2_ll"] for d in results if "native" in d.get("epitope","") and "scram" not in d.get("epitope","")]
scr=[d["b2_ll"] for d in results if "scram" in d.get("epitope","")]
var=[d["b2_ll"] for d in results if "variant" in d.get("epitope","")]

print(f"\n=== B2 Pillar A Validation ===")
if nat:print(f"Native seeds:   mean={np.mean(nat):.4f} values={[f'{x:.4f}' for x in nat]}")
if scr:print(f"Scrambled:      mean={np.mean(scr):.4f} values={[f'{x:.4f}' for x in scr]}")
if var:print(f"A variants:     mean={np.mean(var):.4f} range={np.min(var):.4f}-{np.max(var):.4f}")
if nat and scr:
    sep=np.mean(nat)-np.mean(scr)
    print(f"Native-scram sep: {sep:+.4f}")
if nat and var:
    sep_nv=np.mean(nat)-np.mean(var)
    print(f"Native-variant sep: {sep_nv:+.4f}")
    print(f"Native ABOVE variant: {all(n>np.mean(var) for n in nat)}")

ts=time.strftime("%Y%m%d_%H%M%S")
out=f"idp_design_results/b2_pillar_a_{ts}.json"
with open(out,"w")as f:json.dump(results,f,indent=2)
print(f"Saved {out}")
