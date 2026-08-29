"""V4: AF2-fold designs → extract CDR coords → L4 ESM-IF on REAL structure.
Paradigm shift: from scaffold backbone to folded CDR coordinates.
If this also fails, pure computational metrics have reached their limit."""
import sys,os,json,time,numpy as np,random
os.chdir("/mnt/d/biological/DisorderFlow")
sys.path.insert(0,"/mnt/d/biological/DisorderFlow")
sys.path.insert(0,"/mnt/d/biological/DisorderFlow/modules")
from af2_jax_runner import run_multimer_prediction
import esm,esm.inverse_folding.util as util
from scipy.stats import kendalltau

model,alphabet=esm.pretrained.esm_if1_gvp4_t16_142M_UR50();model.eval()

# Load designs
with open("idp_design_results/p3_scale_top100_20260629_030111.json") as f:
    designs=json.load(f)[:10]  # start with 10 for speed
print(f"V4: AF2-fold {len(designs)} designs → real CDR coords → L4 ESM-IF")
print("="*60)

results=[]
t0=time.time()

# Add native controls
NATIVES=[
    {"sequence":"GFTFSNYRSGGGYCVRYDHY","epitope":"4HIX_native","scaffold":"4HIX"},
    {"sequence":"GFTFSSYAISGSGGSTYYADSVKG","epitope":"5CSZ_native","scaffold":"5CSZ"},
]
for nat in NATIVES:
    s=list(nat["sequence"]);random.shuffle(s)
    NATIVES.append({**nat,"sequence":"".join(s),"epitope":nat["epitope"]+"_scram"})

all_designs=NATIVES+designs

for i,d in enumerate(all_designs):
    # Graft CDR into scaffold sequence
    sp=d.get("scaffold_path","") or f'data/misfolding_targets/{d.get("scaffold","3STB")}.pdb'
    from Bio.PDB import PDBParser
    AA3={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
    if os.path.exists(sp):
        parser=PDBParser(QUIET=True);s=parser.get_structure("s",sp)
        ch=d.get("scaffold_chain","A")
        cs=d.get("cdr_spec","A:26-33,A:51-58,A:97-113")
        cr=[(int(p.split(":")[1].split("-")[0]),int(p.split(":")[1].split("-")[1]))for p in cs.split(",")]
        # Build full scaffold sequence
        scaff_seq=[]
        for res in s[0][ch]:
            if"CA"in res:scaff_seq.append(AA3.get(res.resname.strip(),"X"))
        # Graft CDR
        full_seq=list(scaff_seq);cdr=d["sequence"];pos=0
        for st,en in cr:
            for j in range(st-1,en):
                if pos<len(cdr) and j<len(full_seq):
                    full_seq[j]=cdr[pos];pos+=1
        full_str="".join(full_seq)
    else:
        full_str=d.get("sequence","")  # fallback

    # AF2 fold
    print(f"  [{i+1}/{len(all_designs)}] Folding {len(full_str)}aa...",end=" ",flush=True)
    t1=time.time()
    try:
        res=run_multimer_prediction(full_str,"",num_recycle=1,return_structure=True)
        dt=time.time()-t1
    except Exception as e:
        print(f"FAIL: {e}")
        continue

    if "final_atom_positions" not in res:
        print("no coords")
        continue

    pos=np.array(res["final_atom_positions"])  # (L, 37, 3)
    # CA = index 1, N = 0, C = 2
    # Extract CDR region backbone
    cs=d.get("cdr_spec","A:26-33,A:51-58,A:97-113")
    cr=[(int(p.split(":")[1].split("-")[0]),int(p.split(":")[1].split("-")[1]))for p in cs.split(",")]
    crd=[];cdr_seq=[]
    for st,en in cr:
        for j in range(st-1,min(en,len(pos))):
            crd.append([pos[j,0,:],pos[j,1,:],pos[j,2,:]])  # N,CA,C
            if j<len(full_str):cdr_seq.append(full_str[j])

    if len(crd)<3:
        print(f"CDR too short: {len(crd)}")
        continue

    # L4 ESM-IF on folded coords
    s_str="".join(cdr_seq)
    try:
        ll_full,ll_coord=util.score_sequence(model,alphabet,np.array(crd),s_str)
        print(f"LL={ll_coord:.4f} ({dt:.0f}s)")
    except Exception as e:
        print(f"ESM-IF fail: {e}")
        ll_coord=-999

    d["v4_folded_ll"]=float(ll_coord)
    d["v4_fold_time"]=float(dt)
    results.append(d)

# Report
nat_scores=[d["v4_folded_ll"] for d in results if "native" in d.get("epitope","") and "scram" not in d.get("epitope","")]
scram_scores=[d["v4_folded_ll"] for d in results if "scram" in d.get("epitope","")]
gen_scores=[d["v4_folded_ll"] for d in results if "native" not in d.get("epitope","") and "scram" not in d.get("epitope","")]

print(f"\n=== V4 Fold+Score Results ===")
if nat_scores: print(f"Native mean: {np.mean(nat_scores):.4f}")
if scram_scores: print(f"Scrambled mean: {np.mean(scram_scores):.4f}")
if gen_scores: print(f"Generated mean: {np.mean(gen_scores):.4f} range: {np.min(gen_scores):.4f}-{np.max(gen_scores):.4f}")
if nat_scores and scram_scores:
    sep=np.mean(nat_scores)-np.mean(scram_scores)
    print(f"Native-Scram separation: {sep:+.4f}")
    print(f"V4 Gate: {'PASS' if sep>0.5 else 'FAIL'} (need sep>0.5)")

ts=time.strftime("%Y%m%d_%H%M%S")
out=f"idp_design_results/v4_fold_score_{ts}.json"
with open(out,"w") as f:json.dump(results,f,indent=2)
print(f"Saved {out}")
print(f"Total: {time.time()-t0:.0f}s")
