"""V4 Step 1: AF2-fold 10 designs + 4 native/scram controls, save CDR coords."""
import sys,os,json,time,numpy as np,random
os.chdir("/mnt/d/biological/DisorderFlow")
sys.path.insert(0,"/mnt/d/biological/DisorderFlow")
sys.path.insert(0,"/mnt/d/biological/DisorderFlow/modules")
from af2_jax_runner import run_multimer_prediction
from Bio.PDB import PDBParser
AA3={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

# Load top-5 designs only (AF2 folding is slow)
with open("idp_design_results/p3_scale_top100_20260629_030111.json") as f:
    designs=json.load(f)[:5]

# Native controls
NATIVES=[
    {"sequence":"GFTFSNYRSGGGYCVRYDHY","epitope":"4HIX_native","scaffold":"4HIX","cdr_spec":"A:26-33,A:51-58,A:97-113","scaffold_chain":"A","scaffold_path":"data/anti_abeta_refs/4HIX.pdb"},
    {"sequence":"GFTFSSYAISGSGGSTYYADSVKG","epitope":"5CSZ_native","scaffold":"5CSZ","cdr_spec":"A:26-33,A:51-58,A:97-113","scaffold_chain":"A","scaffold_path":"data/anti_abeta_refs/5CSZ.pdb"},
]
for nat in NATIVES[:]:
    s=list(nat["sequence"]);random.shuffle(s)
    NATIVES.append({**nat,"sequence":"".join(s),"epitope":nat["epitope"]+"_scram"})

all_d = NATIVES + designs
print(f"V4 Step 1: Folding {len(all_d)} antibodies (4 native/scram + 5 generated)")

results = []
for i,d in enumerate(all_d):
    sp = d.get("scaffold_path","") or f'data/misfolding_targets/{d.get("scaffold","3STB")}.pdb'
    if os.path.exists(sp):
        s = PDBParser(QUIET=True).get_structure("s",sp)
        ch = d.get("scaffold_chain","A")
        cs = d.get("cdr_spec","A:26-33,A:51-58,A:97-113")
        cr = [(int(p.split(":")[1].split("-")[0]),int(p.split(":")[1].split("-")[1])) for p in cs.split(",")]
        scaff_seq = []
        for res in s[0][ch]:
            if "CA" in res: scaff_seq.append(AA3.get(res.resname.strip(),"X"))
        full = list(scaff_seq); cdr = d["sequence"]; pos = 0
        for st,en in cr:
            for j in range(st-1, en):
                if pos < len(cdr) and j < len(full): full[j] = cdr[pos]; pos += 1
        full_str = "".join(full)
    else:
        full_str = d.get("sequence","")

    print(f"  [{i+1}/{len(all_d)}] {len(full_str)}aa...", end=" ", flush=True)
    t1 = time.time()
    try:
        res = run_multimer_prediction(full_str, "", num_recycle=1, return_structure=True)
    except Exception as e:
        print(f"FAIL: {e}"); continue

    if "final_atom_positions" not in res:
        print("no coords"); continue

    pos = np.array(res["final_atom_positions"])
    crd = []; cdr_seq = []
    cs2 = d.get("cdr_spec","A:26-33,A:51-58,A:97-113")
    cr2 = [(int(p.split(":")[1].split("-")[0]),int(p.split(":")[1].split("-")[1])) for p in cs2.split(",")]
    for st,en in cr2:
        for j in range(st-1, min(en, len(pos))):
            crd.append([pos[j,0,:].tolist(), pos[j,1,:].tolist(), pos[j,2,:].tolist()])
            if j < len(full_str): cdr_seq.append(full_str[j])

    d["v4_cdr_coords"] = crd
    d["v4_cdr_seq"] = "".join(cdr_seq)
    d["v4_plddt"] = float(res.get("plddt",0))
    d["v4_n_coords"] = len(crd)
    results.append(d)
    print(f"plddt={d['v4_plddt']:.3f} n_cdr={len(crd)} ({time.time()-t1:.0f}s)")

ts = time.strftime("%Y%m%d_%H%M%S")
out = f"idp_design_results/v4_folded_coords_{ts}.json"
with open(out,"w") as f:
    json.dump(results, f, indent=2)
print(f"Saved {out} ({len(results)} designs)")
