"""V4 B1: Template-conditioned AF2 — fix M0 blind refolding failure.
Uses ColabFold with crystal structures as templates so AF2 doesn't
need to blindly refold the short peptide-antibody complex.

B1 hypothesis: with template injection, 4HIX native CDR ipTM should
jump from 0.118 (blind) to >0.6 (template-conditioned).

Usage (WSL):
    python run_v4_b1_template_af2.py
"""
import sys,os,json,time,subprocess,tempfile
os.chdir("/mnt/c/biological/DisorderFlow")

# Step 1: Prepare templates
print("Step 1: Preparing crystal templates...")
from Bio.PDB import PDBParser,PDBIO
os.makedirs("idp_benchmark_results/templates",exist_ok=True)

for pdb_id in ["4HIX","5CSZ"]:
    pdb=f"data/anti_abeta_refs/{pdb_id}.pdb"
    s=PDBParser(QUIET=True).get_structure("s",pdb)
    # Keep antibody H + peptide A/D
    epi_chain="D" if pdb_id=="5CSZ" else "A"
    for m in s:
        for c in list(m):
            if c.id not in ("H",epi_chain):m.detach_child(c.id)
    io=PDBIO();io.set_structure(s)
    out=f"idp_benchmark_results/templates/{pdb_id}_template.pdb"
    io.save(out)
    print(f"  {pdb_id} template: {out}")

# Step 2: Build FASTA inputs
print("\nStep 2: Building FASTA inputs...")
AA3={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

# 4HIX native CDR grafted onto 4HIX scaffold + KLVFFAED peptide
for pdb_id,epi_chain,peptide in [("4HIX","A","KLVFFAED"),("5CSZ","D","DAEFRHDSGY")]:
    pdb=f"data/anti_abeta_refs/{pdb_id}.pdb"
    s=PDBParser(QUIET=True).get_structure("s",pdb)
    # Get antibody H chain sequence
    ab_seq=""
    for res in s[0]["H"]:
        if "CA" in res:ab_seq+=AA3.get(res.resname.strip(),"X")

    fasta=f">{pdb_id}_native_CDR\n{ab_seq}\n>epitope\n{peptide}\n"
    out=f"idp_benchmark_results/templates/{pdb_id}_test.fasta"
    with open(out,"w") as f:f.write(fasta)
    print(f"  {pdb_id}: {len(ab_seq)}aa Ab + {len(peptide)}aa epitope")

# Step 3: Run ColabFold with templates
print("\nStep 3: Running ColabFold with template injection...")
for pdb_id in ["4HIX","5CSZ"]:
    fasta=f"idp_benchmark_results/templates/{pdb_id}_test.fasta"
    template=f"idp_benchmark_results/templates/{pdb_id}_template.pdb"
    outdir=f"idp_benchmark_results/colabfold_{pdb_id}"

    if not os.path.exists(fasta) or not os.path.exists(template):
        print(f"  {pdb_id}: SKIP (missing input)")
        continue

    print(f"  {pdb_id}: colabfold_batch --templates --custom-template-path {template}...")
    t0=time.time()
    cmd=[
        "colabfold_batch",
        fasta,outdir,
        "--templates",
        "--custom-template-path",template,
        "--num-recycle","1",
        "--num-models","1",
        "--use-gpu-relax"
    ]
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=600)
        dt=time.time()-t0
        print(f"    done in {dt:.0f}s (exit {r.returncode})")
        if r.stdout:
            for line in r.stdout.split("\n"):
                if "iptm" in line.lower() or "plddt" in line.lower():
                    print(f"    {line.strip()[:120]}")
        if r.returncode!=0 and r.stderr:
            print(f"    STDERR: {r.stderr[:500]}")
    except Exception as e:
        print(f"    FAIL: {e}")

# Step 4: Parse results
print("\nStep 4: Results...")
for pdb_id in ["4HIX","5CSZ"]:
    outdir=f"idp_benchmark_results/colabfold_{pdb_id}"
    if not os.path.isdir(outdir):continue
    for f in os.listdir(outdir):
        if f.endswith(".json"):
            with open(os.path.join(outdir,f)) as jf:
                data=json.load(jf)
                if "iptm" in str(data) or "plddt" in str(data):
                    print(f"  {pdb_id}/{f}: {json.dumps(data)[:200]}")
                    break

print("\nB1 Template AF2 done.")
