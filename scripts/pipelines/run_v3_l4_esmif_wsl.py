import esm,esm.inverse_folding.util as util,torch,numpy as np,os,random,sys
os.chdir("/mnt/d/biological/DisorderFlow")
model,alphabet=esm.pretrained.esm_if1_gvp4_t16_142M_UR50();model.eval()
from Bio.PDB import PDBParser
AA3={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
for pdb_id in ["4HIX","5CSZ","3UOT"]:
    pdb=f"data/anti_abeta_refs/{pdb_id}.pdb"
    if not os.path.exists(pdb): continue
    s=PDBParser(QUIET=True).get_structure("s",pdb)
    cr=[(26,32),(52,56),(95,102)]
    crd=[];seq=[]
    for res in s[0]["H"]:
        ri=res.id[1]
        if any(sr<=ri<=er for sr,er in cr):
            if all(a in res for a in ["N","CA","C"]):
                crd.append([res["N"].get_coord(),res["CA"].get_coord(),res["C"].get_coord()])
                seq.append(AA3.get(res.resname.strip(),"X"))
    s_str="".join(seq)
    ll_full,ll_coord=util.score_sequence(model,alphabet,np.array(crd),s_str)
    scram="".join(random.sample(list(s_str),len(s_str)))
    ll_s_full,ll_s_coord=util.score_sequence(model,alphabet,np.array(crd),scram)
    sep=ll_coord-ll_s_coord
    print(f"{pdb_id}: native={ll_coord:.4f} scram={ll_s_coord:.4f} sep={sep:+.4f} {'PASS' if sep>0 else 'FAIL'}")
print("ESM-IF L4 native vs scrambled DONE")
