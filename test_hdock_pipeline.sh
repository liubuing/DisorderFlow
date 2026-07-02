#!/bin/bash
cd /mnt/c/biological/DisorderFlow

# Build antibody-peptide PDBs
python3 << 'PYEOF'
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'modules')
from Bio.PDB import PDBParser, PDBIO, Model, Chain, Residue, Atom
import numpy as np, random, os

AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
NATIVES={"4HIX":"GFTFSNYRSGGGYCVRYDHY","5CSZ":"GFTFSSYAISGSGGSTYYADSVKG"}
SCAFFOLD="data/misfolding_targets/3STB.pdb"
CDR_SPEC="A:26-33,A:51-58,A:97-113"
cr=[(26,33),(51,58),(97,113)]

os.makedirs("/tmp/hdock_test",exist_ok=True)

for ab_id, native_cdr in NATIVES.items():
    # Native
    parser=PDBParser(QUIET=True);s=parser.get_structure('s',SCAFFOLD)
    pos=0
    for st,en in cr:
        for j in range(st,en+1):
            if pos<len(native_cdr):
                try:s[0]['A'][j].resname={v:k for k,v in AA3.items()}.get(native_cdr[pos],'GLY')
                except:pass;pos+=1
    # Extract antibody chain only (receptor)
    rec=Model.Model(0)
    for c in s[0]:
        if c.id=='A':c.detach_parent();rec.add(c)
    st_rec=s.copy();st_rec.detach_child('s');st_rec.add(rec)
    io=PDBIO();io.set_structure(st_rec)
    io.save(f"/tmp/hdock_test/{ab_id}_native_rec.pdb")
    
    # Peptide as ligand
    pep_seq="KLVFFAED" if ab_id=="4HIX" else "DAEFRHDSGY"
    pep=Structure.Structure('p');m=Model.Model(0);c=Chain.Chain('P')
    for i,aa in enumerate(pep_seq):
        r=Residue.Residue((' ',i+1,' '),{v:k for k,v in AA3.items()}.get(aa,'GLY'),' ')
        ca=[i*3.8,0,0]
        r.add(Atom.Atom('CA',ca,0,1,' ',' CA ',i+1,'C'))
        r.add(Atom.Atom('N',[ca[0]-1.46,ca[1],ca[2]],0,1,' ',' N  ',i+1,'N'))
        r.add(Atom.Atom('C',[ca[0]+1.52,ca[1],ca[2]],0,1,' ',' C  ',i+1,'C'))
        r.add(Atom.Atom('O',[ca[0]+2.1,ca[1],ca[2]],0,1,' ',' O  ',i+1,'O'))
        c.add(r)
    m.add(c);pep.add(m)
    io2=PDBIO();io2.set_structure(pep)
    io2.save(f"/tmp/hdock_test/{ab_id}_pep.pdb")
    
    # Scrambled
    scram=list(native_cdr);random.shuffle(scram);scram=''.join(scram)
    s2=PDBParser(QUIET=True).get_structure('s2',SCAFFOLD)
    pos=0
    for st,en in cr:
        for j in range(st,en+1):
            if pos<len(scram):
                try:s2[0]['A'][j].resname={v:k for k,v in AA3.items()}.get(scram[pos],'GLY')
                except:pass;pos+=1
    rec2=Model.Model(0)
    for c in s2[0]:
        if c.id=='A':c.detach_parent();rec2.add(c)
    st_rec2=s2.copy();st_rec2.detach_child('s2');st_rec2.add(rec2)
    io3=PDBIO();io3.set_structure(st_rec2)
    io3.save(f"/tmp/hdock_test/{ab_id}_scram_rec.pdb")

print("PDBs ready")
PYEOF

# Run HDOCK on all 4 test cases
for ab in 4HIX 5CSZ; do
  for mode in native scram; do
    echo "=== ${ab}_${mode} ===" 
    ./bin/hdock /tmp/hdock_test/${ab}_${mode}_rec.pdb /tmp/hdock_test/${ab}_pep.pdb \
      -out /tmp/hdock_test/${ab}_${mode}.out 2>&1 | tail -1
    # Extract top score
    head -1 /tmp/hdock_test/${ab}_${mode}.out 2>/dev/null | awk '{print "  Score:", $2}'
  done
done
echo ""
echo "=== NATIVE vs SCRAMBLED ==="
for ab in 4HIX 5CSZ; do
  ns=$(head -1 /tmp/hdock_test/${ab}_native.out 2>/dev/null | awk '{print $2}')
  ss=$(head -1 /tmp/hdock_test/${ab}_scram.out 2>/dev/null | awk '{print $2}')
  echo "$ab: native=$ns scram=$ss"
done
