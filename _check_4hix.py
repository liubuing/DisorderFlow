from Bio.PDB import PDBParser
p = PDBParser(QUIET=True)
s = p.get_structure('x', 'idp_design_results/4hix_clean.pdb')
for chain in s[0]:
    residues = [r for r in chain if r.id[0]==' ']
    first = residues[0].id[1]
    last = residues[-1].id[1]
    print(f"Chain {chain.id}: {len(residues)} residues, {first}-{last}")
    if chain.id == 'H':
        for r in residues:
            if 90 <= r.id[1] <= 110:
                print(f"  H{r.id[1]} {r.resname}")
