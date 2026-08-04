from Bio.PDB import PDBParser, PDBIO, Select

class CleanSelect(Select):
    def accept_residue(self, residue):
        return residue.id[0] == ' '
    def accept_chain(self, chain):
        return chain.id in ('H', 'L', 'A')

parser = PDBParser(QUIET=True)
s = parser.get_structure('x', 'data/anti_abeta_refs/4HIX.pdb')
io = PDBIO()
io.set_structure(s)
io.save('idp_design_results/4hix_clean.pdb', CleanSelect())
print("Cleaned")
for chain in s[0]:
    n = sum(1 for r in chain if r.id[0] == ' ')
    print(f"  Chain {chain.id}: {n} residues")
