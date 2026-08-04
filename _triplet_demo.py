import sys, json, os
sys.path.insert(0, 'modules')
from idp_validation_triplet import score_complex

# Score native complex
native = score_complex(
    'oc_validation_results/_complexes/5IMK_2NAO_model1_A_1-42_complex.pdb',
    cdr_chain='B', epitope_chain='A',
    cdr_regions=[(26,33),(51,58),(97,113)])

print("=== Native 5IMK + Abeta42 Complex ===")
print(f"  CDR contacts: {native['contacts']}")
print(f"  Contact density: {native['density']}")
print(f"  Mean CDR-epitope distance: {native.get('mean_distance', '?')} A")
print(f"  Composite: {native['composite']}")
print()

# Load designs from AF2 run
result = json.load(open('scripts/pipelines/idp_design_results/abeta_af2/idp_design_results.json'))
designs = result['ranked_designs']
print(f"=== Designed CDRs ({len(designs)} designs) ===")
for i, d in enumerate(designs[:3]):
    seq = d['sequence'][:30] + '...'
    af2_ipTM = d.get('af2_iptm', '?')
    af2_plDDT = d.get('af2_plddt', '?')
    af2_iPAE = d.get('af2_interface_pae', '?')
    print(f"  [{i+1}] {seq}")
    print(f"      AF2 ipTM={af2_ipTM:.3f} pLDDT={af2_plDDT:.3f} iPAE={af2_iPAE:.1f}")
    print(f"      BFN pLDDT={d.get('plddt',0):.3f} ipTM={d.get('iptm',0):.3f}")
    print(f"      PPL={d.get('ppl',0):.1f}")

print()
print("=== Summary ===")
print(f"Native contacts: {native['contacts']} (density {native['density']})")
print(f"Native composite: {native['composite']}")
avg_af2_ipTM = sum(d.get('af2_iptm',0) for d in designs) / len(designs)
print(f"Avg AF2 ipTM: {avg_af2_ipTM:.3f} (interface confidence)")
print(f"Interpretation: AF2 ipTM < 0.1 means no predicted interface")
print(f"                (native 4HIX: 0.118, scrambled: 0.116)")
