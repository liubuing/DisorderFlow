"""Design source-aware measurement repairs from sequences, never test affinities.

This writes a proposed assay panel; it does NOT execute wet-lab measurements.
"""
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.diagnose_aayl_failure_v3 import OUT,base,freeze


def build(d,rows,changes):
    panel=[]; summary=[]
    for family,s in d['structures'].items():
        parent=s['sequences'][s['heavy_chain']]
        light=s['sequences'][s['light_chain']]
        train=[i for i,r in enumerate(rows) if r['family']==family and len(changes[i])<=2]
        triples=[i for i,r in enumerate(rows) if r['family']==family and len(changes[i])==3]
        # Exact single and double constructs are needed to estimate clean effects.
        available={r['heavy_sequence']:r for i,r in enumerate(rows) if i in train}
        excluded={r['heavy_sequence']:r for r in d['excluded'] if r['family']==family}
        needs=defaultdict(set)
        kinds={}
        for i in triples:
            for size in [0,1,2]:
                for subset in combinations(sorted(changes[i]),size):
                    seq=list(parent)
                    for pos,aa in subset: seq[pos]=aa
                    seq=''.join(seq)
                    if seq not in available:
                        needs[seq].add(rows[i]['poi'])
                        kinds[seq]=size
        family_rows=[]
        for seq,affected in needs.items():
            source=excluded.get(seq)
            observations=source['observations'] if source else []
            finite=sum(o['original_log10_nM'] is not None for o in observations)
            family_rows.append({'family':family,'role':{0:'parent_anchor',1:'single_effect_anchor',2:'pair_effect_anchor'}[kinds[seq]],
                'heavy_sequence':seq,'light_sequence':light,
                'h3_sequence':''.join(seq[i] for i in s['h3_indices_zero_based_observed_heavy']),
                'mutations':[{'observed_heavy_index_1based':pos+1,'parent':parent[pos],'mutant':aa} for pos,aa in sorted(base.mutations(seq,parent))],
                'affected_triples':sorted(affected),'affected_count':len(affected),
                'prior_source_poi':source['poi'] if source else None,'available_finite_readings':finite,
                'status':'incomplete_source_readings' if source else ('parent_control_not_audited' if kinds[seq]==0 else 'absent_from_H3_candidate_cohort'),
                'action':'source_control_audit_then_matched_assay' if kinds[seq]==0 else 'remeasure_in_matched_assay' if source else 'broader_source_lookup_then_matched_assay'})
        family_rows.sort(key=lambda r:(len(r['mutations']),-r['affected_count'],r['heavy_sequence']))
        for rank,r in enumerate(family_rows,1): r['within_family_priority']=rank
        panel.extend(family_rows)
        summary.append({'family':family,'proposed_constructs':len(family_rows),
            'parent':sum(r['role']=='parent_anchor' for r in family_rows),
            'singles':sum(r['role']=='single_effect_anchor' for r in family_rows),
            'doubles':sum(r['role']=='pair_effect_anchor' for r in family_rows),
            'incomplete_existing':sum(r['status']=='incomplete_source_readings' for r in family_rows),
            'caveat':'Sequence-only prioritization; all missing pair/single controls needed for complete factorial decomposition, not guaranteed useful binding variants'})
    return {'status':'proposed_measurements_not_performed','cohort_sha256':base.digest(base.COHORT),
        'rules':'Parent, exact singles and exact doubles for each current triple; skip complete low-order constructs; sort parent then singles then doubles by number of affected triples; no binding-label prioritization',
        'requirements':'Match assay, target, partner light chain and batch controls; record independent replicates and censoring limits; do not invent values; old exposed triples remain development cases',
        'summary':summary,'records':panel}


if __name__=='__main__':
    d,rows,changes=base.load()
    result=build(d,rows,changes)
    freeze(OUT/'measurement_repair_panel.json',result)
    print(json.dumps(result['summary'],indent=2))
