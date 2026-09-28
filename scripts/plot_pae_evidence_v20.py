"""Static descriptive screening figure, fixed labels and no inferential error bars."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'results/pae_strengthening_v12/evaluation.json').read_text())['results']
out=ROOT/'results/pae_evidence_v20';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
budgets=[.1,.2,.3,.5]
names={'ridge_frozen':'Frozen Ridge','composition_only':'Composition-only Ridge','hydropathy_low':'Lower hydropathy',
       'mpnn_sampling_h3_nll':'MPNN sampling H3 NLL','mpnn_backbone_h3_nll':'MPNN backbone H3 NLL'}
for key,label in names.items():
    axes[0].plot([100*b for b in budgets],[100*data[key][str(b)]['stratum_equal_tie_averaged_recall'] for b in budgets],marker='o',label=label)
axes[0].plot([10,20,30,50],[10,20,30,50],color='gray',linestyle='--',label='Random expectation')
axes[0].set(xlabel='Retained candidate budget (%)',ylabel='Recall of teacher top 20% (%)',ylim=(0,105),xticks=[10,20,30,50],title='A. Four-stratum equal aggregate')
axes[0].legend(fontsize=7,loc='upper left')
cells=data['ridge_frozen']['0.2']['components'];keys=list(cells)
axes[1].bar(range(len(keys)),[100*cells[k]['tie_averaged_recall'] for k in keys],color='#2976a3')
axes[1].axhline(20,color='gray',linestyle='--',label='Random expectation')
axes[1].set(xticks=range(len(keys)),xticklabels=keys,ylim=(0,110),ylabel='Recall at 20% budget (%)',title='B. Frozen Ridge by structural group')
axes[1].tick_params(axis='x',labelsize=8);axes[1].legend(fontsize=8)
for ax in axes: ax.spines[['top','right']].set_visible(False)
fig.suptitle('Retrospective PAE screening: 120 candidates, six groups, four strata',fontsize=12)
fig.savefig(out/'screening_evidence.png',dpi=200)
fig.savefig(out/'screening_evidence.svg')
print(out/'screening_evidence.png')
