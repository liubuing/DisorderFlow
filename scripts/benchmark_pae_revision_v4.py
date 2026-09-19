"""Measure design-PDB-to-ranked-scores time, excluding teacher label access."""
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.harden_pae_surrogate_v4 import (
    OUT, read, write, digest, sources, ScalarHead, chain_residues, locate_h3,
    get_model, build_region_batch, inject_candidate_sequence, Fragment,
)


def main():
    torch.set_num_threads(2)
    manifest = read(OUT/'data_manifest.json')
    rows = [r for r in manifest['rows'] if r['split']=='transfer']
    metadata,_,_ = sources()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dep = read(ROOT/'publication/candidate_interface_pae_deployment_v1.json')['seeds'][0]
    ckpath = ROOT/dep['checkpoint'].replace('\\','/').split('/DisorderFlow/',1)[-1]
    start = time.perf_counter()
    ck = torch.load(ckpath,map_location='cpu',weights_only=False)
    model = get_model(ck['config'].model)
    model.load_state_dict(ck['model'],strict=True)
    model.to(device).eval()
    head = ScalarHead().to(device).eval()
    head.load_state_dict(torch.load(OUT/'full_s2041.pt',map_location=device,weights_only=False)['state_dict'])
    synchronize = lambda: torch.cuda.synchronize() if device=='cuda' else None
    synchronize()
    load_time = time.perf_counter()-start
    timings = []
    for scaffold in sorted({r['scaffold'] for r in rows}):
        selected = [r for r in rows if r['scaffold']==scaffold]
        meta = metadata[scaffold]
        pdb = ROOT/f'data/multiscaffold_confirmatory_v2/generation_work/structures/{scaffold}.pdb'
        def score():
            chains = sorted({l[21] for l in pdb.read_text().splitlines() if l.startswith('ATOM')})
            residues = {c:chain_residues(pdb,c) for c in chains}
            seqs = {c:''.join(r[2] for r in rs) for c,rs in residues.items()}
            hc = next(c for c,s in seqs.items() if meta['vh_sequence'] in s)
            lc = next(c for c,s in seqs.items() if meta['vl_sequence'] in s)
            ac = next(c for c,s in seqs.items() if c not in (hc,lc) and s==meta['antigen_sequence'])
            indices = locate_h3(residues[hc],meta['cdr_h3_sequence'])
            region = hc+':'+','.join(str(residues[hc][i][0]) for i in indices)
            batch = build_region_batch(str(pdb),region,context_chains=[lc,ac],antigen_chains=[ac],
                antigen_context_cap=50,preserve_context_chain_order=True,device=device)
            cm = batch['generate_flag'][0].bool() & batch['mask'][0].bool()
            am = (batch['fragment_type'][0]==int(Fragment.Antigen)) & batch['mask'][0].bool()
            with torch.no_grad():
                _,pair = model.encode(batch,remove_structure=True,remove_sequence=True)
                pooled = pair[0][cm][:,am].mean((0,1))
                features = []
                for row in selected:
                    inject_candidate_sequence(batch,row['h3_sequence'])
                    probs = torch.nn.functional.one_hot(batch['aa'].clamp(0,model.bfn.num_classes-1),model.bfn.num_classes).float()
                    seq = model.bfn.receiver.v12_seq_emb(probs)[0]
                    features.append(torch.cat([seq[cm].mean(0),seq[am].mean(0),pooled]))
                predictions = head(torch.stack(features)).cpu().numpy()
            return predictions
        score()  # one warmup for the fixed shape; excluded and documented
        times = []
        for _ in range(5):
            synchronize()
            start = time.perf_counter()
            scores = score()
            synchronize()
            times.append(time.perf_counter()-start)
        frozen = read(OUT/'predictions.json')
        expected = {eid:v for eid,v in zip(frozen['entities'],frozen['models']['full_s2041'])}
        np.testing.assert_allclose(scores,[expected[r['id']] for r in selected],atol=2e-6)
        timings.append({'scaffold':scaffold,'entities':len(selected),'runs_seconds':times,
                        'median_seconds':float(np.median(times)),
                        'seconds_per_entity_amortized':float(np.median(times)/len(selected))})
    write(OUT/'timing.json',{'device':device,'device_name':torch.cuda.get_device_name() if device=='cuda' else 'cpu',
          'torch_version':torch.__version__,'model_load_seconds':load_time,
          'scope':'PDB parsing, patch building, frozen pair embedding once per scaffold, all candidate sequences, scalar head, CPU score transfer; model loading and one warmup excluded.',
          'teacher_cost_comparability':'Historical AF2 elapsed is not rebenchmarked on identical hardware; no hardware-normalized speedup asserted.',
          'checkpoint_sha256':digest(ckpath),'head_sha256':digest(OUT/'full_s2041.pt'),
          'scaffolds':timings})
    print('Wrote measured PDB-to-score timings with prediction equivalence checks.',flush=True)


if __name__=='__main__':
    main()
