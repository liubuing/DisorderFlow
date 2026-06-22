#!/usr/bin/env python
"""JAX-native AF2 Multimer validation for BFN-designed antibody sequences.

Uses the bfn-msa conda environment (Python 3.10, JAX, alphafold 2.3.1).
No tensorflow required — uses mock TF + JAX/Haiku directly.

Usage:
  conda activate bfn-msa
  python msa_validate_jax2.py --ab EVQLVES... --ag MKGLLL...

Or with FASTA:
  python msa_validate_jax2.py --fasta designs.fasta --antigen VSIG4
"""

import os, sys, json, argparse, time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / 'modules'))

from af2_jax_runner import run_multimer_prediction, AA_TO_IDX


# Common antigen sequences
ANTIGENS = {
    'VSIG4': 'MKGLLLCLFLLFTAQGRPED',
    'PSMA': ('MWNLLHETDSAVATARRPRWLCAGALVLAGGFFLLGFLFGWFIKSSNEATNITPKHNMK'
             'AFLDELKAENIKKFLYNFTQIPHLAGTEQNFQLAKQIQSQWKEFGLDSVELAHYDVLLS'
             'YPNKTHPNYISIINEDGNEIFNTSLFEPPPPGYENVSDIVPPFSAFSPQGMPEGDLVYV'
             'NYARTEDFFKLERDMKINCSGKIVIARYGKVFRGNKVKNAQLAGAKGVILYSDPADYFA'
             'PGVKSYPDGWNLPGGGVQRGNILNLNGAGDPLTPGYPANEYAYRRGIAEAVGLPSIPVH'
             'PIGYYDAQKLLEKMGGSAPPDSSWRGSLKVPYNVGPGFTGNFSTQKVKMHIHSTNEVTR'
             'IYNVIGTLRGAVEPDRYVILGGHRDSWVFGGIDPQSGAAVVHEIVRSFGTLKKEGWRPR'
             'RTILFASWDAEEFGLLGSTEWAEENSRLLQERGVAYINADSSIEGNYTLRVDCTPLMYS'
             'LVHNLTKELKSPDEGFEGKSLYESWTKKSPSPEFSGMPRISKLGSGNDFEVFFQRLGIA'
             'SGRARYTKNWETNKFSGYPLYHSVYETYELVEKFYDPMFKYHLTVAQVRGGMVFELANS'
             'IVLPFDCRDYAVVLRKYADKIYSISMKHPQEMKTYSVSFDSLFSAVKNFTEIASKFSER'
             'LQDFDKSNPIVLRMMNDQLMFLERAFIDPLGLPDRPFYRHVIYAPSSHNKYAGESFPGI'
             'YDALFDIESKVDPSKAWGEVKRQIYVAAFTVQAAAETLSEVA'),
    'STEAP1': ('MESRKDITNQEELWKMKPRRNLEEDDYLHKDTGETSMLKRPVLLHLHQTAHADEFDCP'
               'SELQHTQELFPQWHLPIKIAAIIASLTFLYTLLREVIHPLATSHQQYFYKIPILVINK'
               'VLPMVSITLLALVYLPGVIAAIVQLHNGTKAYKKFPEWL'),
}


def read_fasta(path):
    seqs = []
    with open(path) as f:
        header, seq = None, ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header and seq:
                    seqs.append((header, seq))
                header = line[1:]
                seq = ''
            elif line:
                seq += line
        if header and seq:
            seqs.append((header, seq))
    return seqs


def clean_seq(seq):
    return ''.join(aa for aa in seq.upper() if aa in AA_TO_IDX)


def main():
    p = argparse.ArgumentParser(description='JAX AF2 Multimer Validation')
    p.add_argument('--ab', help='Antibody sequence')
    p.add_argument('--ag', '--antigen-seq', dest='ag', help='Antigen sequence')
    p.add_argument('--fasta', help='FASTA with antibody sequence(s)')
    p.add_argument('--antigen', help='Antigen name: VSIG4, PSMA, STEAP1')
    p.add_argument('--recycle', type=int, default=3)
    p.add_argument('--output', '-o', default=None)
    p.add_argument('--data-dir', default=os.path.expanduser('~/.cache/colabfold'))
    p.add_argument('--quiet', '-q', action='store_true')
    args = p.parse_args()

    # Get antibody sequences
    ab_seqs = []
    if args.ab:
        ab_seqs = [('input', clean_seq(args.ab))]
    elif args.fasta:
        ab_seqs = [(h, clean_seq(s)) for h, s in read_fasta(args.fasta)]
    else:
        print("ERROR: Provide --ab or --fasta")
        return 1

    # Get antigen sequence
    ag_seq = None
    if args.ag:
        ag_seq = clean_seq(args.ag)
    elif args.antigen and args.antigen in ANTIGENS:
        ag_seq = clean_seq(ANTIGENS[args.antigen])
    elif args.antigen:
        print(f"Unknown antigen '{args.antigen}'. Known: {list(ANTIGENS)}")
        return 1
    else:
        ag_seq = clean_seq(ANTIGENS['VSIG4'])
        if not args.quiet:
            print(f'Using default antigen: VSIG4 ({len(ag_seq)} AA)')

    if not args.quiet:
        print(f'JAX AF2 Multimer Validation')
        print(f'  Antigen: {len(ag_seq)} AA')
        print(f'  Designs: {len(ab_seqs)}')
        print(f'  Recycles: {args.recycle}')
        print()

    results = []
    for i, (name, ab_seq) in enumerate(ab_seqs):
        if not args.quiet:
            print(f'[{i+1}/{len(ab_seqs)}] {name[:50]} ({len(ab_seq)} AA)')

        result = run_multimer_prediction(ab_seq, ag_seq,
                                         num_recycle=args.recycle,
                                         data_dir=args.data_dir)
        result['name'] = name
        result['ab_len'] = len(ab_seq)
        result['ag_len'] = len(ag_seq)

        if result['success']:
            if not args.quiet:
                ipae = result.get('interface_pae')
                ipae_str = f'iPAE={ipae:.1f}  ' if ipae is not None else ''
                print(f'  pLDDT={result["plddt"]:.3f}  ipTM={result["iptm"]:.3f}  '
                      f'pTM={result["ptm"]:.3f}  max_pae={result["max_pae"]:.1f}  '
                      f'{ipae_str}({result["elapsed"]:.0f}s)')
        else:
            if not args.quiet:
                print(f'  FAILED: {result.get("error", "?")[:100]}')
        results.append(result)
        if not args.quiet:
            print()

    # Summary
    ok = [r for r in results if r['success']]
    if ok:
        avg_plddt = sum(r['plddt'] for r in ok) / len(ok)
        avg_iptm = sum(r['iptm'] for r in ok) / len(ok)
        avg_ptm = sum(r['ptm'] for r in ok) / len(ok)

        if not args.quiet:
            print(f'---')
            print(f'Summary ({len(ok)}/{len(results)} succeeded):')
            print(f'  Avg pLDDT: {avg_plddt:.3f}')
            print(f'  Avg ipTM:  {avg_iptm:.3f}')
            print(f'  Avg pTM:   {avg_ptm:.3f}')

    # Save
    out_path = args.output or f'af2_validation_{int(time.time())}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'antigen_len': len(ag_seq),
            'n_designs': len(results),
            'n_success': len(ok),
            'avg_plddt': avg_plddt if ok else None,
            'avg_iptm': avg_iptm if ok else None,
            'avg_ptm': avg_ptm if ok else None,
            'results': results,
        }, f, indent=2)

    if not args.quiet:
        print(f'Saved: {out_path}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
