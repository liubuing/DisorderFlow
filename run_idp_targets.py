"""Run ProteinMPNN + AF2 on tau and alpha-synuclein antibody complexes."""
import sys, os, json, re, time, tempfile, subprocess

# ── Config ──
TARGETS = {
    'tau_5MP3': {
        'complex_pdb': 'data/non_abeta_idp/ensembles_v1/tau/5MP3_tau_core_complex.pdb',
        'antibody_chain': 'A',
        'epitope_chain': 'C',
        'cdr_spec': 'A:26-33,A:51-58,A:97-106',
        'fixed_positions_file': 'idp_design_results/tau_fixed.jsonl',
    },
    'alpha_syn_8B9V': {
        'complex_pdb': 'data/non_abeta_idp/ensembles_v1/alpha_synuclein/8B9V_alpha_synuclein_core_complex.pdb',
        'antibody_chain': 'H',
        'epitope_chain': 'A',
        'cdr_spec': 'H:26-33,H:51-58,H:97-106',
        'fixed_positions_file': 'idp_design_results/synuclein_fixed.jsonl',
    },
}

os.makedirs('idp_design_results', exist_ok=True)

for name, cfg in TARGETS.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Check PDB structure
    from Bio.PDB import PDBParser
    p = PDBParser(QUIET=True)
    try:
        s = p.get_structure('x', cfg['complex_pdb'])
        chains = [(c.id, sum(1 for r in c if r.id[0] == ' ')) for c in s[0]]
        print(f"  Chains: {chains}")
    except Exception as e:
        print(f"  PDB ERROR: {e}")
        continue

    # Build fixed positions (framework fixed, only CDR-H3 designable)
    ab_chain = cfg['antibody_chain']
    epi_chain = cfg['epitope_chain']
    chain_lengths = {c.id: sum(1 for r in c if r.id[0] == ' ') for c in s[0]}
    total = sum(chain_lengths.values())
    ab_len = chain_lengths.get(ab_chain, 0)
    epi_len = chain_lengths.get(epi_chain, 0)

    # Design CDR-H3 only: fix residues 1-96 and 107+ on antibody chain
    h_fixed = list(range(1, 97)) + list(range(107, ab_len + 1))
    epi_fixed = list(range(1, epi_len + 1))

    fixed = {ab_chain: h_fixed, epi_chain: epi_fixed}
    # Add other chains as fully fixed
    for cid, clen in chain_lengths.items():
        if cid not in fixed:
            fixed[cid] = list(range(1, clen + 1))

    with open(cfg['fixed_positions_file'], 'w') as f:
        # ProteinMPNN expects PDB basename as key
        pdb_basename = os.path.splitext(os.path.basename(cfg['complex_pdb']))[0]
        json.dump({pdb_basename: fixed}, f)
        f.write('\n')

    designable = ab_len - len(h_fixed)
    print(f"  Ab chain: {ab_len} residues, CDR-H3 designable: {designable}")
    print(f"  Epitope: {epi_len} residues, all fixed")
    print(f"  Total: {total} residues")

    # Run ProteinMPNN
    chain_str = ' '.join(chain_lengths.keys())
    print(f"  Running ProteinMPNN (10 designs)...")
    cmd = [
        sys.executable, 'ProteinMPNN/protein_mpnn_run.py',
        '--pdb_path', cfg['complex_pdb'],
        '--pdb_path_chains', chain_str,
        '--fixed_positions_jsonl', cfg['fixed_positions_file'],
        '--num_seq_per_target', '10',
        '--sampling_temp', '0.5',
        '--seed', '42',
        '--batch_size', '1',
        '--out_folder', f'idp_design_results/{name}_mpnn',
        '--save_score', '1',
        '--path_to_model_weights', 'ProteinMPNN/vanilla_model_weights',
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print(f"  MPNN output: {len(result.stdout)} chars, errors: {len(result.stderr)} bytes")
        if 'Generating sequences' in result.stdout:
            samples_line = [l for l in result.stdout.split('\n') if 'sequences of length' in l]
            if samples_line:
                print(f"  {samples_line[0].strip()}")
    except subprocess.TimeoutExpired:
        print(f"  MPNN TIMEOUT")
        continue

    # Read designs
    fa_path = f'idp_design_results/{name}_mpnn/seqs/{pdb_basename}.fa'
    if not os.path.exists(fa_path):
        print(f"  FASTA not found: {fa_path}")
        continue

    designs = []
    with open(fa_path) as f:
        for line in f:
            if line.startswith('>T=0.5'):
                parts = line.strip().split(',')
                sample = int(parts[1].split('=')[1])
                score = float(parts[2].split('=')[1])
                seq = next(f).strip()
                designs.append({'sample': sample, 'score': score, 'seq': seq})

    designs.sort(key=lambda d: d['score'])
    print(f"\n  Top 5 designs:")
    for d in designs[:5]:
        print(f"    s{d['sample']:02d}: score={d['score']:.3f} "
              f"seq={d['seq'][:40]}...")

    # Save
    report = {
        'target': name,
        'complex_pdb': cfg['complex_pdb'],
        'chains': {c: l for c, l in chain_lengths.items()},
        'designs': [{'sample': d['sample'], 'score': d['score']} for d in designs[:10]],
    }
    with open(f'idp_design_results/{name}_results.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  Saved: idp_design_results/{name}_results.json")

print("\nDone!")
