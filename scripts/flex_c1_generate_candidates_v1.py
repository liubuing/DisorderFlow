"""FLEX-BINDER Phase C1, candidate generation (outsourced to ProteinMPNN).

For each preregistered component: run ProteinMPNN vanilla v_48_020 in sampling
mode on the component PDB (H + P chains parsed), fixing every heavy-chain
residue outside the CDR-H3 region and every peptide residue, temperature 0.2,
50 designs, per-component deterministic seed from the frozen preregistration.
Outputs one candidates.json per component with the sampled H3 sequences.

Resume-able: existing candidates.json files are kept. No scoring here.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

OUT = ROOT / 'results/flex_c1_v1'
MPNN_CONFIG = {
    'script': ROOT / 'ProteinMPNN/protein_mpnn_run.py',
    'weights': ROOT / 'ProteinMPNN/vanilla_model_weights',
    'model_name': 'v_48_020',
}
TEMPERATURE = 0.2
N_DESIGNS = 50
AA = set('ACDEFGHIKLMNPQRSTVWY')


def parse_fasta(path):
    records = []
    header = None
    chunks = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('>'):
            if header is not None:
                records.append((header, ''.join(chunks)))
            header, chunks = line, []
        elif line.strip():
            chunks.append(line.strip())
    if header is not None:
        records.append((header, ''.join(chunks)))
    return records


def chain_lengths(pdb_path):
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    model = next(PDBParser(QUIET=True).get_structure('x', str(pdb_path)).get_models())
    lengths = {}
    for chain in model:
        rows = [r for r in chain.get_residues() if r.id[0] == ' ']
        lengths[chain.id] = len(rows)
    return lengths


def generate_component(component):
    instance = component['instance']
    out_dir = OUT / 'generation' / instance
    candidates_path = out_dir / 'candidates.json'
    if candidates_path.exists():
        return read(candidates_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = ROOT / component['component_pdb']
    lengths = chain_lengths(pdb_path)
    lo, hi = (int(x) for x in component['region_spec'].split(':')[1].split('-'))
    fixed_heavy = [p for p in range(1, lengths['H'] + 1) if not lo <= p <= hi]
    # jsonl key must equal the PDB name ProteinMPNN parses from the filename.
    fixed = {pdb_path.stem: {'H': fixed_heavy, 'P': list(range(1, lengths['P'] + 1))}}
    fixed_path = out_dir / 'fixed_positions.jsonl'
    fixed_path.write_text(json.dumps(fixed) + '\n', encoding='ascii')

    output = out_dir / 'mpnn'
    command = [
        sys.executable,
        MPNN_CONFIG['script'].resolve().as_posix(),
        '--pdb_path', pdb_path.resolve().as_posix(),
        '--pdb_path_chains', 'H P',
        '--fixed_positions_jsonl', fixed_path.resolve().as_posix(),
        '--path_to_model_weights', MPNN_CONFIG['weights'].resolve().as_posix(),
        '--model_name', MPNN_CONFIG['model_name'],
        '--sampling_temp', str(TEMPERATURE),
        '--num_seq_per_target', str(N_DESIGNS),
        '--batch_size', '1',
        '--seed', str(component['mpnn_seed']),
        '--suppress_print', '1',
        '--out_folder', output.resolve().as_posix(),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if completed.returncode:
        raise RuntimeError(f'MPNN failed for {instance}: {completed.stderr[-2000:]}')
    fasta_path = output / 'seqs' / f'{pdb_path.stem}.fa'
    records = parse_fasta(fasta_path)
    native_records = [seq for header, seq in records if 'sample=' not in header]
    design_records = [seq for header, seq in records if 'sample=' in header]
    if len(native_records) != 1 or len(design_records) != N_DESIGNS:
        raise RuntimeError(
            f'{instance}: expected 1 native + {N_DESIGNS} design records, '
            f'got {len(native_records)} + {len(design_records)}')
    # Position contract: the H3 window of the native record must reconstruct
    # the frozen region sequence exactly, or the window indices are wrong.
    native_h3 = native_records[0].split('/')[0][lo - 1:hi]
    if native_h3 != component['region_sequence']:
        raise RuntimeError(
            f'{instance}: native-record H3 {native_h3} != frozen region '
            f"{component['region_sequence']}; window indices unreliable")
    designs = []
    for sequence in design_records:
        design = sequence.split('/')[0][lo - 1:hi]
        invalid = sorted(set(design) - AA)
        if invalid:
            raise RuntimeError(f'{instance}: invalid residues {invalid} in design')
        designs.append(design)
    result = {
        'instance': instance,
        'mpnn_seed': component['mpnn_seed'],
        'temperature': TEMPERATURE,
        'designs': designs,
        'n_designs': len(designs),
        'fasta': str(fasta_path.relative_to(ROOT)),
    }
    freeze(candidates_path, result)
    return result


def main():
    prereg = read(OUT / 'preregistration.json')
    components = prereg['components']
    print('components:', len(components), flush=True)
    all_rows = []
    for component in sorted(components, key=lambda c: c['instance']):
        result = generate_component(component)
        all_rows.append({'instance': result['instance'],
                         'n_designs': result['n_designs'],
                         'n_distinct': len(set(result['designs'])),
                         'equals_native': sum(1 for d in result['designs']
                                              if d == component['region_sequence'])})
        print(result['instance'], 'designs', result['n_designs'],
              'distinct', len(set(result['designs'])), flush=True)
    summary = {
        'components': len(all_rows),
        'total_designs': sum(r['n_designs'] for r in all_rows),
        'distinct_designs': sum(r['n_distinct'] for r in all_rows),
        'designs_equal_native': sum(r['equals_native'] for r in all_rows),
        'rows': all_rows,
    }
    freeze(OUT / 'generation_summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
