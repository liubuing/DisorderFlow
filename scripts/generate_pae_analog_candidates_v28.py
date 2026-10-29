"""Generate analog prescreening candidates and AF2 teacher job lists.

For every v27 split component: writes a contiguously renumbered design PDB
from the frozen v26 component PDB (chain order preserved; numbering 1..N per
chain), runs ProteinMPNN on the receptor chain with the candidate region
flexible and everything else fixed, extracts candidate-region sequences and
generator NLLs, builds native and composition-shuffle controls for eval
components, and emits the six-replicate AF2 job lists (models 1/2, seeds
7103/7111/7121) with ':'-separated receptor chains in the antibody convention
(candidate chain first, partner second).

No AF2 prediction happens here; jobs are written, not run.
"""
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.run_multiscaffold_v2_af2 import shuffled_h3

OUT = ROOT / 'results/pae_analog_generation_v28'
V26 = ROOT / 'results/pae_analog_structures_v26'
V27 = ROOT / 'results/pae_analog_family_split_v27'
SEEDS = [7103, 7111, 7121]
GENERATION_SEED = 20260919
SHUFFLE_SEED = 7307
TEMPERATURE = 0.3
N_EVAL_CANDIDATES = 14
N_TRAIN_CANDIDATES = 7
MPNN_SCRIPT = ROOT / 'ProteinMPNN/protein_mpnn_run.py'
MPNN_WEIGHTS = ROOT / 'ProteinMPNN/soluble_model_weights'

COHORT = {
    'tcr': {'design_chain': 'B', 'partner_chain': 'A', 'region_len': None},
    'vhh': {'design_chain': 'V', 'partner_chain': None, 'region_len': None},
}


def renumber_pdb(source, target, chains):
    """Contiguously renumber each chain 1..N; residue order preserved.
    Returns {chain: length}."""
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Structure import Structure as _Structure
    from Bio.PDB.Model import Model as _Model
    from Bio.PDB.Chain import Chain as _Chain
    import copy as _copy
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('x', str(source))
    model = next(structure.get_models())
    out = _Structure('x')
    out_model = _Model(0)
    out.add(out_model)
    lengths = {}
    for chain_id in chains:
        source_chain = model[chain_id]
        chain_out = _Chain(chain_id)
        out_model.add(chain_out)
        number = 0
        for residue in source_chain.get_residues():
            number += 1
            residue = _copy.deepcopy(residue)
            residue.id = (' ', number, ' ')
            chain_out.add(residue)
        lengths[chain_id] = number
    io = PDBIO()
    io.set_structure(out)
    io.save(str(target))
    return lengths


def run_mpnn(pdb_path, design_chain, region_indices, n_candidates, work_dir):
    """ProteinMPNN on the design chain; returns parsed candidate rows."""
    name = pdb_path.stem
    length = sum(1 for _ in _chain_residues(pdb_path, design_chain))
    fixed = [position for position in range(1, length + 1)
             if position - 1 not in set(region_indices)]
    fixed_path = work_dir / 'fixed_positions.jsonl'
    fixed_path.write_text(json.dumps({name: {design_chain: fixed}}) + '\n', encoding='ascii')
    output = work_dir / 'mpnn'
    fasta = output / 'seqs' / f'{name}.fa'
    if not fasta.exists():
        command = [
            sys.executable, MPNN_SCRIPT.as_posix(),
            '--pdb_path', pdb_path.resolve().as_posix(),
            '--pdb_path_chains', design_chain,
            '--fixed_positions_jsonl', fixed_path.resolve().as_posix(),
            '--path_to_model_weights', MPNN_WEIGHTS.as_posix(),
            '--model_name', 'v_48_020',
            '--num_seq_per_target', str(n_candidates),
            '--batch_size', '1',
            '--sampling_temp', str(TEMPERATURE),
            '--seed', str(GENERATION_SEED),
            '--save_score', '1',
            '--suppress_print', '1',
            '--out_folder', output.resolve().as_posix(),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
        if completed.returncode:
            raise RuntimeError(f'ProteinMPNN failed for {name}: {completed.stderr[-2000:]}')
    rows = []
    lines = fasta.read_text(encoding='utf-8').splitlines()
    for index, line in enumerate(lines):
        if not line.startswith('>T='):
            continue
        full = lines[index + 1].strip().split('/')[0]
        if len(full) != length:
            raise ValueError(f'MPNN chain length {len(full)} != {length} for {name}')
        fields = {}
        for field in line[1:].split(','):
            if '=' in field:
                key, value = field.split('=', 1)
                fields[key.strip()] = value.strip()
        rows.append({'region_sequence': ''.join(full[position] for position in region_indices),
                     'generator_score': float(fields.get('score', 'nan')),
                     'full_design_chain_sequence': full})
    if len(rows) != n_candidates:
        raise ValueError(f'{name}: expected {n_candidates} rows, got {len(rows)}')
    return rows


def _chain_residues(pdb_path, chain_id):
    from Bio.PDB import PDBParser
    structure = PDBParser(QUIET=True).get_structure('x', str(pdb_path))
    model = next(structure.get_models())
    return [r for r in model[chain_id].get_residues() if r.id[0] == ' ' and 'CA' in r]


def inject(sequence, region_indices, candidate):
    """Positional injection; region indices may be non-contiguous (insertion codes)."""
    if len(region_indices) != len(candidate):
        raise ValueError('candidate length != region index count')
    chars = list(sequence)
    for offset, index in enumerate(region_indices):
        chars[index] = candidate[offset]
    return ''.join(chars)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = read(V27 / 'split.json')
    manifest = {r['instance']: r for r in read(V26 / 'manifest.json')['records'] if r['passes']}

    protocol = {
        'classification': 'candidate_generation_and_job_preparation_no_labels',
        'v26_manifest_sha256': digest(V26 / 'manifest.json'),
        'v27_split_sha256': digest(V27 / 'split.json'),
        'generation': {'tool': 'ProteinMPNN v_48_020 soluble', 'temperature': TEMPERATURE,
                       'generation_seed': GENERATION_SEED,
                       'eval_candidates_per_component': N_EVAL_CANDIDATES,
                       'train_candidates_per_component': N_TRAIN_CANDIDATES,
                       'fixed_backbone': 'design chain only; candidate region flexible; contiguous 1..N renumbered PDB derived from frozen v26 PDB'},
        'controls': {'eval_components_only': ['native', 'composition_shuffle'],
                     'shuffle_seed': SHUFFLE_SEED, 'require_non_native_shuffle': True},
        'teacher_jobs': {'models': [1, 2], 'seeds': SEEDS, 'recycles': 3,
                         'ab_seq': 'candidate chain first, partner second (":-separated"); epi_seq = peptide',
                         'estimand': 'mean normalized directional PAE candidate-region rows to all peptide columns / 31'},
        'boundary': 'no AF2 prediction in this script; jobs only',
        'script_sha256': digest(Path(__file__)),
    }
    freeze(OUT / 'protocol.json', protocol)

    design_dir = OUT / 'design_pdbs'
    jobs_dir = OUT
    design_dir.mkdir(parents=True, exist_ok=True)
    jobs = {1: [], 2: []}
    generation = []
    for side, n_candidates, with_controls in (('eval', N_EVAL_CANDIDATES, True),
                                              ('train', N_TRAIN_CANDIDATES, False)):
        for instance in split[f'{side}_components']:
            record = manifest[instance]
            cohort = record['cohort']
            cfg = COHORT[cohort]
            work_dir = OUT / 'work' / instance
            work_dir.mkdir(parents=True, exist_ok=True)
            source = ROOT / record['pdb_path']
            design_pdb = design_dir / f'{instance}.pdb'
            chains = [cfg['design_chain'], cfg['partner_chain']] if cfg['partner_chain'] else [cfg['design_chain']]
            chains += ['P']
            lengths = renumber_pdb(source, design_pdb, chains)
            region_indices = record['candidate_region']['indices_zero_based']
            receptor_seq = record['sequences']['receptor']
            partner_seq = record['sequences'].get('partner')
            peptide = record['sequences']['peptide']
            native_region = record['candidate_region']['sequence']

            rows = run_mpnn(design_pdb, cfg['design_chain'], region_indices, n_candidates, work_dir)
            entities = []
            for i, row in enumerate(rows):
                if len(row['region_sequence']) != len(native_region):
                    raise ValueError(f'{instance}: candidate {i} region length mismatch')
                design_seq = inject(receptor_seq, region_indices, row['region_sequence'])
                entities.append({'entity_id': f'{instance}|candidate|{i:02d}',
                                 'arm': 'candidate', 'slot': i,
                                 'region_sequence': row['region_sequence'],
                                 'design_chain_sequence': design_seq,
                                 'generator_score': row['generator_score']})
            if with_controls:
                shuffle = shuffled_h3(native_region, SHUFFLE_SEED)
                entities.append({'entity_id': f'{instance}|control|native', 'arm': 'native', 'slot': 'native',
                                 'region_sequence': native_region,
                                 'design_chain_sequence': inject(receptor_seq, region_indices, native_region),
                                 'generator_score': None})
                entities.append({'entity_id': f'{instance}|control|composition_shuffle',
                                 'arm': 'composition_shuffle', 'slot': 'shuffle',
                                 'region_sequence': shuffle,
                                 'design_chain_sequence': inject(receptor_seq, region_indices, shuffle),
                                 'generator_score': None})
            for entity in entities:
                ab_seq = (entity['design_chain_sequence'] + ':' + partner_seq) if partner_seq \
                    else entity['design_chain_sequence']
                for model in (1, 2):
                    for seed in SEEDS:
                        job_id = f"{entity['entity_id']}|model{model}|seed{seed}"
                        pae_path = (OUT / 'pae' / f'{job_id}.npz').as_posix()
                        jobs[model].append({'id': job_id, 'seq': ab_seq, 'epi_seq': peptide,
                                            'recycle': 3, 'seed': seed, 'output_pae': pae_path})
            generation.append({'instance': instance, 'cohort': cohort, 'side': side,
                               'design_pdb': design_pdb.relative_to(ROOT).as_posix(),
                               'design_pdb_sha256': digest(design_pdb),
                               'source_pdb_sha256': record['pdb_sha256'],
                               'chain_lengths': lengths,
                               'native_region': native_region,
                               'peptide': peptide,
                               'partner_sequence': partner_seq,
                               'entities': entities})
            print(f'{instance}: {len(entities)} entities', flush=True)

    freeze(OUT / 'generation.json', {'components': generation})
    for model in (1, 2):
        with (jobs_dir / f'jobs_model{model}.jsonl').open('w', encoding='ascii') as handle:
            for job in jobs[model]:
                handle.write(json.dumps(job) + '\n')
    summary = {'components': len(generation),
               'eval_components': len(split['eval_components']),
               'train_components': len(split['train_components']),
               'entities': sum(len(c['entities']) for c in generation),
               'job_slots': {model: len(jobs[model]) for model in jobs}}
    freeze(OUT / 'summary.json', summary)
    freeze(OUT / 'completion.json', {'status': 'completed', 'af2_started': False,
                                     'generation_sha256': digest(OUT / 'generation.json')})
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
