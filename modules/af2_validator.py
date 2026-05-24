

#!/usr/bin/env python
"""AF2 batch validator for protein sequence design results.

Runs ColabFold on each designed sequence independently and returns
structured confidence scores (pLDDT, pTM, ipTM, PAE) for downstream
cascade filtering and iterative refinement.
"""

import os, json, subprocess, re, tempfile, time
from pathlib import Path
from typing import List, Dict, Optional, Callable


PROJECT_DIR = Path(__file__).parent


def _load_af_config():
    """Load AF2 configuration from app_config.yaml."""
    config_path = PROJECT_DIR / 'app_config.yaml'
    if config_path.exists():
        import yaml
        with open(config_path, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return cfg.get('alphafold', {})
    return {}


def validate_sequences(
    sequences: List[str],
    output_dir: Optional[str] = None,
    model_type: str = 'auto',
    num_recycle: int = 1,
    stop_at_score: int = 85,
    timeout: int = 3600,
    progress_cb: Optional[Callable] = None,
) -> List[Dict]:
    """Run ColabFold prediction on each sequence, return structured results.

    Args:
        sequences: list of amino acid sequences to validate
        output_dir: base directory for AF2 outputs (default: alphafold_results/af2_validate/)
        model_type: ColabFold model type (auto, alphafold2_ptm, alphafold2_multimer_v3)
        num_recycle: AF2 recycling steps
        stop_at_score: early stop if pLDDT exceeds this value (0-100)
        timeout: per-sequence timeout in seconds
        progress_cb: optional callback(seq_idx, total, status) for progress reporting

    Returns:
        list of dicts: [{sequence, plddt, ptm, iptm, max_pae, pdb_path, success, error?}, ...]
    """
    af_cfg = _load_af_config()
    venv_path = af_cfg.get('af2', {}).get('venv', str(PROJECT_DIR / 'venv'))
    colabfold_exe = str(Path(venv_path) / 'Scripts' / 'colabfold_batch.exe')
    if not os.path.exists(colabfold_exe):
        colabfold_exe = str(Path(venv_path) / 'Scripts' / 'colabfold_batch')

    base_dir = Path(output_dir or af_cfg.get('output_dir', 'alphafold_results'))
    base_dir.mkdir(exist_ok=True)

    results = []
    n_total = len(sequences)

    for idx, seq in enumerate(sequences):
        seq_name = f'af2_val_{idx}_{hash(seq) % 100000:05d}'
        result_dir = base_dir / seq_name

        if progress_cb:
            progress_cb(idx, n_total, f'AF2 预测 {idx+1}/{n_total}...')

        # Build FASTA
        fasta_path = base_dir / f'{seq_name}.fasta'
        safe_text = f'>design_{idx}\n{seq}'
        with open(fasta_path, 'w', encoding='ascii') as f:
            f.write(safe_text)

        # Build ColabFold command
        cmd = [
            colabfold_exe,
            str(fasta_path), str(result_dir),
            '--num-models', '1',
            '--num-recycle', str(num_recycle),
            '--stop-at-score', str(stop_at_score),
            '--model-type', model_type,
            '--rank', 'auto',
        ]

        try:
            env = os.environ.copy()
            env['PATH'] = str(Path(venv_path) / 'Scripts') + os.pathsep + env.get('PATH', '')
            r = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, cwd=str(PROJECT_DIR), env=env)

            if r.returncode != 0:
                results.append({'sequence': seq, 'success': False,
                               'error': f'exit code {r.returncode}', 'stderr': r.stderr[-500:]})
                continue

            # Find PDB
            pdb_files = sorted(result_dir.glob('*_rank_001_*.pdb'))
            if not pdb_files:
                pdb_files = sorted(result_dir.glob('*.pdb'))
            pdb_path = str(pdb_files[0]) if pdb_files else None

            # Parse scores JSON
            scores_json = sorted(result_dir.glob('*_scores_rank_001_*.json'))
            if scores_json:
                with open(scores_json[0]) as sf:
                    scores = json.load(sf)
                plddt_array = scores.get('plddt', [0])
                mean_plddt = sum(plddt_array) / max(len(plddt_array), 1)
                ptm = scores.get('ptm', 0.0)
                iptm = scores.get('iptm')
                max_pae = scores.get('max_pae', 999.0)

                results.append({
                    'sequence': seq,
                    'success': True,
                    'plddt': mean_plddt / 100.0,  # normalize to [0,1]
                    'ptm': ptm,
                    'iptm': iptm if iptm is not None else None,
                    'max_pae': max_pae,
                    'pdb_path': pdb_path,
                    'result_dir': str(result_dir),
                })
            else:
                results.append({'sequence': seq, 'success': True,
                               'plddt': 0.0, 'ptm': 0.0, 'max_pae': 999.0,
                               'pdb_path': pdb_path, 'warning': 'no scores JSON found'})

        except subprocess.TimeoutExpired:
            results.append({'sequence': seq, 'success': False, 'error': 'timeout'})
        except Exception as e:
            results.append({'sequence': seq, 'success': False, 'error': str(e)})

        # Cleanup temp FASTA
        try:
            os.unlink(fasta_path)
        except Exception:
            pass

    if progress_cb:
        progress_cb(n_total, n_total, 'AF2 验证完成')

    return results


def validate_single(sequence: str, output_name: str = 'af2_single',
                    model_type: str = 'auto', **kwargs) -> Optional[Dict]:
    """Convenience: validate a single sequence. Returns result dict or None."""
    results = validate_sequences([sequence], output_dir=f'alphafold_results/{output_name}',
                                 model_type=model_type, **kwargs)
    return results[0] if results else None


def extract_confidence_from_af2_results(af2_results: List[Dict]) -> List[Dict]:
    """Convert AF2 validation results into cascade-filter-compatible format.

    Maps AF2 pLDDT → plddt, AF2 pTM → ptm, AF2 ipTM → iptm,
    keeps the original sequence and PDB path.
    """
    out = []
    for r in af2_results:
        if r.get('success'):
            entry = {
                'sequence': r['sequence'],
                'plddt': r.get('plddt', 0.0),
                'iptm': r.get('iptm') or r.get('ptm', 0.0),
                'ptm': r.get('ptm', 0.0),
                'max_pae': r.get('max_pae', 999.0),
                'pdb_path': r.get('pdb_path'),
                '_af2_raw': r,
            }
            out.append(entry)
    return out
