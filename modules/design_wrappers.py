#!/usr/bin/env python
"""Factory functions that produce callables for IterativeRefiner and ClosedLoopOrchestrator.

Each factory returns a closure matching the expected callable signature,
so the existing orchestration modules can be used without modification.
"""

import os, sys, logging
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)

# Ensure sibling modules are importable
# 平铺导入兄弟模块：这些模块以"裸文件名"方式相互引用，需保证 modules/ 在 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bfn_loader import run_bfn_design
from idp_antibody_design import _extract_sequence_from_pdb, _parse_cdr_ranges, graft_cdrs
from af2_jax_runner import run_multimer_prediction


def create_bfn_design_fn(
    scaffold_pdb: str,
    scaffold_chain: str,
    mode: str = 'complex',
    device: str = 'cuda',
) -> Callable[[str, str, int], List[Dict]]:
    """Return a design_fn(pdb_path, region_spec, n_samples) -> List[Dict].

    Args:
        scaffold_pdb: path to the scaffold PDB (used as template for all rounds)
        scaffold_chain: chain ID of the scaffold (e.g. 'B')
        mode: 'complex' (antigen visible) or 'fixbb' (scaffold only)
        device: torch device
    """
    context_chains = [] if mode == 'fixbb' else None

    def design_fn(pdb_path: str, region_spec: str, n_samples: int) -> List[Dict]:
        # 闭包固定骨架链与模式，符合编排器 design_fn 签名；
        # fixbb 模式传空 context_chains（不显示抗原），complex 模式让 bfn_loader 自动识别
        logger.info(f'BFN design: {pdb_path} region={region_spec} n={n_samples} mode={mode}')
        results = run_bfn_design(
            pdb_path=pdb_path,
            region_spec=region_spec,
            num_samples=n_samples,
            stochastic=True,
            context_chains=context_chains,
            device=device,
        )
        # Each result already has: sequence, ppl, entropy, plddt, iptm, pae
        return results

    return design_fn


def create_jax_af2_validator(
    scaffold_pdb: str,
    scaffold_chain: str,
    epi_seq: str,
    cdr_spec: str,
    num_recycle: int = 3,
    data_dir: Optional[str] = None,
    scaffold_light_chain: Optional[str] = None,
) -> Callable:
    """Return an af2_validator(sequences, output_dir=None, progress_cb=None, **kwargs) -> List[Dict].

    Uses JAX-native AF2 multimer runner — no tensorflow or colabfold CLI needed.
    pLDDT is scaled to 0-100 scale for cascade_filter compatibility.
    pdb_path is set to None (JAX runner doesn't write PDB files); IterativeRefiner
    falls back gracefully to _template_pdb.

    Args:
        scaffold_pdb: path to the scaffold PDB file
        scaffold_chain: chain ID of the antibody heavy chain
        epi_seq: epitope/antigen amino acid sequence (1-letter)
        cdr_spec: CDR region string, e.g. 'B:26-33,51-58,97-113'
        num_recycle: AF2 recycling steps
        data_dir: AlphaFold params directory
        scaffold_light_chain: if provided (e.g. 'L'), include light chain as
            separate chain in AF2 multimer (VH:VL format)
    """
    if data_dir is None:
        data_dir = os.path.expanduser('~/.cache/colabfold')

    # Pre-compute scaffold sequence and CDR ranges (same for all designs)
    # 骨架序列与 CDR 区间对所有设计相同，提前算好避免每条序列重复解析 PDB
    scaffold_seq = _extract_sequence_from_pdb(scaffold_pdb, scaffold_chain)
    cdr_ranges = _parse_cdr_ranges(cdr_spec)
    light_seq = None
    if scaffold_light_chain:
        light_seq = _extract_sequence_from_pdb(scaffold_pdb, scaffold_light_chain)

    def af2_validator(sequences, output_dir=None, progress_cb=None, **kwargs):
        results = []
        n = len(sequences)
        for i, seq in enumerate(sequences):
            try:
                # 把生成的 CDR 序列嫁接回骨架序列得到完整重链；可选拼上轻链（VH:VL 格式）
                full_ab = graft_cdrs(scaffold_seq, seq, cdr_ranges)
                if light_seq:
                    full_ab = f"{full_ab}:{light_seq}"
                af2_result = run_multimer_prediction(
                    full_ab, epi_seq,
                    num_recycle=num_recycle,
                    data_dir=data_dir,
                )
                if af2_result['success']:
                    results.append({
                        'sequence': seq,
                        'success': True,
                        # JAX runner 输出 0-1 pLDDT，这里放大到 0-100 以兼容 cascade_filter
                        'plddt': af2_result['plddt'] * 100.0,  # 0-100 scale
                        'ptm': af2_result['ptm'],
                        'iptm': af2_result['iptm'],
                        'max_pae': af2_result['max_pae'],
                        'pdb_path': None,  # JAX runner doesn't write PDB
                    })
                else:
                    results.append({'sequence': seq, 'success': False})
            except Exception as e:
                logger.warning(f'AF2 validation failed for seq {i}: {e}')
                results.append({'sequence': seq, 'success': False})

            if progress_cb:
                progress_cb(i + 1, n, f'AF2 {i+1}/{n}')

        return results

    return af2_validator


def create_physics_scorer_safe(
    target_pdb: str,
    antibody_chains: str = 'B',
    antigen_chain: str = 'A',
    enable_relax: bool = False,
) -> Optional[Callable[[List[Dict]], List[Dict]]]:
    """Return a physics scorer, or None if PyRosetta is not installed.

    Args:
        target_pdb: path to target PDB for baseline dG
        antibody_chains: chain IDs for antibody
        antigen_chain: chain ID for antigen
        enable_relax: whether to run FastRelax (expensive)
    """
    try:
        from closed_loop_orchestrator import create_physics_scorer
        return create_physics_scorer(
            target_pdb_path=target_pdb,
            antibody_chains=antibody_chains,
            antigen_chain=antigen_chain,
            enable_relax=enable_relax,
        )
    except ImportError:
        logger.warning('PyRosetta not installed — physics scoring disabled')
        return None
