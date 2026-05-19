#!/usr/bin/env python
"""Iterative refinement loop for protein sequence design.

Implements: design → AF2 validate → filter Top-N → redesign from AF2 structures
Inspired by BindCraft's multi-round refinement strategy.
"""

import os, json, time, copy
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field


@dataclass
class RoundResult:
    round_idx: int
    n_input: int
    n_passed: int
    best_iptm: Optional[float]
    best_ptm: Optional[float]
    best_plddt: Optional[float]
    best_sequence: str
    best_pdb: Optional[str]
    avg_iptm: Optional[float]
    avg_ptm: Optional[float]
    elapsed_seconds: float
    all_results: List[Dict] = field(default_factory=list)
    filter_report: str = ""


@dataclass
class RefinementResult:
    rounds: List[RoundResult] = field(default_factory=list)
    best_overall_sequence: str = ""
    best_overall_iptm: float = 0.0
    best_overall_ptm: float = 0.0
    best_overall_pdb: Optional[str] = None
    total_elapsed: float = 0.0
    converged: bool = False
    convergence_reason: str = ""


class IterativeRefiner:
    """Orchestrate design → validate → filter → redesign cycles.

    Usage:
        refiner = IterativeRefiner(
            design_fn=my_design_fn,
            af2_validator=validate_sequences,
            cascade_fn=apply_cascade_af2,
            n_samples=10, top_k=3, max_rounds=3
        )
        result = refiner.run(initial_pdb='input.pdb', design_region='A:10-30')
    """

    def __init__(
        self,
        design_fn: Callable,
        af2_validator: Callable,
        cascade_fn: Callable,
        n_samples: int = 10,
        top_k: int = 3,
        max_rounds: int = 3,
        convergence_delta: float = 0.02,
        progress_cb: Optional[Callable] = None,
        log_cb: Optional[Callable] = None,
    ):
        """
        Args:
            design_fn: function(pdb_path, region, n_samples) -> List[{sequence, ppl, ...}]
            af2_validator: function(sequences, progress_cb) -> List[{sequence, plddt, ptm, iptm, ...}]
            cascade_fn: function(results, thresholds, weights) -> (filtered, report)
            n_samples: sequences to generate per round
            top_k: number of top sequences to carry to next round
            max_rounds: maximum refinement rounds
            convergence_delta: stop if ipTM improves less than this
            progress_cb: optional callback(round_idx, max_rounds, status)
            log_cb: optional callback(message) for detailed logging
        """
        self.design_fn = design_fn
        self.af2_validator = af2_validator
        self.cascade_fn = cascade_fn
        self.n_samples = n_samples
        self.top_k = top_k
        self.max_rounds = max_rounds
        self.convergence_delta = convergence_delta
        self.progress_cb = progress_cb
        self.log_cb = log_cb

    def _log(self, msg: str):
        if self.log_cb:
            self.log_cb(msg)

    def _progress(self, round_idx: int, status: str):
        if self.progress_cb:
            self.progress_cb(round_idx, self.max_rounds, status)

    def run(self, initial_pdb: str, design_region: str = 'A:10-30',
            native_seq: Optional[str] = None) -> RefinementResult:
        """Execute the iterative refinement loop.

        Args:
            initial_pdb: starting PDB structure
            design_region: region spec (e.g., 'A:10-30')
            native_seq: native sequence for recovery calculation (optional)

        Returns:
            RefinementResult with per-round details and best overall sequence
        """
        result = RefinementResult()
        current_pdbs = [initial_pdb]  # Template PDBs for next round
        prev_best_iptm = -1.0
        t_start = time.time()

        for round_idx in range(1, self.max_rounds + 1):
            self._progress(round_idx, f'第 {round_idx} 轮: 设计 {self.n_samples} 条序列...')
            self._log(f"\n{'='*60}")
            self._log(f"  第 {round_idx} 轮迭代精修")
            self._log(f"{'='*60}")
            self._log(f"  模板数: {len(current_pdbs)}")

            t_round_start = time.time()

            # Step 1: Design sequences on each template PDB
            all_designs = []
            samples_per_template = max(1, self.n_samples // len(current_pdbs))

            for pdb_path in current_pdbs:
                if not os.path.exists(str(pdb_path)):
                    self._log(f"  ⚠ 跳过缺失模板: {pdb_path}")
                    continue
                try:
                    designs = self.design_fn(pdb_path, design_region, samples_per_template)
                    for d in designs:
                        d['_template_pdb'] = pdb_path
                    all_designs.extend(designs)
                except Exception as e:
                    self._log(f"  ⚠ 设计失败 ({pdb_path}): {e}")

            if not all_designs:
                self._log(f"  ❌ 无可设计序列 — 终止")
                break

            self._log(f"  生成 {len(all_designs)} 条序列")

            # Step 2: AF2 validation
            self._progress(round_idx, f'第 {round_idx} 轮: AF2 验证 {len(all_designs)} 条...')
            sequences = [d['sequence'] for d in all_designs]
            af2_results = self.af2_validator(
                sequences,
                output_dir=f'alphafold_results/refine_round{round_idx}',
                progress_cb=lambda i, n, s: self._progress(round_idx, f'AF2: {i+1}/{n}'),
            )

            # Merge AF2 scores with design data
            af2_map = {r['sequence']: r for r in af2_results if r.get('success')}
            merged = []
            for d in all_designs:
                af2 = af2_map.get(d['sequence'], {})
                merged.append({
                    **d,
                    'plddt': af2.get('plddt', 0) * 100,  # back to 0-100 for cascade_af2
                    'ptm': af2.get('ptm', 0),
                    'iptm': af2.get('iptm'),
                    'max_pae': af2.get('max_pae', 999),
                    'pdb_path': af2.get('pdb_path'),
                    '_af2_raw': af2,
                })

            n_success = len([m for m in merged if m['ptm'] > 0])

            # Step 3: Cascade filter
            self._progress(round_idx, f'第 {round_idx} 轮: 级联筛选...')
            filtered, filter_report = self.cascade_fn(merged)
            self._log(filter_report)

            # Extract round stats
            best_iptm = None
            best_ptm = None
            best_plddt = None
            best_seq = ""
            best_pdb = None
            avg_iptm = None
            avg_ptm = None

            if filtered:
                best = filtered[0]
                best_iptm = best.get('iptm') or best.get('ptm', 0)
                best_ptm = best.get('ptm', 0)
                best_plddt = best.get('plddt', 0)
                best_seq = best['sequence']
                best_pdb = best.get('pdb_path')
                avg_iptm = sum((r.get('iptm') or r.get('ptm', 0)) for r in filtered) / len(filtered)
                avg_ptm = sum((r.get('ptm', 0) or 0) for r in filtered) / len(filtered)

                # Update overall best
                if best_iptm and best_iptm > result.best_overall_iptm:
                    result.best_overall_iptm = best_iptm
                    result.best_overall_ptm = best_ptm or 0
                    result.best_overall_sequence = best_seq
                    result.best_overall_pdb = best_pdb

                # Prepare templates for next round: use AF2-predicted structures from top_k
                current_pdbs = []
                for r in filtered[:self.top_k]:
                    pdb = r.get('pdb_path')
                    if pdb and os.path.exists(str(pdb)):
                        current_pdbs.append(pdb)
                    elif r.get('_template_pdb') and os.path.exists(str(r['_template_pdb'])):
                        current_pdbs.append(r['_template_pdb'])

                if not current_pdbs:
                    current_pdbs = [initial_pdb]
            else:
                # All filtered out — keep original template
                current_pdbs = [initial_pdb]
                best_seq = ""

            t_round = time.time() - t_round_start

            round_result = RoundResult(
                round_idx=round_idx,
                n_input=len(all_designs),
                n_passed=len(filtered),
                best_iptm=best_iptm,
                best_ptm=best_ptm,
                best_plddt=best_plddt,
                best_sequence=best_seq,
                best_pdb=best_pdb,
                avg_iptm=avg_iptm,
                avg_ptm=avg_ptm,
                elapsed_seconds=t_round,
                all_results=filtered,
                filter_report=filter_report,
            )
            result.rounds.append(round_result)

            self._log(f"  第 {round_idx} 轮完成 ({t_round:.0f}s)")
            self._log(f"    AF2成功/总数: {n_success}/{len(all_designs)}")
            self._log(f"    通过筛选: {len(filtered)}")
            if best_iptm is not None:
                self._log(f"    最优 ipTM: {best_iptm:.3f}  pTM: {best_ptm:.3f}")

            # Convergence check
            if best_iptm is not None and prev_best_iptm >= 0:
                delta = best_iptm - prev_best_iptm
                if delta < self.convergence_delta and round_idx >= 2:
                    result.converged = True
                    result.convergence_reason = f"ipTM 提升 < {self.convergence_delta} (Δ={delta:.4f})"
                    self._log(f"  ✅ 收敛: {result.convergence_reason}")
                    break

            if best_iptm is not None:
                prev_best_iptm = best_iptm

        result.total_elapsed = time.time() - t_start
        self._log(f"\n{'='*60}")
        self._log(f"  迭代精修完成 ({result.total_elapsed:.0f}s, {len(result.rounds)} 轮)")
        self._log(f"  全局最优: {result.best_overall_sequence}")
        self._log(f"  最优 ipTM: {result.best_overall_iptm:.3f}")
        self._log(f"{'='*60}")

        return result


def format_refinement_report(result: RefinementResult) -> str:
    """Format a human-readable refinement report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  🔄 迭代精修报告")
    lines.append("=" * 70)

    for r in result.rounds:
        lines.append(f"\n  第 {r.round_idx} 轮:")
        lines.append(f"    生成: {r.n_input} 条 | 通过筛选: {r.n_passed} 条 | 耗时: {r.elapsed_seconds:.0f}s")
        if r.best_iptm is not None:
            lines.append(f"    最优 ipTM: {r.best_iptm:.3f}  |  pTM: {r.best_ptm:.3f}  |  pLDDT: {r.best_plddt:.0f}")
            lines.append(f"    最优序列: {r.best_sequence[:60]}{'...' if len(r.best_sequence)>60 else ''}")

    lines.append(f"\n  {'─'*60}")
    lines.append(f"  总耗时: {result.total_elapsed:.0f}s  |  收敛: {'是' if result.converged else '否'}")
    if result.converged:
        lines.append(f"  收敛原因: {result.convergence_reason}")
    lines.append(f"  全局最优 ipTM: {result.best_overall_iptm:.3f}")
    lines.append(f"  全局最优序列: {result.best_overall_sequence[:60]}")
    lines.append("=" * 70)

    return '\n'.join(lines)
