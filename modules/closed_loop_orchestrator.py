#!/usr/bin/env python
"""Closed-loop orchestrator for multi-objective antibody design alignment.

Implements the full cycle:
  Design → AF2 Validate → Physics Score → Multi-Objective Rank → Select → Redesign

Extends IterativeRefiner by adding:
  - PyRosetta physics scoring (dG, ddG, relax_e)
  - Multi-objective Pareto ranking
  - Hypervolume-based convergence tracking
  - Optional model fine-tuning between cycles
"""

import os, json, time, copy, math
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field

import closed_loop_scorer as cls


@dataclass
class CycleResult:
    """Per-cycle statistics."""
    cycle_idx: int
    n_generated: int
    n_af2_passed: int
    n_physics_passed: int
    n_final: int
    # Best metrics
    best_composite: float
    best_iptm: float
    best_plddt: float
    best_dG: Optional[float]
    best_sequence: str
    best_pdb: str
    # Pareto stats
    pareto_front_size: int
    hypervolume: float
    avg_composite_top5: float
    # Convergence
    delta_iptm: float = 0.0
    delta_hypervolume: float = 0.0
    # All per-design data from this cycle
    all_ranked: List[Dict] = field(default_factory=list)
    report: str = ""
    elapsed_seconds: float = 0.0


@dataclass
class ClosedLoopResult:
    """Complete closed-loop run result."""
    cycles: List[CycleResult] = field(default_factory=list)
    best_overall_sequence: str = ""
    best_overall_composite: float = 0.0
    best_overall_iptm: float = 0.0
    best_overall_plddt: float = 0.0
    best_overall_dG: Optional[float] = None
    best_overall_pdb: str = ""
    total_elapsed: float = 0.0
    converged: bool = False
    convergence_reason: str = ""
    finetune_checkpoints: List[str] = field(default_factory=list)


def create_physics_scorer(target_pdb_path: str,
                          antibody_chains: str = 'HL',
                          antigen_chain: str = 'A',
                          enable_relax: bool = False,
                          scorefxn: str = 'ref2015') -> Callable:
    """Factory: create a physics scorer callable for ClosedLoopOrchestrator.

    Returns a function that takes a list of AF2 result dicts and returns
    them enriched with PyRosetta physics scores (dG, ddG, relax_e).

    Gracefully degrades if PyRosetta is unavailable.
    """
    _has_pyrosetta = False
    try:
        import pyrosetta
        pyrosetta.init(' '.join([
            '-mute', 'all', '-use_input_sc',
            '-ignore_unrecognized_res',
            '-ignore_zero_occupancy', 'false',
        ]))
        _has_pyrosetta = True
    except ImportError:
        pass

    def physics_scorer(af2_results: List[Dict]) -> List[Dict]:
        """Enrich AF2 results with Rosetta interface energy scores.

        Args:
            af2_results: list of dicts with at least 'pdb_path', 'success'

        Returns:
            Same list with added 'dG', 'ddG', 'relax_e' keys
        """
        if not _has_pyrosetta:
            for r in af2_results:
                r['dG'] = None
                r['ddG'] = None
                r['relax_e'] = None
                r['_no_rosetta'] = True
            return af2_results

        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

        # Compute reference dG if target PDB available
        dG_ref = None
        if target_pdb_path and os.path.exists(str(target_pdb_path)):
            try:
                ref_pose = pyrosetta.pose_from_pdb(str(target_pdb_path))
                analyzer = InterfaceAnalyzerMover(f'{antibody_chains}_{antigen_chain}')
                analyzer.set_pack_separated(True)
                analyzer.apply(ref_pose)
                dG_ref = ref_pose.scores.get('dG_separated', None)
            except Exception:
                dG_ref = None

        for r in af2_results:
            pdb_path = r.get('pdb_path')
            if not pdb_path or not os.path.exists(str(pdb_path)) or not r.get('success'):
                r['dG'] = None
                r['ddG'] = None
                r['relax_e'] = None
                continue

            try:
                pose = pyrosetta.pose_from_pdb(str(pdb_path))
                interface = f'{antibody_chains}_{antigen_chain}'
                analyzer = InterfaceAnalyzerMover(interface)
                analyzer.set_pack_separated(True)
                analyzer.apply(pose)
                dG_gen = pose.scores.get('dG_separated', None)
                r['dG'] = round(dG_gen, 2) if dG_gen is not None else None
                r['ddG'] = (round(dG_gen - dG_ref, 2)
                            if (dG_gen is not None and dG_ref is not None) else None)
            except Exception:
                r['dG'] = None
                r['ddG'] = None

            # Relaxation (optional)
            r['relax_e'] = None
            if enable_relax:
                try:
                    from pyrosetta.rosetta.protocols.relax import FastRelax
                    from pyrosetta import create_score_function
                    sf = create_score_function(scorefxn)
                    fr = FastRelax()
                    fr.set_scorefxn(sf)
                    fr.max_iter(200)
                    relax_pose = pyrosetta.pose_from_pdb(str(pdb_path))
                    fr.apply(relax_pose)
                    r['relax_e'] = round(relax_pose.energies().total_energy(), 2)
                except Exception:
                    r['relax_e'] = None

        return af2_results

    return physics_scorer


class ClosedLoopOrchestrator:
    """Orchestrate design → validate → score → rank → select → (finetune) → redesign.

    Usage:
        orchestrator = ClosedLoopOrchestrator(
            design_fn=my_design_fn,
            af2_validator=my_validator,
            physics_scorer=create_physics_scorer(target_pdb),
            n_samples=20, top_k=5, max_cycles=3,
        )
        result = orchestrator.run(initial_pdb='input.pdb', design_region='A:25-35')
    """

    def __init__(
        self,
        design_fn: Callable,
        af2_validator: Callable,
        physics_scorer: Optional[Callable] = None,
        mo_ranker: Optional[cls.MultiObjectiveRanker] = None,
        finetune_fn: Optional[Callable] = None,
        n_samples: int = 20,
        top_k: int = 5,
        max_cycles: int = 5,
        convergence_delta: float = 0.02,
        convergence_patience: int = 2,
        finetune_every_n_cycles: int = 2,
        selection_strategy: str = 'pareto_first',
        progress_cb: Optional[Callable] = None,
        log_cb: Optional[Callable] = None,
        yield_cb: Optional[Callable] = None,
    ):
        """
        Args:
            design_fn: function(pdb_path, region, n_samples) -> List[{sequence, ppl, ...}]
            af2_validator: function(sequences, output_dir, progress_cb) -> List[{...}]
            physics_scorer: function(List[Dict]) -> List[Dict] (from create_physics_scorer)
            mo_ranker: MultiObjectiveRanker instance (uses default if None)
            finetune_fn: function(top_designs, cycle) -> new_checkpoint_path (optional)
            n_samples: sequences generated per cycle
            top_k: designs carried forward to next cycle
            max_cycles: maximum closed-loop cycles
            convergence_delta: stop if hypervolume improvement < this
            convergence_patience: stop after N cycles without significant improvement
            finetune_every_n_cycles: fine-tune frequency
            selection_strategy: 'pareto_first' | 'composite' | 'hybrid'
            progress_cb: callback(cycle, max_cycles, status)
            log_cb: callback(message)
            yield_cb: callback(ClosedLoopResult) for Gradio streaming
        """
        self.design_fn = design_fn
        self.af2_validator = af2_validator
        self.physics_scorer = physics_scorer
        self.mo_ranker = mo_ranker or cls.MultiObjectiveRanker()
        self.finetune_fn = finetune_fn
        self.n_samples = n_samples
        self.top_k = top_k
        self.max_cycles = max_cycles
        self.convergence_delta = convergence_delta
        self.convergence_patience = convergence_patience
        self.finetune_every_n_cycles = finetune_every_n_cycles
        self.selection_strategy = selection_strategy
        self.progress_cb = progress_cb
        self.log_cb = log_cb
        self.yield_cb = yield_cb

    def _log(self, msg: str):
        if self.log_cb:
            self.log_cb(msg)

    def _progress(self, cycle: int, status: str):
        if self.progress_cb:
            self.progress_cb(cycle, self.max_cycles, status)

    def _select_top_k(self, ranked: List[Dict], k: int) -> List[Dict]:
        """Select top-k designs from ranked results.

        Strategies:
          - 'pareto_first': fill from Pareto front 0 first, then front 1, etc.
          - 'composite': purely by mo_composite_score
          - 'hybrid': 50% from Pareto, 50% by composite
        """
        if self.selection_strategy == 'composite':
            by_comp = sorted(ranked,
                             key=lambda x: x.get('mo_composite_score', 0),
                             reverse=True)
            return by_comp[:k]

        if self.selection_strategy == 'pareto_first':
            # Ensure Pareto fronts computed
            if not any('pareto_front' in r for r in ranked):
                cls.MultiObjectiveRanker.compute_pareto_fronts(ranked)
            selected = []
            front = 0
            while len(selected) < k:
                front_members = [r for r in ranked if r.get('pareto_front') == front]
                if not front_members:
                    break
                # Within same front, sort by crowding distance
                front_members.sort(
                    key=lambda x: x.get('crowding_distance', 0) or 0, reverse=True)
                needed = k - len(selected)
                selected.extend(front_members[:needed])
                front += 1
            return selected

        # Hybrid
        if not any('pareto_front' in r for r in ranked):
            cls.MultiObjectiveRanker.compute_pareto_fronts(ranked)
        n_pareto = k // 2
        n_composite = k - n_pareto

        # Pareto half
        pf0 = [r for r in ranked if r.get('pareto_front') == 0]
        pf0.sort(key=lambda x: x.get('crowding_distance', 0) or 0, reverse=True)
        pareto_selected = pf0[:n_pareto]
        pareto_seqs = {r['sequence'] for r in pareto_selected}

        # Composite half (excluding already-selected)
        remaining = [r for r in ranked if r['sequence'] not in pareto_seqs]
        remaining.sort(key=lambda x: x.get('mo_composite_score', 0), reverse=True)
        composite_selected = remaining[:n_composite]

        return pareto_selected + composite_selected

    def _check_convergence(self, cycles: List[CycleResult]) -> Tuple[bool, str]:
        """Check convergence based on hypervolume and composite score trends."""
        min_cycles = self.convergence_patience + 1
        if len(cycles) < min_cycles:
            return False, ""

        recent = cycles[-self.convergence_patience:]

        # Check hypervolume stagnation (only if HV values are meaningful)
        hvs = [c.hypervolume for c in recent]
        hv_deltas = [hvs[i] - hvs[i - 1] for i in range(1, len(hvs))]
        all_hv_zero = all(hv == 0.0 for hv in hvs)
        if not all_hv_zero and hv_deltas and all(abs(d) < self.convergence_delta for d in hv_deltas):
            return True, (
                f"Hypervolume Delta < {self.convergence_delta} for "
                f"{self.convergence_patience} consecutive cycles")

        # Check composite score stagnation
        comps = [c.best_composite for c in recent]
        comp_deltas = [comps[i] - comps[i - 1] for i in range(1, len(comps))]
        if comp_deltas and all(d < self.convergence_delta for d in comp_deltas):
            return True, (
                f"Composite score improvement < {self.convergence_delta} for "
                f"{self.convergence_patience} consecutive cycles")

        return False, ""

    def run(self,
            initial_pdb: str,
            design_region: str = 'A:25-35',
            native_seq: Optional[str] = None) -> ClosedLoopResult:
        """Execute the full closed-loop optimization.

        Args:
            initial_pdb: starting PDB structure
            design_region: region spec (e.g., 'A:25-35')
            native_seq: optional native sequence for recovery calculation

        Returns:
            ClosedLoopResult with per-cycle details and overall best
        """
        result = ClosedLoopResult()
        current_pdbs = [initial_pdb]
        t_start = time.time()
        prev_hypervolume = 0.0
        prev_best_iptm = 0.0
        patience_counter = 0
        all_designs_across_cycles = []

        for cycle_idx in range(1, self.max_cycles + 1):
            self._progress(cycle_idx, f'Cycle {cycle_idx}: designing {self.n_samples} sequences...')
            self._log(f"\n{'='*60}")
            self._log(f"  Closed-Loop Cycle {cycle_idx}/{self.max_cycles}")
            self._log(f"{'='*60}")
            self._log(f"  Templates: {len(current_pdbs)}")

            t_cycle_start = time.time()

            # ═══ Step 1: Design ═══
            all_designs = []
            samples_per_template = max(1, self.n_samples // max(len(current_pdbs), 1))

            for pdb_path in current_pdbs:
                pdb_str = str(pdb_path)
                if not os.path.exists(pdb_str):
                    self._log(f"  Skip missing template: {pdb_str}")
                    continue
                try:
                    designs = self.design_fn(pdb_str, design_region, samples_per_template)
                    for d in designs:
                        d['_template_pdb'] = pdb_str
                        d['_cycle'] = cycle_idx
                    all_designs.extend(designs)
                except Exception as e:
                    self._log(f"  Design error ({pdb_str}): {e}")

            if not all_designs:
                self._log("  No designs generated — stopping")
                break

            n_generated = len(all_designs)
            self._log(f"  Generated: {n_generated} sequences")

            # ═══ Step 2: AF2 Validation ═══
            self._progress(cycle_idx, f'Cycle {cycle_idx}: AF2 validating {n_generated} sequences...')

            sequences = [d['sequence'] for d in all_designs]
            try:
                af2_results = self.af2_validator(
                    sequences,
                    output_dir=f'alphafold_results/cl_cycle{cycle_idx}',
                )
            except Exception as e:
                self._log(f"  AF2 validation error: {e}")
                # Create minimal results without AF2
                af2_results = [{'sequence': seq, 'success': False, 'error': str(e)}
                               for seq in sequences]

            # Merge AF2 with design data
            af2_map = {r['sequence']: r for r in af2_results if r.get('success')}
            merged = []
            for d in all_designs:
                af2 = af2_map.get(d['sequence'], {})
                merged.append({
                    **d,
                    'plddt': af2.get('plddt', 0),      # 0-100 scale or 0-1
                    'ptm': af2.get('ptm', 0),
                    'iptm': af2.get('iptm', 0) or 0,
                    'max_pae': af2.get('max_pae', 999),
                    'pdb_path': af2.get('pdb_path'),
                    'success': af2.get('success', False),
                })

            n_af2_passed = len([m for m in merged if m['success']])
            self._log(f"  AF2 success: {n_af2_passed}/{n_generated}")

            # ═══ Step 3: Physics Scoring ═══
            self._progress(cycle_idx, f'Cycle {cycle_idx}: physics scoring...')

            if self.physics_scorer:
                try:
                    merged = self.physics_scorer(merged)
                except Exception as e:
                    self._log(f"  Physics scoring error: {e}")
                    for m in merged:
                        m['dG'] = None
                        m['ddG'] = None
                        m['relax_e'] = None
            else:
                for m in merged:
                    m['dG'] = None
                    m['ddG'] = None
                    m['relax_e'] = None

            n_physics = len([m for m in merged if m.get('dG') is not None])
            self._log(f"  Physics scored: {n_physics}/{n_generated}")

            # ═══ Step 4: Multi-Objective Ranking ═══
            self._progress(cycle_idx, f'Cycle {cycle_idx}: Pareto ranking...')

            # Hard threshold filter
            passed, rej = self.mo_ranker.apply_hard_thresholds(merged)
            self._log(f"  Hard threshold pass: {len(passed)}/{len(merged)}"
                      + (f" (rejected: {rej['total']})" if rej['total'] > 0 else ""))

            if not passed:
                self._log("  All designs rejected — using unfiltered")
                passed = merged

            # Compute Pareto fronts and weighted composite
            ranked = self.mo_ranker.compute_weighted_composite(
                cls.MultiObjectiveRanker.compute_pareto_fronts(passed)
            )

            if not ranked:
                self._log("  No designs after ranking — stopping")
                break

            # ═══ Step 5: Extract Stats ═══
            best = ranked[0]
            best_comp = best.get('mo_composite_score', 0)
            best_iptm = best.get('iptm', 0) or 0
            best_plddt = best.get('plddt', 0) or 0
            best_dg = best.get('dG')
            best_seq = best.get('sequence', '')
            best_pdb = best.get('pdb_path', '')

            pf0 = [r for r in ranked if r.get('pareto_front') == 0]
            pf0_size = len(pf0)
            hv = cls.MultiObjectiveRanker.compute_hypervolume(ranked, dims=['iptm', 'dG'])
            avg_top5 = (sum(r.get('mo_composite_score', 0) for r in ranked[:5]) /
                        min(len(ranked), 5))

            delta_iptm = best_iptm - prev_best_iptm if prev_best_iptm > 0 else 0
            delta_hv = hv - prev_hypervolume

            # Update global best
            if best_iptm > result.best_overall_iptm:
                result.best_overall_iptm = best_iptm
                result.best_overall_plddt = best_plddt
                result.best_overall_sequence = best_seq
                result.best_overall_dG = best_dg
                result.best_overall_pdb = best_pdb
            if best_comp > result.best_overall_composite:
                result.best_overall_composite = best_comp

            # ═══ Step 6: Select Top-K Templates ═══
            selected = self._select_top_k(ranked, self.top_k)
            current_pdbs = []
            for r in selected:
                pdb = r.get('pdb_path')
                if pdb and os.path.exists(str(pdb)):
                    current_pdbs.append(pdb)
                elif r.get('_template_pdb') and os.path.exists(str(r['_template_pdb'])):
                    current_pdbs.append(r['_template_pdb'])

            if not current_pdbs:
                current_pdbs = [initial_pdb]
                self._log(f"  Fallback to initial template")

            self._log(f"  Selected {len(selected)} designs, {len(current_pdbs)} PDB templates")

            # ═══ Step 7: Fine-Tuning (optional) ═══
            if (self.finetune_fn and
                cycle_idx > 0 and
                cycle_idx % self.finetune_every_n_cycles == 0):
                self._progress(cycle_idx, f'Cycle {cycle_idx}: fine-tuning model...')
                try:
                    new_ckpt = self.finetune_fn(ranked[:self.top_k], cycle_idx)
                    if new_ckpt:
                        result.finetune_checkpoints.append(new_ckpt)
                        self._log(f"  Fine-tuned checkpoint: {new_ckpt}")
                except Exception as e:
                    self._log(f"  Fine-tuning failed: {e}")

            # ═══ Record Cycle ═══
            t_cycle = time.time() - t_cycle_start

            cycle_result = CycleResult(
                cycle_idx=cycle_idx,
                n_generated=n_generated,
                n_af2_passed=n_af2_passed,
                n_physics_passed=n_physics,
                n_final=len(ranked),
                best_composite=best_comp,
                best_iptm=best_iptm,
                best_plddt=best_plddt,
                best_dG=best_dg,
                best_sequence=best_seq,
                best_pdb=best_pdb or '',
                pareto_front_size=pf0_size,
                hypervolume=hv,
                avg_composite_top5=avg_top5,
                delta_iptm=delta_iptm,
                delta_hypervolume=delta_hv,
                all_ranked=ranked,
                report=cls.format_mo_ranking_report(ranked, top_n=10),
                elapsed_seconds=t_cycle,
            )
            result.cycles.append(cycle_result)

            self._log(f"  Cycle {cycle_idx} complete ({t_cycle:.0f}s)")
            self._log(f"    Best: composite={best_comp:.3f}, ipTM={best_iptm:.3f}")
            if best_dg is not None:
                self._log(f"    dG={best_dg:.1f}, PF0={pf0_size}, HV={hv:.3f}")

            prev_hypervolume = hv
            prev_best_iptm = best_iptm
            all_designs_across_cycles.extend(ranked)

            # Yield intermediate result for streaming UIs
            if self.yield_cb:
                self.yield_cb(result)

            # ═══ Convergence Check ═══
            converged, reason = self._check_convergence(result.cycles)
            if converged:
                result.converged = True
                result.convergence_reason = reason
                self._log(f"  ✅ Converged: {reason}")
                break

        result.total_elapsed = time.time() - t_start

        self._log(f"\n{'='*60}")
        self._log(f"  Closed-Loop Complete ({result.total_elapsed:.0f}s, "
                   f"{len(result.cycles)} cycles)")
        self._log(f"  Best: {result.best_overall_sequence[:60]}")
        self._log(f"  ipTM={result.best_overall_iptm:.3f}, "
                   f"composite={result.best_overall_composite:.3f}")
        if result.best_overall_dG is not None:
            self._log(f"  dG={result.best_overall_dG:.1f}")
        self._log(f"{'='*60}")

        return result


def format_closed_loop_report(result: ClosedLoopResult) -> str:
    """Format the full closed-loop result as a human-readable text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  Multi-Objective Closed-Loop Alignment Report")
    lines.append("=" * 70)

    for c in result.cycles:
        lines.append(f"\n  Cycle {c.cycle_idx}:")
        lines.append(f"    Generated: {c.n_generated} | AF2 pass: {c.n_af2_passed} | "
                     f"Final: {c.n_final} | Time: {c.elapsed_seconds:.0f}s")
        lines.append(f"    Best composite: {c.best_composite:.3f} | "
                     f"ipTM: {c.best_iptm:.3f} | pLDDT: {c.best_plddt:.1f}")
        if c.best_dG is not None:
            lines.append(f"    Best dG: {c.best_dG:.1f} REU | "
                         f"Pareto-0: {c.pareto_front_size} | "
                         f"HV: {c.hypervolume:.3f}")
        if c.delta_iptm != 0:
            lines.append(f"    ΔipTM: {c.delta_iptm:+.3f} | "
                         f"ΔHV: {c.delta_hypervolume:+.3f}")
        lines.append(f"    Best seq: {c.best_sequence[:50]}...")

    lines.append(f"\n  {'─'*60}")
    lines.append(f"  Total: {result.total_elapsed:.0f}s | "
                 f"Converged: {'yes' if result.converged else 'no'}")
    if result.converged:
        lines.append(f"  Reason: {result.convergence_reason}")
    lines.append(f"  Global best ipTM: {result.best_overall_iptm:.3f} | "
                 f"composite: {result.best_overall_composite:.3f}")
    if result.best_overall_dG is not None:
        lines.append(f"  Global best dG: {result.best_overall_dG:.1f} REU")
    lines.append(f"  Global best: {result.best_overall_sequence[:60]}")
    if result.finetune_checkpoints:
        lines.append(f"  Fine-tuned checkpoints: {len(result.finetune_checkpoints)}")
    lines.append("=" * 70)

    return '\n'.join(lines)
