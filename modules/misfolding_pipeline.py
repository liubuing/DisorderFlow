#!/usr/bin/env python
"""Misfolding disease antibody design pipeline.

Orchestrates the full workflow:
  target preparation -> epitope analysis -> constrained design ->
  AF2 multimer validation -> iterative refinement

Reuses heavily from:
  - target_design_helpers (tdh)  : epitope scoring, contact analysis, interface scoring
  - cascade_filter (cf)          : misfolding-aware cascade filtering
  - iterative_refiner             : iterative design-refine loop
"""

import sys
import os
import json
import time
import tempfile
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent

import target_design_helpers as tdh
import cascade_filter as cf
import misfolding_knowledge_base as mkb
import misfolding_data_loader as mdl


class MisfoldingDesignPipeline:
    """Orchestrator for misfolding disease antibody design."""

    def __init__(self, config=None):
        self.config = config or {}
        self.mf_config = self.config.get('misfolding_disease', {})

    # ── Step 0: Target preparation ──

    def prepare_target(self, disease_key, conformation_key=None):
        """Load or download target PDB structure.

        Returns:
            dict with pdb_path, chain_id, target_name, idp_warning, etc.
        """
        target = mkb.get_target(disease_key)
        if target is None:
            return {'success': False, 'error': f'Unknown disease key: {disease_key}'}

        result = mdl.prepare_misfolding_target(disease_key, target, conformation_key)
        result['disease_key'] = disease_key
        result['target_data'] = target
        result['conformation_key'] = conformation_key or mkb.get_default_conformation(disease_key)
        result['is_idp'] = target.get('idp_flag', False)
        return result

    # ── Step 1: Epitope analysis ──

    def analyze_epitope(self, pdb_path, chain_id, top_n=20, weights=None,
                        known_epitopes=None):
        """Score epitope residues on the target.

        Args:
            pdb_path: path to target PDB
            chain_id: chain to analyze
            top_n: number of top epitope residues to return
            weights: dict for SASA/hydrophilicity/protrusion weights
            known_epitopes: list of literature epitope dicts

        Returns:
            dict with epitope_data, formatted_table, suggested_region, known_epitopes
        """
        if weights is None:
            weights = {'sasa': 0.4, 'hydrophilicity': 0.35, 'protrusion': 0.25}

        epitope_data = tdh.score_epitope_residues(pdb_path, chain_id, weights)
        if not epitope_data:
            return {
                'epitope_data': [],
                'formatted_table': 'Failed to compute epitope scores.',
                'suggested_region': None,
                'known_epitopes': known_epitopes or [],
            }

        # Format display
        formatted = tdh.format_epitope_table(epitope_data, top_n)

        # Build suggested region from top-N surface residues
        surface = [d for d in epitope_data if d.get('rel_sasa', 0) > 0.15]
        top_resseqs = [d['resseq'] for d in surface[:top_n]]
        suggested_region = tdh.residues_to_region_spec(chain_id, top_resseqs)

        return {
            'epitope_data': epitope_data,
            'formatted_table': formatted,
            'suggested_region': suggested_region,
            'known_epitopes': known_epitopes or [],
        }

    def suggest_region_from_known_epitopes(self, chain_id, known_epitopes):
        """Build a region spec from literature-known epitope ranges.

        Args:
            chain_id: target chain ID
            known_epitopes: list of {'region': (start,end), 'ab': str, 'source': str}

        Returns:
            region_spec string like "A:1-16,18-26"
        """
        if not known_epitopes:
            return ''

        all_resseqs = set()
        for ep in known_epitopes:
            region = ep.get('region')
            if region and len(region) == 2:
                all_resseqs.update(range(region[0], region[1] + 1))

        return tdh.residues_to_region_spec(chain_id, sorted(all_resseqs))

    # ── Step 2: Antibody template ──

    def get_scaffold_info(self, scaffold_key):
        """Get nanobody scaffold sequence and info.

        Returns:
            dict with sequence, length, cdr_definitions, notes
        """
        scaffold = mkb.get_scaffold(scaffold_key)
        if scaffold is None:
            return None
        return scaffold

    def generate_antibody_template(self, scaffold_key, output_dir=None):
        """Generate an antibody template structure.

        For scaffolds with PDB ID: downloads the PDB.
        For scaffolds with sequence only: returns sequence for AF2 folding.

        Returns:
            dict with pdb_path (or None), sequence, cdr_definitions
        """
        scaffold = mkb.get_scaffold(scaffold_key)
        if scaffold is None:
            return {'error': f'Unknown scaffold: {scaffold_key}'}

        seq = scaffold.get('sequence')
        pdb_id = scaffold.get('pdb_id')

        pdb_path = None
        if pdb_id:
            pdb_path = mdl.fetch_rcsb_structure(pdb_id)

        return {
            'scaffold_key': scaffold_key,
            'name': scaffold['name'],
            'sequence': seq,
            'pdb_path': str(pdb_path) if pdb_path else None,
            'cdr_definitions': scaffold.get('cdr_definitions', {}),
            'length': scaffold.get('length'),
        }

    # ── Step 3: Constrained design region ──

    def compute_design_region(self, ab_pdb, ab_chain, ag_pdb, ag_chain,
                              epitope_resseqs, constraint_cutoff=8.0):
        """Find antibody residues facing the target epitope.

        This requires a pre-docked antibody-antigen complex PDB.
        If no complex exists, falls back to full CDR design.

        Returns:
            (region_spec, facing_resseqs, message)
        """
        if not ab_pdb or not os.path.exists(str(ab_pdb)):
            return None, [], 'Antibody PDB not available — full CDR design mode.'

        # Check if antibody and antigen are in the same PDB
        # If not, we're doing template-only design
        try:
            facing = tdh.find_residues_facing_region(
                str(ab_pdb), ag_chain, epitope_resseqs, ab_chain, constraint_cutoff
            )
        except Exception as e:
            return None, [], f'Contact analysis failed: {e} — using default CDRs.'

        if not facing:
            return None, [], f'No residues within {constraint_cutoff}A of epitope — using default CDRs.'

        region_spec = tdh.residues_to_region_spec(ab_chain, facing)
        msg = f'{len(facing)} residues facing epitope within {constraint_cutoff}A'
        return region_spec, facing, msg

    # ── Step 4: Design orchestration ──

    def design_sequences(self, design_fn, pdb_path, design_region, n_samples=10,
                         **fn_kwargs):
        """Run sequence design using the provided design function.

        Args:
            design_fn: callable(pdb_path, region, n_samples, **kwargs) -> list[dict]
            pdb_path: path to antibody PDB
            design_region: region spec string (e.g. 'H:95-102')
            n_samples: number of sequences to generate

        Returns:
            list of dicts with sequence, ppl, plddt, iptm, etc.
        """
        results = design_fn(pdb_path, design_region, n_samples, **fn_kwargs)
        return results

    # ── Step 5: AF2 multimer validation ──

    def validate_with_af2(self, ab_seq, ag_seq, output_dir, num_recycle=3,
                          ab_name='design', ag_name='target',
                          colabfold_exe='colabfold_batch',
                          model_type='alphafold2_multimer_v3'):
        """Run AF2 multimer and extract interface quality metrics.

        Args:
            ab_seq: antibody sequence (designed)
            ag_seq: antigen sequence (target)
            output_dir: output directory for AF2 results
            num_recycle: AF2 recycle count
            ab_name: label for antibody chain
            ag_name: label for antigen chain
            colabfold_exe: path to colabfold_batch
            model_type: AF2 model preset

        Returns:
            dict with plddt, ptm, iptm, max_pae, pdb_path, interface_pae, success, error
        """
        os.makedirs(output_dir, exist_ok=True)

        # Build multimer FASTA
        fasta_content = tdh.build_multimer_fasta(ab_seq, ag_seq, ab_name, ag_name)
        fasta_path = os.path.join(output_dir, f'{ab_name}_{ag_name}.fasta')
        with open(fasta_path, 'w') as f:
            f.write(fasta_content)

        # Build colabfold command
        cmd = (
            f'"{colabfold_exe}" '
            f'--num-models 1 '
            f'--num-recycle {num_recycle} '
            f'--model-type {model_type} '
            f'--rank iptm '
            f'"{fasta_path}" "{output_dir}"'
        )

        try:
            import subprocess
            result = subprocess.run(cmd, shell=True, capture_output=True,
                                    text=True, timeout=3600, cwd=output_dir)
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'AF2 multimer timed out (1 hour).'}
        except FileNotFoundError:
            return {
                'success': False,
                'error': f'ColabFold not found at "{colabfold_exe}".',
            }

        if result.returncode != 0:
            return {
                'success': False,
                'error': f'ColabFold failed (code {result.returncode}): {result.stderr[:300]}',
            }

        # Parse results
        parsed = self._parse_af2_output(output_dir, ab_name, ag_name, len(ab_seq))
        parsed['ab_seq'] = ab_seq
        parsed['ag_seq'] = ag_seq
        return parsed

    def _parse_af2_output(self, output_dir, ab_name, ag_name, ab_len):
        """Parse ColabFold multimer output files."""
        output_path = Path(output_dir)
        complex_name = f'complex_{ab_name}_{ag_name}'

        # Find the scores JSON and ranked PDB
        scores_files = sorted(output_path.glob(f'{complex_name}*scores_rank_001*.json'))
        pdb_files = sorted(output_path.glob(f'{complex_name}*unrelaxed_rank_001*.pdb'))
        if not pdb_files:
            pdb_files = sorted(output_path.glob(f'{complex_name}*relaxed_rank_001*.pdb'))

        if not scores_files:
            return {'success': False, 'error': 'No AF2 scores JSON found.'}

        try:
            with open(scores_files[0]) as f:
                scores = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return {'success': False, 'error': f'Failed to parse scores: {e}'}

        plddt_array = np.array(scores.get('plddt', []))
        ptm = scores.get('ptm', 0.0)
        iptm = scores.get('iptm', 0.0)
        max_pae = scores.get('max_pae', 999.0)

        # Compute interface PAE
        pae_matrix = np.array(scores.get('pae', []))
        interface_pae = None
        if pae_matrix.size > 0:
            interface_pae, _ = tdh.compute_interface_pae(pae_matrix, ab_len)

        pdb_path = str(pdb_files[0]) if pdb_files else None

        return {
            'success': True,
            'plddt': float(np.mean(plddt_array)) if len(plddt_array) > 0 else 0.0,
            'ptm': ptm,
            'iptm': iptm,
            'max_pae': max_pae,
            'interface_pae': interface_pae,
            'pdb_path': pdb_path,
            'scores_json': str(scores_files[0]),
            'error': None,
        }

    # ── Step 6: Full pipeline ──

    def run(self, disease_key, conformation_key=None, scaffold_key=None,
            design_fn=None, n_samples=10, top_k=5, constraint_cutoff=8.0,
            run_af2=True, num_recycle=3, colabfold_exe='colabfold_batch',
            progress_cb=None):
        """Run the complete misfolding disease design pipeline.

        Args:
            disease_key: key in MISFOLDING_TARGETS
            conformation_key: specific conformation to target
            scaffold_key: nanobody scaffold to use
            design_fn: callable for sequence design
            n_samples: sequences to generate
            top_k: top designs to AF2-validate
            constraint_cutoff: distance for epitope-facing residues
            run_af2: whether to run AF2 validation
            num_recycle: AF2 recycle count
            colabfold_exe: path to colabfold_batch
            progress_cb: optional progress callback(done, total, message)

        Returns:
            dict with full pipeline results and formatted report
        """
        def _progress(step, total, msg):
            if progress_cb:
                progress_cb(step, total, msg)

        pipeline_result = {
            'disease_key': disease_key,
            'steps': {},
            'success': False,
            'report': '',
        }

        # Step 0: Prepare target
        _progress(0, 6, 'Loading target structure...')
        target = self.prepare_target(disease_key, conformation_key)
        pipeline_result['target'] = target
        if not target['success']:
            pipeline_result['report'] = f'Target preparation failed: {target.get("error")}'
            return pipeline_result

        # Step 1: Epitope analysis
        _progress(1, 6, 'Analyzing epitope residues...')
        known_eps = mkb.get_known_epitope_regions(disease_key)
        epitope_result = self.analyze_epitope(
            target['pdb_path'], target['chain_id'],
            known_epitopes=known_eps
        )
        pipeline_result['epitope'] = epitope_result

        # Determine design region
        if known_eps:
            suggested = self.suggest_region_from_known_epitopes(
                target['chain_id'], known_eps
            )
        else:
            suggested = epitope_result.get('suggested_region')

        pipeline_result['design_region'] = suggested

        # Step 2: Scaffold
        _progress(2, 6, 'Loading antibody scaffold...')
        if scaffold_key:
            scaffold = self.get_scaffold_info(scaffold_key)
            pipeline_result['scaffold'] = scaffold
        else:
            pipeline_result['scaffold'] = None

        # Step 3: Design constraint
        _progress(3, 6, 'Computing design constraints...')
        # For misfolding targets, we design the scaffold CDRs
        # If scaffold has CDR definitions, use those
        design_region = None
        if scaffold_key:
            scf = mkb.get_scaffold(scaffold_key)
            if scf and scf.get('cdr_definitions'):
                cdrs = scf['cdr_definitions']
                # Build region from all CDRs
                all_resseqs = set()
                for cdr_name, (start, end) in cdrs.items():
                    all_resseqs.update(range(start, end + 1))
                chain = 'A'  # nanobody single chain
                design_region = tdh.residues_to_region_spec(chain, sorted(all_resseqs))
        pipeline_result['effective_design_region'] = design_region or 'A:26-33,51-58,97-110'

        # Step 4: Design
        _progress(4, 6, f'Designing {n_samples} sequences...')
        if design_fn is not None:
            ab_pdb = target.get('pdb_path')  # target structure — for template
            # Actually for fixed-backbone design we need antibody structure
            # Use scaffold PDB if available
            if scaffold_key:
                scaffold_info = self.generate_antibody_template(scaffold_key)
                ab_pdb = scaffold_info.get('pdb_path')
            if not ab_pdb:
                pipeline_result['report'] = 'No antibody template PDB available for design.'
                return pipeline_result

            designs = design_fn(ab_pdb, pipeline_result['effective_design_region'],
                               n_samples)
            pipeline_result['designs_raw'] = designs

            # Apply misfolding cascade
            filtered, casc_report = cf.apply_cascade_misfolding(designs, use_af2=False)
            pipeline_result['designs_filtered'] = filtered
            pipeline_result['cascade_report'] = casc_report

            # IDP warning
            if target.get('idp_warning'):
                idp_note = cf.format_idp_score_warning(
                    target['target_name'], target['idp_warning']
                )
                pipeline_result['cascade_report'] += '\n\n' + idp_note
        else:
            pipeline_result['designs_raw'] = []
            pipeline_result['designs_filtered'] = []
            pipeline_result['cascade_report'] = 'No design function provided (design_fn=None).'

        # Step 5: AF2 validation
        pipeline_result['af2_results'] = []
        if run_af2 and pipeline_result.get('designs_filtered'):
            _progress(5, 6, f'AF2 validation (top {top_k} designs)...')
            top_designs = pipeline_result['designs_filtered'][:top_k]
            output_dir = os.path.join(
                os.getcwd(),
                self.mf_config.get('output_dir', 'misfolding_results'),
                disease_key,
            )

            # Get target sequence
            ag_seq = tdh.extract_seq_from_pdb(target['pdb_path'], target['chain_id'])

            af2_results = []
            for i, d in enumerate(top_designs):
                ab_seq = d['sequence']
                _progress(5, 6, f'AF2 validation ({i+1}/{len(top_designs)})...')
                result = self.validate_with_af2(
                    ab_seq, ag_seq, output_dir,
                    num_recycle=num_recycle,
                    colabfold_exe=colabfold_exe,
                    ab_name=f'design_{i+1}',
                    ag_name=disease_key,
                )
                result['rank'] = d.get('rank', i + 1)
                af2_results.append(result)

            pipeline_result['af2_results'] = af2_results

            # AF2 cascade re-ranking
            af2_casc_input = []
            for r in af2_results:
                if r['success']:
                    af2_casc_input.append({
                        'sequence': r.get('ab_seq', ''),
                        'plddt': r['plddt'],
                        'ptm': r['ptm'],
                        'iptm': r['iptm'],
                        'max_pae': r['max_pae'],
                        'interface_pae': r.get('interface_pae'),
                        'pdb_path': r.get('pdb_path'),
                    })

            if af2_casc_input:
                af2_filtered, af2_casc_report = cf.apply_cascade_misfolding(
                    af2_casc_input, use_af2=True
                )
                pipeline_result['af2_filtered'] = af2_filtered
                pipeline_result['af2_cascade_report'] = af2_casc_report

        # Step 6: Build final report
        _progress(6, 6, 'Generating report...')
        report = self._format_pipeline_report(pipeline_result)
        pipeline_result['report'] = report
        pipeline_result['success'] = True
        return pipeline_result

    def _format_pipeline_report(self, pr):
        """Build a comprehensive Markdown report from pipeline results."""
        lines = []
        lines.append('# Misfolding Disease Antibody Design Report')
        lines.append('')

        target = pr.get('target', {})
        lines.append('## Target Information')
        lines.append(f'- **Disease**: {target.get("target_name", "?")}')
        lines.append(f'- **PDB**: {target.get("pdb_path", "N/A")}')
        lines.append(f'- **Chain**: {target.get("chain_id", "?")}')
        if target.get('idp_warning'):
            lines.append(f'- **IDP Warning**: {target["idp_warning"]}')
        lines.append('')

        epitope = pr.get('epitope', {})
        lines.append('## Epitope Analysis')
        lines.append(f'- **Suggested Region**: `{pr.get("design_region", "N/A")}`')
        known = epitope.get('known_epitopes', [])
        if known:
            lines.append('- **Known Literature Epitopes**:')
            for ep in known:
                lines.append(f'  - {ep.get("ab", "?")}: region {ep.get("region")} ({ep.get("source", "")})')
        lines.append('')

        lines.append('## Design Results')
        casc_report = pr.get('cascade_report', 'No design results.')
        lines.append('```')
        lines.append(casc_report)
        lines.append('```')
        lines.append('')

        af2_results = pr.get('af2_results', [])
        if af2_results:
            lines.append('## AF2 Multimer Validation')
            for r in af2_results:
                status = 'OK' if r['success'] else 'FAILED'
                lines.append(f'- **Rank {r.get("rank", "?")}** [{status}] '
                           f'ipTM={r.get("iptm", 0):.3f} pTM={r.get("ptm", 0):.3f} '
                           f'pLDDT={r.get("plddt", 0):.1f}')
                if r.get('interface_pae') is not None:
                    lines.append(f'  - Interface PAE: {r["interface_pae"]:.2f}')
                if r.get('error'):
                    lines.append(f'  - Error: {r["error"]}')
            lines.append('')

        if pr.get('af2_cascade_report'):
            lines.append('## AF2 Re-Ranking')
            lines.append('```')
            lines.append(pr['af2_cascade_report'])
            lines.append('```')

        return '\n'.join(lines)


def quick_design(project_dir, disease_key, tool='bfn', n_samples=10,
                 scaffold_key='nanobody_cAb1'):
    """Quick design entry point — minimal setup.

    Loads the BFN model and runs the pipeline with default settings.
    """
    sys.path.insert(0, str(project_dir))

    # Lazy-import design modules to avoid circular deps
    # The design_fn is provided by the caller (app.py)
    # This function is a convenience wrapper

    pipeline = MisfoldingDesignPipeline()

    target = pipeline.prepare_target(disease_key)
    if not target['success']:
        print(f'ERROR: {target.get("error")}')
        return None

    print(f'Target: {target["target_name"]}')
    print(f'PDB: {target["pdb_path"]}')
    print(f'Chain: {target["chain_id"]}')
    if target.get('idp_warning'):
        print(f'IDP Warning: {target["idp_warning"]}')

    # Epitope analysis
    epitope = pipeline.analyze_epitope(target['pdb_path'], target['chain_id'])
    print(f'\nEpitope:\n{epitope["formatted_table"]}')
    print(f'Suggested region: {epitope["suggested_region"]}')

    return pipeline, target, epitope
