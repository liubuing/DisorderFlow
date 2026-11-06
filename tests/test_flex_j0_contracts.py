"""FLEX-BINDER Gate J0 pipeline contract tests.

Pins the properties the J0 gate silently depends on:
1. every shuffle is a true composition-matched permutation of its native
   region (multiset equality) - otherwise the ECLS contrast measures something
   other than sequence order;
2. the cohort's shuffles reproduce bit-exactly from the preregistered RNG and
   component order - otherwise the frozen cohort is not the frozen cohort;
3. the written component PDBs really contain chain H and chain P, and the
   declared H3 region spec extracts exactly the frozen region sequence;
4. build_region_batch honours the region spec and the context-chain contract
   on one real component (complex has peptide context, stripped does not);
5. when antigen_chains is passed, the peptide really lands in
   Fragment.Antigen and the antibody chain in Fragment.Heavy - the FLEX
   frozen numbers were produced by a batch where H was encoded as padding
   and P as Heavy, leaving mask_antigen empty (see
   .dsh-research/flex-distance/code-fix-report.md section 2).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_pae_dense_v6 import read


@pytest.fixture(scope='module')
def cohort():
    return read(ROOT / 'results/flex_j0_v1/cohort.json')


def test_shuffles_are_composition_matched(cohort):
    for entry in cohort['components']:
        native = entry['region_sequence']
        for shuffle in entry['shuffles']:
            assert sorted(shuffle) == sorted(native)


def test_shuffles_reproduce_from_preregistered_rng(cohort):
    rng = np.random.default_rng(cohort['rng_seed'])
    for entry in sorted(cohort['components'], key=lambda c: c['instance']):
        reproduced = []
        for _ in range(cohort['n_shuffles']):
            letters = list(entry['region_sequence'])
            rng.shuffle(letters)
            reproduced.append(''.join(letters))
        assert reproduced == entry['shuffles']


def test_component_pdbs_contain_named_chains_and_region(cohort):
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    parser = PDBParser(QUIET=True)
    for entry in cohort['components'][:5]:
        structure = parser.get_structure(entry['instance'],
                                         str(ROOT / entry['component_pdb']))
        model = next(structure.get_models())
        chain_ids = {chain.id for chain in model}
        assert chain_ids == {'H', 'P'}
        lo, hi = entry['region_spec'].split(':')[1].split('-')
        residues = [residue for residue in model['H'] if residue.id[0] == ' '
                    and int(lo) <= residue.id[1] <= int(hi)]
        sequence = ''.join(seq1(residue.resname) for residue in residues)
        assert sequence == entry['region_sequence']


def test_region_batch_honours_region_and_context_contract(cohort):
    pytest.importorskip('torch')
    from modules.bfn_loader import build_region_batch, inject_candidate_sequence
    entry = sorted(cohort['components'], key=lambda c: c['instance'])[0]
    complex_batch = build_region_batch(str(ROOT / entry['component_pdb']),
                                       entry['region_spec'],
                                       context_chains=['P'], device='cpu')
    stripped_batch = build_region_batch(str(ROOT / entry['component_pdb']),
                                        entry['region_spec'],
                                        context_chains=[], device='cpu')
    complex_gen = complex_batch['generate_flag'].bool()
    stripped_gen = stripped_batch['generate_flag'].bool()
    assert int(complex_gen.sum()) == len(entry['region_sequence'])
    assert int(stripped_gen.sum()) == len(entry['region_sequence'])
    # The complex context carries strictly more residues than the stripped one,
    # and the difference covers the peptide.
    complex_len = int(complex_batch['mask'].bool().sum())
    stripped_len = int(stripped_batch['mask'].bool().sum())
    assert complex_len > stripped_len
    assert complex_len - stripped_len <= len(entry['peptide_sequence'])
    inject_candidate_sequence(complex_batch, entry['region_sequence'])


def test_antigen_roles_batch_marks_peptide_as_antigen(cohort):
    # Regression pin for the FLEX chain-role encoding bug: without
    # antigen_chains the peptide was encoded as Fragment.Heavy and
    # mask_antigen was empty, silently disabling the antigen branch.
    pytest.importorskip('torch')
    from modules.bfn_loader import build_region_batch
    from disorderflow.utils.protein.constants import Fragment
    entry = sorted(cohort['components'], key=lambda c: c['instance'])[0]
    batch = build_region_batch(str(ROOT / entry['component_pdb']),
                               entry['region_spec'],
                               context_chains=['P'], antigen_chains=['P'],
                               device='cpu')
    roles = batch['fragment_type'][0].tolist()
    peptide_len = len(entry['peptide_sequence'])
    antigen_count = sum(1 for role in roles if role == int(Fragment.Antigen))
    heavy_count = sum(1 for role in roles if role == int(Fragment.Heavy))
    assert antigen_count == peptide_len
    assert heavy_count > 0
