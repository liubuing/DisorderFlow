#!/usr/bin/env python
"""Epitope Structure Builder — build structural models for epitope segments.

Supports two modes:
  1. De novo: generate alpha-helix template from sequence
  2. Extraction: extract epitope region from an existing PDB structure

The resulting PDB files serve as structural context for BFN antibody CDR design.
"""

import os
import tempfile
import math


def build_epitope_helix(sequence, chain_id='A', start_res=1, output_dir=None):
    """Generate a PDB file with alpha-helix backbone for an epitope sequence.

    Uses standard alpha-helix geometry (~3.6 res/turn, 1.5 Å rise).
    Compatible with BFN, ProteinMPNN, and ESM-IF.

    Args:
        sequence: amino acid sequence string
        chain_id: PDB chain identifier
        start_res: starting residue number
        output_dir: directory for output PDB (default: temp dir)

    Returns:
        pdb_path: path to the generated PDB file
    """
    from app import generate_pdb_from_sequence

    pdb_content = generate_pdb_from_sequence(sequence, chain_id, start_res)
    if pdb_content is None:
        raise ValueError(f"Cannot generate PDB for sequence: {sequence[:20]}...")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='epitope_')
    os.makedirs(output_dir, exist_ok=True)

    pdb_path = os.path.join(output_dir, f'epitope_{chain_id}.pdb')
    with open(pdb_path, 'w') as f:
        f.write(pdb_content)

    return pdb_path


def extract_epitope_pdb(pdb_path, chain_id, resseq_range, output_dir=None,
                        output_chain=None):
    """Extract an epitope region from an existing PDB structure.

    Args:
        pdb_path: path to source PDB file
        chain_id: chain to extract from
        resseq_range: (start, end) inclusive residue numbers
        output_dir: output directory (default: same as source)
        output_chain: rewrite chain ID in output (default: keep original)

    Returns:
        pdb_path: path to the extracted PDB file
    """
    start, end = resseq_range
    if output_dir is None:
        output_dir = os.path.dirname(pdb_path) or '.'
    if output_chain is None:
        output_chain = chain_id

    extracted_atoms = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                cid = line[21:22].strip()
                if cid != chain_id:
                    continue
                try:
                    resid = int(line[22:26])
                except ValueError:
                    continue
                if start <= resid <= end:
                    extracted_atoms.append(line)

    if not extracted_atoms:
        raise ValueError(
            f"No atoms found in {pdb_path} chain {chain_id} "
            f"range {start}-{end}"
        )

    # Rewrite chain ID if needed
    if output_chain != chain_id:
        rewritten = []
        for line in extracted_atoms:
            rewritten.append(line[:21] + output_chain + line[22:])
        extracted_atoms = rewritten

    base = os.path.splitext(os.path.basename(pdb_path))[0]
    out_path = os.path.join(output_dir,
                            f'{base}_{chain_id}_{start}_{end}.pdb')
    with open(out_path, 'w') as f:
        for line in extracted_atoms:
            f.write(line)
        f.write(f"{'TER':6s}{'':4s}{'':3s} {output_chain:1s}{end:4d}\n")
        f.write('END\n')

    return out_path


def build_epitope_structure(segment, target_seq=None, target_pdb=None,
                            chain_id='A', output_dir=None):
    """Build epitope structure from a segment dict.

    The segment dict comes from find_ordered_segments() and contains
    'start', 'end', 'residues' keys. If target_pdb is provided, extracts
    the region from the real structure. Otherwise generates a helix.

    Args:
        segment: dict with 'start', 'end' keys (1-indexed residue numbers)
        target_seq: full target sequence (needed for helix mode)
        target_pdb: optional path to real structure for extraction
        chain_id: chain identifier
        output_dir: output directory

    Returns:
        dict with keys: pdb_path, sequence, mode ('helix' or 'extracted')
    """
    if target_pdb and os.path.exists(target_pdb):
        pdb_path = extract_epitope_pdb(
            target_pdb, chain_id,
            (segment['start'], segment['end']),
            output_dir, output_chain='B'
        )
        # Read the extracted sequence from the generated PDB
        from Bio.PDB import PDBParser
        aa3to1 = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
                  'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
                  'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
                  'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
        try:
            parser = PDBParser(QUIET=True)
            pid = os.path.basename(pdb_path).replace('.pdb', '')
            s = parser.get_structure(pid, pdb_path)[0]['B']
            seq = ''.join(aa3to1.get(r.get_resname().strip(), 'X')
                         for r in s.get_residues() if r.get_resname().strip() in aa3to1)
        except Exception:
            seq = ''
        mode = 'extracted'
    else:
        if target_seq is None:
            raise ValueError("target_seq required when no PDB is provided")
        # Extract subsequence for the segment
        start_idx = segment['start'] - 1
        end_idx = segment['end']
        seg_seq = target_seq[start_idx:end_idx]
        pdb_path = build_epitope_helix(
            seg_seq, chain_id, segment['start'], output_dir
        )
        seq = seg_seq
        mode = 'helix'

    return {
        'pdb_path': pdb_path,
        'sequence': seq,
        'mode': mode,
        'start': segment['start'],
        'end': segment['end'],
    }
