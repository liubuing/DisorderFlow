"""Generic protein structure preprocessing (no antibody concepts)."""
import logging
from Bio import PDB
from Bio.PDB import PDBExceptions
import torch

from ..utils.protein import parsers


def preprocess_protein_structure(pdb_path, chain_ids=None):
    """
    Parse any PDB file into a structure dict for general protein design.

    Args:
        pdb_path: Path to PDB file
        chain_ids: Optional list of chain IDs to process. If None, all chains are processed.

    Returns:
        dict with keys:
            - id: PDB ID (filename without .pdb)
            - chains: list of dicts, each with {'data': EasyDict, 'seqmap': dict, 'chain_id': str}
            - num_chains: int
            - all_chain_ids: list of str
        Returns None on parse failure.
    """
    import os
    parser = PDB.PDBParser(QUIET=True)
    pdb_id = os.path.basename(pdb_path).replace('.pdb', '')

    try:
        structure = parser.get_structure(pdb_id, pdb_path)
        model = structure[0]
        all_chain_ids = sorted([c.id for c in model.get_chains()])

        if chain_ids is not None:
            target_chains = [cid for cid in chain_ids if cid in all_chain_ids]
        else:
            target_chains = all_chain_ids

        if len(target_chains) == 0:
            raise ValueError(f'No valid chains found in {pdb_path}. '
                             f'Requested: {chain_ids}, Available: {all_chain_ids}')

        chains = []
        for ci, chain_id in enumerate(target_chains):
            chain = model[chain_id]
            data, seqmap = parsers.parse_biopython_structure(chain)
            # Set CDR flag to zeros (no CDR concept in generic proteins)
            data['cdr_flag'] = torch.zeros_like(data['aa'])
            chains.append({
                'data': data,
                'seqmap': seqmap,
                'chain_id': chain_id,
                'chain_idx': ci,
            })

        return {
            'id': pdb_id,
            'chains': chains,
            'num_chains': len(chains),
            'all_chain_ids': target_chains,
        }

    except (
        PDBExceptions.PDBConstructionException,
        parsers.ParsingException,
        KeyError,
        ValueError,
    ) as e:
        logging.warning(f'[{pdb_id}] {e.__class__.__name__}: {str(e)}')
        return None
