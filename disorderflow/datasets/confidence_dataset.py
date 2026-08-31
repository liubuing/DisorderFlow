"""Dataset for BFN confidence head fine-tuning.

Loads pre-computed LMDB entries that pair BFN-preprocessed protein batches
with AF2-derived ground-truth confidence scores (pLDDT, ipTM, PAE).
"""

import os
import pickle

import torch
from torch.utils.data import Dataset

from disorderflow.datasets._base import register_dataset
from disorderflow.utils.protein.constants import Fragment

_LMDB_ENV_CACHE = {}


def _acquire_lmdb(path):
    normalized = os.path.abspath(path)
    cached = _LMDB_ENV_CACHE.get(normalized)
    if cached is None:
        import lmdb

        cached = [lmdb.open(normalized, readonly=True, lock=False, readahead=False), 0]
        _LMDB_ENV_CACHE[normalized] = cached
    cached[1] += 1
    return cached[0], normalized


def _release_lmdb(path):
    cached = _LMDB_ENV_CACHE.get(path)
    if cached is None:
        return
    cached[1] -= 1
    if cached[1] <= 0:
        cached[0].close()
        del _LMDB_ENV_CACHE[path]


@register_dataset("confidence_regression")
class ConfidenceRegressionDataset(Dataset):
    """Load BFN batch + AF2 confidence ground truth from LMDB.

    Each entry in the LMDB is a dict with keys:
        pdb_id, sequence, batch, af2_plddt, af2_iptm, af2_pae_matrix

    The batch dict contains standard BFN input tensors (aa, pos_heavyatom,
    mask_heavyatom, generate_flag, etc.) prepared by MaskRegion →MergeProtein
    →PatchProtein transform pipeline.
    """

    def __init__(self, cfg, transform=None, **kwargs):
        super().__init__()
        # Accept both EasyDict/config dict and plain string path
        if isinstance(cfg, str):
            self.db_path = cfg
            cfg = {}
        else:
            self.db_path = cfg.db_path if hasattr(cfg, "db_path") else cfg["db_path"]
        self.candidate_interface_v1 = bool(
            cfg.get("candidate_interface_v1", False))
        self.max_residues = cfg.get("max_residues", 0) if isinstance(cfg, dict) else 0
        self.idp_sample_weight = cfg.get("idp_sample_weight", 1.0) if isinstance(cfg, dict) else 1.0
        # transform is ignored —data is already preprocessed in the LMDB

        if os.path.isdir(self.db_path) and not os.path.exists(
            os.path.join(self.db_path, "data.mdb")
        ):
            # Pickle directory fallback
            self._use_lmdb = False
            meta_path = os.path.join(self.db_path, "meta.json")
            if os.path.exists(meta_path):
                import json

                with open(meta_path) as f:
                    meta = json.load(f)
                self._length = meta["n_entries"]
            else:
                # Count .pkl files
                self._length = len([f for f in os.listdir(self.db_path) if f.endswith(".pkl")])
            self._pkl_dir = self.db_path
            self._valid_indices = list(range(self._length))
        else:
            self._use_lmdb = True
            self._env, self._env_cache_key = _acquire_lmdb(self.db_path)
            with self._env.begin() as txn:
                encoded_length = txn.get(b"__len__")
                self._length = (
                    pickle.loads(encoded_length)
                    if encoded_length is not None
                    else txn.stat()["entries"]
                )
            self._valid_indices = list(range(self._length))

        self.min_plddt = cfg.get("min_af2_plddt", 0.0) if isinstance(cfg, dict) else 0.0

        # Filter by max_residues to avoid CUDA OOM on large proteins
        if self.max_residues > 0 and self._use_lmdb:
            self._valid_indices = []
            with self._env.begin() as txn:
                for i in range(self._length):
                    key = f"{i:08d}".encode()
                    entry = pickle.loads(txn.get(key))
                    seq_len = len(entry.get("sequence", ""))
                    if seq_len <= self.max_residues:
                        self._valid_indices.append(i)
            n_filtered = self._length - len(self._valid_indices)
            if n_filtered > 0:
                print(
                    f"[ConfidenceRegression] Filtered {n_filtered} entries > {self.max_residues} residues "
                    f"({len(self._valid_indices)}/{self._length} retained)"
                )

        # Quality filter: remove entries with low mean AF2 pLDDT (V10)
        if self.min_plddt > 0.0:
            plddt_before = len(self._valid_indices)
            plddt_filtered = []
            plddt_threshold = self.min_plddt / 100.0  # config uses 0-100 scale, data uses 0-1
            if self._use_lmdb:
                with self._env.begin() as txn:
                    for idx in self._valid_indices:
                        entry = pickle.loads(txn.get(f"{idx:08d}".encode()))
                        if (
                            entry.get("af2_plddt", torch.tensor([0.0])).mean().item()
                            >= plddt_threshold
                        ):
                            plddt_filtered.append(idx)
            else:
                for idx in self._valid_indices:
                    pkl_path = os.path.join(self._pkl_dir, f"{idx:08d}.pkl")
                    with open(pkl_path, "rb") as f:
                        entry = pickle.load(f)
                    if entry.get("af2_plddt", torch.tensor([0.0])).mean().item() >= plddt_threshold:
                        plddt_filtered.append(idx)
            n_removed = plddt_before - len(plddt_filtered)
            self._valid_indices = plddt_filtered
            if n_removed > 0:
                print(
                    f"[ConfidenceRegression] Quality filter (>={self.min_plddt} pLDDT): "
                    f"removed {n_removed} low-quality entries "
                    f"({len(self._valid_indices)}/{plddt_before} retained)"
                )

        # Build sample weights for IDP-stratified sampling
        self._sample_weights = None
        self._scaffold_ids = []  # scaffold_id per valid_index — for grouped batch sampler
        if self.idp_sample_weight > 1.0:
            self._sample_weights = torch.ones(len(self._valid_indices), dtype=torch.float64)
            n_idp = 0
            for idx, real_idx in enumerate(self._valid_indices):
                if self._use_lmdb:
                    with self._env.begin() as txn:
                        entry = pickle.loads(txn.get(f"{real_idx:08d}".encode()))
                else:
                    pkl_path = os.path.join(self._pkl_dir, f"{real_idx:08d}.pkl")
                    with open(pkl_path, "rb") as f:
                        entry = pickle.load(f)
                if entry.get("is_idp", False):
                    self._sample_weights[idx] = self.idp_sample_weight
                    n_idp += 1
            if n_idp > 0:
                print(
                    f"[ConfidenceRegression] IDP-weighted sampling: {n_idp} IDP entries "
                    f"(weight={self.idp_sample_weight:.1f}x, {n_idp}/{len(self._valid_indices)})"
                )
            else:
                self._sample_weights = None  # No IDPs found, fall back to uniform

        # Build scaffold_id index for grouped batch sampler (V14+)
        self._scaffold_ids = []
        for idx in self._valid_indices:
            if self._use_lmdb:
                with self._env.begin() as txn:
                    entry = pickle.loads(txn.get(f"{idx:08d}".encode()))
            else:
                pkl_path = os.path.join(self._pkl_dir, f"{idx:08d}.pkl")
                with open(pkl_path, "rb") as f:
                    entry = pickle.load(f)
            self._scaffold_ids.append(int(entry.get("scaffold_id", idx)))

        self.requires_complete_groups = bool(self.candidate_interface_v1)
        if self.requires_complete_groups:
            groups = {}
            for dataset_index, scaffold_id in enumerate(self._scaffold_ids):
                groups.setdefault(scaffold_id, []).append(dataset_index)
            self.group_indices = list(groups.values())

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, index):
        real_idx = self._valid_indices[index]
        if self._use_lmdb:
            with self._env.begin() as txn:
                key = f"{real_idx:08d}".encode()
                entry = pickle.loads(txn.get(key))
        else:
            pkl_path = os.path.join(self._pkl_dir, f"{real_idx:08d}.pkl")
            with open(pkl_path, "rb") as f:
                entry = pickle.load(f)

        # Reconstruct batch and attach AF2 ground truth
        batch = entry["batch"]
        aa_len = batch["aa"].shape[0]
        if self.candidate_interface_v1:
            if entry.get("schema_version") != "candidate_interface_confidence_v1":
                raise ValueError(
                    "candidate_interface_v1 requires successor dataset schema")
            generate_flag = batch.get("generate_flag")
            fragment_type = batch.get("fragment_type")
            if generate_flag is None or not generate_flag.bool().any():
                raise ValueError(
                    "candidate_interface_v1 requires a nonempty candidate mask")
            if (fragment_type is None
                    or not (fragment_type == int(Fragment.Antigen)).any()):
                raise ValueError(
                    "candidate_interface_v1 requires explicit antigen residues")
            af2_plddt = entry["af2_plddt"]
            af2_pae = entry["af2_pae_matrix"]
            if af2_plddt.shape[0] != aa_len:
                raise ValueError(
                    "candidate_interface_v1 pLDDT must cover the full complex")
            if af2_pae.shape != (aa_len, aa_len):
                raise ValueError(
                    "candidate_interface_v1 PAE must cover the full complex")
            batch["af2_plddt"] = af2_plddt
            batch["af2_iptm"] = entry["af2_iptm"]
            batch["af2_pae_matrix"] = af2_pae
            batch["af2_pae_normalized"] = torch.tensor(
                bool(entry["af2_pae_normalized"]), dtype=torch.bool)
            batch["confidence_sample_weight"] = torch.tensor(
                float(entry.get("confidence_sample_weight", 1.0)),
                dtype=torch.float32)
            for key in (
                    "af2_iptm_std", "af2_candidate_plddt_std",
                    "af2_interface_pae_normalized_std", "af2_iptm_sem",
                    "af2_candidate_plddt_sem",
                    "af2_interface_pae_normalized_sem"):
                if key in entry:
                    batch[key] = torch.tensor(float(entry[key]), dtype=torch.float32)
        else:
            self._attach_legacy_confidence_targets(batch, entry, aa_len)
            # Legacy confidence regression treats the complete structure as context.
            batch["generate_flag"] = torch.zeros(
                batch["aa"].shape[0], dtype=torch.bool)

        batch["pdb_id"] = entry.get("pdb_id", "")
        batch["construct_id"] = entry.get("construct_id", "")
        batch["protocol_id"] = entry.get("protocol_id", "")
        batch["af2_seed"] = torch.tensor(
            int(entry.get("af2_seed", -1)), dtype=torch.long)
        batch["scaffold_family"] = entry.get("scaffold_family", "")
        batch["is_idp"] = entry.get("is_idp", False)
        batch["source"] = entry.get("source", "")

        # V14 grouped-contrastive mode: design variants of the same scaffold carry
        # a scaffold_id (int) so the grouped margin/variance losses
        # (disorderflow/modules/bfn/core.py) can compare designs that share a
        # backbone. Stored per-entry by build_design_variant_dataset.py.
        # Legacy entries (no scaffold_id) get a unique id -> singletons.
        batch["scaffold_id"] = torch.tensor(
            int(entry.get("scaffold_id", real_idx)), dtype=torch.long
        )

        # Per-residue disorder labels from EBI MobiDB-lite (Phase 2).
        # Falls back to AF2 pLDDT < 50 for entries where API was unavailable.
        disorder_mask = batch.get("disorder_mask", None)
        if disorder_mask is not None:
            batch["disorder_label"] = disorder_mask.float()
        else:
            # Legacy: no disorder_mask, use is_idp broadcast (Phase 1 style)
            is_idp = batch.get("is_idp", False)
            if isinstance(is_idp, torch.Tensor):
                is_idp = is_idp.item() if is_idp.numel() == 1 else is_idp[0].item()
            seq_len = batch["aa"].shape[0]
            batch["disorder_label"] = (
                torch.ones(seq_len, dtype=torch.float32)
                if is_idp
                else torch.zeros(seq_len, dtype=torch.float32)
            )

        return batch

    @staticmethod
    def _attach_legacy_confidence_targets(batch, entry, aa_len):
        # AF2 multimer runs on antibody + epitope → plddt/pae cover the FULL
        # complex. Truncate to the antibody portion for training.
        af2_plddt = entry["af2_plddt"]
        if af2_plddt.shape[0] > aa_len:
            af2_plddt = af2_plddt[:aa_len]
        batch["af2_plddt"] = af2_plddt
        batch["af2_iptm"] = entry["af2_iptm"]
        af2_pae = entry["af2_pae_matrix"]
        if af2_pae.dim() >= 2 and af2_pae.shape[0] > aa_len:
            af2_pae = af2_pae[:aa_len, :aa_len]
        batch["af2_pae_matrix"] = af2_pae

    def set_max_residues(self, max_residues):
        """Dynamically update the max_residues filter (for length curriculum).

        Rebuilds _valid_indices using the original full index set, then
        re-filters for the new max_residues. Safe to call mid-training when
        num_workers=0.
        """
        self.max_residues = max_residues
        full_indices = list(range(self._length))
        if max_residues <= 0:
            self._valid_indices = full_indices
            return
        self._valid_indices = []
        if self._use_lmdb:
            with self._env.begin() as txn:
                for i in full_indices:
                    key = f"{i:08d}".encode()
                    entry = pickle.loads(txn.get(key))
                    if len(entry.get("sequence", "")) <= max_residues:
                        self._valid_indices.append(i)
        else:
            for i in full_indices:
                pkl_path = os.path.join(self._pkl_dir, f"{i:08d}.pkl")
                with open(pkl_path, "rb") as f:
                    entry = pickle.load(f)
                if len(entry.get("sequence", "")) <= max_residues:
                    self._valid_indices.append(i)

    def close(self):
        if self._use_lmdb and hasattr(self, "_env"):
            _release_lmdb(self._env_cache_key)
            del self._env
