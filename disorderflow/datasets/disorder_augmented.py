"""Disorder-Augmented Dataset Wrapper (WAY4 V7 P0-fix-B).

Wraps Phase3Dataset to inject per-residue disorder profiles (epitope_disorder_profile)
into each training sample. The per-residue disorder comes from a pre-computed lookup
dict mapping SAbDab sample IDs to disorder arrays.

Used by: train.py with --disorder-lookup flag
"""

import numpy as np
import torch


def _supervision_arrays(entry):
    """Return values, evidence mask, and confidence for old or v3 lookups."""
    if entry is None:
        return None, None, None
    if isinstance(entry, dict) and 'values' in entry:
        values = np.asarray(entry['values'], dtype=np.float32)
        mask = np.asarray(entry.get('mask', np.isfinite(values)), dtype=bool)
        confidence = np.asarray(entry.get('confidence', np.ones_like(values)), dtype=np.float32)
        return values, mask, confidence
    values = np.asarray(entry, dtype=np.float32)
    return values, np.isfinite(values), np.ones_like(values, dtype=np.float32)


class DisorderAugmentedDataset:
    """Wraps a Phase3Dataset to add per-residue epitope disorder profiles.

    For each sample:
    - Looks up the pre-computed per-residue disorder array by sample ID
    - Builds epitope_disorder_profile: zeros for antibody, disorder for antigen
    - The profile has shape (L,) where L is the patched sequence length
    """

    def __init__(self, base_dataset, disorder_lookup, sample_ids, balance_boost=0.0):
        """
        Args:
            base_dataset: Phase3Dataset instance
            disorder_lookup: dict {sample_id: np.ndarray of per-residue disorder [0,1]}
            sample_ids: list of sample IDs in same order as base_dataset
        """
        self._base = base_dataset
        self._lookup = disorder_lookup
        self._ids = sample_ids
        self._sample_weights = self._build_sample_weights(float(balance_boost))

    def _build_sample_weights(self, boost):
        if boost <= 0:
            return getattr(self._base, '_sample_weights', None)
        weights = np.ones(len(self._ids), dtype=np.float64)
        for index, sample_id in enumerate(self._ids):
            entry = self._lookup.get(sample_id)
            if entry is None:
                entry = self._lookup.get(str(sample_id).casefold())
            values, mask, confidence = _supervision_arrays(entry)
            if values is None or not mask.any():
                continue
            coverage = float(mask.mean())
            evidence = float(confidence[mask].mean())
            disorder = float(values[mask].mean())
            weights[index] += boost * coverage * evidence * disorder
        return weights

    def __getitem__(self, idx):
        sid = self._ids[idx]
        disorder_arr = self._lookup.get(sid) if self._lookup else None
        if disorder_arr is None and self._lookup:
            disorder_arr = self._lookup.get(str(sid).casefold())

        # Phase3 transforms merge and crop chains. Attach the profile to the raw
        # antigen first so those transforms preserve the exact residue mapping.
        if hasattr(self._base, 'get_structure') and hasattr(self._base, 'transform'):
            structure = self._base.get_structure(idx)
            if structure is not None and structure.get('antigen') is not None:
                antigen = structure['antigen']
                profile = torch.zeros(antigen['aa'].shape[0], dtype=torch.float32)
                evidence_mask = torch.zeros_like(profile, dtype=torch.bool)
                confidence = torch.zeros_like(profile)
                values, source_mask, source_confidence = _supervision_arrays(disorder_arr)
                if values is not None and len(values) > 0:
                    n_copy = min(profile.shape[0], len(values))
                    profile[:n_copy] = torch.as_tensor(
                        values[:n_copy], dtype=torch.float32)
                    evidence_mask[:n_copy] = torch.as_tensor(source_mask[:n_copy], dtype=torch.bool)
                    confidence[:n_copy] = torch.as_tensor(source_confidence[:n_copy], dtype=torch.float32)
                antigen['epitope_disorder_profile'] = profile
                antigen['disorder_supervision_mask'] = evidence_mask
                antigen['disorder_confidence'] = confidence
                data = self._base.transform(structure) if self._base.transform else structure
                if data is not None and 'aa' in data:
                    # Inject disorder_label for the disorder head loss
                    if 'epitope_disorder_profile' in data:
                        data['disorder_label'] = data['epitope_disorder_profile'].clone()
                        data['disorder_supervision_mask'] = data.get(
                            'disorder_supervision_mask', torch.zeros_like(data['aa'], dtype=torch.bool))
                        data['disorder_confidence'] = data.get(
                            'disorder_confidence', torch.zeros_like(data['disorder_label']))
                    else:
                        data['disorder_label'] = torch.zeros(
                            data['aa'].shape[0], dtype=torch.float32)
                    return self._add_negative_profiles(self._add_scalar_mean(data), idx)
                # Transform failed — fall through to fallback path with retry

        data = self._base[idx]

        # Phase3Dataset transforms can fail (return None); retry with neighbor
        if data is None or 'aa' not in data:
            import random
            for _retry in range(5):
                alt_idx = random.randint(0, len(self._base) - 1)
                data = self._base[alt_idx]
                if data is not None and 'aa' in data:
                    sid = self._ids[alt_idx] if alt_idx < len(self._ids) else sid
                    disorder_arr = self._lookup.get(sid) if self._lookup else None
                    break
            else:
                raise RuntimeError(f'All retries failed for sample {idx}')

        # Always add profile (all zeros if no disorder data available)
        # This ensures PaddingCollate._get_common_keys finds the field in ALL samples
        fragment_type = data.get('fragment_type')
        L = data['aa'].shape[0]
        profile = torch.zeros(L, dtype=torch.float32)
        supervision_mask = torch.zeros(L, dtype=torch.bool)
        confidence = torch.zeros(L, dtype=torch.float32)

        values, source_mask, source_confidence = _supervision_arrays(disorder_arr)
        if values is not None and len(values) > 0:
            ag_indices = torch.empty(0, dtype=torch.long)
            if fragment_type is not None:
                ag_indices = (fragment_type == 3).nonzero(as_tuple=True)[0]
            # Confidence/IDP LMDB records are single proteins rather than
            # antibody-antigen complexes, so full-length evidence maps directly.
            target_indices = (ag_indices if len(ag_indices) else
                              torch.arange(L) if len(values) == L else ag_indices)
            if len(target_indices):
                n_copy = min(len(target_indices), len(values))
                profile[target_indices[:n_copy]] = torch.tensor(
                    values[:n_copy], dtype=torch.float32
                )
                supervision_mask[target_indices[:n_copy]] = torch.as_tensor(
                    source_mask[:n_copy], dtype=torch.bool)
                confidence[target_indices[:n_copy]] = torch.as_tensor(
                    source_confidence[:n_copy], dtype=torch.float32)

        data['epitope_disorder_profile'] = profile
        # Also inject disorder_label for the disorder HEAD loss (core.py lines 394-432).
        # Without this key in every sample, PaddingCollate drops it and the loss never fires.
        data['disorder_label'] = profile.clone()
        data['disorder_supervision_mask'] = supervision_mask
        data['disorder_confidence'] = confidence

        return self._add_negative_profiles(self._add_scalar_mean(data), idx)

    @staticmethod
    def _add_scalar_mean(data):
        profile = data['epitope_disorder_profile']
        fragment_type = data.get('fragment_type')
        if fragment_type is not None:
            ag_mask = (fragment_type == 3)
            ag_profile = profile[ag_mask]
            ag_mean = (ag_profile.mean().item() if ag_profile.numel() > 0
                       else profile.mean().item())
        else:
            ag_mean = profile.mean().item()
        data['epitope_disorder'] = torch.tensor([[ag_mean]], dtype=torch.float32)
        return data

    def _add_negative_profiles(self, data, idx):
        """Attach deterministic position-shuffled and cross-sample profiles."""
        factual = data['epitope_disorder_profile']
        fragment_type = data.get('fragment_type')
        shuffled = torch.zeros_like(factual)
        if fragment_type is not None:
            antigen_indices = torch.where(fragment_type == 3)[0]
            values = factual[antigen_indices]
            if len(values) > 1:
                # A half-length derangement preserves the exact value
                # distribution while avoiding near-identical one-step shifts
                # on locally smooth disorder profiles.
                shuffled[antigen_indices] = values.roll(max(1, len(values) // 2))
            elif len(values) == 1:
                shuffled[antigen_indices] = values
        data['epitope_disorder_shuffled_profile'] = shuffled

        donor_idx = (idx + max(1, len(self) // 2)) % len(self)
        donor_id = self._ids[donor_idx]
        donor = self._lookup.get(donor_id) if self._lookup else None
        if donor is None and self._lookup:
            donor = self._lookup.get(str(donor_id).casefold())
        mismatch = torch.zeros_like(data['epitope_disorder_profile'])
        donor_values, donor_mask, _ = _supervision_arrays(donor)
        if donor_values is not None and donor_mask.any() and fragment_type is not None:
            antigen_indices = torch.where(fragment_type == 3)[0]
            donor_values = donor_values[donor_mask]
            if len(antigen_indices):
                source_x = np.linspace(0.0, 1.0, len(donor_values))
                target_x = np.linspace(0.0, 1.0, len(antigen_indices))
                mapped = np.interp(target_x, source_x, donor_values)
                # Match the factual empirical distribution exactly. Donor rank
                # order remains cross-sample, but global mean/variance can no
                # longer reveal which arm is negative.
                factual_values = factual[antigen_indices].cpu().numpy()
                donor_order = np.argsort(mapped, kind='stable')
                matched = np.empty_like(mapped, dtype=np.float32)
                matched[donor_order] = np.sort(factual_values)
                mismatch[antigen_indices] = torch.as_tensor(matched, dtype=torch.float32)
        data['epitope_disorder_mismatched_profile'] = mismatch
        return data

    def __len__(self):
        return len(self._base)

    def __getattr__(self, name):
        # Delegate all other attributes to base dataset
        # This ensures _scaffold_ids, _sample_weights, set_max_residues, etc. work
        return getattr(self._base, name)


class ContrastiveNegativeDataset:
    """Attach a real CDR from another complex as an antigen-mismatched negative."""

    def __init__(self, base_dataset):
        self._base = base_dataset

    def __getitem__(self, idx):
        data = self._base[idx]
        donor_idx = (idx + max(1, len(self) // 2)) % len(self)
        donor = self._base[donor_idx]
        current_idx = torch.where(data['generate_flag'].bool())[0]
        donor_aa = donor['aa'][donor['generate_flag'].bool()]
        negative = data['aa'].clone()
        if len(current_idx) and len(donor_aa):
            source_idx = torch.linspace(
                0, len(donor_aa) - 1, len(current_idx)).round().long()
            negative[current_idx] = donor_aa[source_idx]
        data['contrastive_negative_aa'] = negative
        return data

    def __len__(self):
        return len(self._base)

    def __getattr__(self, name):
        return getattr(self._base, name)


class DisorderBalancedSampler:
    """Weighted sampler that oversamples high-disorder entries.

    Addresses the severe class imbalance (0.74% disordered residues) by
    assigning higher sampling probability to entries with flexible antigens.

    Usage with DataLoader:
        sampler = DisorderBalancedSampler(dataset, lookup, ids, boost=5.0)
        loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(self, dataset, disorder_lookup, sample_ids, boost=5.0,
                 batch_size=8, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        n = len(dataset)

        # Compute per-sample mean disorder
        weights = np.ones(n, dtype=np.float64)
        for i in range(n):
            sid = sample_ids[i] if i < len(sample_ids) else None
            arr = disorder_lookup.get(sid) if (disorder_lookup and sid) else None
            values, mask, _ = _supervision_arrays(arr)
            if values is not None and mask.any():
                mean_d = float(np.mean(values[mask]))
                weights[i] = 1.0 + boost * mean_d

        # Normalize to probability distribution
        self.weights = weights / weights.sum()
        self.n = n

    def __iter__(self):
        # Draw indices with replacement according to weights
        indices = np.random.choice(self.n, size=self.n, replace=True, p=self.weights)
        if self.shuffle:
            np.random.shuffle(indices)
        # Yield batches
        for start in range(0, len(indices), self.batch_size):
            yield indices[start:start + self.batch_size].tolist()

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


def load_disorder_lookup(path, expected_split=None, expected_ids=None, require_envelope=False):
    """Load a disorder lookup and validate split-scoped artifacts when present."""
    import hashlib
    import pickle
    with open(path, 'rb') as f:
        artifact = pickle.load(f)
    if not isinstance(artifact, dict):
        raise ValueError(f'Invalid disorder lookup artifact: {path}')
    if 'profiles' not in artifact:
        if require_envelope:
            raise ValueError(f'Disorder lookup must be a split-scoped artifact: {path}')
        return artifact
    if expected_split is not None and artifact.get('split') != expected_split:
        raise ValueError(
            f"Disorder lookup split mismatch: expected {expected_split}, got {artifact.get('split')}")
    profiles = artifact['profiles']
    if expected_ids is not None:
        normalized = [str(value) for value in expected_ids]
        expected_set = set(map(str.casefold, normalized))
        profile_set = set(map(str.casefold, profiles))
        if artifact.get('schema_version', 1) >= 3:
            if not expected_set & profile_set:
                raise ValueError(f'Disorder lookup has no IDs in the requested dataset: {path}')
        else:
            digest = hashlib.sha256('\n'.join(normalized).encode('utf-8')).hexdigest()
            if artifact.get('ids_sha256') != digest:
                raise ValueError(f'Disorder lookup ID fingerprint mismatch: {path}')
            if expected_set != profile_set:
                raise ValueError(f'Disorder lookup ID set mismatch: {path}')
    if artifact.get('schema_version', 1) >= 3:
        return profiles
    source = artifact.get('source_contract', {}).get('primary_source')
    confidence = {
        'charge_hydropathy_heuristic_v1': 0.10,
    }.get(source, 0.25 if source else 1.0)
    return {
        key: {
            'values': np.asarray(value, dtype=np.float32),
            'mask': np.isfinite(value),
            'confidence': np.full(len(value), confidence, dtype=np.float32),
        }
        for key, value in profiles.items()
    }
