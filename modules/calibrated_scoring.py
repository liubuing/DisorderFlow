#!/usr/bin/env python
"""Integrated calibrated scoring module for BFN design pipeline.

Drop-in replacement for raw BFN confidence scores. Combines:
  - Isotonic calibration for ipTM magnitude (OC 6x→1x)
  - Sequence calibrator MLP for pLDDT (Spearman=0.65)
  - CDR Transformer for sequence-context-aware prediction (pLDDT=0.584, ipTM=0.486)
  - PPL as primary ranking signal (orthogonal to confidence)

Ensemble strategy:
  pLDDT = 0.5 * seq_MLP + 0.5 * CDR_Transformer  (both sequence-based, complementary)
  ipTM  = 0.5 * isotonic + 0.5 * CDR_Transformer  (magnitude + discrimination)

Usage in design pipeline:
    from modules.calibrated_scoring import CalibratedScorer

    scorer = CalibratedScorer()  # auto-loads artifacts
    results = scorer.rerank(bfn_results_list)
    # Each result now has: cal_plddt, cal_iptm, composite_score

Or score a single design:
    cal_plddt, cal_iptm, score = scorer.score(
        bfn_plddt=0.267, bfn_iptm=0.338, ppl=61.0, sequence="QVQLVES..."
    )
"""

import pickle
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
ART_DIR = PROJECT / "calibration_artifacts"


class CalibratedScorer:
    """Post-hoc calibration layer for BFN confidence outputs."""

    def __init__(self, artifacts_dir=None, device='cpu'):
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else ART_DIR
        self.device = device
        self._isotonic = None
        self._ridge = None
        self._seq_model = None
        self._cdr_transformer = None
        self._load_artifacts()

    def _load_artifacts(self):
        # Isotonic + Ridge
        pkl_path = self.artifacts_dir / "calibration_functions.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                calibrators = pickle.load(f)
            self._isotonic = {
                'plddt': calibrators['isotonic_plddt'],
                'iptm': calibrators['isotonic_iptm'],
            }
            self._ridge = {
                'plddt': calibrators['ridge_plddt'],
                'iptm': calibrators['ridge_iptm'],
            }

        # Sequence calibrator
        for name in ["sequence_calibrator_best.pt", "sequence_calibrator_final.pt"]:
            pt_path = self.artifacts_dir / name
            if pt_path.exists():
                try:
                    import sys
                    sys.path.insert(0, str(PROJECT))
                    from scripts.finetune_confidence_head_v2 import SequenceCalibrator
                    ckpt = torch.load(pt_path, map_location=self.device, weights_only=False)
                    cfg = ckpt['config']
                    model = SequenceCalibrator(
                        hidden_dim=cfg['hidden_dim'],
                        num_layers=cfg['num_layers'],
                        dropout=0.0)
                    model.load_state_dict(ckpt['model_state'])
                    model.eval()
                    self._seq_model = model
                    break
                except Exception:
                    pass

        # CDR Transformer
        cdr_path = self.artifacts_dir / "cdr_transformer_best.pt"
        if cdr_path.exists():
            try:
                import sys
                sys.path.insert(0, str(PROJECT))
                from scripts.cdr_transformer_calibrator import CDRTransformerCalibrator
                ckpt = torch.load(cdr_path, map_location=self.device, weights_only=False)
                cfg = ckpt['config']
                model = CDRTransformerCalibrator(
                    d_model=cfg['d_model'], nhead=cfg['nhead'],
                    num_layers=cfg['num_layers'], dropout=0.0,
                    max_len=cfg['max_len'])
                model.load_state_dict(ckpt['model_state'])
                model.eval()
                self._cdr_transformer = model
            except Exception:
                pass

    def calibrate_plddt(self, bfn_plddt, sequence=None):
        """Calibrate pLDDT via ensemble of sequence-based methods.

        Ensemble: 0.5 * seq_MLP (Spearman=0.65) + 0.5 * CDR_Transformer (0.584).
        Falls back to whichever is available; last resort = isotonic.
        """
        scores = []
        if self._seq_model is not None and sequence and len(sequence) > 5:
            scores.append(self._seq_predict_plddt(sequence))
        if self._cdr_transformer is not None and sequence and len(sequence) > 5:
            p, _ = self._cdr_transformer_predict(sequence)
            scores.append(p)
        if scores:
            return float(np.mean(scores))
        if self._isotonic:
            return float(self._isotonic['plddt'].predict(np.array([bfn_plddt]))[0])
        return bfn_plddt

    def calibrate_iptm(self, bfn_iptm, sequence=None):
        """Calibrate ipTM via ensemble of isotonic + CDR Transformer.

        Ensemble: 0.5 * isotonic (magnitude fix, Spearman=0.62)
                + 0.5 * CDR_Transformer (sequence discrimination, 0.486).
        """
        scores = []
        if self._isotonic:
            scores.append(float(self._isotonic['iptm'].predict(np.array([bfn_iptm]))[0]))
        if self._cdr_transformer is not None and sequence and len(sequence) > 5:
            _, i = self._cdr_transformer_predict(sequence)
            scores.append(i)
        if scores:
            return float(np.mean(scores))
        return bfn_iptm

    def _cdr_transformer_predict(self, sequence):
        """Run CDR Transformer for (pLDDT, ipTM) prediction from sequence."""
        import torch.nn.functional as F
        aa_letters = 'ACDEFGHIKLMNPQRSTVWY'
        max_len = self._cdr_transformer.max_len
        seq = sequence[:max_len]
        L = len(seq)

        # One-hot encode
        aa_idx = torch.tensor([[aa_letters.index(c) if c in aa_letters else 0
                                for c in seq] + [0] * (max_len - L)],
                              dtype=torch.long, device=self.device)
        aa_onehot = F.one_hot(aa_idx.clamp(0, 19), num_classes=20).float()

        # CDR type assignment (canonical VHH positions)
        cdr_types = torch.zeros(1, max_len, dtype=torch.long, device=self.device)
        for start, end, cdr_id in [(25, 33, 1), (50, 58, 2), (96, 113, 3)]:
            if start < L:
                cdr_types[0, start:min(end, L)] = cdr_id

        # Mask
        mask = torch.zeros(1, max_len, dtype=torch.bool, device=self.device)
        mask[0, :L] = True

        with torch.no_grad():
            p, i = self._cdr_transformer(aa_onehot, cdr_types, mask)
        return float(p.item()), float(i.item())

    def _seq_predict_plddt(self, sequence):
        """Run sequence calibrator for pLDDT prediction."""
        aa_letters = 'ACDEFGHIKLMNPQRSTVWY'
        seq = sequence[:512]
        aa = torch.tensor([[aa_letters.index(c) if c in aa_letters else 0
                           for c in seq]], dtype=torch.long, device=self.device)
        mask = torch.ones(1, len(seq), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            p, _ = self._seq_model(aa, mask)
        return float(p.item())

    def score(self, bfn_plddt, bfn_iptm, ppl, sequence=None):
        """Compute calibrated scores and composite ranking score.

        Returns:
            (cal_plddt, cal_iptm, composite_score)
        """
        cal_plddt = self.calibrate_plddt(bfn_plddt, sequence)
        cal_iptm = self.calibrate_iptm(bfn_iptm, sequence)

        # Composite: weighted combination favoring PPL (most discriminative)
        # PPL is inverted (lower = better), normalize to [0,1] range
        # Typical PPL range: 20-150 for BFN designs
        ppl = ppl if ppl is not None else 50.0
        ppl_score = 1.0 - np.clip((np.log1p(ppl) - np.log1p(20)) /
                                   (np.log1p(150) - np.log1p(20)), 0, 1)

        composite = (0.35 * ppl_score
                     + 0.35 * cal_iptm
                     + 0.30 * cal_plddt)

        return cal_plddt, cal_iptm, float(composite)

    def rerank(self, results_list, sort_key='composite_score', descending=True):
        """Re-score and re-rank a list of BFN design results.

        Each dict in results_list should have: plddt, iptm, ppl, sequence.
        Adds: cal_plddt, cal_iptm, composite_score.
        Returns sorted list.
        """
        for r in results_list:
            cal_p, cal_i, comp = self.score(
                bfn_plddt=r.get('plddt', 0.5),
                bfn_iptm=r.get('iptm', 0.5),
                ppl=r.get('ppl', 50.0),
                sequence=r.get('sequence', None),
            )
            r['cal_plddt'] = cal_p
            r['cal_iptm'] = cal_i
            r['composite_score'] = comp

        results_list.sort(key=lambda x: x.get(sort_key, 0), reverse=descending)
        return results_list

    @property
    def available_methods(self):
        methods = []
        if self._isotonic:
            methods.append("isotonic")
        if self._ridge:
            methods.append("ridge_ppl")
        if self._seq_model:
            methods.append("sequence_calibrator")
        if self._cdr_transformer:
            methods.append("cdr_transformer")
        return methods

    def __repr__(self):
        return (f"CalibratedScorer(methods={self.available_methods}, "
                f"artifacts={self.artifacts_dir})")


# ─── Quick self-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    scorer = CalibratedScorer()
    print(scorer)
    print(f"Available methods: {scorer.available_methods}")

    # Test with V14 design #1
    cal_p, cal_i, comp = scorer.score(
        bfn_plddt=0.2668, bfn_iptm=0.3379, ppl=61.0,
        sequence="QVQLVESGGGLVQAGGSLRLSCAASDDASAERPMGWFRQAPGKEREFVAADLKDLKWLYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCWSQNWLYTDTGDRPTDVYWGQGTQVTVSS"
    )
    print(f"\nV14 design #1:")
    print(f"  Raw BFN:      pLDDT=0.2668, ipTM=0.3379, PPL=61.0")
    print(f"  AF2 truth:    pLDDT=0.2661, ipTM=0.0528")
    print(f"  Calibrated:   pLDDT={cal_p:.4f}, ipTM={cal_i:.4f}")
    print(f"  Composite:    {comp:.4f}")
