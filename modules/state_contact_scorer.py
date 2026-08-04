#!/usr/bin/env python
"""Structure/contact-aware state specificity scorer.

This v2 scorer uses the contact topology from a known antibody-peptide complex.
Unlike the sequence-only v1 scorer, it is position sensitive: a candidate is
scored at each native paratope contact position against the epitope residues it
contacts. Composition-matched scrambled controls should lose score when hotspot
chemistry is moved to the wrong positions.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


AA = "ACDEFGHIKLMNPQRSTVWY"
AROMATIC = set("FWY")
HYDROPHOBIC = set("AILMFWYV")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
POLAR = set("STNQYH")

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Minimal BLOSUM-like identity/conservative substitution signal. Kept small so
# interface chemistry and distance-weighted contacts remain primary.
CONSERVATIVE_GROUPS = [
    set("FWY"), set("ILMV"), set("KRH"), set("DE"), set("STNQ"), set("AG"),
]


@dataclass(frozen=True)
class Contact:
    paratope_index: int
    paratope_chain: str
    paratope_resid: int
    native_aa: str
    epitope_index: int
    epitope_chain: str
    epitope_resid: int
    epitope_aa: str
    distance: float


ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


def select_abeta_like_chain(chains: Dict[str, List[Dict]], min_identity: float = 0.75) -> Dict:
    """Select the chain that best matches an A-beta42 contiguous fragment.

    Returns metadata including the chosen chain. If no chain is A-beta-like,
    falls back to the shortest chain but marks `is_abeta_like=False`.
    """
    scored = []
    for cid, residues in chains.items():
        seq = "".join(r["aa"] for r in residues)
        info = abeta_fragment_info(seq)
        scored.append({"chain": cid, "sequence": seq, "length": len(seq), **info})
    abeta_like = [s for s in scored if s["is_abeta_like"]]
    if abeta_like:
        # Prefer exact/high identity A-beta fragments, then shorter peptide-like chains.
        chosen = sorted(abeta_like, key=lambda s: (-s["best_identity"], s["length"]))[0]
    else:
        chosen = sorted(scored, key=lambda s: s["length"])[0]
    return {"chosen_chain": chosen["chain"], "chains": scored, "chosen": chosen}


def abeta_fragment_info(seq: str) -> Dict:
    """Return whether seq is an A-beta42-like contiguous fragment."""
    seq = _clean(seq)
    if not seq:
        return {"is_abeta_like": False, "abeta_start": None, "abeta_end": None,
                "abeta_coverage": 0.0, "best_identity": 0.0}
    idx = ABETA42.find(seq)
    if idx >= 0:
        return {"is_abeta_like": True, "abeta_start": idx + 1,
                "abeta_end": idx + len(seq), "abeta_coverage": round(len(seq) / 42, 4),
                "best_identity": 1.0}
    if len(seq) <= len(ABETA42):
        best = 0.0
        best_start = None
        for i in range(len(ABETA42) - len(seq) + 1):
            window = ABETA42[i:i + len(seq)]
            ident = sum(a == b for a, b in zip(seq, window)) / len(seq)
            if ident > best:
                best = ident
                best_start = i + 1
        # Require a reasonably peptide-like chain and high local identity.
        is_like = best >= 0.75 and len(seq) <= 42
        return {"is_abeta_like": is_like, "abeta_start": best_start,
                "abeta_end": (best_start + len(seq) - 1) if best_start else None,
                "abeta_coverage": round(min(len(seq), 42) / 42, 4),
                "best_identity": round(best, 4)}
    best = 0.0
    best_start = None
    for i in range(len(seq) - len(ABETA42) + 1):
        window = seq[i:i + len(ABETA42)]
        ident = sum(a == b for a, b in zip(window, ABETA42)) / 42
        if ident > best:
            best = ident
            best_start = i + 1
    return {"is_abeta_like": best >= 0.75, "abeta_start": best_start,
            "abeta_end": (best_start + 41) if best_start else None,
            "abeta_coverage": 1.0, "best_identity": round(best, 4)}


def extract_contact_map(pdb_path: str, cutoff: float = 8.0, peptide_chain: str = None) -> Dict:
    """Extract peptide-contacting paratope residues and contact pairs.

    The shortest protein chain is treated as the peptide/epitope chain.
    Paratope positions are all antibody residues with at least one peptide
    residue within cutoff, ordered by (chain, residue number).
    """
    import numpy as np
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    chains = {}
    for model in structure:
        for chain in model:
            residues = []
            for res in chain:
                aa = AA3_TO_1.get(res.resname.strip())
                if not aa:
                    continue
                atom = res["CB"] if "CB" in res else res["CA"] if "CA" in res else None
                if atom is None:
                    continue
                residues.append({
                    "chain": chain.id.strip() or "_",
                    "resid": res.id[1],
                    "aa": aa,
                    "coord": atom.get_coord(),
                })
            if residues:
                chains[chain.id.strip() or "_"] = residues
        break
    if len(chains) < 2:
        raise ValueError(f"Need at least two protein chains in {pdb_path}")

    chain_selection = select_abeta_like_chain(chains)
    peptide_chain = peptide_chain or chain_selection["chosen_chain"]
    if peptide_chain not in chains:
        raise ValueError(f"Peptide chain {peptide_chain!r} not found in {pdb_path}")
    peptide = chains[peptide_chain]
    pep_coords = np.array([r["coord"] for r in peptide])

    paratope_res = []
    contact_pairs: List[Contact] = []
    paratope_lookup = {}
    for cid, residues in chains.items():
        if cid == peptide_chain:
            continue
        for r in residues:
            dists = np.linalg.norm(pep_coords - r["coord"], axis=1)
            close = [(j, float(d)) for j, d in enumerate(dists) if d <= cutoff]
            if not close:
                continue
            key = (r["chain"], r["resid"])
            if key not in paratope_lookup:
                paratope_lookup[key] = len(paratope_res)
                paratope_res.append(r)
            pidx = paratope_lookup[key]
            for j, d in close:
                e = peptide[j]
                contact_pairs.append(Contact(
                    paratope_index=pidx,
                    paratope_chain=r["chain"],
                    paratope_resid=r["resid"],
                    native_aa=r["aa"],
                    epitope_index=j,
                    epitope_chain=e["chain"],
                    epitope_resid=e["resid"],
                    epitope_aa=e["aa"],
                    distance=d,
                ))

    # paratope_lookup was insertion-order by chain scan. Reorder positions to be
    # deterministic and remap contact indices accordingly.
    ordered = sorted(enumerate(paratope_res), key=lambda x: (x[1]["chain"], x[1]["resid"]))
    remap = {old_i: new_i for new_i, (old_i, _) in enumerate(ordered)}
    ordered_res = [r for _, r in ordered]
    ordered_contacts = [
        Contact(
            paratope_index=remap[c.paratope_index],
            paratope_chain=c.paratope_chain,
            paratope_resid=c.paratope_resid,
            native_aa=c.native_aa,
            epitope_index=c.epitope_index,
            epitope_chain=c.epitope_chain,
            epitope_resid=c.epitope_resid,
            epitope_aa=c.epitope_aa,
            distance=c.distance,
        )
        for c in contact_pairs
    ]
    ordered_contacts.sort(key=lambda c: (c.paratope_index, c.distance, c.epitope_index))

    return {
        "pdb_path": pdb_path,
        "peptide_chain": peptide_chain,
        "peptide_sequence": "".join(r["aa"] for r in peptide),
        "paratope_sequence": "".join(r["aa"] for r in ordered_res),
        "paratope_residues": [
            {"chain": r["chain"], "resid": r["resid"], "aa": r["aa"]}
            for r in ordered_res
        ],
        "contacts": ordered_contacts,
        "chain_lengths": {cid: len(v) for cid, v in chains.items()},
        "chain_selection": chain_selection,
        "cutoff": cutoff,
    }


def score_sequence_on_contact_map(sequence: str, contact_map: Dict) -> Dict:
    """Score a paratope sequence against a native contact topology."""
    seq = _clean(sequence)
    native = contact_map["paratope_sequence"]
    if len(seq) != len(native):
        raise ValueError(f"Candidate length {len(seq)} != contact-map length {len(native)}")
    contacts: Sequence[Contact] = contact_map["contacts"]
    if not contacts:
        return _empty_result(seq)

    total_w = 0.0
    total = 0.0
    native_total = 0.0
    hotspot_total = 0.0
    hotspot_w = 0.0
    per_position = [0.0 for _ in seq]
    per_position_w = [0.0 for _ in seq]

    for c in contacts:
        aa = seq[c.paratope_index]
        w = _distance_weight(c.distance)
        chem = _pair_score(aa, c.epitope_aa)
        native_chem = _pair_score(c.native_aa, c.epitope_aa)
        subst = _substitution_score(aa, c.native_aa)
        # Native-like chemistry at known contact positions is rewarded, but a
        # conservative replacement can still score if it preserves pair chemistry.
        score = 0.70 * chem + 0.30 * subst * native_chem
        total += w * score
        native_total += w * native_chem
        total_w += w
        per_position[c.paratope_index] += w * score
        per_position_w[c.paratope_index] += w
        if native_chem >= 0.45 or c.distance <= 5.0:
            hotspot_total += w * score
            hotspot_w += w

    contact_score = total / max(total_w, 1e-9)
    native_reference = native_total / max(total_w, 1e-9)
    hotspot_score = hotspot_total / max(hotspot_w, 1e-9) if hotspot_w else contact_score
    position_scores = [
        round(per_position[i] / per_position_w[i], 4) if per_position_w[i] else 0.0
        for i in range(len(seq))
    ]
    identity = sum(1 for a, b in zip(seq, native) if a == b) / max(1, len(seq))
    conservative = sum(_substitution_score(a, b) for a, b in zip(seq, native)) / max(1, len(seq))
    final = 0.65 * contact_score + 0.25 * hotspot_score + 0.10 * conservative
    return {
        "sequence": seq,
        "native_sequence": native,
        "contact_score": round(contact_score, 4),
        "hotspot_score": round(hotspot_score, 4),
        "native_reference_score": round(native_reference, 4),
        "identity_to_native": round(identity, 4),
        "conservative_similarity": round(conservative, 4),
        "state_contact_score": round(final, 4),
        "n_contacts": len(contacts),
        "n_paratope_positions": len(seq),
        "position_scores": position_scores,
    }


def scrambled_controls(seq: str, n: int = 100, seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    chars = list(_clean(seq))
    seen = {"".join(chars)}
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 80:
        attempts += 1
        c = chars[:]
        rng.shuffle(c)
        s = "".join(c)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def benchmark_native_vs_scrambled(contact_map: Dict, n_scramble: int = 100, seed: int = 42) -> Dict:
    native_seq = contact_map["paratope_sequence"]
    native = score_sequence_on_contact_map(native_seq, contact_map)
    scrambles = [
        score_sequence_on_contact_map(s, contact_map)
        for s in scrambled_controls(native_seq, n_scramble, seed)
    ]
    vals = [s["state_contact_score"] for s in scrambles]
    hs = [s["hotspot_score"] for s in scrambles]
    n = max(1, len(vals))
    native_percentile = 1.0 - sum(1 for v in vals if v >= native["state_contact_score"]) / n
    hotspot_percentile = 1.0 - sum(1 for v in hs if v >= native["hotspot_score"]) / n
    return {
        "native": native,
        "scrambles": scrambles,
        "summary": {
            "native_score": native["state_contact_score"],
            "native_hotspot": native["hotspot_score"],
            "scramble_mean_score": round(sum(vals) / n, 4),
            "scramble_max_score": round(max(vals) if vals else 0.0, 4),
            "scramble_mean_hotspot": round(sum(hs) / n, 4),
            "scramble_max_hotspot": round(max(hs) if hs else 0.0, 4),
            "native_percentile": round(native_percentile, 4),
            "native_hotspot_percentile": round(hotspot_percentile, 4),
            "n_scramble": len(scrambles),
        },
    }


def generate_decoys(contact_map: Dict, n_scramble: int = 100, seed: int = 42) -> List[Dict]:
    """Generate multiple decoy classes for a native contact map."""
    native = contact_map["paratope_sequence"]
    decoys: List[Dict] = []
    for i, seq in enumerate(scrambled_controls(native, n_scramble, seed), 1):
        decoys.append({"decoy_id": f"scramble_{i:03d}", "decoy_type": "scrambled", "sequence": seq})
    decoys.append({"decoy_id": "poly_alanine", "decoy_type": "alanine", "sequence": "A" * len(native)})
    decoys.append({"decoy_id": "poly_glycine", "decoy_type": "glycine", "sequence": "G" * len(native)})
    decoys.append({"decoy_id": "hotspot_breaking", "decoy_type": "hotspot_breaking", "sequence": hotspot_breaking_mutant(contact_map)})
    decoys.append({"decoy_id": "conservative", "decoy_type": "conservative", "sequence": conservative_mutant(native)})
    return decoys


def benchmark_native_vs_decoys(contact_map: Dict, n_scramble: int = 100, seed: int = 42) -> Dict:
    """Benchmark native against scrambled and designed negative decoys."""
    native = score_sequence_on_contact_map(contact_map["paratope_sequence"], contact_map)
    decoys = []
    for d in generate_decoys(contact_map, n_scramble=n_scramble, seed=seed):
        score = score_sequence_on_contact_map(d["sequence"], contact_map)
        decoys.append({**d, **score})

    values = [d["state_contact_score"] for d in decoys]
    hotspot_values = [d["hotspot_score"] for d in decoys]
    n = max(1, len(decoys))
    native_percentile = 1.0 - sum(1 for v in values if v >= native["state_contact_score"]) / n
    hotspot_percentile = 1.0 - sum(1 for v in hotspot_values if v >= native["hotspot_score"]) / n
    by_type = {}
    for d in decoys:
        by_type.setdefault(d["decoy_type"], []).append(d)
    type_summary = {}
    for dtype, rows in by_type.items():
        vals = [r["state_contact_score"] for r in rows]
        hvals = [r["hotspot_score"] for r in rows]
        type_summary[dtype] = {
            "n": len(rows),
            "mean_score": round(sum(vals) / len(vals), 4),
            "max_score": round(max(vals), 4),
            "mean_hotspot": round(sum(hvals) / len(hvals), 4),
            "max_hotspot": round(max(hvals), 4),
        }
    return {
        "native": native,
        "decoys": decoys,
        "summary": {
            "native_score": native["state_contact_score"],
            "native_hotspot": native["hotspot_score"],
            "decoy_mean_score": round(sum(values) / n, 4),
            "decoy_max_score": round(max(values) if values else 0.0, 4),
            "decoy_mean_hotspot": round(sum(hotspot_values) / n, 4),
            "decoy_max_hotspot": round(max(hotspot_values) if hotspot_values else 0.0, 4),
            "native_percentile": round(native_percentile, 4),
            "native_hotspot_percentile": round(hotspot_percentile, 4),
            "n_decoys": len(decoys),
            "type_summary": type_summary,
        },
    }


def hotspot_breaking_mutant(contact_map: Dict) -> str:
    """Mutate hotspot contact positions to weak residues while preserving length."""
    seq = list(contact_map["paratope_sequence"])
    hotspot_positions = set()
    for c in contact_map["contacts"]:
        if _pair_score(c.native_aa, c.epitope_aa) >= 0.45 or c.distance <= 5.0:
            hotspot_positions.add(c.paratope_index)
    for i in hotspot_positions:
        seq[i] = _weak_residue_for_position(seq[i])
    return "".join(seq)


def conservative_mutant(seq: str) -> str:
    """Create a conservative mutant preserving broad chemistry."""
    mapping = {
        "F": "Y", "Y": "F", "W": "Y",
        "I": "L", "L": "V", "V": "I", "M": "L",
        "K": "R", "R": "K", "H": "K",
        "D": "E", "E": "D",
        "S": "T", "T": "S", "N": "Q", "Q": "N",
        "A": "G", "G": "A", "P": "A", "C": "S",
    }
    return "".join(mapping.get(a, "A") for a in _clean(seq))


def generate_contact_guided_variants(
    contact_map: Dict,
    n: int = 100,
    max_mutations: int = 4,
    seed: int = 42,
    allowed_positions: Optional[Iterable[int]] = None,
) -> List[Dict]:
    """Generate paratope variants guided by contact topology.

    Variants keep the native contact-map length and mutate a bounded number of
    positions. Hotspot positions are mutated less often and only to conservative
    or contact-compatible residues. This is a candidate-library generator, not a
    structural grafting model.
    """
    if max_mutations < 1:
        raise ValueError("max_mutations must be at least 1")
    rng = random.Random(seed)
    native = contact_map["paratope_sequence"]
    positions_pool = list(range(len(native))) if allowed_positions is None else sorted(set(allowed_positions))
    if not positions_pool:
        raise ValueError("allowed_positions must contain at least one position")
    if positions_pool[0] < 0 or positions_pool[-1] >= len(native):
        raise ValueError("allowed_positions contains an out-of-range paratope index")
    hotspots = _hotspot_positions(contact_map)
    position_options = [_position_allowed_residues(contact_map, i, hotspots) for i in range(len(native))]
    variants = []
    seen = {native}
    attempts = 0
    while len(variants) < n and attempts < n * 100:
        attempts += 1
        seq = list(native)
        n_mut = rng.randint(1, min(max_mutations, len(positions_pool)))
        weights = [0.25 if i in hotspots else 1.0 for i in positions_pool]
        positions = _weighted_sample_without_replacement(rng, positions_pool, weights, n_mut)
        mutations = []
        for pos in positions:
            choices = [a for a in position_options[pos] if a != native[pos]]
            if not choices:
                continue
            new_aa = rng.choice(choices)
            seq[pos] = new_aa
            mutations.append(f"{native[pos]}{pos + 1}{new_aa}")
        s = "".join(seq)
        if s in seen or not mutations:
            continue
        seen.add(s)
        variants.append({
            "candidate_id": f"cg_{len(variants) + 1:04d}",
            "sequence": s,
            "mutations": ";".join(mutations),
            "n_mutations": len(mutations),
            "generator": "contact_guided_v1",
        })
    return variants


def score_contact_guided_variants(contact_map: Dict, variants: Iterable[Dict]) -> List[Dict]:
    """Score and rank contact-guided variants with lightweight filters."""
    rows = []
    for v in variants:
        score = score_sequence_on_contact_map(v["sequence"], contact_map)
        complexity = _sequence_complexity(v["sequence"])
        immuno = _immunogenicity_proxy(v["sequence"])
        native_score = score_sequence_on_contact_map(contact_map["paratope_sequence"], contact_map)
        retention = score["state_contact_score"] / max(native_score["state_contact_score"], 1e-9)
        composite = (
            0.55 * retention
            + 0.20 * score["hotspot_score"]
            + 0.15 * complexity
            + 0.10 * (1.0 - immuno)
        )
        failures = []
        if retention < 0.70:
            failures.append("low_contact_retention")
        if complexity < 0.55:
            failures.append("low_complexity")
        if immuno > 0.65:
            failures.append("high_immunogenicity_proxy")
        rows.append({
            **v,
            **score,
            "contact_retention": round(retention, 4),
            "complexity": round(complexity, 4),
            "immunogenicity_proxy": round(immuno, 4),
            "candidate_score": round(composite, 4),
            "passes_filters": not failures,
            "filter_failures": ";".join(failures),
        })
    rows.sort(key=lambda r: (r["passes_filters"], r["candidate_score"], r["contact_retention"]), reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def _clean(seq: str) -> str:
    return "".join(a for a in str(seq).upper() if a in AA)


def _distance_weight(distance: float) -> float:
    # Strongly weight close contacts while keeping 5-8 A context informative.
    return max(0.05, math.exp(-max(0.0, distance - 3.5) / 2.0))


def _pair_score(pa: str, ea: str) -> float:
    score = 0.05
    if ea in HYDROPHOBIC and pa in HYDROPHOBIC:
        score += 0.35
    if ea in AROMATIC and pa in AROMATIC:
        score += 0.25
    if (ea in POSITIVE and pa in NEGATIVE) or (ea in NEGATIVE and pa in POSITIVE):
        score += 0.45
    if ea in POLAR and pa in POLAR:
        score += 0.20
    if ea == pa:
        score += 0.08
    if pa == "C":
        score -= 0.20
    return max(0.0, min(1.0, score))


def _substitution_score(aa: str, native: str) -> float:
    if aa == native:
        return 1.0
    for group in CONSERVATIVE_GROUPS:
        if aa in group and native in group:
            return 0.65
    if (aa in POLAR and native in POLAR) or (aa in HYDROPHOBIC and native in HYDROPHOBIC):
        return 0.35
    return 0.0


def _weak_residue_for_position(native: str) -> str:
    if native in HYDROPHOBIC or native in AROMATIC:
        return "S"
    if native in POSITIVE or native in NEGATIVE:
        return "A"
    if native in POLAR:
        return "G"
    return "A"


def _hotspot_positions(contact_map: Dict) -> set:
    out = set()
    for c in contact_map["contacts"]:
        if _pair_score(c.native_aa, c.epitope_aa) >= 0.45 or c.distance <= 5.0:
            out.add(c.paratope_index)
    return out


def _position_allowed_residues(contact_map: Dict, pos: int, hotspots: set) -> List[str]:
    native = contact_map["paratope_sequence"][pos]
    contacted = [c.epitope_aa for c in contact_map["contacts"] if c.paratope_index == pos]
    allowed = set()
    # Always allow conservative substitutions.
    for group in CONSERVATIVE_GROUPS:
        if native in group:
            allowed.update(group)
    # Add residues compatible with contacted epitope chemistry.
    for ea in contacted:
        if ea in HYDROPHOBIC:
            allowed.update("YWFILV")
        if ea in AROMATIC:
            allowed.update("YWF")
        if ea in POSITIVE:
            allowed.update("DEYQ")
        if ea in NEGATIVE:
            allowed.update("KRHY")
        if ea in POLAR:
            allowed.update("STNQYH")
    allowed.add(native)
    if pos in hotspots:
        # Keep hotspots conservative/contact-compatible and avoid Cys/Pro.
        allowed.discard("C")
        allowed.discard("P")
    else:
        allowed.update("ASTNGQ")
        allowed.discard("C")
    return sorted(a for a in allowed if a in AA)


def _weighted_sample_without_replacement(rng: random.Random, items: List[int], weights: List[float], k: int) -> List[int]:
    chosen = []
    pool = list(items)
    w = list(weights)
    for _ in range(min(k, len(pool))):
        total = sum(w)
        if total <= 0:
            break
        x = rng.random() * total
        acc = 0.0
        idx = 0
        for i, wi in enumerate(w):
            acc += wi
            if acc >= x:
                idx = i
                break
        chosen.append(pool.pop(idx))
        w.pop(idx)
    return chosen


def _sequence_complexity(seq: str) -> float:
    seq = _clean(seq)
    if len(seq) < 3:
        return 1.0
    counts = {a: seq.count(a) for a in set(seq)}
    n = len(seq)
    entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
    entropy /= math.log(min(20, max(2, n)))
    max_freq = max(counts.values()) / n
    return float(max(0.0, min(1.0, entropy - max(0.0, max_freq - 0.45))))


def _immunogenicity_proxy(seq: str) -> float:
    seq = _clean(seq)
    if not seq:
        return 0.0
    hydrophobic = sum(1 for a in seq if a in HYDROPHOBIC) / len(seq)
    aromatic = sum(1 for a in seq if a in AROMATIC) / len(seq)
    cys = 0.3 if "C" in seq else 0.0
    return float(max(0.0, min(1.0, 0.55 * hydrophobic + 0.25 * aromatic + cys)))


def _empty_result(seq: str) -> Dict:
    return {
        "sequence": seq,
        "native_sequence": seq,
        "contact_score": 0.0,
        "hotspot_score": 0.0,
        "native_reference_score": 0.0,
        "identity_to_native": 0.0,
        "conservative_similarity": 0.0,
        "state_contact_score": 0.0,
        "n_contacts": 0,
        "n_paratope_positions": len(seq),
        "position_scores": [],
    }
