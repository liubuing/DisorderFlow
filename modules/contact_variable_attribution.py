#!/usr/bin/env python
"""Feature attribution for emergent IDP antibody contact variables."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List


SMALL = set("GAS")
MEDIUM = set("CTDNVP")
LARGE = set("EQHILKMFYRW")
AROMATIC = set("FWY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
POLAR = set("STNQYH")
AA = "ACDEFGHIKLMNPQRSTVWY"


def volume_class(aa: str) -> str:
    aa = aa.upper()
    if aa in SMALL:
        return "small"
    if aa in MEDIUM:
        return "medium"
    if aa in LARGE:
        return "large"
    return "other"


def chemistry_class(aa: str) -> str:
    aa = aa.upper()
    if aa in AROMATIC:
        return "aromatic"
    if aa in POSITIVE:
        return "positive"
    if aa in NEGATIVE:
        return "negative"
    if aa in POLAR:
        return "polar"
    if aa in "AILMV":
        return "hydrophobic"
    if aa == "G":
        return "flexible"
    if aa == "P":
        return "proline"
    if aa == "C":
        return "cys"
    return "other"


def residues_for_classes(volume: str, chemistry: str) -> List[str]:
    return [aa for aa in AA if volume_class(aa) == volume and chemistry_class(aa) == chemistry]


def residue_matches_classes(aa: str, volume: str, chemistry: str) -> bool:
    return volume_class(aa) == volume and chemistry_class(aa) == chemistry


def compile_guidance_rules(signal_rows: Iterable[Dict], min_abs_delta: float = 0.20) -> Dict[str, Dict[int, Dict]]:
    """Convert high/low-gap enrichment deltas into per-position design rules."""
    rules: Dict[str, Dict[int, Dict]] = defaultdict(dict)
    for row in signal_rows:
        delta = float(row.get("fraction_delta", 0.0))
        if abs(delta) < min_abs_delta:
            continue
        ref = row["reference_pdb"]
        pos = int(row["paratope_index"]) - 1
        rule = rules[ref].setdefault(pos, {"preferred": [], "avoid": [], "chain": row.get("chain", ""), "resid": row.get("resid", "")})
        cls = {
            "volume_class": row["volume_class"],
            "chemistry_class": row["chemistry_class"],
            "fraction_delta": round(delta, 4),
        }
        if delta > 0:
            rule["preferred"].append(cls)
        elif delta < 0:
            rule["avoid"].append(cls)
    return {ref: dict(pos_rules) for ref, pos_rules in rules.items()}


def generate_attribution_guided_variants(
    contact_map: Dict,
    parent_rows: Iterable[Dict],
    rules: Dict[int, Dict],
    n: int = 100,
    max_mutations: int = 4,
    seed: int = 42,
    reference_pdb: str = "",
) -> List[Dict]:
    """Generate v3 candidates by protecting preferred classes and avoiding depleted classes."""
    rng = random.Random(seed)
    native = contact_map["paratope_sequence"]
    parents = list(parent_rows) or []
    parents.append({"candidate_id": "native", "sequence": native})
    seen = {native}
    variants = []
    attempts = 0
    while len(variants) < n and attempts < n * 1000:
        attempts += 1
        parent = rng.choice(parents)
        seq = list(parent["sequence"])
        changes = apply_guidance_rules(seq, native, rules, rng)
        mutate_extra_positions(seq, native, rules, rng, max_mutations=max_mutations, changes=changes)
        s = "".join(seq)
        if s in seen or s == native:
            continue
        seen.add(s)
        variants.append({
            "candidate_id": f"v3_{len(variants) + 1:04d}",
            "sequence": s,
            "mutations": mutation_string(native, s),
            "n_mutations": sum(1 for a, b in zip(native, s) if a != b),
            "generator": "attribution_guided_v3",
            "parent_candidate_id": parent.get("candidate_id", ""),
            "reference_pdb": reference_pdb,
            "rule_hits": ";".join(changes),
        })
    return variants


def apply_guidance_rules(seq: List[str], native: str, rules: Dict[int, Dict], rng: random.Random) -> List[str]:
    changes = []
    for pos, rule in rules.items():
        if pos < 0 or pos >= len(seq):
            continue
        if preferred_match(seq[pos], rule):
            changes.append(f"protect:{pos + 1}:{seq[pos]}")
            continue
        if avoid_match(seq[pos], rule) or rule.get("preferred"):
            choices = preferred_residues(rule)
            if choices:
                new_aa = rng.choice(choices)
                if new_aa != seq[pos]:
                    seq[pos] = new_aa
                    changes.append(f"repair:{pos + 1}:{new_aa}")
            elif avoid_match(seq[pos], rule) and not avoid_match(native[pos], rule):
                seq[pos] = native[pos]
                changes.append(f"restore_native:{pos + 1}:{native[pos]}")
    return changes


def mutate_extra_positions(seq: List[str], native: str, rules: Dict[int, Dict], rng: random.Random, max_mutations: int, changes: List[str]) -> None:
    current = sum(1 for a, b in zip(seq, native) if a != b)
    budget = max(0, max_mutations - current)
    if budget <= 0:
        return
    protected = {pos for pos, rule in rules.items() if 0 <= pos < len(seq) and preferred_match(seq[pos], rule)}
    candidates = [i for i in range(len(seq)) if i not in protected]
    rng.shuffle(candidates)
    for pos in candidates[:rng.randint(1, max(1, budget))]:
        choices = [aa for aa in AA if aa != seq[pos] and aa != "C" and not avoid_match(aa, rules.get(pos, {}))]
        if not choices:
            continue
        seq[pos] = rng.choice(choices)
        changes.append(f"explore:{pos + 1}:{seq[pos]}")


def preferred_match(aa: str, rule: Dict) -> bool:
    return any(residue_matches_classes(aa, c["volume_class"], c["chemistry_class"]) for c in rule.get("preferred", []))


def avoid_match(aa: str, rule: Dict) -> bool:
    return any(residue_matches_classes(aa, c["volume_class"], c["chemistry_class"]) for c in rule.get("avoid", []))


def preferred_residues(rule: Dict) -> List[str]:
    out = []
    for cls in rule.get("preferred", []):
        out.extend(residues_for_classes(cls["volume_class"], cls["chemistry_class"]))
    return sorted(set(out) - {"C"})


def mutation_string(native: str, seq: str) -> str:
    return ";".join(f"{a}{i + 1}{b}" for i, (a, b) in enumerate(zip(native, seq)) if a != b)


def per_position_features(sequence: str, contact_map: Dict, group: str, candidate_id: str = "") -> List[Dict]:
    rows = []
    residues = contact_map.get("paratope_residues", [])
    for i, aa in enumerate(sequence):
        res = residues[i] if i < len(residues) else {}
        rows.append({
            "candidate_id": candidate_id,
            "group": group,
            "paratope_index": i + 1,
            "chain": res.get("chain", ""),
            "resid": res.get("resid", ""),
            "aa": aa,
            "volume_class": volume_class(aa),
            "chemistry_class": chemistry_class(aa),
        })
    return rows


def compare_high_low_gap(rows: Iterable[Dict], contact_map: Dict, top_fraction: float = 0.25) -> Dict:
    ranked = sorted(rows, key=lambda r: float(r.get("specificity_gap", 0.0)), reverse=True)
    n = max(1, int(len(ranked) * top_fraction))
    high = ranked[:n]
    low = ranked[-n:]
    feature_rows = []
    for group, subset in (("high_gap", high), ("low_gap", low)):
        for r in subset:
            feature_rows.extend(per_position_features(r["sequence"], contact_map, group, r.get("candidate_id", "")))
    enrichment = summarize_position_classes(feature_rows)
    return {"feature_rows": feature_rows, "enrichment_rows": enrichment, "n_high": len(high), "n_low": len(low)}


def summarize_position_classes(feature_rows: Iterable[Dict]) -> List[Dict]:
    counts = defaultdict(int)
    totals = defaultdict(int)
    for r in feature_rows:
        key = (r["group"], r["paratope_index"], r["chain"], r["resid"], r["volume_class"], r["chemistry_class"])
        counts[key] += 1
        totals[(r["group"], r["paratope_index"])] += 1
    out = []
    for key, count in sorted(counts.items()):
        group, pidx, chain, resid, vol, chem = key
        total = max(1, totals[(group, pidx)])
        out.append({
            "group": group,
            "paratope_index": pidx,
            "chain": chain,
            "resid": resid,
            "volume_class": vol,
            "chemistry_class": chem,
            "count": count,
            "fraction": round(count / total, 4),
        })
    return out
