#!/usr/bin/env python
"""State ensemble primitives for IDP antibody design.

An IDP epitope is represented as positive and negative contact states, not as a
single sequence. Negative states are lightweight topology perturbations used for
computational contrast, not claims about exact monomer structures.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from state_contact_scorer import Contact


@dataclass(frozen=True)
class IDPState:
    # 一个"状态"= 一份接触图快照；state_type 区分正例（观测到的真实界面）
    # 与负例（合成扰动，用于对比打分）
    name: str
    state_type: str
    contact_map: Dict
    description: str = ""


def positive_state(name: str, contact_map: Dict, description: str = "") -> IDPState:
    return IDPState(name=name, state_type="positive", contact_map=contact_map, description=description)


def build_negative_states(contact_map: Dict, prefix: str, seed: int = 1) -> List[IDPState]:
    """Build synthetic negative states by perturbing contact topology/chemistry."""
    # 三种轻量负例扰动：表位身份洗牌（保持几何、破坏化学匹配）、
    # paratope 索引错位（破坏接触拓扑）、热点弱化（把近接触推远）
    return [
        IDPState(f"{prefix}_epitope_shuffle", "negative", shuffle_epitope_contacts(contact_map, seed), "same contacts, shuffled epitope residue identities"),
        IDPState(f"{prefix}_paratope_shift", "negative", shift_paratope_contacts(contact_map, 1), "contact topology shifted along paratope"),
        IDPState(f"{prefix}_hotspot_diluted", "negative", dilute_hotspot_contacts(contact_map), "native hotspots made less decisive"),
    ]


def clone_contact_map(contact_map: Dict, contacts: List[Contact]) -> Dict:
    out = dict(contact_map)
    out["contacts"] = contacts
    return out


def shuffle_epitope_contacts(contact_map: Dict, seed: int) -> Dict:
    # 表位侧残基身份随机重排（保持配对几何与距离不变）；
    # 固定 seed 保证同一 contact_map 生成的负例可复现
    rng = random.Random(seed)
    aas = [c.epitope_aa for c in contact_map["contacts"]]
    rng.shuffle(aas)
    contacts = []
    for c, aa in zip(contact_map["contacts"], aas):
        contacts.append(Contact(
            paratope_index=c.paratope_index,
            paratope_chain=c.paratope_chain,
            paratope_resid=c.paratope_resid,
            native_aa=c.native_aa,
            epitope_index=c.epitope_index,
            epitope_chain=c.epitope_chain,
            epitope_resid=c.epitope_resid,
            epitope_aa=aa,
            distance=c.distance,
        ))
    return clone_contact_map(contact_map, contacts)


def shift_paratope_contacts(contact_map: Dict, shift: int) -> Dict:
    # paratope 索引整体平移（模序列长度）：模拟"接触长在别的环区"的拓扑负例
    n = len(contact_map["paratope_sequence"])
    contacts = []
    for c in contact_map["contacts"]:
        contacts.append(Contact(
            paratope_index=(c.paratope_index + shift) % n,
            paratope_chain=c.paratope_chain,
            paratope_resid=c.paratope_resid,
            native_aa=c.native_aa,
            epitope_index=c.epitope_index,
            epitope_chain=c.epitope_chain,
            epitope_resid=c.epitope_resid,
            epitope_aa=c.epitope_aa,
            distance=c.distance,
        ))
    return clone_contact_map(contact_map, contacts)


def dilute_hotspot_contacts(contact_map: Dict) -> Dict:
    contacts = []
    for c in contact_map["contacts"]:
        # Push close contacts toward weaker, less decisive distances.
        # 近接触统一抬到 7.5Å：保留接触存在性、削弱"决定性热点"信号
        distance = max(c.distance, 7.5)
        contacts.append(Contact(
            paratope_index=c.paratope_index,
            paratope_chain=c.paratope_chain,
            paratope_resid=c.paratope_resid,
            native_aa=c.native_aa,
            epitope_index=c.epitope_index,
            epitope_chain=c.epitope_chain,
            epitope_resid=c.epitope_resid,
            epitope_aa=c.epitope_aa,
            distance=distance,
        ))
    return clone_contact_map(contact_map, contacts)
