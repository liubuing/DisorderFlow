#!/usr/bin/env python
"""Curated knowledge base of protein misfolding disease targets for antibody design.

Each target includes:
  - UniProt ID, gene name, disease association
  - Pathogenic conformations with PDB structures
  - Known antibody epitopes from literature
  - IDP flagging and design recommendations

Usage:
  from misfolding_knowledge_base import get_target, list_diseases, get_known_epitope_regions
"""

# ── Misfolding disease targets ──
# fmt: off
MISFOLDING_TARGETS = {
    # ═══════════════════════════════════════════════════════════════════
    # Alzheimer's Disease
    # ═══════════════════════════════════════════════════════════════════
    'alzheimer_abeta': {
        'disease_cn': "阿尔茨海默病 — Aβ",
        'disease_en': "Alzheimer's Disease — Amyloid-beta",
        'protein': 'Amyloid-beta (Aβ42)',
        'uniprot_id': 'P05067',
        'gene': 'APP',
        'pathogenic_form': 'Aβ42 amyloid fibril',
        'normal_function': 'Synaptic plasticity, neuronal survival (monomeric)',
        'conformations': {
            'fibril_2nao': {
                'pdb_id': '2NAO',
                'type': 'fibril (paired β-sheet)',
                'n_chains': 12,
                'chain': 'A',
                'region': (1, 42),
                'resolution': 'NMR',
                'description': 'Aβ42 fibril — archetypal Alzheimer\'s fold',
            },
            'fibril_5oqv': {
                'pdb_id': '5OQV',
                'type': 'fibril (LS-shaped)',
                'n_chains': 9,
                'chain': 'A',
                'region': (1, 42),
                'resolution': 'cryo-EM 3.1Å',
                'description': 'Aβ42 fibril — distinct polymorph, brain-derived',
            },
            'fibril_2mxu': {
                'pdb_id': '2MXU',
                'type': 'fibril (3-fold)',
                'n_chains': 15,
                'chain': 'A',
                'region': (1, 40),
                'resolution': 'ssNMR',
                'description': 'Aβ40 fibril — alternative polymorph',
            },
        },
        'known_epitopes': [
            {'region': (1, 16), 'ab': 'aducanumab', 'source': 'Sevigny et al. 2016 (PMID:27582221)'},
            {'region': (3, 12), 'ab': 'bapineuzumab (3D6)', 'source': 'N-terminus'},
            {'region': (18, 26), 'ab': 'solanezumab', 'source': 'mAb266, central domain'},
            {'region': (19, 42), 'ab': 'gantenerumab', 'source': 'conformational epitope'},
            {'region': (30, 40), 'ab': 'crenezumab', 'source': 'C-terminal fibril-specific'},
        ],
        'structured_core': (1, 42),
        'idp_flag': True,
        'idp_warning': 'Aβ monomer is intrinsically disordered. Design against FIBRIL conformations only (2NAO, 5OQV). BFN pLDDT scores on the monomeric form are unreliable.',
        'design_note': 'Target fibril β-sheet face for aggregation-blocking antibodies. N-terminal region (1-16) is exposed in fibrils and is the validated epitope for aducanumab.',
        'scaffold_candidates': ['nanobody_cAb1', 'nanobody_4xm4', 'nanobody_3gd8'],
    },

    'alzheimer_tau': {
        'disease_cn': "阿尔茨海默病 — Tau",
        'disease_en': "Alzheimer's Disease — Tau",
        'protein': 'Microtubule-associated protein Tau',
        'uniprot_id': 'P10636',
        'gene': 'MAPT',
        'pathogenic_form': 'Paired helical filaments (PHF)',
        'normal_function': 'Microtubule stabilization (axons)',
        'conformations': {
            'phf_5o3l': {
                'pdb_id': '5O3L',
                'type': 'Paired helical filament',
                'n_chains': 10,
                'chain': 'A',
                'region': (306, 378),
                'resolution': 'cryo-EM 3.3Å',
                'description': 'Tau PHF core (R3-R4 repeats) — Alzheimer\'s disease',
            },
            'sf_5o3t': {
                'pdb_id': '5O3T',
                'type': 'Straight filament',
                'n_chains': 10,
                'chain': 'A',
                'region': (306, 378),
                'resolution': 'cryo-EM 3.4Å',
                'description': 'Tau SF core — Alzheimer\'s disease',
            },
            'cte_6nwp': {
                'pdb_id': '6NWP',
                'type': 'Type I CTE filament',
                'n_chains': 10,
                'chain': 'A',
                'region': (274, 380),
                'resolution': 'cryo-EM 3.2Å',
                'description': 'Tau filament from chronic traumatic encephalopathy (CTE)',
            },
            'pick_6gx5': {
                'pdb_id': '6GX5',
                'type': 'Pick\'s disease filament',
                'n_chains': 10,
                'chain': 'A',
                'region': (254, 378),
                'resolution': 'cryo-EM 3.2Å',
                'description': 'Tau filament from Pick\'s disease (3R tau)',
            },
        },
        'known_epitopes': [
            {'region': (306, 320), 'ab': 'Anti-phospho-tau (AT8)', 'source': 'pS202/pT205 epitope region'},
            {'region': (350, 378), 'ab': 'Conformation-specific (MC1)', 'source': 'Pathological conformation'},
            {'region': (306, 378), 'ab': 'Various tau immunotherapies', 'source': 'Fibril core is the common target'},
        ],
        'structured_core': (306, 378),
        'idp_flag': True,
        'idp_warning': 'Full-length Tau (441 aa) is an IDP. Only the fibril core (R3-R4 repeats) is structured. Design against PDB 5O3L/5O3T core, not full-length.',
        'design_note': 'Tau has distinct fibril polymorphs across diseases (AD, CTE, PiD). Consider disease-specific conformation targeting.',
        'scaffold_candidates': ['nanobody_cAb1', 'nanobody_4xm4', 'nanobody_5imk'],
    },

    # ═══════════════════════════════════════════════════════════════════
    # Parkinson's Disease & Synucleinopathies
    # ═══════════════════════════════════════════════════════════════════
    'parkinson_alpha_synuclein': {
        'disease_cn': "帕金森病 — α-突触核蛋白",
        'disease_en': "Parkinson's Disease — α-Synuclein",
        'protein': 'Alpha-synuclein',
        'uniprot_id': 'P37840',
        'gene': 'SNCA',
        'pathogenic_form': 'α-synuclein amyloid fibril (Lewy body)',
        'normal_function': 'Synaptic vesicle trafficking (presynaptic)',
        'conformations': {
            'fibril_6cu7': {
                'pdb_id': '6CU7',
                'type': 'α-syn fibril (rod polymorph)',
                'n_chains': 8,
                'chain': 'A',
                'region': (38, 97),
                'resolution': 'cryo-EM 3.1Å',
                'description': 'Full-length α-synuclein fibril — rod polymorph',
            },
            'fibril_6cu8': {
                'pdb_id': '6CU8',
                'type': 'α-syn fibril (twister polymorph)',
                'n_chains': 8,
                'chain': 'A',
                'region': (38, 97),
                'resolution': 'cryo-EM 3.8Å',
                'description': 'Full-length α-synuclein fibril — twister polymorph',
            },
            'msa_6xyo': {
                'pdb_id': '6XYO',
                'type': 'MSA-type fibril',
                'n_chains': 8,
                'chain': 'A',
                'region': (38, 97),
                'resolution': 'cryo-EM 3.4Å',
                'description': 'α-synuclein fibril from multiple system atrophy (MSA)',
            },
        },
        'known_epitopes': [
            {'region': (38, 60), 'ab': 'Many conformational antibodies', 'source': 'N-terminal β-sheet region of fibril'},
            {'region': (60, 80), 'ab': 'NAC region antibodies', 'source': 'Central hydrophobic core'},
            {'region': (80, 97), 'ab': 'C-terminal fibril core antibodies', 'source': 'C-terminal β-sheet'},
        ],
        'structured_core': (38, 97),
        'idp_flag': True,
        'idp_warning': 'Monomeric α-synuclein is an IDP. Structure is only resolved in the fibril form. Design against PDB 6CU7.',
        'design_note': 'The NAC region (61-95) is critical for aggregation. Fibril-specific conformational antibodies targeting distinct polymorphs (rod vs twister) could provide strain-selective targeting.',
        'scaffold_candidates': ['nanobody_cAb1', 'nanobody_4xm4', 'nanobody_3gd8'],
    },

    # ═══════════════════════════════════════════════════════════════════
    # ALS / FTD
    # ═══════════════════════════════════════════════════════════════════
    'als_sod1': {
        'disease_cn': "肌萎缩侧索硬化症 — SOD1",
        'disease_en': "ALS — SOD1",
        'protein': 'Superoxide dismutase 1',
        'uniprot_id': 'P00441',
        'gene': 'SOD1',
        'pathogenic_form': 'Misfolded SOD1 dimer/oligomer',
        'normal_function': 'Superoxide radical detoxification (antioxidant)',
        'conformations': {
            'wt_1spd': {
                'pdb_id': '1SPD',
                'type': 'Wild-type folded dimer',
                'n_chains': 2,
                'chain': 'A',
                'region': None,
                'resolution': 'X-ray 2.0Å',
                'description': 'WT SOD1 — Cu/Zn-bound native fold (antiparallel β-barrel)',
            },
            'misfolded_6fz5': {
                'pdb_id': '6FZ5',
                'type': 'A4V mutant (misfolding-prone)',
                'n_chains': 2,
                'chain': 'A',
                'region': None,
                'resolution': 'X-ray 1.8Å',
                'description': 'ALS-linked A4V mutant — partially destabilized dimer',
            },
        },
        'known_epitopes': [
            {'region': (40, 60), 'ab': 'Conformation-specific (C4F6)', 'source': 'Misfolded SOD1 exposed hydrophobic patch'},
            {'region': (80, 100), 'ab': 'DSE2-α', 'source': 'Exposed dimer interface loop'},
            {'region': (1, 30), 'ab': 'Various', 'source': 'N-terminal β-strand (buried in native, exposed in misfolded)'},
        ],
        'structured_core': None,
        'idp_flag': False,
        'idp_warning': None,
        'design_note': 'SOD1 is folded (β-barrel), not an IDP. The challenge is distinguishing pathogenic misfolded conformers from WT. Target exposed hydrophobic patches unique to the misfolded state. The dimer interface (loop IV, residues 49-53, and Greek key loop VII, residues 115-143) are key regions.',
        'scaffold_candidates': ['nanobody_cAb1', 'nanobody_4xm4', 'nanobody_5imk'],
    },

    'als_tdp43': {
        'disease_cn': "肌萎缩侧索硬化症/额颞叶痴呆 — TDP-43",
        'disease_en': "ALS/FTD — TDP-43",
        'protein': 'TAR DNA-binding protein 43',
        'uniprot_id': 'Q13148',
        'gene': 'TARDBP',
        'pathogenic_form': 'C-terminal amyloid-like aggregates',
        'normal_function': 'RNA splicing and processing (nuclear)',
        'conformations': {
            'lcd_6n37': {
                'pdb_id': '6N37',
                'type': 'TDP-43 LCD fibril segment',
                'n_chains': 1,
                'chain': 'A',
                'region': (310, 340),
                'resolution': 'X-ray 1.3Å',
                'description': 'TDP-43 low-complexity domain amyloid core (aa 311-360)',
            },
            'lcd_6n3c': {
                'pdb_id': '6N3C',
                'type': 'TDP-43 LCD steric zipper',
                'n_chains': 2,
                'chain': 'A',
                'region': (310, 340),
                'resolution': 'X-ray 1.1Å',
                'description': 'TDP-43 LCD segment — NFGAILS, aa 312-324',
            },
        },
        'known_epitopes': [
            {'region': (310, 360), 'ab': 'C-terminal domain antibodies', 'source': 'LCD amyloid core'},
        ],
        'structured_core': (310, 340),
        'idp_flag': True,
        'idp_warning': 'TDP-43 C-terminal LCD is partially disordered. Design against crystallized fibril core (6N37). BFN confidence heads unreliable on full-length TDP-43.',
        'design_note': 'TDP-43 pathology spreads in a prion-like manner. Targeting aggregation-prone segments in the LCD region (especially NFGAILS motif at 312-319 and the 332-343 region) may block templated aggregation.',
        'scaffold_candidates': ['nanobody_cAb1', 'nanobody_4xm4', 'nanobody_3gd8'],
    },

    'als_fus': {
        'disease_cn': "肌萎缩侧索硬化症 — FUS",
        'disease_en': "ALS — FUS",
        'protein': 'Fused in sarcoma RNA-binding protein',
        'uniprot_id': 'P35637',
        'gene': 'FUS',
        'pathogenic_form': 'Cytoplasmic aggregates with prion-like domain',
        'normal_function': 'RNA metabolism (nuclear)',
        'conformations': {
            'lcd_6bwz': {
                'pdb_id': '6BWZ',
                'type': 'FUS LCD fibril segment',
                'n_chains': 1,
                'chain': 'A',
                'region': (37, 55),
                'resolution': 'X-ray 1.4Å',
                'description': 'FUS low-complexity domain fibril core (SYSGYS motif)',
            },
            'prld_6xfg': {
                'pdb_id': '6XFG',
                'type': 'FUS prion-like domain hydrogel',
                'n_chains': 2,
                'chain': 'A',
                'region': (1, 163),
                'resolution': 'X-ray 1.8Å',
                'description': 'FUS prion-like domain — labile cross-β polymer',
            },
        },
        'known_epitopes': [
            {'region': (37, 55), 'ab': 'Anti-FUS conformational', 'source': 'SYSGYS repeat region'},
        ],
        'structured_core': (37, 55),
        'idp_flag': True,
        'idp_warning': 'FUS N-terminal prion-like domain (aa 1-165) is largely disordered. Only short fibril segments are structurally resolved.',
        'design_note': 'Target the SYSGYS repeat motif (aa 37-55) which drives FUS liquid-liquid phase separation and subsequent aggregation.',
        'scaffold_candidates': ['nanobody_cAb1', 'nanobody_4xm4'],
    },

    # ═══════════════════════════════════════════════════════════════════
    # Huntington's Disease
    # ═══════════════════════════════════════════════════════════════════
    'huntington_htt': {
        'disease_cn': "亨廷顿病 — Huntingtin",
        'disease_en': "Huntington's Disease — Huntingtin",
        'protein': 'Huntingtin exon 1 (polyQ-expanded)',
        'uniprot_id': 'P42858',
        'gene': 'HTT',
        'pathogenic_form': 'polyQ-expanded exon1 amyloid fibrils',
        'normal_function': 'Multiple cellular functions (scaffold protein)',
        'conformations': {
            'exon1_6ezf': {
                'pdb_id': '6EZF',
                'type': 'HTTex1-Q46 amyloid fibril',
                'n_chains': 15,
                'chain': 'A',
                'region': (1, 46),
                'resolution': 'cryo-EM 4.9Å',
                'description': 'polyQ-expanded huntingtin exon1 fibril',
            },
            'htt_polyq_6xje': {
                'pdb_id': '6XJE',
                'type': 'polyQ peptide fibril',
                'n_chains': 1,
                'chain': 'A',
                'region': (1, 30),
                'resolution': 'cryo-EM 2.6Å',
                'description': 'Polyglutamine peptide fibril model',
            },
        },
        'known_epitopes': [
            {'region': (1, 17), 'ab': 'N17-specific (3B5H10, MW1)', 'source': 'N-terminal 17 aa (pre-polyQ)'},
            {'region': (1, 46), 'ab': 'Conformation-specific', 'source': 'polyQ fibril targeting'},
        ],
        'structured_core': (1, 46),
        'idp_flag': True,
        'idp_warning': 'Huntingtin exon1 outside the fibril core is disordered. Design against the fibril structure (6EZF).',
        'design_note': 'The N17 domain (aa 1-17) is a critical regulatory region that modulates aggregation. The polyQ tract length determines disease onset. Conformation-specific antibodies targeting the expanded polyQ fibril could be therapeutic.',
        'scaffold_candidates': ['nanobody_cAb1', 'nanobody_4xm4', 'nanobody_5imk'],
    },

    # ═══════════════════════════════════════════════════════════════════
    # Prion Diseases
    # ═══════════════════════════════════════════════════════════════════
    'prion_prp': {
        'disease_cn': "朊病毒病 — PrP",
        'disease_en': "Prion Disease — PrP",
        'protein': 'Prion protein (PrP^Sc)',
        'uniprot_id': 'P04156',
        'gene': 'PRNP',
        'pathogenic_form': 'Scrapie-associated prion fibril (PrP^Sc)',
        'normal_function': 'Copper binding, neuroprotection (PrP^C)',
        'conformations': {
            'prpc_1qm0': {
                'pdb_id': '1QM0',
                'type': 'PrP^C — normal cellular form',
                'n_chains': 1,
                'chain': 'A',
                'region': (121, 230),
                'resolution': 'NMR',
                'description': 'WT PrP^C — folded globular domain (α-helical)',
            },
            'prpsc_6uur': {
                'pdb_id': '6UUR',
                'type': 'PrP^Sc — scrapie fibril',
                'n_chains': 1,
                'chain': 'A',
                'region': (94, 178),
                'resolution': 'cryo-EM 3.1Å',
                'description': 'Human PrP^Sc amyloid fibril — parallel in-register β-structure',
            },
        },
        'known_epitopes': [
            {'region': (121, 165), 'ab': 'Conformation-specific (ICSM18, POM1)', 'source': 'PrP^C globular domain'},
            {'region': (144, 155), 'ab': 'β1-α1 loop — species barrier', 'source': 'Key structural difference PrP^C vs PrP^Sc'},
        ],
        'structured_core': (121, 230),
        'idp_flag': False,
        'idp_warning': None,
        'design_note': 'PrP^C is well-folded. PrP^Sc adopts a radically different β-sheet-rich architecture. The key challenge is designing antibodies that bind PrP^Sc but NOT PrP^C (must be conformation-specific). Target the refolded β-sheet regions unique to PrP^Sc (6UUR).',
        'scaffold_candidates': ['nanobody_cAb1', 'nanobody_4xm4', 'nanobody_5imk'],
    },

    # ═══════════════════════════════════════════════════════════════════
    # Cerebral Amyloid Angiopathy
    # ═══════════════════════════════════════════════════════════════════
    'caa_notch3': {
        'disease_cn': "脑淀粉样血管病 — NOTCH3",
        'disease_en': "Cerebral Autosomal Dominant Arteriopathy (CADASIL) — NOTCH3",
        'protein': 'Notch homolog 3 (NOTCH3 EGF repeats)',
        'uniprot_id': 'Q9UM47',
        'gene': 'NOTCH3',
        'pathogenic_form': 'NOTCH3 EGF-repeat aggregates in vascular smooth muscle',
        'normal_function': 'Vascular smooth muscle signaling (Notch pathway)',
        'conformations': {
            'egf_4zl5': {
                'pdb_id': '4ZL5',
                'type': 'NOTCH3 EGF repeats 7-9',
                'n_chains': 1,
                'chain': 'A',
                'region': (257, 410),
                'resolution': 'X-ray 2.1Å',
                'description': 'NOTCH3 EGF7-9 — region containing common CADASIL mutations',
            },
        },
        'known_epitopes': [
            {'region': (257, 410), 'ab': 'EGF-repeat region', 'source': 'Mutation cluster region'},
        ],
        'structured_core': None,
        'idp_flag': False,
        'idp_warning': None,
        'design_note': 'CADASIL is caused by cysteine-altering mutations in NOTCH3 EGF repeats that lead to misfolding and aggregation in vascular smooth muscle. Target EGF repeats 7-9 where most pathogenic mutations cluster.',
        'scaffold_candidates': ['nanobody_cAb1', 'nanobody_4xm4', 'nanobody_3gd8'],
    },

    # ═══════════════════════════════════════════════════════════════════
    # Transthyretin Amyloidosis
    # ═══════════════════════════════════════════════════════════════════
    'attr_ttr': {
        'disease_cn': "转甲状腺素蛋白淀粉样变性 — TTR",
        'disease_en': "ATTR Amyloidosis — Transthyretin",
        'protein': 'Transthyretin',
        'uniprot_id': 'P02766',
        'gene': 'TTR',
        'pathogenic_form': 'TTR amyloid fibril',
        'normal_function': 'Thyroxine and retinol transport',
        'conformations': {
            'wt_1tta': {
                'pdb_id': '1TTA',
                'type': 'WT TTR tetramer (native)',
                'n_chains': 4,
                'chain': 'A',
                'region': None,
                'resolution': 'X-ray 1.7Å',
                'description': 'Native TTR tetramer — all-β sandwich fold',
            },
            'v30m_4tne': {
                'pdb_id': '4TNE',
                'type': 'V30M mutant tetramer',
                'n_chains': 4,
                'chain': 'A',
                'region': None,
                'resolution': 'X-ray 1.6Å',
                'description': 'TTR V30M — most common ATTR mutation',
            },
            'amyloid_6sdz': {
                'pdb_id': '6SDZ',
                'type': 'TTR amyloid fibril',
                'n_chains': 8,
                'chain': 'A',
                'region': (10, 123),
                'resolution': 'cryo-EM 3.6Å',
                'description': 'Ex vivo TTR amyloid fibril (V30M)',
            },
        },
        'known_epitopes': [
            {'region': (50, 80), 'ab': 'Conformational (TabFH4, Tafamidis binding region)', 'source': 'TTR dimer-dimer interface'},
            {'region': (10, 35), 'ab': 'N-terminal — exposed in monomeric/misfolded', 'source': 'Hidden in native tetramer, exposed upon dissociation'},
        ],
        'structured_core': None,
        'idp_flag': False,
        'idp_warning': None,
        'design_note': 'TTR is a well-folded β-sandwich tetramer. Misfolding is triggered by tetramer dissociation → monomer misfolding → amyloid aggregation. Target epitopes hidden in the native tetramer but exposed in the misfolded monomer (e.g., N-terminal strand A). The dimer-dimer interface (Tafamidis binding site) is stabilization target.',
        'scaffold_candidates': ['nanobody_cAb1', 'nanobody_4xm4', 'nanobody_5imk'],
    },

    # ═══════════════════════════════════════════════════════════════════
    # Type 2 Diabetes — Islet Amyloid Polypeptide
    # ═══════════════════════════════════════════════════════════════════
    't2d_iapp': {
        'disease_cn': "2型糖尿病 — 胰岛淀粉样多肽",
        'disease_en': "Type 2 Diabetes — IAPP (Amylin)",
        'protein': 'Islet Amyloid Polypeptide (Amylin)',
        'uniprot_id': 'P10997',
        'gene': 'IAPP',
        'pathogenic_form': 'IAPP amyloid deposits in pancreatic islets',
        'normal_function': 'Glucose homeostasis (co-secreted with insulin)',
        'conformations': {
            'fibril_6uc3': {
                'pdb_id': '6UC3',
                'type': 'IAPP amyloid fibril',
                'n_chains': 14,
                'chain': 'A',
                'region': (1, 37),
                'resolution': 'cryo-EM 3.9Å',
                'description': 'Human IAPP S20G mutant fibril',
            },
        },
        'known_epitopes': [
            {'region': (1, 10), 'ab': 'N-terminal — exposed in fibril', 'source': 'N-terminal disulfide region'},
            {'region': (20, 29), 'ab': 'Core amyloidogenic region (SNNFGAILSS)', 'source': 'Central amyloid core'},
        ],
        'structured_core': (1, 37),
        'idp_flag': True,
        'idp_warning': 'IAPP monomer is disordered. Design against the fibril structure (6UC3).',
        'design_note': 'The SNNFGAILSS region (20-29) is the key amyloidogenic core. N-terminal disulfide bond (C2-C7) is preserved in fibrils. Target fibril-specific epitopes, avoiding cross-reactivity with monomeric IAPP (normal function).',
        'scaffold_candidates': ['nanobody_cAb1', 'nanobody_4xm4', 'nanobody_3gd8'],
    },

    # ═══════════════════════════════════════════════════════════════════
    # Additional targets (brief entries)
    # ═══════════════════════════════════════════════════════════════════
    'als_c9orf72': {
        'disease_cn': "ALS/FTD — C9orf72 DPR",
        'disease_en': "ALS/FTD — C9orf72 dipeptide repeat proteins",
        'protein': 'C9orf72 DPRs (poly-GA, poly-GR, poly-GP)',
        'uniprot_id': 'Q96LT7',
        'gene': 'C9orf72',
        'pathogenic_form': 'DPR aggregates from GGGGCC repeat expansion',
        'normal_function': 'Endosomal trafficking (normal C9orf72)',
        'conformations': {
            'polyga_6bzm': {
                'pdb_id': '6BZM',
                'type': 'poly-GA amyloid fibril',
                'n_chains': 1,
                'chain': 'A',
                'region': None,
                'resolution': 'X-ray 1.1Å',
                'description': 'poly-GA dipeptide repeat fibril segment',
            },
        },
        'known_epitopes': [],  # No well-characterized therapeutic epitopes yet
        'structured_core': None,
        'idp_flag': True,
        'idp_warning': 'DPR peptides are disordered in isolation. Design against the fibril-like assemblies.',
        'design_note': 'The GGGGCC repeat expansion produces toxic DPRs via RAN translation. poly-GA is most aggregating, poly-GR is most toxic. Target the β-sheet assemblies of poly-GA.',
        'scaffold_candidates': ['nanobody_cAb1', 'nanobody_4xm4'],
    },
}

# ── Nanobody scaffolds for antibody design ──
NANOBODY_SCAFFOLDS = {
    'nanobody_cAb1': {
        'name': 'cAb1 (4W6Y)',
        'pdb_id': '4W6Y',
        'type': 'VHH (Camelid nanobody)',
        'sequence': (
            "QVQLQESGGGLVQAGGSLRLSCAASGRTFPSTYAMGWFRQAPGKEREFVAAIRWS"
            "GGSTYYTDSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAATYLRMYYDYAD"
            "EYDYWGQGTQVTVSS"
        ),
        'length': 124,
        'cdr_definitions': {
            'H_CDR1': (26, 33),
            'H_CDR2': (51, 58),
            'H_CDR3': (97, 110),
        },
        'notes': 'Well-characterized lysozyme-binding nanobody. Good scaffold stability.',
    },
    'nanobody_4xm4': {
        'name': '4XM4 VHH',
        'pdb_id': '4XM4',
        'type': 'VHH (Camelid nanobody)',
        'sequence': (
            "QVQLQESGGGLVQPGGSLRLSCAASGFTFSSYWMYWVRQAPGKGLEWVSAINTD"
            "GSTTYADSVKGRFTISRDNAKNTLYLQMNSLKPEDTAVYFCARDGTTPTNNWGQ"
            "GTQVTVSS"
        ),
        'length': 113,
        'cdr_definitions': {
            'H_CDR1': (26, 33),
            'H_CDR2': (51, 57),
            'H_CDR3': (96, 106),
        },
        'notes': 'Generic VHH scaffold. Good thermodynamic stability. Suitable for CDR grafting.',
    },
    'nanobody_3gd8': {
        'name': '3GD8 (anti-AQP4 scaffold)',
        'pdb_id': '3GD8',
        'type': 'Engineered VHH',
        'sequence': None,   # loaded from PDB
        'length': None,
        'cdr_definitions': {
            'H_CDR1': (26, 33),
            'H_CDR2': (51, 58),
            'H_CDR3': (97, 110),
        },
        'notes': 'Existing project scaffold for AQP4. Sequence loaded from PDB at runtime.',
    },
    'nanobody_5imk': {
        'name': '5IMK VHH (anti-fibril)',
        'pdb_id': '5IMK',
        'type': 'VHH (Camelid nanobody)',
        'sequence': (
            "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYGMGWFRQAPGKEREFVAAIRWN"
            "GGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAGRWDKYGSSF"
            "QDEYDYWGQGTQVTVSS"
        ),
        'length': 125,
        'cdr_definitions': {
            'H_CDR1': (26, 33),
            'H_CDR2': (51, 58),
            'H_CDR3': (97, 113),
        },
        'notes': 'Nanobody optimized for amyloid fibril binding. Extended CDR3 for groove recognition. '
                 'Sequence corrected 2026-05-31 from PDB chain B (was 118AA, now 125AA).',
    },
}

# ── Known therapeutic antibodies (for epitope reference) ──
THERAPEUTIC_ANTIBODIES = {
    'aducanumab': {
        'target': 'alzheimer_abeta',
        'type': 'Human IgG1',
        'epitope': (3, 7),
        'epitope_desc': 'Aβ N-terminus (EFRH motif)',
        'specificity': 'Fibrillar > monomeric Aβ',
        'status': 'FDA approved (controversial, withdrawn 2024)',
        'pdb_available': False,
    },
    'lecanemab': {
        'target': 'alzheimer_abeta',
        'type': 'Humanized IgG1',
        'epitope': (1, 16),
        'epitope_desc': 'Aβ protofibrils (soluble aggregates)',
        'specificity': 'Protofibrils',
        'status': 'FDA approved (Leqembi, 2023)',
        'pdb_available': False,
    },
    'donanemab': {
        'target': 'alzheimer_abeta',
        'type': 'Humanized IgG1',
        'epitope': (3, 7),
        'epitope_desc': 'N-terminal pyroglutamate Aβ (pE3-Aβ)',
        'specificity': 'Plaque-specific',
        'status': 'FDA approved (Kisunla, 2024)',
        'pdb_available': False,
    },
    'prasinezumab': {
        'target': 'parkinson_alpha_synuclein',
        'type': 'Humanized IgG1',
        'epitope': (60, 80),
        'epitope_desc': 'C-terminal region of α-synuclein fibrils',
        'specificity': 'Aggregated α-synuclein',
        'status': 'Phase 2 clinical trials',
        'pdb_available': False,
    },
    'cinpanemab': {
        'target': 'parkinson_alpha_synuclein',
        'type': 'Human IgG1',
        'epitope': (1, 40),
        'epitope_desc': 'N-terminal α-synuclein',
        'specificity': 'Extracellular aggregated α-synuclein',
        'status': 'Phase 2 (discontinued, insufficient efficacy)',
        'pdb_available': False,
    },
}

# ── Helper functions ──


def list_diseases():
    """Return list of (key, disease_cn, disease_en) for all targets."""
    return [
        (k, v['disease_cn'], v['disease_en'])
        for k, v in MISFOLDING_TARGETS.items()
    ]


def get_target(disease_key):
    """Get full target data dict for a disease key. Returns None if not found."""
    return MISFOLDING_TARGETS.get(disease_key)


def get_conformations(disease_key):
    """Return dict of {conf_key: conf_data} for a target."""
    target = MISFOLDING_TARGETS.get(disease_key)
    if target is None:
        return {}
    return target.get('conformations', {})


def get_conformation(disease_key, conf_key):
    """Return single conformation data or None."""
    confs = get_conformations(disease_key)
    return confs.get(conf_key)


def get_known_epitope_regions(disease_key):
    """Return list of epitope dicts with {'region': (start, end), 'ab': str, 'source': str}."""
    target = MISFOLDING_TARGETS.get(disease_key)
    if target is None:
        return []
    return target.get('known_epitopes', [])


def get_scaffold(scaffold_key):
    """Return scaffold data dict or None."""
    return NANOBODY_SCAFFOLDS.get(scaffold_key)


def list_scaffolds():
    """Return list of (scaffold_key, name, type, length) tuples."""
    return [
        (k, v['name'], v['type'], v.get('length', '?'))
        for k, v in NANOBODY_SCAFFOLDS.items()
    ]


def get_therapeutic_antibodies_for_target(disease_key):
    """Return list of therapeutic antibody dicts known for this disease target."""
    return [
        ab for ab in THERAPEUTIC_ANTIBODIES.values()
        if ab['target'] == disease_key
    ]


def is_idp_target(disease_key):
    """Check if target is flagged as IDP-prone."""
    target = MISFOLDING_TARGETS.get(disease_key)
    if target is None:
        return False
    return target.get('idp_flag', False)


def get_idp_warning(disease_key):
    """Return IDP warning string or None."""
    target = MISFOLDING_TARGETS.get(disease_key)
    if target is None:
        return None
    return target.get('idp_warning')


def get_design_note(disease_key):
    """Return design guidance string or empty string."""
    target = MISFOLDING_TARGETS.get(disease_key)
    if target is None:
        return ''
    return target.get('design_note', '')


def get_default_conformation(disease_key):
    """Return the first (default) conformation key for a target."""
    confs = get_conformations(disease_key)
    if not confs:
        return None
    return list(confs.keys())[0]


def get_target_summary(disease_key):
    """Return a formatted English summary string for the target."""
    target = MISFOLDING_TARGETS.get(disease_key)
    if target is None:
        return f'Unknown target: {disease_key}'
    lines = [
        f"Disease: {target['disease_en']} ({target['disease_cn']})",
        f"Protein: {target['protein']}",
        f"Gene: {target['gene']} | UniProt: {target['uniprot_id']}",
        f"Pathogenic form: {target['pathogenic_form']}",
        f"Normal function: {target['normal_function']}",
        f"IDP: {'YES' if target['idp_flag'] else 'No'} | Structured core: {target['structured_core'] or 'Full protein'}",
        f"Conformations available: {len(target['conformations'])}",
    ]
    if target['known_epitopes']:
        lines.append(f"Known epitopes: {len(target['known_epitopes'])} (from literature)")
    return '\n'.join(lines)
