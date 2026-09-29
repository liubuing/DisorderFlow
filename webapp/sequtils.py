"""app.py split: sequtils (auto-generated 2026-09-27, bodies verbatim)."""
#!/usr/bin/env python
"""
蛋白质序列设计平台 — Gradio Web 界面（管理增强版）

访问: http://127.0.0.1:7860  （仅本机访问，外部无法连接）
管理: python manage.py start|stop|restart|status|batch|config|test
"""
import os, json, subprocess, tempfile, re, sys, yaml, time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import gradio as gr
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'modules'))
import target_design_helpers as tdh
import cascade_filter as cf
import misfolding_knowledge_base as mkb
import misfolding_pipeline as mfp
import closed_loop_scorer as cls
import closed_loop_orchestrator as clo
from runtime_environment import build_colabfold_command, colabfold_environment

from webapp.baselines import AA_NAMES


def detect_chains(pdb_path):
    chains = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                c = line[21:22].strip()
                if c and c not in chains:
                    chains.append(c)
    return chains
def get_chain_info(pdb_path):
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    pid = os.path.basename(pdb_path).replace('.pdb', '')
    s = parser.get_structure(pid, pdb_path)[0]
    aa3to1 = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
              'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
              'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}
    info = {}
    for ch in s.get_chains():
        cid = ch.id
        res = [r for r in ch.get_residues() if r.get_resname().strip() in aa3to1]
        seq = ''.join(aa3to1[r.get_resname().strip()] for r in res)
        ids = [str(r.get_id()[1]) for r in res]
        info[cid] = {'seq': seq, 'len': len(seq), 'first': ids[0] if ids else '?', 'last': ids[-1] if ids else '?'}
    return info
def on_upload(pdb_file):
    if pdb_file is None:
        return "请上传 PDB 文件", "", ""
    try:
        chains = detect_chains(pdb_file.name)
        info = get_chain_info(pdb_file.name)
        lines = [f"检测到 {len(chains)} 条链: {', '.join(chains)}"]
        for c in chains:
            ci = info.get(c, {})
            s = ci.get('seq', '')
            lines.append(f"  链 {c}: {ci.get('len','?')}残基 [{ci.get('first','?')}-{ci.get('last','?')}] {s[:60]}{'...' if len(s)>60 else ''}")
        return '\n'.join(lines), ', '.join(chains), f"{chains[0]}:1-10" if chains else ""
    except Exception as e:
        return f"解析错误: {e}", "", ""
def _fmt_coord(v):
    """Format a coordinate to exactly 8 chars for PDB (cols 31-38/39-46/47-54).
    Python f'{v:8.3f}' overflows when v >= 10000, corrupting PDB column alignment."""
    s = f"{v:8.3f}"
    if len(s) <= 8:
        return s
    s = f"{v:8.2f}"
    if len(s) <= 8:
        return s
    s = f"{v:8.1f}"
    if len(s) <= 8:
        return s
    return f"{v:8.0f}"[:8]
BB_ATOMS = ['N', 'CA', 'C', 'O']
def generate_pdb_from_sequence(seq, chain_id='A', start_res=1):
    """Generate a PDB file with backbone atoms in a realistic alpha-helix conformation.

    Uses standard alpha-helix parameters (~3.6 residues/turn, 1.5 A rise, 2.3 A radius)
    so that CA-CA distances (~3.8 A) match real proteins. This ensures compatibility
    with ProteinMPNN and other structure-based design tools.
    """
    import math
    aa_list = [AA_NAMES.get(a, 'ALA') for a in seq if a in AA_NAMES]
    if not aa_list:
        return None

    # Standard alpha-helix parameters
    residues_per_turn = 3.6
    rise_per_residue = 1.5   # Angstroms
    helix_radius = 2.3       # Angstroms

    # Standard bond lengths (Engh & Huber)
    bond_n_ca = 1.47
    bond_ca_c = 1.53
    bond_c_o = 1.23

    n_residues = len(aa_list)
    pdb_lines = []

    def add_atom(serial, name, res_name, res_seq, x, y, z, element):
        if len(name) == 1:
            name4 = f" {name}  "
        elif len(name) == 2:
            name4 = f" {name} "
        elif len(name) == 3:
            name4 = f" {name}"
        else:
            name4 = f"{name:4s}"
        pdb_lines.append(
            f"ATOM  {serial:5d} {name4} {res_name:3s} {chain_id:1s}{res_seq:4d}    "
            f"{_fmt_coord(x)}{_fmt_coord(y)}{_fmt_coord(z)}  1.00  0.00          {element:>2s}  "
        )

    # Phase 1: compute CA positions along a proper alpha helix
    ca_positions = []
    for i in range(n_residues):
        angle = i * 2.0 * math.pi / residues_per_turn
        x = helix_radius * math.cos(angle)
        y = helix_radius * math.sin(angle)
        z = i * rise_per_residue
        ca_positions.append((x, y, z))

    # Phase 2: compute N, C, O positions using local backbone frame.
    # N-CA-C ≈ 111° (standard tetrahedral geometry).
    # N and C straddle the forward (tangent) direction in the plane formed
    # by forward × helix-axis.  Since the peptide plane parallels the helix
    # axis in an α-helix, the perpendicular is (0, 0, 1).
    half_n_ca_c = math.radians(55.5)  # half of 111°
    cos_half = math.cos(half_n_ca_c)   # ≈ 0.566
    sin_half = math.sin(half_n_ca_c)   # ≈ 0.824

    atom_num = 0
    for i in range(n_residues):
        res_num = start_res + i
        res_name = aa_list[i]
        cx, cy, cz = ca_positions[i]

        # Local forward direction (CA[i-1] -> CA[i+1])
        if i == 0:
            fx, fy, fz = ca_positions[1][0] - cx, ca_positions[1][1] - cy, ca_positions[1][2] - cz
        elif i == n_residues - 1:
            fx, fy, fz = cx - ca_positions[i - 1][0], cy - ca_positions[i - 1][1], cz - ca_positions[i - 1][2]
        else:
            fx = ca_positions[i + 1][0] - ca_positions[i - 1][0]
            fy = ca_positions[i + 1][1] - ca_positions[i - 1][1]
            fz = ca_positions[i + 1][2] - ca_positions[i - 1][2]

        f_norm = math.sqrt(fx * fx + fy * fy + fz * fz)
        if f_norm > 1e-6:
            fx, fy, fz = fx / f_norm, fy / f_norm, fz / f_norm
        else:
            fx, fy, fz = 0.0, 0.0, 1.0

        # Perpendicular is the helix axis (z-direction); the peptide plane
        # is approximately parallel to the helix axis in an α-helix.
        px, py, pz = 0.0, 0.0, 1.0

        # C_dir = cos_half * forward + sin_half * z_axis
        c_dir_x = cos_half * fx + sin_half * px
        c_dir_y = cos_half * fy + sin_half * py
        c_dir_z = cos_half * fz + sin_half * pz
        c_norm = math.sqrt(c_dir_x**2 + c_dir_y**2 + c_dir_z**2)
        c_dir_x, c_dir_y, c_dir_z = c_dir_x / c_norm, c_dir_y / c_norm, c_dir_z / c_norm

        # N_dir = cos_half * forward - sin_half * z_axis
        n_dir_x = cos_half * fx - sin_half * px
        n_dir_y = cos_half * fy - sin_half * py
        n_dir_z = cos_half * fz - sin_half * pz
        n_norm = math.sqrt(n_dir_x**2 + n_dir_y**2 + n_dir_z**2)
        n_dir_x, n_dir_y, n_dir_z = n_dir_x / n_norm, n_dir_y / n_norm, n_dir_z / n_norm

        x_c = cx + bond_ca_c * c_dir_x
        y_c = cy + bond_ca_c * c_dir_y
        z_c = cz + bond_ca_c * c_dir_z

        x_n = cx + bond_n_ca * n_dir_x
        y_n = cy + bond_n_ca * n_dir_y
        z_n = cz + bond_n_ca * n_dir_z

        # O: placed trans to N relative to CA-C, ~120° from CA-C bond.
        # CA-C-O angle ≈ 120° (standard peptide geometry).
        ca_to_c_x = x_c - cx
        ca_to_c_y = y_c - cy
        ca_to_c_z = z_c - cz
        cc_norm = math.sqrt(ca_to_c_x**2 + ca_to_c_y**2 + ca_to_c_z**2)
        ca_to_c_x, ca_to_c_y, ca_to_c_z = ca_to_c_x / cc_norm, ca_to_c_y / cc_norm, ca_to_c_z / cc_norm

        # O direction: rotate CA→C by ~120° in the peptide plane (forward × helix-axis)
        ox = fy * pz - fz * py  # forward × z_axis
        oy = fz * px - fx * pz
        oz = fx * py - fy * px
        o_norm = math.sqrt(ox * ox + oy * oy + oz * oz)
        if o_norm > 1e-6:
            ox, oy, oz = ox / o_norm, oy / o_norm, oz / o_norm
        else:
            ox, oy, oz = 1.0, 0.0, 0.0

        # C→O = bond_c_o * (cos_60 * u + sin_60 * w) where u = CA→C, w ⟂ peptide
        # This gives CA-C-O ≈ 120° (standard peptide geometry).
        cos60 = 0.5
        sin60 = math.sqrt(3.0) / 2.0  # ≈ 0.866
        x_o = x_c + bond_c_o * (cos60 * ca_to_c_x + sin60 * ox)
        y_o = y_c + bond_c_o * (cos60 * ca_to_c_y + sin60 * oy)
        z_o = z_c + bond_c_o * (cos60 * ca_to_c_z + sin60 * oz)

        add_atom(atom_num + 1, 'N', res_name, res_num, x_n, y_n, z_n, 'N')
        add_atom(atom_num + 2, 'CA', res_name, res_num, cx, cy, cz, 'C')
        add_atom(atom_num + 3, 'C', res_name, res_num, x_c, y_c, z_c, 'C')
        add_atom(atom_num + 4, 'O', res_name, res_num, x_o, y_o, z_o, 'O')
        atom_num += 4

    pdb_lines.append(f"TER   {atom_num + 1:5d}      {aa_list[-1]:3s} {chain_id:1s}{start_res + n_residues - 1:4d}")
    pdb_lines.append("END")
    return '\n'.join(pdb_lines)
def fasta_to_pdb_file(fasta_text):
    """Convert FASTA text to PDB file, return (pdb_path, info_text)."""
    if not fasta_text or not fasta_text.strip():
        return None, "请提供 FASTA 序列"

    seqs = parse_fasta(fasta_text)
    if not seqs:
        return None, "无效的 FASTA 格式（需要 > 开头）"

    # Use the first sequence
    header, seq = seqs[0]
    pdb_content = generate_pdb_from_sequence(seq)

    if pdb_content is None:
        return None, "序列中没有有效的氨基酸"

    # Use header as filename base
    name = re.sub(r'[^A-Za-z0-9_-]', '_', header.split()[0] if header else 'protein')
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False,
                                      encoding='utf-8', prefix=f'{name}_')
    tmp.write(pdb_content)
    tmp.close()

    info = (f"✅ PDB 已生成: {name}\n"
            f"   链 A: {len(seq)} 个残基\n"
            f"   文件: {tmp.name}\n"
            f"   注意: 此 PDB 为 α-螺旋模板骨架，非真实折叠结构\n"
            f"   可用于 BFN / ProteinMPNN / ESM-IF 进行序列设计")
    return tmp.name, info
def parse_fasta(text):
    seqs = []
    cur_header, cur_seq = None, []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if cur_header:
                seqs.append((cur_header, ''.join(cur_seq)))
            cur_header = line[1:].strip()
            cur_seq = []
        else:
            cur_seq.append(line)
    if cur_header:
        seqs.append((cur_header, ''.join(cur_seq)))
    return seqs
def _save_fasta(fasta_text):
    """Write FASTA text to a temporary file accepted by Gradio File outputs."""
    if not fasta_text:
        return None
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.fasta', delete=False, encoding='utf-8')
    tmp.write(fasta_text)
    tmp.close()
    return tmp.name
def on_fasta_input(fasta_text, fasta_file):
    if fasta_file is not None:
        try:
            with open(fasta_file.name, encoding='utf-8') as f:
                fasta_text = f.read()
        except Exception:
            pass
    if not fasta_text or not fasta_text.strip():
        return "", "请提供 FASTA 序列或上传 .fasta 文件"
    seqs = parse_fasta(fasta_text)
    if not seqs:
        return "", "未检测到有效 FASTA 格式序列（需要 > 开头）"
    lines = []
    for h, s in seqs:
        lines.append(f"{h}: {len(s)} aa")
        lines.append(f"  {s[:80]}{'...' if len(s) > 80 else ''}")
    return fasta_text, '\n'.join(lines)
