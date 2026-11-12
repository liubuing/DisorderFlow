"""操作层：把界面的一次请求翻译成一次可运行的科研任务。

这一层不依赖任何 UI 框架，输入是普通 dict / 文件路径，输出是可直接 JSON
序列化的 `OpResult`。Web 服务、CLI、测试都调用它。

设计约定：
- 所有路径由调用方（服务端）解析后传入，操作层不信任原始字符串；
- 结果表统一为 ``columns + rows``，避免把 pandas 对象泄漏到传输层；
- 产物写入调用方给定的 ``outdir``，返回相对文件名，不回传绝对路径。
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

Progress = Callable[..., None]


@dataclass
class OpResult:
    status: str = "ok"                 # ok | warn | error
    summary: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    log: str = ""
    artifacts: list[Path] = field(default_factory=list)

    def as_dict(self, artifact_names: list[str] | None = None) -> dict:
        return {
            "status": self.status,
            "summary": self.summary,
            "columns": self.columns,
            "rows": self.rows,
            "log": self.log,
            "artifacts": artifact_names or [],
        }


# ── 工具函数 ────────────────────────────────────────────────────────────────

def _frame(frame: Any) -> tuple[list[str], list[list[Any]]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return [], []
    columns = [str(c) for c in frame.columns]
    clean = frame.astype(object).where(pd.notna(frame), "")
    return columns, [[_cell(v) for v in row] for row in clean.values.tolist()]


def _cell(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _stash(path: str | os.PathLike | None, outdir: Path, name: str) -> Path | None:
    """把临时产物落到 run 目录，返回新路径。"""
    if not path:
        return None
    src = Path(path)
    if not src.exists():
        return None
    dest = outdir / name
    try:
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return dest
    except OSError:
        return src


def _noop(*_args, **_kwargs) -> None:
    return None


def _safe_label(value: Any, fallback: str = "target") -> str:
    label = re.sub(r"[^A-Za-z0-9_-]", "_", str(value or "")).strip("_")
    return label[:64] or fallback


def _count(rows: list[list]) -> int:
    return len(rows)


# ── 设计 ────────────────────────────────────────────────────────────────────

def design_single(pdb_path: str, params: dict, outdir: Path,
                  progress: Progress = _noop) -> OpResult:
    from webapp.pipelines import run_design

    tool = params.get("tool", "BFN (通用蛋白设计)")
    progress(0.1, desc="设计进行中")
    text, fasta_path, frame = run_design(
        pdb_path, tool,
        params.get("heavy", "H"), params.get("light", "L"),
        int(params.get("samples", 3)), bool(params.get("stochastic", False)),
        bool(params.get("eval_mode", True)),
        params.get("region", "A:10-25"),
        int(params.get("samples", 3)), bool(params.get("stochastic", False)),
        bool(params.get("eval_mode", True)),
        params.get("chains", "A"), float(params.get("temperature", 0.1)),
        int(params.get("samples", 3)), int(params.get("seed", 42)),
        params.get("omit", ""),
        params.get("chain", "A"), float(params.get("temperature", 0.1)),
        int(params.get("samples", 3)),
    )
    columns, rows = _frame(frame)
    artifact = _stash(fasta_path, outdir, "design.fasta")

    if not fasta_path:
        return OpResult("warn", "未产生候选序列", columns, rows, text)

    return OpResult(
        "ok",
        f"完成 · {_count(rows)} 条候选 · {tool.split(' (')[0]}",
        columns, rows, text,
        [artifact] if artifact else [],
    )


def design_batch(files: list[dict], params: dict, outdir: Path,
                 progress: Progress = _noop) -> OpResult:
    from webapp.pipelines import run_batch

    if not files:
        return OpResult("error", "没有输入文件")

    shims = [type("F", (), {"name": item["path"]})() for item in files]
    tool = params.get("tool", "BFN (通用蛋白设计)")
    text, summary_path, frame = run_batch(
        shims, tool,
        params.get("region", "A:10-25"),
        params.get("chains", "A"),
        float(params.get("temperature", 0.1)),
        int(params.get("samples", 3)),
        progress,
    )
    columns, rows = _frame(frame)
    artifact = _stash(summary_path, outdir, "batch_summary.json")
    return OpResult(
        "ok",
        f"批量完成 · {len(files)} 个文件 · {_count(rows)} 条候选",
        columns, rows, text,
        [artifact] if artifact else [],
    )


def design_target(params: dict, outdir: Path, progress: Progress = _noop) -> OpResult:
    from webapp.pipelines import run_target_design

    target_pdb = params.get("target_pdb")
    antibody_pdb = params.get("antibody_pdb")
    region = (params.get("epitope") or "").strip()
    if not target_pdb:
        return OpResult("error", "请先上传靶点结构")
    if not region:
        return OpResult("error", "请先分析并确认表位区域")
    if not antibody_pdb:
        return OpResult("error", "请先上传抗体结构")

    tool = params.get("tool", "BFN (抗体CDR设计)")
    text, fasta_text, validation, frame = run_target_design(
        target_pdb, params.get("chain", "A"), region,
        antibody_pdb, params.get("heavy", "H"), params.get("light", "L"),
        tool, bool(params.get("use_constraint", True)),
        float(params.get("cutoff", 10)),
        int(params.get("samples", 3)), bool(params.get("stochastic", False)),
        bool(params.get("eval_mode", True)),
        float(params.get("temperature", 0.1)), int(params.get("samples", 3)),
        int(params.get("seed", 42)), params.get("omit", ""),
        float(params.get("temperature", 0.1)), int(params.get("samples", 3)),
    )
    columns, rows = _frame(frame)
    log = text if not validation else f"{text}\n\n{'=' * 50}\n{validation}"
    artifacts: list[Path] = []
    if fasta_text:
        path = outdir / "design.fasta"
        path.write_text(fasta_text, encoding="utf-8")
        artifacts.append(path)
        return OpResult("ok", f"完成 · {_count(rows)} 条候选 · 表位 {region}",
                        columns, rows, log, artifacts)
    return OpResult("warn", "未产生候选序列", columns, rows, log)


def design_disease(params: dict, outdir: Path, progress: Progress = _noop) -> OpResult:
    import cascade_filter as cf
    from webapp.baselines import run_esmif, run_mpnn
    from webapp.design import run_bfn_protein
    from webapp.sequtils import fasta_to_pdb_file

    sequence = (params.get("sequence") or "").strip()
    target_pdb = params.get("target_pdb")
    region = (params.get("region") or "").strip() or "A:26-33,51-58,97-110"
    tool = params.get("tool", "BFN (通用蛋白设计)")
    samples = int(params.get("samples", 8))

    if not sequence:
        return OpResult("error", "请先选择支架或输入抗体序列")
    if not target_pdb:
        return OpResult("error", "请先加载靶点结构")

    ab_pdb, conversion = fasta_to_pdb_file(f">scaffold\n{sequence}")
    if not ab_pdb:
        return OpResult("error", f"无法生成抗体模板 — {conversion}")

    progress(0.15, desc="设计抗体序列")
    if tool.startswith("BFN"):
        report, fasta_text, results = run_bfn_protein(ab_pdb, region, samples, False, True)
    elif tool == "ProteinMPNN":
        chain = region.split(":", 1)[0]
        report, fasta_text, results = run_mpnn(ab_pdb, chain, samples, 0.1, 42, "")
    else:
        chain = region.split(":", 1)[0]
        report, fasta_text, results = run_esmif(ab_pdb, chain, 0.1, samples)

    filtered, cascade_report = results, ""
    try:
        filtered, cascade_report = cf.apply_cascade_misfolding(results, use_af2=False)
    except Exception:  # noqa: BLE001 - 级联失败不阻断产出
        filtered = results

    frame = pd.DataFrame([{
        "Rank": index,
        "Sequence": (item.get("sequence") or "")[:40],
        "PPL": round(item["ppl"], 2) if item.get("ppl") is not None else "-",
        "pLDDT": round(item["plddt"], 3) if item.get("plddt") is not None else "-",
        "ipTM": round(item["iptm"], 3) if item.get("iptm") is not None else "-",
    } for index, item in enumerate(filtered or results, 1)])
    columns, rows = _frame(frame)

    artifacts: list[Path] = []
    if fasta_text:
        path = outdir / "design.fasta"
        path.write_text(fasta_text, encoding="utf-8")
        artifacts.append(path)

    log = f"{report}\n\n{cascade_report}".strip()
    if not filtered:
        return OpResult("warn", "设计完成但没有候选通过级联", columns, rows, log)
    return OpResult("ok", f"完成 · {len(filtered)} 条候选通过级联",
                    columns, rows, log, artifacts)


# ── 验证 ────────────────────────────────────────────────────────────────────

def validate_af2(fasta_text: str, params: dict, outdir: Path,
                 progress: Progress = _noop) -> OpResult:
    from webapp.af2 import run_alphafold_prediction

    if not fasta_text or not fasta_text.strip():
        return OpResult("error", "请提供 FASTA 序列")

    log, pdb_path, _result_dir = run_alphafold_prediction(
        fasta_text, None,
        int(params.get("models", 1)), int(params.get("recycles", 1)),
        bool(params.get("dropout", False)), progress,
    )
    if not pdb_path:
        return OpResult("warn", "预测结束但没有 PDB 输出", [], [], log)

    artifact = _stash(pdb_path, outdir, Path(pdb_path).name)
    return OpResult("ok", "预测完成 · 已生成最佳结构", [], [], log,
                    [artifact] if artifact else [])


def validate_confidence(pdb_path: str, params: dict, outdir: Path,
                        progress: Progress = _noop) -> OpResult:
    from webapp.design import run_bfn_confidence_evaluation

    region = params.get("region", "A:10-25")
    text, frame = run_bfn_confidence_evaluation(pdb_path, region)
    columns, rows = _frame(frame)
    if not rows:
        return OpResult("warn", "评估未返回逐残基结果", [], [], text)
    return OpResult("ok", f"评估完成 · {len(rows)} 个残基", columns, rows, text)


def validate_iterative(pdb_path: str, params: dict, outdir: Path,
                       progress: Progress = _noop) -> OpResult:
    from af2_validator import validate_sequences
    from cascade_filter import apply_cascade_af2
    from disorderflow.datasets.protein import preprocess_protein_structure
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.misc import seed_all
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.transforms import get_transform
    from iterative_refiner import IterativeRefiner, format_refinement_report

    import torch

    from webapp.state import DEVICE, load_bfn

    region = (params.get("region") or "").strip()
    if not region:
        return OpResult("error", "请填写设计区域")
    samples = int(params.get("samples", 5))
    top_k = int(params.get("top_k", 2))
    rounds = int(params.get("rounds", 2))
    recycles = int(params.get("recycles", 1))

    def _regions(spec: str) -> dict[str, list[int]]:
        parsed: dict[str, list[int]] = {}
        for chain, body in re.findall(r"([A-Za-z0-9]+):([0-9,\-\s]+)", spec):
            indices: list[int] = []
            for segment in body.split(","):
                segment = segment.strip()
                if not segment:
                    continue
                if "-" in segment:
                    start, end = segment.split("-")
                    indices.extend(range(int(start), int(end) + 1))
                else:
                    indices.append(int(segment))
            parsed[chain] = sorted(set(indices))
        return parsed

    def design_fn(pdb, region_spec, count):
        regions = _regions(region_spec)
        if not regions:
            raise ValueError(f"区域格式无效: {region_spec}")
        model, config = load_bfn()
        seed_all(getattr(config.sampling, "seed", 42))
        structure = preprocess_protein_structure(pdb, chain_ids=list(regions))
        transform = get_transform([
            {"type": "mask_region", "regions": regions},
            {"type": "merge_protein"},
            {"type": "patch_protein"},
        ])
        batch = recursive_to(PaddingCollate()([transform(structure)]), DEVICE)
        mask = batch["generate_flag"][0].bool()
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        results = []
        for _ in range(count):
            with torch.no_grad():
                traj = model.sample(batch, sample_opt={"deterministic": False,
                                                       "num_recycles": 3})
            pred = traj[0][2][0][mask]
            logits = traj["pred_logits"][0][mask]
            logp = torch.log_softmax(logits[..., :20], dim=-1)
            nll = -logp[range(len(pred)), pred].mean()
            results.append({
                "sequence": "".join(alphabet[a] if a < 20 else "X" for a in pred.cpu()),
                "ppl": torch.exp(nll).item(),
                "entropy": -(torch.exp(logp) * logp).sum(dim=-1).mean().item(),
                "plddt": traj["plddt"][0][mask].mean().item(),
                "iptm": traj["iptm"][0].item(),
            })
        return results

    def af2_validator(sequences, output_dir=None, progress_cb=None):
        # IterativeRefiner 以 output_dir= 关键字调用（每轮独立目录）；
        # 单独调用时回落到本操作的 af2/ 目录。
        target_dir = str(output_dir) if output_dir is not None else str(outdir / "af2")
        return validate_sequences(sequences, output_dir=target_dir,
                                  num_recycle=recycles, progress_cb=progress_cb)

    refiner = IterativeRefiner(
        design_fn=design_fn, af2_validator=af2_validator,
        cascade_fn=apply_cascade_af2, n_samples=samples, top_k=top_k,
        max_rounds=rounds,
        progress_cb=lambda r, m, s: progress((r - 1) / m, desc=f"第 {r} 轮: {s}"),
        log_cb=lambda m: None,
    )
    result = refiner.run(pdb_path, region)
    report = format_refinement_report(result)
    best = getattr(result, "best_overall_sequence", "") or ""
    artifacts: list[Path] = []
    if best:
        path = outdir / "best_iterative.fasta"
        path.write_text(f">best_iterative rounds={rounds} "
                        f"ipTM={getattr(result, 'best_overall_iptm', 0):.3f}\n{best}\n",
                        encoding="utf-8")
        artifacts.append(path)
        return OpResult("ok", f"精修完成 · {rounds} 轮", [], [], report, artifacts)
    return OpResult("warn", "精修未产生结果", [], [], report)


def validate_loop(pdb_path: str, params: dict, outdir: Path,
                  progress: Progress = _noop) -> OpResult:
    import closed_loop_orchestrator as clo
    from af2_validator import validate_sequences

    import target_design_helpers as tdh
    from webapp.baselines import run_esmif, run_mpnn
    from webapp.design import run_bfn_protein

    tool = params.get("tool", "BFN (通用蛋白设计)")
    region = params.get("region", "A:25-35")
    ag_chain = params.get("ag_chain", "A")
    ab_chain = params.get("ab_chain", "B")

    def design_fn(pdb, region_spec, count):
        if tool.startswith("BFN"):
            _, _, results = run_bfn_protein(pdb, region_spec, count, True, True)
        elif tool == "ProteinMPNN":
            chains = re.findall(r"([A-Za-z0-9]+):", region_spec)
            _, _, results = run_mpnn(pdb, " ".join(chains) or "A", count, 0.1, 42, "")
        else:
            chain = region_spec.split(":")[0] if ":" in region_spec else "A"
            _, _, results = run_esmif(pdb, chain, 0.1, count)
        return results

    def af2_validator(sequences, output_dir, **kwargs):
        try:
            antigen = tdh.extract_seq_from_pdb(str(pdb_path), str(ag_chain))
            return validate_sequences(sequences, antigen_sequences=antigen,
                                      output_dir=output_dir,
                                      model_type="alphafold2_multimer_v3",
                                      num_recycle=1, progress_cb=None)
        except Exception as exc:  # noqa: BLE001
            return [{"sequence": s, "success": False, "error": str(exc),
                     "pdb_path": None} for s in sequences]

    physics = clo.create_physics_scorer(
        target_pdb_path=str(pdb_path), antibody_chains=ab_chain,
        antigen_chain=ag_chain, enable_relax=bool(params.get("relax", False)),
    )
    orchestrator = clo.ClosedLoopOrchestrator(
        design_fn=design_fn, af2_validator=af2_validator, physics_scorer=physics,
        finetune_fn=None,
        n_samples=int(params.get("samples", 15)),
        top_k=int(params.get("top_k", 5)),
        max_cycles=int(params.get("cycles", 3)),
        convergence_delta=float(params.get("convergence", 0.02)),
        selection_strategy="pareto_first",
        log_cb=lambda m: None,
        progress_cb=lambda c, m, s: progress((c - 1) / m + 0.1 / m,
                                             desc=f"Cycle {c}/{m}: {s}"),
    )
    result = orchestrator.run(initial_pdb=str(pdb_path), design_region=region)
    report = clo.format_closed_loop_report(result)

    rows = []
    for cycle in result.cycles:
        for item in cycle.all_ranked[:5]:
            rows.append({
                "Cycle": cycle.cycle_idx,
                "Rank": item.get("mo_rank", "?"),
                "Composite": item.get("mo_composite_score", 0),
                "ipTM": item.get("iptm", "") or "",
                "pLDDT": item.get("plddt", "") or "",
                "dG": item.get("dG", "") or "",
                "Sequence": (item.get("sequence", "") or "")[:40],
            })
    columns = ["Cycle", "Rank", "Composite", "ipTM", "pLDDT", "dG", "Sequence"]
    body = [[row.get(c, "") for c in columns] for row in rows]
    if not body:
        return OpResult("warn", "闭环没有产生记录", columns, body, report)
    return OpResult("ok", f"闭环完成 · {len(result.cycles)} 轮 · {len(body)} 条记录",
                    columns, body, report)


# ── 靶点 / 知识库 ───────────────────────────────────────────────────────────

def epitope_scan(pdb_path: str, chain: str, top_n: int = 20) -> dict:
    import target_design_helpers as tdh

    data = tdh.score_epitope_residues(
        pdb_path, chain,
        weights={"sasa": 0.4, "hydrophilicity": 0.35, "protrusion": 0.25},
    )
    table = tdh.format_epitope_table(data, top_n=int(top_n))
    region = (tdh.residues_to_region_spec(chain, [d["resseq"] for d in data[:5]])
              if data else "")
    rows = [[
        d.get("resseq"),
        d.get("resname"),
        round(float(d.get("combined_score", 0)), 3),
        round(float(d.get("rel_sasa", 0)), 2),
        round(float(d.get("hydrophilicity", 0)), 1),
        round(float(d.get("protrusion", 0)), 2),
    ] for d in (data or [])[:int(top_n)]]
    return {"table": table, "region": region,
            "columns": ["残基号", "残基", "综合分", "相对 SASA", "亲水性", "凸出度"],
            "rows": rows}


def chain_summary(pdb_path: str) -> dict:
    from webapp.sequtils import get_chain_info

    info = get_chain_info(pdb_path)
    chains = [{"id": cid, "length": meta.get("len", 0),
               "first": meta.get("first", "?"), "last": meta.get("last", "?"),
               "sequence": (meta.get("seq") or "")[:60]}
              for cid, meta in info.items()]
    return {"chains": chains}


def write_temp_pdb(sequence: str, outdir: Path) -> Path:
    """从 FASTA 文本生成 α-螺旋模板，落进 run 目录。"""
    from webapp.sequtils import fasta_to_pdb_file

    path, message = fasta_to_pdb_file(sequence)
    if not path:
        raise ValueError(message)
    dest = outdir / "template.pdb"
    shutil.copy2(path, dest)
    return dest


def tempdir_for(upload_root: Path) -> Path:
    return Path(tempfile.mkdtemp(prefix="df_", dir=str(upload_root)))


# ── 结构预览 ────────────────────────────────────────────────────────────────

def structure_preview(pdb_path: str, max_points: int = 480) -> dict:
    """抽取每条链的 CA 坐标，供前端画骨架投影。

    只返回坐标，不做任何渲染决策；降采样在服务端做，避免把 10 万行的
    PDB 整个丢给浏览器。
    """
    chains: dict[str, list[list[float]]] = {}
    for line in Path(pdb_path).read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        chain = (line[21:22] or "A").strip() or "A"
        try:
            resseq = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        chains.setdefault(chain, []).append([resseq, round(x, 2), round(y, 2), round(z, 2)])

    out = []
    for chain_id, atoms in chains.items():
        total = len(atoms)
        if total > max_points:
            step = max(1, total // max_points)
            atoms = atoms[::step]
        out.append({
            "id": chain_id,
            "count": total,
            "first": atoms[0][0] if atoms else 0,
            "last": atoms[-1][0] if atoms else 0,
            "atoms": atoms,
        })
    return {"chains": out}
