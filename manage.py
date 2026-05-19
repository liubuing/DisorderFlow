#!/usr/bin/env python
"""
蛋白质序列设计平台 — 管理工具

用法:
  python manage.py start             启动 Web 服务
  python manage.py start --bg        后台启动
  python manage.py stop              停止服务
  python manage.py restart           重启服务
  python manage.py status            查看服务状态
  python manage.py batch <dir>       批量处理 PDB 目录
  python manage.py batch <dir> --tool bfn_protein --region "A:10-30"
  python manage.py config            显示当前配置
  python manage.py config --edit     在编辑器中打开配置
  python manage.py test              运行自检
"""
import os, sys, json, signal, time, yaml, subprocess, argparse
from pathlib import Path

# Fix Unicode output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_DIR = Path(__file__).parent
PID_FILE = PROJECT_DIR / '.server.pid'
LOG_FILE = PROJECT_DIR / 'server.log'
CONFIG_FILE = PROJECT_DIR / 'app_config.yaml'
DEFAULT_CONFIG = PROJECT_DIR / 'configs' / 'demo_design.yml'


def load_app_config():
    with open(CONFIG_FILE, encoding='utf-8') as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════
# 服务管理
# ═══════════════════════════════════════════

def cmd_start(args):
    """启动 Web 服务器"""
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        if _is_running(pid):
            print(f"[!] 服务器已在运行 (PID: {pid})")
            print(f"    地址: http://127.0.0.1:7860")
            return

    config = load_app_config()
    host = config['server']['host']
    port = config['server']['port']
    auto_open = config['server'].get('auto_open', True)

    if args.bg:
        # 后台启动
        print(f"[*] 后台启动服务器...")
        log_f = open(LOG_FILE, 'w')
        proc = subprocess.Popen(
            [sys.executable, 'app.py'],
            stdout=log_f, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_DIR),
        )
        PID_FILE.write_text(str(proc.pid))
        time.sleep(3)
        if _is_running(proc.pid):
            print(f"[✓] 服务器已启动 (PID: {proc.pid})")
            print(f"    地址: http://{host}:{port}")
            print(f"    日志: {LOG_FILE}")
            print(f"    停止: python manage.py stop")
        else:
            print(f"[✗] 启动失败，查看日志: {LOG_FILE}")
    else:
        # 前台启动
        print(f"[*] 启动服务器: http://{host}:{port}")
        print(f"    按 Ctrl+C 停止")
        os.chdir(str(PROJECT_DIR))
        os.execv(sys.executable, [sys.executable, 'app.py'])


def cmd_stop(args):
    """停止 Web 服务器"""
    if not PID_FILE.exists():
        print("[!] 未找到运行中的服务器 (PID 文件不存在)")
        # 尝试通过端口查找
        pid = _find_by_port()
        if pid:
            print(f"[*] 通过端口找到进程 PID:{pid}，正在终止...")
            _kill_process(pid)
        return

    pid = int(PID_FILE.read_text().strip())
    if _is_running(pid):
        print(f"[*] 停止服务器 (PID: {pid})...")
        _kill_process(pid)
        time.sleep(1)
        if not _is_running(pid):
            print(f"[✓] 服务器已停止")
            PID_FILE.unlink()
        else:
            print(f"[✗] 无法停止，尝试强制终止...")
            _kill_process(pid, force=True)
    else:
        print(f"[!] 进程 {pid} 已不存在，清理 PID 文件")
        PID_FILE.unlink()


def cmd_restart(args):
    """重启 Web 服务器"""
    print("[*] 重启服务器...")
    cmd_stop(args)
    time.sleep(2)
    args.bg = True
    cmd_start(args)


def cmd_status(args):
    """查看服务状态"""
    config = load_app_config()
    host = config['server']['host']
    port = config['server']['port']

    # 检查 PID 文件
    pid = None
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())

    running = _is_running(pid) if pid else False

    # 检查端口
    import socket
    port_open = False
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        port_open = s.connect_ex((host, port)) == 0
        s.close()
    except:
        pass

    print("=" * 50)
    print("  蛋白质序列设计平台 — 服务状态")
    print("=" * 50)
    print(f"  地址:         http://{host}:{port}")
    print(f"  服务运行中:   {'✓ 是' if running else '✗ 否'}")
    print(f"  端口响应:     {'✓ 是' if port_open else '✗ 否'}")
    if pid:
        print(f"  进程 PID:     {pid}")
    print(f"  配置文件:     {CONFIG_FILE}")
    print(f"  日志文件:     {LOG_FILE}")
    if LOG_FILE.exists():
        size = LOG_FILE.stat().st_size
        print(f"  日志大小:     {size/1024:.1f} KB")
    print("=" * 50)

    # 打印最近日志
    if LOG_FILE.exists() and args.verbose:
        print("\n最近日志 (最后20行):")
        print("-" * 50)
        lines = LOG_FILE.read_text().split('\n')[-20:]
        for line in lines:
            print(f"  {line}")


def _is_running(pid):
    """检查进程是否运行"""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _kill_process(pid, force=False):
    """终止进程"""
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.kill(pid, sig)
    except:
        pass


def _find_by_port(port=7860):
    """通过端口查找进程 (Windows)"""
    try:
        result = subprocess.run(
            ['netstat', '-ano'], capture_output=True, text=True
        )
        for line in result.stdout.split('\n'):
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                return int(parts[-1])
    except:
        pass
    return None


# ═══════════════════════════════════════════
# 批量处理
# ═══════════════════════════════════════════

def cmd_batch(args):
    """批量处理 PDB 文件"""
    import torch
    import numpy as np

    pdb_dir = Path(args.directory)
    if not pdb_dir.exists():
        print(f"[✗] 目录不存在: {pdb_dir}")
        return

    pdb_files = sorted(pdb_dir.glob('*.pdb'))
    if not pdb_files:
        print(f"[✗] 目录中没有 PDB 文件: {pdb_dir}")
        return

    config = load_app_config()
    output_dir = Path(config['batch']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    tool = args.tool
    print(f"[*] 批量处理 {len(pdb_files)} 个 PDB 文件")
    print(f"    工具: {tool}")
    print(f"    输出: {output_dir}")

    results = []
    for i, pdb_path in enumerate(pdb_files, 1):
        pdb_name = pdb_path.stem
        print(f"\n[{i}/{len(pdb_files)}] {pdb_name}")

        try:
            result = _run_single_job(str(pdb_path), tool, args)
            result['pdb'] = pdb_name
            results.append(result)

            # 单独保存
            if config['batch']['save_individual']:
                out_file = output_dir / f'{pdb_name}_{tool}.json'
                with open(out_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
            print(f"  ✓ 完成")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            results.append({'pdb': pdb_name, 'error': str(e)})

    # 汇总
    print(f"\n{'='*60}")
    print(f"  批量处理完成: {len(results)} 个文件")
    success = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    print(f"  成功: {len(success)}  失败: {len(failed)}")

    if config['batch']['save_summary']:
        _save_summary(results, output_dir, config)

    print(f"  结果保存在: {output_dir}")


def _run_single_job(pdb_path, tool, args):
    """执行单个设计任务"""
    if tool == 'bfn_protein':
        return _batch_bfn_protein(pdb_path, args)
    elif tool == 'bfn_antibody':
        return _batch_bfn_antibody(pdb_path, args)
    elif tool == 'proteinmpnn':
        return _batch_mpnn(pdb_path, args)
    elif tool == 'esmif':
        return _batch_esmif(pdb_path, args)
    elif tool == 'all':
        return _batch_all(pdb_path, args)
    else:
        raise ValueError(f"未知工具: {tool}")


def _batch_bfn_protein(pdb_path, args):
    from antibodydesignbfn.datasets.protein import preprocess_protein_structure
    from antibodydesignbfn.utils.train import recursive_to
    from antibodydesignbfn.utils.misc import load_config as _lc, seed_all
    from antibodydesignbfn.utils.data import PaddingCollate
    from antibodydesignbfn.utils.transforms import get_transform
    from antibodydesignbfn.models import get_model
    import torch, re

    # Parse region
    region_spec = args.region or 'A:10-25'
    regions = {}
    for cid, spec in re.findall(r'([A-Za-z0-9]+):([0-9,\-\s]+)', region_spec):
        indices = []
        for seg in spec.split(','):
            seg = seg.strip()
            if not seg: continue
            if '-' in seg:
                a, b = seg.split('-')
                indices.extend(range(int(a.strip()), int(b.strip())+1))
            else:
                indices.append(int(seg))
        regions[cid] = sorted(set(indices))

    gconfig, _ = _lc(DEFAULT_CONFIG)
    ckpt = torch.load(gconfig.model.checkpoint, map_location='cpu', weights_only=False)
    mc = ckpt['config'].model
    if hasattr(ckpt['config'], 'train') and hasattr(ckpt['config'].train, 'loss_weights'):
        mc['loss_weight'] = dict(ckpt['config'].train.loss_weights)
    model = get_model(mc).to('cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    seed_all(42)

    structure = preprocess_protein_structure(pdb_path, chain_ids=list(regions.keys()))
    transform = get_transform([
        {'type': 'mask_region', 'regions': regions},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    batch = recursive_to(PaddingCollate()([transform(structure)]), 'cpu')
    gen_mask = batch['generate_flag'][0].bool()

    native_aa = batch['aa'][0][gen_mask]
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    native_seq = ''.join(AA[a] if a < 20 else 'X' for a in native_aa.cpu())

    seqs = []
    for i in range(args.samples or 3):
        with torch.no_grad():
            traj = model.sample(batch, sample_opt={'deterministic': not args.stochastic})
        pred_aa = traj[0][2][0][gen_mask]
        seq = ''.join(AA[a] if a < 20 else 'X' for a in pred_aa.cpu())
        logits = traj['pred_logits'][0][gen_mask]
        lp = torch.log_softmax(logits[..., :20], dim=-1)
        nll = -lp[range(len(pred_aa)), pred_aa].mean()
        ppl = torch.exp(nll).item()
        rec = sum(1 for a,b in zip(seq, native_seq) if a==b)/len(native_seq)
        seqs.append({'sample': i+1, 'sequence': seq, 'ppl': round(ppl,2), 'recovery': round(rec,3)})

    best = min(seqs, key=lambda x: x['ppl'])
    return {
        'tool': 'bfn_protein', 'region': region_spec,
        'native': native_seq, 'best_seq': best['sequence'], 'best_ppl': best['ppl'],
        'samples': seqs,
    }


def _batch_bfn_antibody(pdb_path, args):
    from antibodydesignbfn.datasets.custom import preprocess_antibody_structure
    from antibodydesignbfn.utils.train import recursive_to
    from antibodydesignbfn.utils.misc import load_config as _lc, seed_all
    from antibodydesignbfn.utils.data import PaddingCollate
    from antibodydesignbfn.utils.transforms import get_transform
    from antibodydesignbfn.models import get_model
    import torch

    gconfig, _ = _lc(DEFAULT_CONFIG)
    ckpt = torch.load(gconfig.model.checkpoint, map_location='cpu', weights_only=False)
    mc = ckpt['config'].model
    if hasattr(ckpt['config'], 'train') and hasattr(ckpt['config'].train, 'loss_weights'):
        mc['loss_weight'] = dict(ckpt['config'].train.loss_weights)
    model = get_model(mc).to('cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    seed_all(42)

    heavy = args.heavy or 'H'
    light = args.light or 'L'
    structure = preprocess_antibody_structure(pdb_path, heavy_chain=heavy, light_chain=light)

    cdrs = ['H_CDR1','H_CDR2','H_CDR3','L_CDR1','L_CDR2','L_CDR3']
    transform = get_transform([
        {'type': 'mask_cdr', 'sample_cdr': cdrs, 'mode': 'all'},
        {'type': 'merge_antibody'},
        {'type': 'patch_around_anchor'},
    ])
    batch = recursive_to(PaddingCollate()([transform(structure)]), 'cpu')
    gen_mask = batch['generate_flag'][0].bool()

    native_aa = batch['aa'][0][gen_mask]
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    native_seq = ''.join(AA[a] if a < 20 else 'X' for a in native_aa.cpu())

    seqs = []
    for i in range(args.samples or 3):
        with torch.no_grad():
            traj = model.sample(batch, sample_opt={'deterministic': not args.stochastic})
        pred_aa = traj[0][2][0][gen_mask]
        seq = ''.join(AA[a] if a < 20 else 'X' for a in pred_aa.cpu())
        logits = traj['pred_logits'][0][gen_mask]
        lp = torch.log_softmax(logits[..., :20], dim=-1)
        nll = -lp[range(len(pred_aa)), pred_aa].mean()
        ppl = torch.exp(nll).item()
        rec = sum(1 for a,b in zip(seq, native_seq) if a==b)/len(native_seq)
        seqs.append({'sample': i+1, 'sequence': seq, 'ppl': round(ppl,2), 'recovery': round(rec,3)})

    best = min(seqs, key=lambda x: x['ppl'])
    return {
        'tool': 'bfn_antibody', 'heavy': heavy, 'light': light,
        'native': native_seq, 'best_seq': best['sequence'], 'best_ppl': best['ppl'],
        'samples': seqs,
    }


def _batch_mpnn(pdb_path, args):
    out_dir = Path('batch_mpnn_temp')
    out_dir.mkdir(exist_ok=True)
    cmd = [
        sys.executable, 'ProteinMPNN/protein_mpnn_run.py',
        '--pdb_path', pdb_path,
        '--pdb_path_chains', args.chains or 'A',
        '--num_seq_per_target', str(args.samples or 3),
        '--sampling_temp', args.temperature or '0.1',
        '--seed', str(args.seed or 42),
        '--out_folder', str(out_dir),
        '--save_score', '1',
        '--path_to_model_weights', 'ProteinMPNN/vanilla_model_weights',
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:500])

    pid = Path(pdb_path).stem
    fa = out_dir / 'seqs' / f'{pid}.fa'
    seqs = []
    if fa.exists():
        with open(fa) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>T='):
                    parts = {p.split('=')[0].strip(): p.split('=')[1].strip()
                             for p in line.split(',') if '=' in p}
                    seq = next(f).strip()
                    seqs.append({'sample': parts.get('sample'), 'sequence': seq,
                                 'score': parts.get('score'), 'recovery': parts.get('seq_recovery')})
    return {'tool': 'proteinmpnn', 'samples': seqs}


def _batch_esmif(pdb_path, args):
    from esm.pretrained import esm_if1_gvp4_t16_142M_UR50
    from esm.inverse_folding import util as if_util
    import torch

    model, _ = esm_if1_gvp4_t16_142M_UR50()
    model = model.to('cpu').eval()
    chain = args.chain or 'A'
    temp = float(args.temperature or 0.1)
    coords, native = if_util.load_coords(pdb_path, chain)

    seqs = []
    for i in range(args.samples or 3):
        s = model.sample(coords, temperature=temp)
        rec = sum(1 for a,b in zip(s, native) if a==b)/len(native)
        seqs.append({'sample': i+1, 'sequence': s, 'recovery': round(rec, 3)})
    return {'tool': 'esmif', 'chain': chain, 'native': native, 'samples': seqs}


def _batch_all(pdb_path, args):
    """Run all three tools on one PDB"""
    results = {}
    for tool_name, fn in [('bfn_protein', _batch_bfn_protein),
                           ('proteinmpnn', _batch_mpnn),
                           ('esmif', _batch_esmif)]:
        try:
            results[tool_name] = fn(pdb_path, args)
        except Exception as e:
            results[tool_name] = {'error': str(e)}
    return results


def _save_summary(results, output_dir, config):
    """保存汇总表"""
    fmt = config['batch']['summary_format']

    # CSV
    if fmt == 'csv':
        import csv
        csv_path = output_dir / 'summary.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['PDB', 'Tool', 'Best_Seq', 'Best_PPL', 'Recovery', 'Native'])
            for r in results:
                if 'error' in r:
                    writer.writerow([r['pdb'], '', 'ERROR', '', '', ''])
                elif 'bfn_protein' in str(r.get('tool', '')):
                    writer.writerow([r['pdb'], r['tool'], r.get('best_seq',''),
                                     r.get('best_ppl',''), r.get('samples',[{}])[0].get('recovery',''),
                                     r.get('native','')])
                elif 'samples' in r:
                    best = min(r['samples'], key=lambda x: float(x.get('score', x.get('ppl', 99))))
                    writer.writerow([r['pdb'], r.get('tool',''),
                                     best.get('sequence','')[:60],
                                     best.get('score', best.get('ppl','')),
                                     best.get('recovery',''),
                                     r.get('native','')])

    # JSON
    json_path = output_dir / 'summary.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  汇总: {csv_path if fmt=='csv' else ''} {json_path}")


# ═══════════════════════════════════════════
# 配置管理
# ═══════════════════════════════════════════

def cmd_config(args):
    """显示或编辑配置"""
    if args.edit:
        import platform
        config_str = CONFIG_FILE.read_text(encoding='utf-8')
        if platform.system() == 'Windows':
            os.startfile(str(CONFIG_FILE))
        else:
            editor = os.environ.get('EDITOR', 'nano')
            subprocess.call([editor, str(CONFIG_FILE)])
        return

    config = load_app_config()
    print("=" * 60)
    print("  当前配置")
    print("=" * 60)
    _print_config(config, indent=2)


def _print_config(d, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{' ' * indent}{k}:")
            _print_config(v, indent + 2)
        elif isinstance(v, list):
            print(f"{' ' * indent}{k}: [{', '.join(str(x) for x in v)}]")
        else:
            print(f"{' ' * indent}{k}: {v}")


# ═══════════════════════════════════════════
# 自检
# ═══════════════════════════════════════════

def cmd_test(args):
    """运行环境自检"""
    print("[*] 环境自检...")
    errors = []

    # Python
    print(f"  Python: {sys.version}")

    # PyTorch
    try:
        import torch
        print(f"  PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except Exception as e:
        errors.append(f"PyTorch: {e}")

    # NumPy
    try:
        import numpy
        print(f"  NumPy: {numpy.__version__}")
    except Exception as e:
        errors.append(f"NumPy: {e}")

    # Biopython
    try:
        import Bio
        print(f"  Biopython: {Bio.__version__}")
    except Exception as e:
        errors.append(f"Biopython: {e}")

    # Gradio
    try:
        import gradio
        print(f"  Gradio: {gradio.__version__}")
    except Exception as e:
        errors.append(f"Gradio: {e}")

    # ESM
    try:
        import esm
        print(f"  fair-esm: OK")
    except Exception as e:
        errors.append(f"fair-esm: {e}")

    # BFN model
    config = load_app_config()
    bfn_ckpt = config['models']['bfn']['checkpoint']
    if os.path.exists(bfn_ckpt):
        print(f"  BFN 模型: OK ({bfn_ckpt})")
    else:
        errors.append(f"BFN 模型不存在: {bfn_ckpt}")

    # ProteinMPNN
    mpnn_weights = Path(config['models']['proteinmpnn']['weights_dir'])
    if mpnn_weights.exists():
        pts = list(mpnn_weights.glob('*.pt'))
        print(f"  ProteinMPNN: OK ({len(pts)} 个权重文件)")
    else:
        errors.append(f"ProteinMPNN 权重目录不存在: {mpnn_weights}")

    # AlphaFold2
    af_cfg = config.get('alphafold', {})
    af2_venv = af_cfg.get('af2', {}).get('venv', '')
    if af2_venv:
        af2_exe = Path(af2_venv) / 'Scripts' / 'colabfold_batch.exe'
        if af2_exe.exists():
            print(f"  AlphaFold2 (ColabFold): OK ({af2_exe})")
        else:
            print(f"  AlphaFold2 (ColabFold): 未找到 ({af2_exe})")
    else:
        print(f"  AlphaFold2: 未配置")

    # Config
    if CONFIG_FILE.exists():
        print(f"  配置文件: OK ({CONFIG_FILE})")
    else:
        errors.append(f"配置文件不存在: {CONFIG_FILE}")

    print()
    if errors:
        print(f"[✗] {len(errors)} 个问题:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("[✓] 所有检查通过！可以运行 python manage.py start")


# ═══════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='蛋白质序列设计平台 — 管理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python manage.py start                前台启动 Web 服务
  python manage.py start --bg           后台启动
  python manage.py stop                 停止服务
  python manage.py restart              重启服务
  python manage.py status               查看状态
  python manage.py batch ./my_pdbs      批量处理目录中所有 PDB
  python manage.py batch ./pdbs --tool proteinmpnn --chains "A B"
  python manage.py config               查看当前配置
  python manage.py config --edit        编辑配置文件
  python manage.py test                 运行环境自检
        """
    )
    sub = parser.add_subparsers(dest='command')

    # start
    p_start = sub.add_parser('start', help='启动 Web 服务')
    p_start.add_argument('--bg', action='store_true', help='后台运行')

    # stop
    sub.add_parser('stop', help='停止 Web 服务')

    # restart
    sub.add_parser('restart', help='重启 Web 服务')

    # status
    p_status = sub.add_parser('status', help='查看服务状态')
    p_status.add_argument('-v', '--verbose', action='store_true', help='显示最近日志')

    # batch
    p_batch = sub.add_parser('batch', help='批量处理 PDB 文件')
    p_batch.add_argument('directory', help='包含 PDB 文件的目录')
    p_batch.add_argument('--tool', default='bfn_protein',
                         choices=['bfn_protein', 'bfn_antibody', 'proteinmpnn', 'esmif', 'all'],
                         help='设计工具 (默认: bfn_protein)')
    p_batch.add_argument('--region', default='A:10-25', help='BFN 设计区域')
    p_batch.add_argument('--chains', default='A', help='ProteinMPNN 链')
    p_batch.add_argument('--chain', default='A', help='ESM-IF 链')
    p_batch.add_argument('--heavy', default='H', help='BFN抗体 重链')
    p_batch.add_argument('--light', default='L', help='BFN抗体 轻链')
    p_batch.add_argument('--samples', type=int, default=3, help='每PDB生成序列数')
    p_batch.add_argument('--temperature', default='0.1', help='采样温度')
    p_batch.add_argument('--seed', type=int, default=42, help='随机种子')
    p_batch.add_argument('--stochastic', action='store_true', help='随机采样')

    # config
    p_config = sub.add_parser('config', help='配置管理')
    p_config.add_argument('--edit', action='store_true', help='编辑配置文件')

    # test
    sub.add_parser('test', help='环境自检')

    args = parser.parse_args()

    if args.command == 'start' or args.command is None:
        cmd_start(args if args.command else argparse.Namespace(bg=False))
    elif args.command == 'stop':
        cmd_stop(args)
    elif args.command == 'restart':
        cmd_restart(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'batch':
        cmd_batch(args)
    elif args.command == 'config':
        cmd_config(args)
    elif args.command == 'test':
        cmd_test(args)


if __name__ == '__main__':
    main()
