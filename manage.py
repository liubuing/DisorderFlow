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

# 运行期状态文件的固定位置：PID/日志用于后台模式的服务管理，
# CONFIG_FILE 是 Web 平台的主配置，DEFAULT_CONFIG 是批量设计时使用的 BFN 模型配置
PROJECT_DIR = Path(__file__).parent
PID_FILE = PROJECT_DIR / '.server.pid'
LOG_FILE = PROJECT_DIR / 'server.log'
CONFIG_FILE = PROJECT_DIR / 'app_config.yaml'
DEFAULT_CONFIG = PROJECT_DIR / 'configs' / 'demo_design.yml'


def load_app_config():
    # 读取平台主配置 app_config.yaml（服务器地址、模型检查点、批量输出等）；
    # 返回值：yaml 解析后的嵌套 dict
    with open(CONFIG_FILE, encoding='utf-8') as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════
# 服务管理
# ═══════════════════════════════════════════

def cmd_start(args):
    """启动 Web 服务器"""
    # PID 文件存在且进程存活 => 已有实例在跑，避免重复启动
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        if _is_running(pid):
            print(f"[!] 服务器已在运行 (PID: {pid})")
            print(f"    地址: http://127.0.0.1:7860")
            return

    config = load_app_config()
    host = config['server']['host']
    port = config['server']['port']
    entry = 'app_web.py'

    if args.bg:
        # 后台启动
        # 预检：端口若已被占用，新进程必然绑定失败（uvicorn 会立即退出），
        # 而端口探测只能说明"有人"在监听，不能说明是我们拉起的那个进程
        if _port_is_open(host, port):
            print(f"[✗] 端口 {host}:{port} 已被占用，请先 python manage.py stop")
            print(f"    若不是本服务占用：netstat -ano | findstr :{port} 后自行结束该进程")
            return

        # 以分离进程运行 app_web.py，stdout/stderr 重定向到 server.log，
        # 轮询端口（最长 90 秒）确认服务真正可用后才算启动成功
        print(f"[*] 后台启动服务器...")
        log_f = open(LOG_FILE, 'w')
        # Windows 需要显式创建新进程组并分离控制台，否则子进程随父进程一起退出；
        # 非 Windows 平台 creationflags 保持 0（该参数仅 Windows 有效）
        creationflags = 0
        if sys.platform == 'win32':
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        proc = subprocess.Popen(
            [sys.executable, entry],
            stdout=log_f, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_DIR),
            creationflags=creationflags,
            close_fds=True,
        )
        log_f.close()
        PID_FILE.write_text(str(proc.pid))
        deadline = time.time() + 90
        while time.time() < deadline:
            if not _is_running(proc.pid):
                break                      # 子进程已退出（多为绑定失败/导入报错）
            if _port_is_open(host, port):
                break
            time.sleep(1)
        # 端口通了之后再等 3 秒复核：致命错误（端口冲突等）通常在启动后
        # 一两秒内出现，立即宣布成功会误报
        if _is_running(proc.pid) and _port_is_open(host, port):
            time.sleep(3)
        if _is_running(proc.pid) and _port_is_open(host, port):
            print(f"[✓] 服务器已启动 (PID: {proc.pid})")
            print(f"    地址: http://{host}:{port}")
            print(f"    日志: {LOG_FILE}")
            print(f"    停止: python manage.py stop")
        else:
            if PID_FILE.exists():
                PID_FILE.unlink()
            print(f"[✗] 启动失败（PID {proc.pid}），日志尾部：")
            try:
                lines = LOG_FILE.read_text(
                    encoding='utf-8', errors='replace').splitlines()[-8:]
                for line in lines:
                    print("    " + line)
            except OSError:
                print(f"    （无法读取 {LOG_FILE}）")
    else:
        # 前台启动
        print(f"[*] 启动服务器: http://{host}:{port}")
        print(f"    按 Ctrl+C 停止")
        os.chdir(str(PROJECT_DIR))
        # 前台模式用 os.execv 原地替换为 app_web.py 进程（不产生额外父进程）
        os.execv(sys.executable, [sys.executable, entry])


def cmd_stop(args):
    """停止 Web 服务器"""
    # PID 文件缺失时退化为按端口扫描（netstat），尽力找到残留进程
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
    # 注意：os.kill(pid, 0) 在 Windows 上不是存在性探测——0 恰是 CTRL_C_EVENT，
    # 走 GenerateConsoleCtrlEvent；对 DETACHED_PROCESS（无控制台）必然失败，
    # 导致明明活着的进程被误判为"不存在"（stop 因此从不停不掉服务）。
    # Windows 用 OpenProcess + GetExitCodeProcess 探测，POSIX 才能用 os.kill(pid, 0)。
    if sys.platform == 'win32':
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _kill_process(pid, force=False):
    """终止进程"""
    # 常规停止用 SIGTERM（允许清理），强制停止用 SIGKILL；
    # 失败（如进程恰好退出）静默吞掉，由调用方在后续检查中处理
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.kill(pid, sig)
    except:
        pass


def _find_by_port(port=7860):
    """通过端口查找进程 (Windows)"""
    # 依赖 Windows 的 netstat -ano 输出：找到 LISTENING 行后取最后一列（PID）
    try:
        # 不能用 text=True：中文 Windows 的 netstat 表头是 GBK，而 text=True
        # 的解码发生在子进程读取线程里，UnicodeDecodeError 不归本函数的
        # try 管，会直接以线程异常的形式刷屏。改为按字节取回、replace 解码，
        # 我们只关心 ASCII 的端口/PID 列
        result = subprocess.run(
            ['netstat', '-ano'], capture_output=True
        )
        for line in result.stdout.decode('utf-8', errors='replace').split('\n'):
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                return int(parts[-1])
    except:
        pass
    return None


def _port_is_open(host, port):
    # 用一次短超时 TCP 连接探测服务是否就绪；任何 OSError（拒绝/超时）都视为未就绪
    import socket

    try:
        with socket.create_connection((host, int(port)), timeout=1):
            return True
    except OSError:
        return False


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
    # 按 --tool 选择分发；各实现函数返回可 JSON 序列化的结果 dict
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
    # 离线批量执行 BFN 区域重设计：加载冻结检查点 -> 遮蔽指定区域 -> 采样多条序列，
    # 返回包含困惑度（ppl）与序列恢复率（recovery）的统计结果
    from disorderflow.datasets.protein import preprocess_protein_structure
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.misc import load_config as _lc, seed_all
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.transforms import get_transform
    from disorderflow.models import get_model
    import torch, re

    # Parse region
    # 解析 "A:10-25,30 B:1-5" 形式的区域描述 -> {链 ID: 残基编号（去重升序）}
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
    # weights_only=False 是必须的：检查点内保存了 OmegaConf 配置对象而非纯张量
    ckpt = torch.load(gconfig.model.checkpoint, map_location='cpu', weights_only=False)
    # 模型超参与损失权重以训练时保存的配置为准，避免与当前代码默认值不一致
    mc = ckpt['config'].model
    if hasattr(ckpt['config'], 'train') and hasattr(ckpt['config'].train, 'loss_weights'):
        mc['loss_weight'] = dict(ckpt['config'].train.loss_weights)
    model = get_model(mc).to('cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    # 固定随机种子保证批量结果可复现
    seed_all(42)

    structure = preprocess_protein_structure(pdb_path, chain_ids=list(regions.keys()))
    # 预处理流水线：遮蔽设计区域 -> 合并链 -> 裁剪 patch，再 padding 成批
    transform = get_transform([
        {'type': 'mask_region', 'regions': regions},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    batch = recursive_to(PaddingCollate()([transform(structure)]), 'cpu')
    gen_mask = batch['generate_flag'][0].bool()

    native_aa = batch['aa'][0][gen_mask]
    # 20 标准氨基酸字母表；索引 >= 20（未知/掩码残基）统一显示为 X
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    native_seq = ''.join(AA[a] if a < 20 else 'X' for a in native_aa.cpu())

    seqs = []
    # 每条采样序列计算：ppl = exp(被生成残基的平均负对数似然)（越低越好），
    # recovery = 与天然序列的逐位一致比例
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
    # 抗体 CDR 重设计版本：与 _batch_bfn_protein 流程一致，
    # 区别在于用抗体专用预处理（H/L 链）并遮蔽 6 个 CDR 区域
    from disorderflow.datasets.custom import preprocess_antibody_structure
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.misc import load_config as _lc, seed_all
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.transforms import get_transform
    from disorderflow.models import get_model
    import torch

    gconfig, _ = _lc(DEFAULT_CONFIG)
    # weights_only=False：检查点包含配置对象；其余逻辑与蛋白版完全相同
    ckpt = torch.load(gconfig.model.checkpoint, map_location='cpu', weights_only=False)
    mc = ckpt['config'].model
    if hasattr(ckpt['config'], 'train') and hasattr(ckpt['config'].train, 'loss_weights'):
        mc['loss_weight'] = dict(ckpt['config'].train.loss_weights)
    model = get_model(mc).to('cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    seed_all(42)

    # 重/轻链 ID 可通过命令行覆盖，默认按常见命名取 H 与 L
    heavy = args.heavy or 'H'
    light = args.light or 'L'
    structure = preprocess_antibody_structure(pdb_path, heavy_chain=heavy, light_chain=light)

    # 同时遮蔽重链与轻链的 6 个 CDR 区域进行联合生成
    cdrs = ['H_CDR1','H_CDR2','H_CDR3','L_CDR1','L_CDR2','L_CDR3']
    transform = get_transform([
        {'type': 'mask_cdr', 'sample_cdr': cdrs, 'mode': 'all'},
        {'type': 'merge_antibody'},
        {'type': 'patch_around_anchor'},
    ])
    batch = recursive_to(PaddingCollate()([transform(structure)]), 'cpu')
    gen_mask = batch['generate_flag'][0].bool()

    native_aa = batch['aa'][0][gen_mask]
    # 20 标准氨基酸字母表；索引 >= 20（未知/掩码残基）统一显示为 X
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    native_seq = ''.join(AA[a] if a < 20 else 'X' for a in native_aa.cpu())

    seqs = []
    # 每条采样序列计算：ppl = exp(被生成残基的平均负对数似然)（越低越好），
    # recovery = 与天然序列的逐位一致比例
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
    # 调用仓库内 ProteinMPNN 官方脚本作为对比基线；解析其 FASTA 输出
    # （每个 ">T=" 头部记录 score/seq_recovery 等指标，序列在下一行）
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
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=300)
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
    # ESM-IF1 逆折叠基线：按单链坐标采样序列，统计与天然序列的恢复率
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
    # 递归打印嵌套 dict 配置；list 压成一行，便于在终端快速查看
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
    warnings = []

    # Python
    print(f"  Python: {sys.version}")
    if sys.version_info < (3, 14):
        warnings.append("冻结发布代码需要 Biopython 1.87 API；本机已验证的主运行时是 Python 3.14")

    # PyTorch
    try:
        import torch
        print(f"  PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            memory_gb = props.total_memory / 1024 ** 3
            print(f"  GPU: {props.name} ({memory_gb:.1f} GB, CUDA {torch.version.cuda})")
            if memory_gb < 16:
                warnings.append("GPU 显存低于 16 GB，AF2 Multimer 应使用单任务和低 recycle")
        else:
            warnings.append("Windows PyTorch 未检测到 CUDA；BFN 将回退到 CPU")
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

    # FastAPI
    try:
        import fastapi
        print(f"  FastAPI: {fastapi.__version__}")
    except Exception as e:
        errors.append(f"FastAPI: {e}")

    # ESM
    try:
        import esm
        print(f"  fair-esm: OK")
    except Exception as e:
        errors.append(f"fair-esm: {e}")

    # BFN model
    config = load_app_config()
    bfn_ckpt = Path(config['models']['bfn']['checkpoint']).expanduser()
    if not bfn_ckpt.is_absolute():
        bfn_ckpt = PROJECT_DIR / bfn_ckpt
    if bfn_ckpt.exists():
        print(f"  BFN 模型: OK ({bfn_ckpt})")
    else:
        errors.append(f"BFN 模型不存在: {bfn_ckpt}")

    # ProteinMPNN
    mpnn_weights = Path(config['models']['proteinmpnn']['weights_dir'])
    if not mpnn_weights.is_absolute():
        mpnn_weights = PROJECT_DIR / mpnn_weights
    if mpnn_weights.exists():
        pts = list(mpnn_weights.glob('*.pt'))
        print(f"  ProteinMPNN: OK ({len(pts)} 个权重文件)")
    else:
        errors.append(f"ProteinMPNN 权重目录不存在: {mpnn_weights}")

    # AlphaFold2
    af_runtime = config.get('alphafold', {}).get('af2', {})
    try:
        from modules.runtime_environment import resolve_colabfold_executable
        af2_exe = resolve_colabfold_executable(af_runtime, PROJECT_DIR)
    except Exception as e:
        af2_exe = None
        errors.append(f"AlphaFold2 配置无效: {e}")
    if af2_exe:
        print(f"  AlphaFold2 (ColabFold): OK ({af_runtime.get('backend', 'native')}: {af2_exe})")
    else:
        errors.append("AlphaFold2 (ColabFold) 可执行文件未找到")

    if af_runtime.get('backend') == 'wsl':
        distro = af_runtime.get('distribution', 'Ubuntu-24.04-D')
        environment = af_runtime.get('environment', 'venv_wsl')
        project_wsl = f"/mnt/{PROJECT_DIR.drive[0].lower()}{PROJECT_DIR.as_posix()[2:]}"
        python_wsl = f"{project_wsl}/{environment}/bin/python"
        try:
            result = subprocess.run(
                ["wsl.exe", "-d", distro, "--", python_wsl, "-c",
                 "import jax; print(jax.__version__); print(jax.devices())"],
                capture_output=True, text=True, timeout=30,
                encoding='utf-8', errors='replace',
            )
            if result.returncode == 0 and "CudaDevice" in result.stdout:
                print(f"  WSL JAX: GPU OK ({result.stdout.strip().replace(chr(10), ' | ')})")
            else:
                errors.append(f"WSL JAX 未检测到 CUDA: {(result.stderr or result.stdout).strip()}")
        except Exception as e:
            errors.append(f"WSL JAX 检查失败: {e}")

    try:
        from openmm import Platform
        platforms = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
        print(f"  OpenMM: {', '.join(platforms)}")
        if 'CUDA' not in platforms:
            warnings.append("Windows OpenMM 没有 CUDA 平台；正式 GPU MD 建议在 WSL 环境运行")
    except Exception as e:
        warnings.append(f"OpenMM 不可用: {e}")

    # Config
    if CONFIG_FILE.exists():
        print(f"  配置文件: OK ({CONFIG_FILE})")
    else:
        errors.append(f"配置文件不存在: {CONFIG_FILE}")

    print()
    if warnings:
        print(f"[!] {len(warnings)} 个提示:")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    if errors:
        print(f"[✗] {len(errors)} 个问题:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("[✓] 必要检查通过！可以运行 python manage.py start")


# ═══════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════

def main():
    # CLI 入口：start/stop/restart/status/batch/config/test 七个子命令；
    # 无参数时默认执行 start（保持旧行为兼容）
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
