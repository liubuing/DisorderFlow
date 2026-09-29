"""One-off: split app.py (3216 lines) into the webapp/ package.

Run once from the DisorderFlow root with the project venv:
    python scripts/oneoff/split_app_20260927.py

Keeps every function body verbatim; only module boundaries, import headers and
two nested-handler `global` mutations in create_ui are touched. The old app.py
stays recoverable via git.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'app.py'

text = SRC.read_text(encoding='utf-8')
lines = text.splitlines(keepends=True)
tree = ast.parse(text)

# ── collect top-level nodes ─────────────────────────────────────────────
spans = {}  # name -> (start, end) 1-based inclusive
order = []
for node in tree.body:
    start, end = node.lineno, getattr(node, 'end_lineno', node.lineno)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        name = node.name
    elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
    elif isinstance(node, ast.If):
        name = '__main__'
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        name = f'__import_{node.lineno}'
    elif isinstance(node, ast.Expr):
        name = f'__expr_{node.lineno}'
    else:
        name = f'__other_{node.lineno}'
    # pull directly preceding comment-only lines into the span (section banners)
    while start > 1 and lines[start - 2].lstrip().startswith('#'):
        start -= 1
    spans[name] = (start, end)
    order.append(name)

IMPORT_SPANS = [n for n in order if n.startswith('__import_')]
DOCSTRING = next(n for n in order if n.startswith('__expr_'))

# contiguous header: line 1 through the line before PROJECT_DIR
HDR_END = spans['PROJECT_DIR'][0] - 1  # exclusive
header_lines = lines[:HDR_END]
header = ''.join(header_lines)
pkg_header = header.replace(
    "sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))",
    "sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'modules'))",
)
assert 'sys.path.insert' in pkg_header, 'header lost the sys.path bootstrap'

PARTITION = {
    'state': ['PROJECT_DIR', 'CONFIG_FILE', 'DEFAULT_MODEL_CONFIG', 'AA_LETTERS',
              '_detect_device', 'DEVICE', '_bfn_model', '_bfn_config', '_esmif_model',
              '_app_config', 'load_app_config', 'get_bfn_ckpt', 'load_bfn', 'load_esmif',
              'CONFIDENCE_DEFAULTS', 'get_config_display', 'save_config_from_text',
              'get_system_status'],
    'sequtils': ['detect_chains', 'get_chain_info', 'on_upload', '_fmt_coord', 'BB_ATOMS',
                 'generate_pdb_from_sequence', 'fasta_to_pdb_file', 'parse_fasta',
                 '_save_fasta', 'on_fasta_input'],
    'design': ['confidence_quality_label', 'diagnose_design_results',
               'build_results_dataframe', 'run_bfn_confidence_evaluation',
               'run_bfn_antibody', 'run_bfn_protein'],
    'baselines': ['run_mpnn', 'run_esmif', 'AA_NAMES'],
    'af2': ['_predicted_pdb', 'run_alphafold_prediction'],
    'pipelines': ['run_batch', 'run_target_design', 'run_design',
                  'run_unified_pipeline', 'run_unified_pipeline_for_ui'],
    'ui': ['create_ui'],
    '__main__': ['__main__'],
}
owner = {name: mod for mod, names in PARTITION.items() for name in names}
missing = [n for n in order if not n.startswith('__') and n not in owner]
assert not missing, f'unassigned top-level names: {missing}'

# every node must be inside the header range or explicitly assigned
HDR_NODE_NAMES = {n for n in order if spans[n][1] <= HDR_END}
uncovered = [n for n in order
             if n not in owner and n not in HDR_NODE_NAMES and spans[n][0] > HDR_END]
assert not uncovered, f'uncovered top-level nodes: {uncovered}'

# ── per-module name tables for cross-imports ───────────────────────────
defined = {mod: set(names) for mod, names in PARTITION.items()}
mod_of = {name: mod for mod, names in PARTITION.items() for name in names}
header_names = set()
for n in IMPORT_SPANS:
    s, e = spans[n]
    node = ast.parse(''.join(lines[s - 1:e])).body[0]
    if isinstance(node, ast.Import):
        header_names |= {a.asname or a.name.split('.')[0] for a in node.names}
    else:
        header_names |= {a.asname or node.module.split('.')[0] if a.name == '*' else (a.asname or a.name) for a in node.names}
builtins_set = set(dir(__builtins__)) | {'__file__', '__name__', '__doc__'}

PUBLIC = {}  # mod -> list of public names it defines (for facade)


def render_module(mod: str) -> str:
    names = PARTITION[mod]
    body = []
    for name in names:
        s, e = spans[name]
        body.append(''.join(lines[s - 1:e]))
        if not body[-1].endswith('\n'):
            body[-1] += '\n'
    module_src = ''.join(body)

    # webapp/ is one level below the project root: correct the state globals' base dir
    if mod == 'state':
        module_src = module_src.replace(
            "PROJECT_DIR = Path(__file__).parent\n",
            "PROJECT_DIR = Path(__file__).resolve().parent.parent\n", 1)

    # free names referenced but not defined/imported here
    parsed = ast.parse(module_src)
    used = {n.id for n in ast.walk(parsed) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    own = set(names)
    need = {}
    for name in sorted(used - own - header_names - builtins_set):
        src_mod = mod_of.get(name)
        if src_mod and src_mod != mod:
            need.setdefault(src_mod, []).append(name)
    import_block = ''
    for src_mod, names_list in sorted(need.items()):
        import_block += f'from webapp.{src_mod} import ' + ', '.join(names_list) + '\n'

    out = f'"""app.py split: {mod} (auto-generated 2026-09-27, bodies verbatim)."""\n'
    out += pkg_header + import_block + '\n\n' + module_src
    PUBLIC[mod] = [n for n in names if not n.startswith('_')]
    return out


webapp = ROOT / 'webapp'
webapp.mkdir(exist_ok=True)
(webapp / '__init__.py').write_text('"""DisorderFlow web app package (split from app.py, 2026-09-27)."""\n', encoding='utf-8')

for mod in ['state', 'sequtils', 'design', 'baselines', 'af2', 'pipelines', 'ui']:
    (webapp / f'{mod}.py').write_text(render_module(mod), encoding='utf-8')

# ── facade: re-export every public name for `from app import X` users ──
facade = '"""Aggregated public surface of the old app.py (auto-generated)."""\n'
for mod, names in PUBLIC.items():
    if names:
        facade += f'from webapp.{mod} import ' + ', '.join(sorted(names)) + '\n'
(webapp / 'facade.py').write_text(facade, encoding='utf-8')

# ── surgical edits in ui.py: nested handlers must mutate state module ──
ui_path = webapp / 'ui.py'
ui = ui_path.read_text(encoding='utf-8')
ui = ui.replace(
    '                def do_reload_config():\n'
    '                    global _app_config\n'
    '                    _app_config = None\n',
    '                def do_reload_config():\n'
    '                    state._app_config = None\n',
)
ui = ui.replace(
    '                def do_reload_bfn():\n'
    '                    global _bfn_model, _bfn_config\n'
    '                    _bfn_model = None; _bfn_config = None\n',
    '                def do_reload_bfn():\n'
    '                    state._bfn_model = None; state._bfn_config = None\n',
)
ui = ui.replace('"""app.py split: ui', '"""app.py split: ui', 1)
ui = ui.replace('from webapp.af2 import', 'from webapp import state\nfrom webapp.af2 import', 1)
ui_path.write_text(ui, encoding='utf-8')

# ── new thin app.py ──
s, e = spans['__main__']
tail = ''.join(lines[s - 1:e])
shim = (
    '#!/usr/bin/env python\n'
    '"""\n蛋白质序列设计平台 — Gradio Web 界面（管理增强版）\n\n'
    '2026-09-27 拆分：实现位于 webapp/ 包（state/sequtils/design/baselines/af2/pipelines/ui）。\n'
    '本文件只是入口 + 兼容再导出（modules/ 与 scripts/ 里的 `from app import X` 不受影响）。\n\n'
    '访问: http://127.0.0.1:7860  （仅本机访问，外部无法连接）\n'
    '管理: python manage.py start|stop|restart|status|batch|config|test\n'
    '"""\n'
    + header
    + 'from webapp.facade import *  # noqa: F401,F403\n'
    + 'from webapp.state import load_app_config, load_bfn\n'
    + 'from webapp.ui import create_ui\n\n'
    + tail
)
SRC.write_text(shim, encoding='utf-8')
print('split complete:', [p.name for p in sorted(webapp.glob('*.py'))])
