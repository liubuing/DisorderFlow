#!/usr/bin/env python
"""Fix CJK/Unicode encoding corruption across the DisorderFlow codebase.

Run: python fix_encoding.py --dry-run  (preview only)
     python fix_encoding.py             (apply fixes)
"""

import os
import sys
import argparse
from pathlib import Path

# Corrupted hex bytes -> correct string
# Hex values verified via: PYTHONIOENCODING=utf-8 python -c "print(c.encode('utf-8').hex())"
REPLACEMENTS_RAW = [
    # Emoji
    ('e9a683e5b986', '\U0001f3af'),           # 🎯 dartboard
    ('e9a683e6919d', '\U0001f4e6'),           # 📦 package
    ('e9a683e695ad', '\U0001f52e'),           # 🔮 crystal ball
    ('e988bfe6ac99e7ac8d', '⚙️'),   # ⚙️ gear
    ('e9a683e694a7', '\U0001f504'),           # 🔄 refresh
    ('e9a683e690b3', '\U0001f4ca'),           # 📊 chart
    ('e9a683e6ae8c', '\U0001f680'),           # 🚀 rocket

    # CJK phrases (longest first)
    ('e7bc81e786b6e7abb4e5aeb8e383a4e7b694e5a8b4', '统一工作流'),  # 统一工作流
    ('e7bc83e4bf8ae690b4efb981e7988e', '置信度评估'),              # 置信度评估
    ('e98d90e591afe79686e7bc83e4bf8ae690b4', '内置置信度'),        # 内置置信度
    ('e7bc81e692b4e780afe6a3b0e58bace7a5b4', '结构预测'),              # 结构预测
    ('e98eb5e5bd92e599bae6beb6e58bade6828a', '批量处理'),              # 批量处理
    ('e98d97e69b9fe79281e68d90', '单步设计'),                          # 单步设计
    ('e99788e58981e581a3e79281e68d90', '靶点设计'),                    # 靶点设计
    ('e69da9e594ace7bbaee58fa5e685a8', '迭代精修'),                    # 迭代精修
    ('e79281e589a7e79686', '设置'),                                            # 设置
    ('e98f83e383a5e7b994', '日志'),                                            # 日志
    ('e5aeb8e68f92e4bb9b', '已做'),                                            # 已做

    # Box drawing: 鈹€ (e9 88 b9 e2 82 ac) -> ─
    ('e988b9e282ac', '─'),

    # Additional symbol corruptions
    ('e884b3', '×'),         # 脳 -> × (multiplication sign)
    ('e988a6e5aea63f', '⁻¹'),  # 鈦宦? -> ⁻¹ (superscript -1)

    # Special chars with trailing byte
    ('e988a53f', '—'),       # 鈥? -> —
    ('e988a520', '— '),      # 鈥?(space) -> —
    ('e988ab3f', '→'),       # 鈫? -> →
    ('e988ab20', '→ '),      # 鈫?(space) -> →
    ('e8849c', 'Å'),         # 脜 -> Å
    ('e989813f', '⚠'),       # 鉁? -> ⚠

    # Standalone (without trailing context)
    ('e988a5', '—'),         # 鈥 -> —
    ('e988ab', '→'),         # 鈫 -> →
]

INCLUDE_EXTS = {'.py', '.md', '.yml', '.yaml', '.txt', '.rst', '.sh'}
EXCLUDE_DIRS = {'.git', '__pycache__', '.claude', 'venv', 'logs', 'data',
                'alphafold_results', 'batch_results', 'closed_loop_results',
                'misfolding_results', 'oc_validation_results', 'diffab',
                'ProteinMPNN'}
EXCLUDE_FILES = {'fix_encoding.py'}


def build_replacements():
    """Convert raw hex patterns to (str, str) pairs for replacement."""
    result = []
    for hex_bytes, correct_str in REPLACEMENTS_RAW:
        raw = bytes.fromhex(hex_bytes)
        corrupted_str = raw.decode('utf-8')
        result.append((corrupted_str, correct_str))
    # Sort by length (longest first) to avoid partial matches
    result.sort(key=lambda x: len(x[0]), reverse=True)
    return result


REPLACEMENTS = build_replacements()


def fix_content(content):
    """Apply all known corrections to content string."""
    for corrupted_str, correct_str in REPLACEMENTS:
        content = content.replace(corrupted_str, correct_str)
    return content


def count_corruptions(content):
    """Count total corruptions in content."""
    total = 0
    for corrupted_str, _ in REPLACEMENTS:
        total += content.count(corrupted_str)
    return total


def find_corruptions(content):
    """Find corruption samples for reporting."""
    issues = []
    for corrupted_str, correct_str in REPLACEMENTS:
        count = content.count(corrupted_str)
        if count > 0 and len(issues) < 5:
            preview = corrupted_str[:30]
            corr_preview = correct_str[:30]
            issues.append((preview, corr_preview, count))
    return issues


def should_process(filepath):
    """Check if file should be processed."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in INCLUDE_EXTS:
        return False
    if os.path.basename(filepath) in EXCLUDE_FILES:
        return False
    parts = Path(filepath).parts
    for part in parts:
        if part in EXCLUDE_DIRS:
            return False
    return True


def scan_directory(root_dir):
    """Scan for files with encoding corruptions."""
    affected = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if not should_process(fpath):
                continue
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except (UnicodeDecodeError, IOError):
                continue
            issue_count = count_corruptions(content)
            if issue_count > 0:
                samples = find_corruptions(content)
                affected.append((fpath, issue_count, samples))
    return affected


def main():
    parser = argparse.ArgumentParser(description='Fix CJK/Unicode encoding corruption')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Preview only, do not modify files')
    parser.add_argument('--dir', default='.', help='Root directory to scan')
    args = parser.parse_args()

    root = os.path.abspath(args.dir)
    print(f"Scanning: {root}")
    print(f"Mode: {'DRY RUN (preview only)' if args.dry_run else 'APPLY FIXES'}")
    print()

    affected = scan_directory(root)

    if not affected:
        print("No encoding corruptions found!")
        return

    total_issues = sum(n for _, n, _ in affected)
    print(f"Found {len(affected)} files with {total_issues} corruptions:\n")

    for fpath, n_issues, samples in affected:
        relpath = os.path.relpath(fpath, root)
        print(f"  {relpath} ({n_issues} issues)")
        for old, new, count in samples:
            print(f"    x{count}: '{old}' -> '{new}'")

    if args.dry_run:
        print(f"\n[DRY RUN] Would fix {total_issues} corruptions in {len(affected)} files.")
        print("Run without --dry-run to apply fixes.")
        return

    # Apply fixes
    fixed_count = 0
    for fpath, _, _ in affected:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, IOError):
            print(f"  SKIP (cannot read): {fpath}")
            continue
        fixed = fix_content(content)
        if fixed != content:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(fixed)
            fixed_count += 1
            print(f"  FIXED: {os.path.relpath(fpath, root)}")

    print(f"\nFixed {fixed_count} files.")


if __name__ == '__main__':
    main()
