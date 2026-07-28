#!/usr/bin/env python3
"""Fix import paths in moved scripts.

For each .py file in scripts/ subdirectories, ensure it can find:
  - The project root (for build_design_variant_dataset, modules, etc.)
  - Other scripts in sibling directories (e.g., probes importing from probes/)
"""

import os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT, 'scripts')

# Subdirectories that contain moved scripts
SUBDIRS = ['probes', 'pipelines', 'validation', 'build', 'utils']

def fix_file(filepath):
    """Add or fix sys.path.insert to include project root."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Remove old `sys.path.insert(0, '.')` that won't work in subdirs
    # Replace with project-root-relative path
    old_patterns = [
        "sys.path.insert(0, '.')",
        "sys.path.insert(0, 'modules')",
        "sys.path.insert(0, '.'); sys.path.insert(0, 'modules')",
    ]

    # Check if file already has a proper relative path setup
    if "os.path.join(os.path.dirname(__file__)" in content:
        # Already has relative path setup, just ensure root is in path
        if "sys.path.insert(0, os.path.join(os.path.dirname(__file__)" not in content:
            # Add project root insertion before existing sys.path lines
            pass  # Let the generic fix handle it
        return False  # Already OK

    # Strategy: find the first sys.path.insert or import line, add our fix before it
    lines = content.split('\n')
    new_lines = []
    inserted = False
    has_os_import = 'import os' in content or 'from os' in content

    for i, line in enumerate(lines):
        if not inserted and line.strip().startswith('import ') and not inserted:
            # Insert our path fix after imports but before any other code
            continue_flag = False

        if not inserted and (line.strip().startswith('sys.path') or line.strip().startswith('# sys.path')):
            # Replace old path insertions
            indent = line[:len(line) - len(line.lstrip())]
            rel_depth = os.path.relpath(ROOT, os.path.dirname(filepath))
            new_lines.append(f"{indent}# Auto-fixed path: project root at {rel_depth}")
            new_lines.append(f"{indent}_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), {repr(rel_depth)})")
            new_lines.append(f"{indent}sys.path.insert(0, _PROJECT_ROOT)")
            new_lines.append(f"{indent}sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'modules'))")
            inserted = True
            continue

        new_lines.append(line)

    if not inserted:
        # No existing sys.path.insert found — add after first import block
        final_lines = []
        added = False
        for i, line in enumerate(new_lines):
            final_lines.append(line)
            if not added and line.strip() == '' and i > 3:
                # After first blank line following imports
                rel_depth = os.path.relpath(ROOT, os.path.dirname(filepath))
                final_lines.append(f"import sys, os")
                final_lines.append(f"_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), {repr(rel_depth)})")
                final_lines.append(f"sys.path.insert(0, _PROJECT_ROOT)")
                final_lines.append(f"sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'modules'))")
                final_lines.append('')
                added = True
        new_lines = final_lines

    content = '\n'.join(new_lines)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def main():
    fixed = 0
    for subdir in SUBDIRS:
        dirpath = os.path.join(SCRIPTS_DIR, subdir)
        if not os.path.isdir(dirpath):
            continue
        for fname in os.listdir(dirpath):
            if fname.endswith('.py') and not fname.startswith('__'):
                fpath = os.path.join(dirpath, fname)
                if fix_file(fpath):
                    print(f'  Fixed: scripts/{subdir}/{fname}')
                    fixed += 1

    print(f'\nTotal files fixed: {fixed}')


if __name__ == '__main__':
    main()
