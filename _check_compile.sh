#!/bin/bash
cd /mnt/c/biological/DisorderFlow
source venv_wsl/bin/activate
python -c "
import py_compile
py_compile.compile('scripts/generate_h3_peptide_t21.py', doraise=True)
py_compile.compile('scripts/benchmark_h3_t2.1_recovery.py', doraise=True)
print('Compile OK')
"
