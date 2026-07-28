#!/bin/bash
# HDOCK Validation: Dock 3STB nanobody scaffold against Abeta42 epitope
# Compares baseline scaffold docking with designed variants
#
# Usage: bash scripts/run_hdock_validation.sh

set -e

HDOCK_DIR="${HDOCK_DIR:-/opt/HDOCKlite-v1.1}"
HDOCK="$HDOCK_DIR/hdock"
CREATEPL="$HDOCK_DIR/createpl"
DATA_DIR="./data/misfolding_targets"
OUTPUT_DIR="/tmp/hdock_validation"

mkdir -p "$OUTPUT_DIR"

echo "=== HDOCK Validation: 3STB vs Abeta42 ==="

# Prepare receptor (Abeta42 epitope) and ligand (3STB nanobody)
# Use absolute paths - HDOCKlite Fortran binary requires them
RECEPTOR_ABS="$(realpath "$DATA_DIR/2NAO_model1_A_1-42.pdb")"
LIGAND_ABS="$(realpath "$DATA_DIR/3STB.pdb")"

if [ ! -f "$RECEPTOR_ABS" ]; then
    echo "ERROR: Receptor not found: $RECEPTOR_ABS"
    exit 1
fi
if [ ! -f "$LIGAND_ABS" ]; then
    echo "ERROR: Ligand not found: $LIGAND_ABS"
    exit 1
fi

echo "Receptor: $RECEPTOR_ABS"
echo "Ligand: $LIGAND_ABS"

# Copy PDBs to output dir to avoid Fortran path issues
cp "$RECEPTOR_ABS" "$OUTPUT_DIR/receptor.pdb"
cp "$LIGAND_ABS" "$OUTPUT_DIR/ligand.pdb"

# Step 1: Run HDOCK docking
echo ""
echo "[Step 1] Running HDOCK docking..."
cd "$OUTPUT_DIR"
"$HDOCK" receptor.pdb ligand.pdb -out Hdock.out 2>&1 | tee hdock_log.txt

if [ ! -f "Hdock.out" ]; then
    echo "ERROR: HDOCK failed to produce output"
    exit 1
fi

# Step 2: Generate top 100 complex models
echo ""
echo "[Step 2] Creating complex models..."
"$CREATEPL" Hdock.out top100.pdb -nmax 100 -complex -models 2>&1 | tee createpl_log.txt

# Step 3: Score top models with local interface scorer
echo ""
echo "[Step 3] Scoring top models..."

# Find the script directory for absolute paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
/usr/bin/python3 -c "
import os, sys, json, glob
sys.path.insert(0, '.')
from modules.interface_scorer import score_complex

output_dir = '$OUTPUT_DIR'
model_files = sorted(
    [f for f in glob.glob(os.path.join(output_dir, 'model_*.pdb')) if 'top100' not in f],
    key=lambda x: int(x.split('_')[-1].replace('.pdb', ''))
)

if not model_files:
    # Try alternative naming pattern from createpl
    for f in sorted(glob.glob(os.path.join(output_dir, '*.pdb'))):
        print(f'  Found PDB: {os.path.basename(f)}')

print(f'Found {len(model_files)} model files')

results = []
for i, model_pdb in enumerate(model_files[:20]):
    try:
        scores = score_complex(model_pdb, scaffold_chain='A', epitope_chain='B')
        scores['rank'] = i + 1
        scores['model'] = os.path.basename(model_pdb)
        results.append(scores)
        print(f'  Model {i+1}: combined={scores[\"combined_interface_score\"]:.4f} BSA={scores[\"total_bsa\"]:.0f}')
    except Exception as e:
        print(f'  [WARN] Model {i+1}: {e}')

# Sort by combined score
results.sort(key=lambda x: x.get('combined_interface_score', 0), reverse=True)

with open(os.path.join(output_dir, 'docking_scores.json'), 'w') as f:
    json.dump(results, f, indent=2)

if results:
    print(f'\n=== Top 5 Docking Results ===')
    for r in results[:5]:
        print(f\"  Rank {r['rank']:2d}: combined={r['combined_interface_score']:.4f}  BSA={r['total_bsa']:.0f}  Sc={r['Sc']:.4f}  EC={r['EC']:.4f}  contacts={r['contact_pairs']}\")
else:
    print('\nNo valid models to score')

print(f'\nFull results saved to: {output_dir}/docking_scores.json')
"

echo ""
echo "=== Done ==="
echo "Output: $OUTPUT_DIR"
