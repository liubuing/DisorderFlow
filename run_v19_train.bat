@echo off
cd /d C:\biological\DisorderFlow
python train.py configs/train/bfn_v19_diversity_xpu.yml --finetune logs/bfn_v18_v7_per_residue_xpu_2026_07_02__00_48_49_v18_v7_run/checkpoints/best.pt --tag v19_final --device cpu
