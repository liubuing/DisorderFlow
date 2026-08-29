@echo off
cd /d D:\biological\DisorderFlow
python train.py configs/train/bfn_v20_amplify_xpu.yml --finetune logs/bfn_v19_diversity_xpu_2026_07_02__14_23_57_v19_final/checkpoints/best.pt --tag v20_amp --device cpu
