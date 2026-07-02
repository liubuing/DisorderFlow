@echo off
cd /d C:\biological\DisorderFlow
echo Starting V20 P2 training...
echo %date% %time%
python train.py configs/train/bfn_v20_amplify_xpu.yml --finetune logs/bfn_v19_diversity_xpu_2026_07_02__14_23_57_v19_final/checkpoints/best.pt --tag v20_amp --device cpu >> logs\v20_launch.log 2>&1
echo Training exited with code %ERRORLEVEL%
pause
