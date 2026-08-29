@echo off
setlocal
set DISTRO=Ubuntu-24.04-D

wsl -d %DISTRO% bash -lc "cd /mnt/d/biological/DisorderFlow; exec bash scripts/run_v5_1_training_pipeline.sh --run"
exit /b %errorlevel%
