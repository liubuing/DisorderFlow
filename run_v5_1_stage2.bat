@echo off
setlocal
set DISTRO=Ubuntu-24.04

wsl -d %DISTRO% bash -lc "cd /mnt/c/biological/DisorderFlow; exec bash scripts/run_v5_1_training_pipeline.sh --run"
exit /b %errorlevel%
