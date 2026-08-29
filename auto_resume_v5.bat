@echo off
:: ============================================================
:: DisOrderFlow v5 多构象数据集构建 - 开机自动恢复
:: 触发：Windows 计划任务（ONLOGON，开机登录后运行）
:: 作用：启动 WSL2 看门狗，自动从上次断点 resume 构建
:: ============================================================
setlocal

set PROJECT=%~dp0
set WATCHDOG=%PROJECT%\_watchdog_v5.sh
set LOG=%PROJECT%\_boot_resume.log
if not defined DISORDERFLOW_WSL_DISTRO set DISORDERFLOW_WSL_DISTRO=Ubuntu-24.04-D

if exist "%PROJECT%\.v5_build_stopped" (
    echo [%date% %time%] Build intentionally stopped; skip boot resume >> "%LOG%"
    exit /b 0
)

echo [%date% %time%] === Boot resume triggered === >> "%LOG%"

:: 1. 等待网络/WSL 服务就绪（开机后 60 秒）
timeout /t 60 /nobreak >nul

:: 2. 确认 WSL 可用（最多重试 5 次，每次间隔 30 秒）
set TRIES=0
:check_wsl
wsl -l --running 2>nul | findstr /i "Ubuntu" >nul
if errorlevel 1 (
    wsl -d %DISORDERFLOW_WSL_DISTRO% -- echo ok >nul 2>&1
)
if errorlevel 1 (
    set /a TRIES+=1
    if %TRIES% geq 5 (
        echo [%date% %time%] WSL not ready after 5 tries, aborting >> "%LOG%"
        exit /b 1
    )
    echo [%date% %time%] WSL not ready, retry %TRIES%/5 in 30s >> "%LOG%"
    timeout /t 30 /nobreak >nul
    goto check_wsl
)

:: 3. 检查构建是否已在运行（避免重复启动）
wsl -d %DISORDERFLOW_WSL_DISTRO% bash -lc "pgrep -f 'build_conformation_cf' > /dev/null 2>&1 && echo RUNNING || echo IDLE" > "%TEMP%\_v5_status.txt" 2>&1
findstr /c:"RUNNING" "%TEMP%\_v5_status.txt" >nul
if not errorlevel 1 (
    echo [%date% %time%] Build already running, skip launch >> "%LOG%"
    exit /b 0
)

:: 4. 启动看门狗（setsid 脱离，后台独立运行）
echo [%date% %time%] Launching watchdog... >> "%LOG%"
for /f "delims=" %%P in ('wsl -d %DISORDERFLOW_WSL_DISTRO% wslpath -a "%WATCHDOG%"') do set WSL_WATCHDOG=%%P
wsl -d %DISORDERFLOW_WSL_DISTRO% bash -c "setsid bash '%WSL_WATCHDOG%' > /dev/null 2>&1 < /dev/null &"

:: 5. 等待并确认进程启动
timeout /t 10 /nobreak >nul
wsl -d %DISORDERFLOW_WSL_DISTRO% bash -lc "pgrep -f 'build_conformation_cf' > /dev/null 2>&1 && echo 'LAUNCHED OK' || echo 'LAUNCH FAILED'" >> "%LOG%" 2>&1

echo [%date% %time%] === Boot resume done === >> "%LOG%"
exit /b 0
