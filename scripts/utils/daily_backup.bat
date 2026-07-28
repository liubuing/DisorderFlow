@echo off
REM Daily 1:00 AM backup for BFN IDP pipeline
REM Run by Windows Task Scheduler

set PROJECT_DIR=C:\biological\disorderflow-main
set BACKUP_DIR=%PROJECT_DIR%\logs\backups
set LOG_FILE=%PROJECT_DIR%\logs\conformation_build.log
set DATE_STAMP=%date:~0,4%%date:~5,2%%date:~8,2%

mkdir "%BACKUP_DIR%" 2>nul

echo [%date% %time%] Daily Backup START >> "%BACKUP_DIR%\backup_schedule.log"

REM 1. Copy conformation log
if exist "%LOG_FILE%" (
    copy /Y "%LOG_FILE%" "%BACKUP_DIR%\conformation_log_%DATE_STAMP%.log" >> "%BACKUP_DIR%\backup_schedule.log" 2>&1
    echo   Log copied >> "%BACKUP_DIR%\backup_schedule.log"
)

REM 2. Count LMDB entries
C:\cf\Scripts\python.exe -c "import pickle,lmdb; env=lmdb.open(r'%PROJECT_DIR%\data\confidence_conformation_v3\confidence_train.lmdb',readonly=True,lock=False); n=pickle.loads(env.begin().get(b'__len__')); print(f'  LMDB entries: {n}'); env.close()" >> "%BACKUP_DIR%\backup_schedule.log" 2>&1

REM 3. Check process
tasklist /FI "IMAGENAME eq python.exe" 2>nul | findstr /I python >> "%BACKUP_DIR%\backup_schedule.log" 2>&1
echo   Process check done >> "%BACKUP_DIR%\backup_schedule.log"

REM 4. Line count of build log
if exist "%LOG_FILE%" (
    for /f %%i in ('type "%LOG_FILE%" ^| find /c /v ""') do set LINES=%%i
    echo   Build log lines: %LINES% >> "%BACKUP_DIR%\backup_schedule.log"
)

echo [%date% %time%] Daily Backup END >> "%BACKUP_DIR%\backup_schedule.log"
