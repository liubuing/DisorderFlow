#!/bin/bash
# 看门狗包装：崩溃自动重启，指数退避
set -u
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
LOG="${DISORDERFLOW_BUILD_LOG:-$PROJECT_DIR/_build_v5.log}"
BACKOFF=15
CRASHES=()

if [[ -f "$PROJECT_DIR/.v5_build_stopped" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Build intentionally stopped; watchdog exiting" >> "$LOG"
    exit 0
fi

while true; do
    if [[ -f "$PROJECT_DIR/.v5_build_stopped" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Build stop marker found; watchdog exiting" >> "$LOG"
        exit 0
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Launching build ===" >> "$LOG"
    bash "$PROJECT_DIR/_run_v5_build.sh" >> "$LOG" 2>&1
    CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Build exited code=$CODE" >> "$LOG"

    if [ $CODE -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Build completed successfully." >> "$LOG"
        exit 0
    fi

    NOW=$(date +%s)
    CRASHES+=("$NOW")
    # 只保留最近 10 分钟的崩溃
    RECENT=()
    for t in "${CRASHES[@]}"; do
        if [ $((NOW - t)) -lt 600 ]; then
            RECENT+=("$t")
        fi
    done
    CRASHES=("${RECENT[@]}")

    if [ ${#CRASHES[@]} -ge 3 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] !! ${#CRASHES[@]} crashes in 10 min — sleeping 30 min" >> "$LOG"
        sleep 1800
        CRASHES=()
        BACKOFF=15
        continue
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Backing off ${BACKOFF}s..." >> "$LOG"
    sleep $BACKOFF
    BACKOFF=$((BACKOFF * 2 < 300 ? BACKOFF * 2 : 300))
done
