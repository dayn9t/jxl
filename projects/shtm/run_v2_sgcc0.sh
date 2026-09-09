#!/usr/bin/env bash
# Launch SHTM detector V2 training on sgcc0, detached from the ssh session.
set -e
LOG=/home/jiang/ws/trash/cabin/v2_train.log
setsid nohup uv run --project /home/jiang/cc/py/jxl python /home/jiang/ws/trash/cabin/train_v2.py >"$LOG" 2>&1 < /dev/null &
echo "launched, pid=$! log=$LOG"
