#!/usr/bin/env bash
# Headless Locust runs at 1, 5, 10, 20 users. Pass the locustfile as $1.
# Example: ./run_locust.sh locustfile_sync.py
set -euo pipefail
LOCUSTFILE="${1:?usage: $0 <locustfile.py>}"
NAME="${LOCUSTFILE%.py}"
HOST="${HOST:-http://localhost:8000}"
mkdir -p results

for N in 1 5 10 20; do
  echo "=== $NAME users=$N ==="
  locust -f "$LOCUSTFILE" --host "$HOST" --headless \
    -u "$N" -r "$N" -t 60s \
    --csv "results/${NAME}_u${N}"
done
