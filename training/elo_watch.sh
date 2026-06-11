#!/bin/zsh
# Hourly Elo snapshots during an overnight run: rates the run's newest
# checkpoint against the anchors + the previous best. Appends to a log so
# the morning review sees an Elo trajectory, not one endpoint.
# Usage: ./elo_watch.sh <run_dir> <hours>

set -u
RUN_DIR=$1
HOURS=${2:-8}
LOG=/tmp/elo-overnight.log

cd "$(dirname "$0")"
BASELINE=$(readlink -f runs/20260611-0049-stage2b-mixed/checkpoints/best.pt)
[ -f "$BASELINE" ] || { echo "baseline missing: $BASELINE" >> "$LOG"; exit 1; }
echo "=== elo watch started $(date) for $RUN_DIR ===" >> "$LOG"
for i in $(seq 1 "$HOURS"); do
  sleep 3600
  # Resolve the symlink: step files are immutable, so no race with training.
  CK=$(readlink -f "$RUN_DIR/checkpoints/latest.pt" 2>/dev/null) || continue
  [ -f "$CK" ] || continue
  echo "--- hour $i: $(date '+%H:%M') $(basename "$CK") ---" >> "$LOG"
  PYTHONPATH=. python3 elo.py tournament "$CK" "$BASELINE" \
    --tables 24 --games-per-table 16 --seed $((100 + i)) >> "$LOG" 2>&1
done
echo "=== elo watch done $(date) ===" >> "$LOG"
