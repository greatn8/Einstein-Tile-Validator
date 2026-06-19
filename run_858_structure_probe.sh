#!/usr/bin/env bash
set -euo pipefail

# Run from the same directory as forced_structure_probe.py.
# This first profile is deliberately moderate: it should produce useful
# structural evidence and progress artefacts without becoming a multi-day job.

CANDIDATE="/home/nshorter/cosc330/einstein/records_holefree_restart_v1/candidate_0000858.txt"
OUT="858_structure_probe_run1"

mkdir -p "$OUT"

python3 forced_structure_probe.py "$CANDIDATE" \
  --out "$OUT" \
  --radius 70 \
  --trials 3 \
  --max-tiles 1200 \
  --seed 858 \
  --allow-reflections 1 \
  --heartbeat-sec 10 \
  --snapshot-every 400 \
  --boundary-sample-every 200 \
  --boundary-sample-size 60 \
  --forced-scan-limit 70 \
  | tee "$OUT/live.log"
