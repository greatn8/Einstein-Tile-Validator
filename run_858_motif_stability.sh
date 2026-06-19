#!/usr/bin/env bash
set -euo pipefail

# Moderate first screen for candidate 0000858.
# Run from the directory containing:
#   motif_stability_probe.py
#   forced_structure_probe.py
#
# This runs 9 independent finite constructions:
#   3 seeds × 3 growth policies × reflections allowed
#
# It keeps the job reasonable while asking the meaningful question:
# Are the same motifs present under forced, mixed, and random growth?

CANDIDATE="/home/nshorter/cosc330/einstein/records_holefree_restart_v1/candidate_0000858.txt"
OUT="858_motif_stability_screen1"

mkdir -p "$OUT"

python3 motif_stability_probe.py "$CANDIDATE" \
  --out "$OUT" \
  --seeds 858,271828,314159 \
  --modes forced,mixed,random \
  --reflection-modes 1 \
  --radius 60 \
  --max-tiles 800 \
  --analysis-margin 12 \
  --edge-guard 10 \
  --forced-scan-limit 80 \
  --fallback-scan-limit 70 \
  --mixed-forced-probability 0.55 \
  --motif-radii 3,4,5,6 \
  --motif-margin 14 \
  --min-motif-members 3 \
  --per-run-min-occurrences 3 \
  --stability-fraction 0.67 \
  --heartbeat-sec 10 \
  | tee "$OUT/live.log"
