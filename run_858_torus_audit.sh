#!/usr/bin/env bash
set -euo pipefail

# Per-(W,H) finite torus audit for candidate 0000858.
# This does NOT do patch-growth tests. It only logs the periodic exact-cover
# cases, writing a durable row after every case.

CANDIDATE="/home/nshorter/cosc330/einstein/records_holefree_restart_v1/candidate_0000858.txt"
OUT="audit_0858_period22"

# Compile if needed.
g++ -O3 -std=c++17 -Wall -Wextra -pedantic \
  einstein_torus_audit.cpp \
  -o einstein_torus_audit

# The same period/node scale as the earlier p22 run, but with per-case logging.
# --resume makes a later restart skip all completed cases already in audit.csv.
./einstein_torus_audit \
  --input "$CANDIDATE" \
  --out "$OUT" \
  --max-period 22 \
  --nodes 50000000 \
  --mode both \
  --progress-sec 15 \
  --resume
