#!/usr/bin/env bash
set -euo pipefail

bin_path="${1:-./cmake-build-release/clife}"
cpu_set="${CLIFE_BENCH_CPUSET:-}"
seed="${CLIFE_BENCH_SEED:-1}"
threads="${CLIFE_BENCH_THREADS:-0}"
min_seconds="${CLIFE_BENCH_MIN_SECONDS:-}"

run_case() {
  local label="$1"
  local warmup="$2"
  local frames="$3"
  local width="$4"
  local height="$5"
  local density="$6"
  local mode="$7"

  echo "CASE=${label}"
  if [[ -n "${cpu_set}" ]]; then
    if [[ -n "${min_seconds}" ]]; then
      taskset -c "${cpu_set}" env \
        CLIFE_BENCH_MIN_SECONDS="${min_seconds}" \
        CLIFE_BENCH_SEED="${seed}" \
        CLIFE_BENCH_WARMUP="${warmup}" \
        CLIFE_BENCH_FRAMES="${frames}" \
        CLIFE_BENCH_WIDTH="${width}" \
        CLIFE_BENCH_HEIGHT="${height}" \
        CLIFE_BENCH_DENSITY="${density}" \
        CLIFE_BENCH_THREADS="${threads}" \
        CLIFE_BENCH_MODE="${mode}" \
        "${bin_path}"
    else
      taskset -c "${cpu_set}" env \
        CLIFE_BENCH_SEED="${seed}" \
        CLIFE_BENCH_WARMUP="${warmup}" \
        CLIFE_BENCH_FRAMES="${frames}" \
        CLIFE_BENCH_WIDTH="${width}" \
        CLIFE_BENCH_HEIGHT="${height}" \
        CLIFE_BENCH_DENSITY="${density}" \
        CLIFE_BENCH_THREADS="${threads}" \
        CLIFE_BENCH_MODE="${mode}" \
        "${bin_path}"
    fi
  else
    if [[ -n "${min_seconds}" ]]; then
      env \
        CLIFE_BENCH_MIN_SECONDS="${min_seconds}" \
        CLIFE_BENCH_SEED="${seed}" \
        CLIFE_BENCH_WARMUP="${warmup}" \
        CLIFE_BENCH_FRAMES="${frames}" \
        CLIFE_BENCH_WIDTH="${width}" \
        CLIFE_BENCH_HEIGHT="${height}" \
        CLIFE_BENCH_DENSITY="${density}" \
        CLIFE_BENCH_THREADS="${threads}" \
        CLIFE_BENCH_MODE="${mode}" \
        "${bin_path}"
    else
      env \
        CLIFE_BENCH_SEED="${seed}" \
        CLIFE_BENCH_WARMUP="${warmup}" \
        CLIFE_BENCH_FRAMES="${frames}" \
        CLIFE_BENCH_WIDTH="${width}" \
        CLIFE_BENCH_HEIGHT="${height}" \
        CLIFE_BENCH_DENSITY="${density}" \
        CLIFE_BENCH_THREADS="${threads}" \
        CLIFE_BENCH_MODE="${mode}" \
        "${bin_path}"
    fi
  fi
  echo
}

run_case small_update 20 12000 257 263 0.37 update
run_case small_combined 20 12000 257 263 0.37 combined
run_case medium_update 10 4000 2048 2048 0.5 update
run_case medium_combined 10 4000 2048 2048 0.5 combined
run_case large_dense_update 5 400 8192 8192 0.5 update
run_case large_dense_combined 5 220 8192 8192 0.5 combined
run_case large_sparse_update 5 600 8192 8192 0.05 update
run_case large_sparse_combined 5 220 8192 8192 0.05 combined
run_case default_update 3 400 1000 1000 0.05 update
run_case default_combined 3 400 1000 1000 0.05 combined
