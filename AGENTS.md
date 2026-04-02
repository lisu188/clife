# Contribution Notes

## README Updates
- Whenever you introduce new features, options, or dependencies, update `README.md` accordingly so users can build and run the project.
- Keep build and installation instructions in `README.md` in sync with the project state.

## Build Verification
- Ensure the project still builds after any change. Configure and build the release variant:
  ```bash
  cmake -B./cmake-build-release -H. -DCMAKE_BUILD_TYPE=Release
  cmake --build ./cmake-build-release
  ```
- Run the correctness test executable with:
  ```bash
  ctest --test-dir ./cmake-build-release --output-on-failure
  ```

## Optimization Workflow
- Prefer deterministic headless benchmarks before changing hot code. Keep `CLIFE_BENCH_SEED`, `CLIFE_BENCH_WIDTH`, `CLIFE_BENCH_HEIGHT`, `CLIFE_BENCH_DENSITY`, and `CLIFE_BENCH_THREADS` fixed so before/after results are comparable.
- For valgrind, callgrind, and disassembly work, use a non-stripped release build:
  ```bash
  cmake -B./cmake-build-inspect -H. -DCMAKE_BUILD_TYPE=Release -DCLIFE_ENABLE_STRIP=OFF
  cmake --build ./cmake-build-inspect
  ```
- Split profiling by workload instead of profiling only the default interactive path:
  ```bash
  env CLIFE_BENCH_MODE=update CLIFE_BENCH_WIDTH=2048 CLIFE_BENCH_HEIGHT=2048 \
    CLIFE_BENCH_DENSITY=0.5 CLIFE_BENCH_FRAMES=10 CLIFE_BENCH_WARMUP=2 \
    CLIFE_BENCH_THREADS=1 CLIFE_BENCH_SEED=1 \
    valgrind --tool=callgrind --callgrind-out-file=callgrind.update.out \
    ./cmake-build-inspect/clife

  env CLIFE_BENCH_MODE=render CLIFE_BENCH_WIDTH=2048 CLIFE_BENCH_HEIGHT=2048 \
    CLIFE_BENCH_DENSITY=0.5 CLIFE_BENCH_FRAMES=10 CLIFE_BENCH_WARMUP=2 \
    CLIFE_BENCH_THREADS=1 CLIFE_BENCH_SEED=1 \
    valgrind --tool=callgrind --callgrind-out-file=callgrind.render.out \
    ./cmake-build-inspect/clife

  env CLIFE_BENCH_MODE=combined CLIFE_BENCH_WIDTH=2048 CLIFE_BENCH_HEIGHT=2048 \
    CLIFE_BENCH_DENSITY=0.5 CLIFE_BENCH_FRAMES=10 CLIFE_BENCH_WARMUP=2 \
    CLIFE_BENCH_THREADS=1 CLIFE_BENCH_SEED=1 \
    valgrind --tool=callgrind --callgrind-out-file=callgrind.combined.out \
    ./cmake-build-inspect/clife
  ```
- Inspect the hottest symbols first:
  ```bash
  callgrind_annotate --auto=yes callgrind.combined.out | head -n 80
  nm -C ./cmake-build-inspect/clife | rg 'BitPackedEngine::|render_packed_row|evaluate_chunk'
  objdump -drwC -Mintel --disassemble='(anonymous namespace)::BitPackedEngine::evaluate_chunk(unsigned long, LifeBoard::RenderTarget const*)' ./cmake-build-inspect/clife
  ```
- After each optimization, re-run both callgrind and a wall-clock benchmark, for example:
  ```bash
  env CLIFE_BENCH_MODE=combined CLIFE_BENCH_WIDTH=4096 CLIFE_BENCH_HEIGHT=4096 \
    CLIFE_BENCH_DENSITY=0.5 CLIFE_BENCH_FRAMES=60 CLIFE_BENCH_MIN_SECONDS=1.0 \
    CLIFE_BENCH_WARMUP=3 CLIFE_BENCH_THREADS=1 CLIFE_BENCH_SEED=1 \
    ./cmake-build-inspect/clife
  ```
- Only keep assembly-sensitive changes that improve both the hot instruction counts and the real benchmark throughput.
