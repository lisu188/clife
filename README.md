# clife

A Linux/X11 implementation of Conway's Game of Life written in C++.

## Features
- Toroidal Game of Life simulation, so patterns wrap around the map edges.
- Mixed dense backends: a tuned bit-packed path for update-heavy and very large workloads, plus a byte backend where it still wins on mid-sized combined/render runs.
- Linux-native X11/XShm presentation path with no SDL dependency.
- Headless fixed-frame benchmark mode with deterministic sizing, density, seed, mode, thread, and backend controls.
- Automated correctness checks that compare the optimized backends against a scalar reference implementation.
- Example screenshot included in `sample.png`.

## Requirements
- C++23 compiler
- CMake 3.3 or later
- X11 development libraries (`libx11-dev` and `libxext-dev` on Debian/Ubuntu)

## Building
1. Run `./configure.sh` to install required packages on Ubuntu if needed.
2. Configure the release build:
   ```bash
   cmake -B./cmake-build-release -H. -DCMAKE_BUILD_TYPE=Release
   ```
3. Build the release configuration:
   ```bash
   cmake --build ./cmake-build-release
   ```
   The executables `clife` and `clife_tests` will be produced in `cmake-build-release`.
   On GCC and Clang, release builds enable `-march=native -mtune=native`.
   `CLIFE_ENABLE_IPO` is available as an opt-in experiment and defaults to `OFF`.

Useful build options:
```bash
cmake -B./cmake-build-release -H. -DCMAKE_BUILD_TYPE=Release \
  -DCLIFE_ENABLE_NATIVE=ON \
  -DCLIFE_ENABLE_IPO=OFF \
  -DCLIFE_ENABLE_STRIP=OFF \
  -DCLIFE_ENABLE_TESTS=ON
```

Run the automated verification target with:
```bash
ctest --test-dir ./cmake-build-release --output-on-failure
```

## Running
Launch the program after building:
```bash
./cmake-build-release/clife
```
A window will open on an X11 display and show the simulation running. The default viewport is `1000x1000`.

For a fixed-frame headless benchmark, set `CLIFE_BENCH_FRAMES`:
```bash
CLIFE_BENCH_FRAMES=240 ./cmake-build-release/clife
```
This path skips X11 window creation and prints the selected mode, backend, timing, and throughput.

Useful benchmark controls:
```bash
CLIFE_BENCH_WIDTH=8192
CLIFE_BENCH_HEIGHT=8192
CLIFE_BENCH_DENSITY=0.5
CLIFE_BENCH_SEED=1
CLIFE_BENCH_THREADS=0       # 0 = auto
CLIFE_BENCH_WARMUP=3
CLIFE_BENCH_MIN_SECONDS=1.0
CLIFE_BENCH_MODE=combined   # combined | update | render
CLIFE_BENCH_BACKEND=bitpack # bitpack | byte | reference
CLIFE_BACKEND=auto          # auto | bitpack | byte | reference
CLIFE_RENDER_BACKEND=auto   # auto | scalar | avx2
```

Example update-only benchmark on a dense large board:
```bash
CLIFE_BENCH_MODE=update \
CLIFE_BENCH_WIDTH=8192 \
CLIFE_BENCH_HEIGHT=8192 \
CLIFE_BENCH_DENSITY=0.5 \
CLIFE_BENCH_FRAMES=60 \
CLIFE_BENCH_MIN_SECONDS=1.0 \
CLIFE_BENCH_SEED=1 \
./cmake-build-release/clife
```

`CLIFE_BENCH_THREADS=0` lets the runtime size the worker count from the board dimensions so medium boards do not get oversubscribed.

`CLIFE_BENCH_MIN_SECONDS` extends a benchmark run past `CLIFE_BENCH_FRAMES` when needed, which makes short workloads much easier to compare reproducibly.

`CLIFE_BACKEND=auto` keeps the optimized bit-packed path on update-heavy, small, and very large workloads, and falls back to the byte backend on the mid-sized combined/render range where it still benchmarks faster.

For objdump, valgrind, or callgrind work, configure a non-stripped release build with `-DCLIFE_ENABLE_STRIP=OFF`.

## License
This project is distributed under the MIT License. See [LICENSE.md](LICENSE.md) for the full license text.

![sample.png](./sample.png?raw=true)
