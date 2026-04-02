# clife

A Linux/X11 implementation of Conway's Game of Life written in C++.

## Features
- Toroidal Game of Life simulation, so patterns wrap around the map edges.
- Packed simulation backend with optimized Conway fast paths plus generic Life-like `B/S` rule support on the 8-neighbor Moore neighborhood.
- Linux-native X11/XShm presentation path with no SDL dependency.
- Lightweight always-visible X11 rules panel for live Birth/Survival edits while the simulation keeps running.
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
A resizable `1000x1000` X11 window will open and show the simulation running. The default board remains `10000x10000`.
The initial rule is Conway `B3/S23`, and the rules panel starts with `Birth=3` and `Survival=23`.

The initial view starts centered on the board. Resize the window to reveal more of the board, hold the middle mouse button to drag the viewport, drag the horizontal and vertical scrollbars, or use the arrow keys / `W`, `A`, `S`, `D`. Mouse-wheel scrolling is disabled.

Use the rules panel on the right side of the window to edit Life-like rules while the simulation runs:
- `Birth` accepts digits `0` through `8` and may be empty.
- `Survival` accepts digits `0` through `8` and may be empty.
- Inputs are normalized to sorted unique digits.
- Click `Apply`, or press `Enter` while a field is focused, to queue the new rule.
- Click `Reset To Default` to restore Conway `B3/S23`.
- The current active rule is shown as `B.../S...` in the panel.

Rule changes preserve the current board state and take effect on the next generation boundary, so the app does not need to restart to switch to rules such as `B36/S23` or `B2/S`.

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
CLIFE_BENCH_BACKEND=bitpack # packed simulation backend
CLIFE_BACKEND=auto          # resolves to the packed simulation backend
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

For the full reproducible matrix used during optimization work:
```bash
CLIFE_BENCH_CPUSET=0-7 ./scripts/bench_matrix.sh
```

`CLIFE_BENCH_THREADS=0` lets the runtime size the worker count from the board dimensions so medium boards do not get oversubscribed.

`CLIFE_BENCH_MIN_SECONDS` extends a benchmark run past `CLIFE_BENCH_FRAMES` when needed, which makes short workloads much easier to compare reproducibly.

`CLIFE_BACKEND=auto` selects the packed simulation backend. Legacy `byte` and `reference` spellings are still accepted as compatibility aliases, but the runtime simulation path is packed-only.

For objdump, valgrind, or callgrind work, configure a non-stripped release build with `-DCLIFE_ENABLE_STRIP=OFF`.

## License
This project is distributed under the MIT License. See [LICENSE.md](LICENSE.md) for the full license text.

![sample.png](./sample.png?raw=true)
