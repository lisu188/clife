# clife

A Linux/X11 implementation of Conway's Game of Life written in C++.

## Features
- Toroidal Game of Life simulation, so patterns wrap around the map edges.
- Dense multithreaded update pipeline tuned for large boards.
- Linux-native X11/XShm presentation path with no SDL dependency.
- Headless fixed-frame benchmark mode that does not require an X server.
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
   The executable `clife` will be produced in `cmake-build-release`.
   On GCC and Clang, release builds enable `-march=native -mtune=native`; CMake 3.9+ also enables LTO/IPO when the toolchain supports it.

## Running
Launch the program after building:
```bash
./cmake-build-release/clife
```
A window will open on an X11 display and show the simulation running.

For a fixed-frame headless benchmark, set `CLIFE_BENCH_FRAMES`:
```bash
CLIFE_BENCH_FRAMES=240 ./cmake-build-release/clife
```
This path skips X11 window creation, prints the elapsed time and FPS, and exits.

For a fixed-frame headless benchmark, set `CLIFE_BENCH_FRAMES` and use SDL's dummy video driver:
```bash
CLIFE_BENCH_FRAMES=240 SDL_VIDEODRIVER=dummy ./cmake-build-release/clife
```
The program will print the elapsed time and FPS, then exit.

## License
This project is distributed under the MIT License. See [LICENSE.md](LICENSE.md) for the full license text.

![sample.png](./sample.png?raw=true)
