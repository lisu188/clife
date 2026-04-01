# clife

A simple implementation of Conway's Game of Life written in C++ using SDL2.

## Features
- Classic Game of Life simulation.
- Minimal SDL2 graphics output.
- Example screenshot included in `sample.png`.

## Requirements
- C++23 compiler
- CMake 3.3 or later
- Boost libraries
- SDL2 development libraries

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

## Running
Launch the program after building:
```bash
./cmake-build-release/clife
```
A window will open showing the simulation running.

## License
This project is distributed under the MIT License. See [LICENSE.md](LICENSE.md) for the full license text.

![sample.png](./sample.png?raw=true)
