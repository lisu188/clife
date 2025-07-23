# clife

A simple implementation of Conway's Game of Life written in C++ using SDL2.

## Features
- Classic Game of Life simulation.
- Minimal SDL2 graphics output.
- Example screenshot included in `sample.png`.

## Requirements
- C++17 compiler
- CMake 3.3 or later
- Boost libraries
- SDL2 development libraries

## Building
1. Run `./configure.sh` to install required packages (Ubuntu) and to generate the build directories.
2. Build the release configuration:
   ```bash
   cd cmake-build-release
   make
   ```
   The executable `clife` will be produced in this directory.

## Running
Launch the program after building:
```bash
./clife
```
A window will open showing the simulation running.

## License
This project is distributed under the MIT License. See [LICENSE.md](LICENSE.md) for the full license text.

![sample.png](./sample.png?raw=true)
