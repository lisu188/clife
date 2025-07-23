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
- There are currently no automated tests.
