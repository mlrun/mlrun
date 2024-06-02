# How to build and release

The Makefile simplifies the process of building and releasing MLRun KFP Python packages to PyPI. 

## Prerequisites
Before using this Makefile, ensure you have the following installed:

- Python
- `twine` package (`python -m pip install twine`)


## Getting Started

1. **Build Packages**: Run the following command to build your Python packages:
    ```bash
    make build
    ```
    
    This command will build each package listed in PACKAGES.

2. **Release Packages**: After building, you can release the packages to PyPI by running:
    ```bash
    make release
    ```

    This command will upload the built distribution files to PyPI using twine.

3. **Cleaning Up**: If needed, you can clean up the build artifacts by running:

    ```bash
    make clean
    ```
    
    This command will remove the dist, build, and *.egg-info directories from each package directory.


## Additional Notes

- Ensure your TWINE_USERNAME and TWINE_PASSWORD are properly set before releasing to PyPI.
- Make sure your package directories contain valid setup.py files.
- Always review the release process and verify that the correct packages are being released.
