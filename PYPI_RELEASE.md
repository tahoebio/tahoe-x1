# PyPI Release Guide for Tahoe-x1

This guide explains how to publish the `tahoex` package to PyPI.

## Prerequisites

1. **PyPI Account**: Ensure you have an account on [PyPI](https://pypi.org/) and are added as a maintainer to the `tahoex` organization/project
2. **API Token**: Generate an API token from PyPI account settings (recommended over password)
3. **Build Tools**: Install required build tools:
   ```bash
   pip install build twine
   ```

## Release Process

### 1. Update Version

Update the version in `pyproject.toml`:
```toml
[project]
version = "1.0.0"  # Update to your desired version
```

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality (backward compatible)
- **PATCH** version for bug fixes (backward compatible)

### 2. Clean Previous Builds

Remove any previous build artifacts:
```bash
rm -rf build/ dist/ *.egg-info/
```

### 3. Build the Distribution

Build both source distribution (sdist) and wheel:
```bash
python -m build
```

This creates:
- `dist/tahoex-1.0.0.tar.gz` (source distribution)
- `dist/tahoex-1.0.0-py3-none-any.whl` (wheel)

### 4. Check the Build

Inspect the built packages:
```bash
twine check dist/*
```

This validates that the package description will render correctly on PyPI.

### 5. Test Upload to TestPyPI (Optional but Recommended)

First, upload to TestPyPI to verify everything works:

```bash
twine upload --repository testpypi dist/*
```

You'll be prompted for credentials:
- Username: `__token__`
- Password: Your TestPyPI API token (starts with `pypi-`)

Then test installation from TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps tahoex
```

### 6. Upload to PyPI

Once verified, upload to the real PyPI:

```bash
twine upload dist/*
```

You'll be prompted for credentials:
- Username: `__token__`
- Password: Your PyPI API token (starts with `pypi-`)

### 7. Verify Installation

Test that the package installs correctly:
```bash
pip install tahoex
```

Visit the PyPI page: https://pypi.org/project/tahoex/

## Using API Tokens

For automated releases or to avoid entering credentials:

1. Create a `~/.pypirc` file:
```ini
[pypi]
username = __token__
password = pypi-YOUR_API_TOKEN_HERE

[testpypi]
username = __token__
password = pypi-YOUR_TEST_API_TOKEN_HERE
```

2. Set restrictive permissions:
```bash
chmod 600 ~/.pypirc
```

Then you can upload without entering credentials:
```bash
twine upload dist/*
```

## Troubleshooting

### Version Already Exists
- PyPI doesn't allow re-uploading the same version
- Bump the version number and rebuild

### Missing Files in Package
- Check `MANIFEST.in` includes all necessary files
- Verify with: `tar -tzf dist/tahoex-1.0.0.tar.gz`

### Large Package Size
- Consider excluding unnecessary files in `pyproject.toml` `[tool.setuptools.packages.find]`
- Current exclusions: `.github`, `envs`, `tutorials`, `tests`, `scripts`, `mcli`, `runai`

### Dependency Issues
- Users will need compatible CUDA and PyTorch versions
- Flash-attention requires compilation and compatible GPU
- Consider documenting platform-specific requirements in README

## Continuous Integration (Future)

For automated releases, consider setting up GitHub Actions:

1. Create `.github/workflows/pypi-publish.yml`
2. Store PyPI API token as GitHub Secret
3. Trigger on tag push (e.g., `v1.0.0`)

Example workflow:
```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

## Notes

- The package name on PyPI is `tahoex` (import: `import tahoex`)
- Large dependencies (PyTorch, flash-attention) may require users to pre-install
- Consider providing pre-built wheels for common platforms if compile-time dependencies cause issues
