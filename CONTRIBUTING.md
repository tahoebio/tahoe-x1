# Contributing to Tahoe-x1

Thank you for your interest in contributing to Tahoe-x1! We welcome contributions from the community, whether it's bug reports, feature requests, documentation improvements, or code contributions.

## Table of Contents

- [Getting Started](#getting-started)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Code Style and Quality](#code-style-and-quality)
- [Commit Message Conventions](#commit-message-conventions)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)
- [Getting Help](#getting-help)

## Getting Started

### Ways to Contribute

We encourage all types of contributions:

- 🐛 **Report bugs**: Open an issue describing the problem
- 💡 **Suggest features**: Share ideas for new features or improvements
- 📝 **Improve documentation**: Fix typos, clarify instructions, add examples
- 🔬 **Share research**: Contribute benchmarks, experiments, or use cases
- 💻 **Submit code**: Fix bugs, implement features, improve performance
- ✅ **Review PRs**: Help review other contributors' pull requests

### Creating Issues

**Bug Reports** should include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU, etc.)
- Error messages or stack traces

**Feature Requests** should include:
- Clear description of the proposed feature
- Use case and motivation
- Potential implementation approach (optional)
- Any alternatives considered

**Template:**
```markdown
## Description
[Clear description of the issue or feature]

## Environment (for bugs)
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10]
- CUDA version: [e.g., 12.4]
- Package version: [e.g., 1.0.4]

## Steps to Reproduce (for bugs)
1. [First step]
2. [Second step]
3. [See error]

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What actually happened]
```

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/tahoe-x1.git
cd tahoe-x1

# Add upstream remote
git remote add upstream https://github.com/tahoebio/tahoe-x1.git
```

### 2. Create a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n tahoe-x1 python=3.10
conda activate tahoe-x1
```

### 3. Install Dependencies

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Or using uv (recommended for faster installs)
uv pip install -e ".[dev]" --no-build-isolation-package flash-attn
```

**Note**: Flash-attention requires CUDA and may need special installation. See [Installation Guide](README.md#installation).

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

This installs git hooks that automatically check code quality before each commit.

## Code Style and Quality

We use automated tools to maintain consistent code quality. All checks run automatically via **pre-commit hooks** before each commit.

### Code Formatting

#### Black
- **Line length**: 88 characters (Black default)
- **Automatically formats** Python code for consistency
- Configuration in `pyproject.toml`:
  ```toml
  [tool.black]
  line-length = 88
  ```

**Example:**
```python
# Before Black
def long_function_name(param1,param2,param3,param4,param5):
    return param1+param2+param3

# After Black
def long_function_name(
    param1, param2, param3, param4, param5
):
    return param1 + param2 + param3
```

#### isort
- **Import sorting**: Organizes imports alphabetically and by type
- **Profile**: `black` (compatible with Black formatting)
- Configuration in `pyproject.toml`:
  ```toml
  [tool.isort]
  profile = "black"
  ```

**Example:**
```python
# After isort
import os
import sys
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from tahoex.model import TahoeModel
from tahoex.utils import load_config
```

### Code Linting

#### Ruff
- **Fast Python linter** (replaces flake8, pylint, etc.)
- **Auto-fixes** many issues
- Checks for:
  - Code quality issues (complexity, unused variables)
  - Performance anti-patterns
  - Common bugs
  - Simplification opportunities

**Selected rules** (see `pyproject.toml`):
- `C4` - Comprehensions
- `LOG` - Logging
- `PERF` - Performance
- `PL` - Pylint
- `E` - Errors
- `F` - Pyflakes
- `COM812` - Trailing commas
- `SIM` - Simplification
- `RUF` - Ruff-specific
- `ERA` - Commented-out code

**Ignored rules**:
- `PLR0913` - Too many function arguments (allowed for ML code)
- `PLR0915` - Too many statements (allowed for ML code)
- `E501` - Line too long (Black handles this)

#### Pycln
- **Removes unused imports** automatically
- Keeps code clean and reduces dependencies

#### Docformatter
- **Formats docstrings** to Google style
- Wraps at 80 characters
- Ensures consistent documentation format

**Example:**
```python
def train_model(model, data_loader, epochs):
    """Train the model on the provided dataset.

    Args:
        model: The model to train.
        data_loader: DataLoader providing training data.
        epochs: Number of training epochs.

    Returns:
        Trained model with updated weights.

    Raises:
        ValueError: If epochs is negative.
    """
    pass
```

### Additional Checks

#### License Headers
- Automatically adds copyright header to all `.py` and `.sh` files:
  ```python
  # Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  ```

#### Security Scanning
- **TruffleHog**: Scans for accidentally committed secrets (API keys, tokens, etc.)
- Fails if verified secrets are found

#### Other Checks
- File size limits (no large files)
- Valid Python syntax (AST check)
- No merge conflict markers
- Valid JSON, YAML, TOML
- No debug statements (`print()`, `pdb`, etc.)
- Executable scripts have shebangs

### Running Checks Manually

```bash
# Run all pre-commit hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
pre-commit run ruff --all-files

# Run checks without committing
git add <files>
pre-commit run
```

### Type Checking (Optional)

We use **Pyright** for type checking, but it's currently disabled in pre-commit. You can run it manually:

```bash
pyright
```

Configuration in `pyproject.toml` under `[tool.pyright]`.

## Commit Message Conventions

We use **Conventional Commits** with **Gitmoji** for automated versioning and changelog generation.

### Format

```
<gitmoji> <type>(<optional-scope>): <description>

[optional body]

[optional footer]
```

### Types and Version Bumps

| Type | Emoji | Version Impact | Usage |
|------|-------|----------------|-------|
| `feat` | ✨ | Minor (1.0.0 → 1.1.0) | New features |
| `fix` | 🐛 | Patch (1.0.0 → 1.0.1) | Bug fixes |
| `perf` | ⚡ | Patch (1.0.0 → 1.0.1) | Performance improvements |
| `docs` | 📝 | None | Documentation only |
| `style` | 💄 | None | Code style changes |
| `refactor` | ♻️ | None | Code refactoring |
| `test` | ✅ | None | Tests |
| `build` | 🏗️ | None | Build system |
| `ci` | 👷 | None | CI configuration |
| `chore` | 🔧 | None | Maintenance |

### Examples

```bash
# Feature
git commit -m "✨ feat: add support for Qwen2 models"
git commit -m "✨ feat(tokenizer): add custom gene vocabulary"

# Bug fix
git commit -m "🐛 fix: resolve CUDA memory leak in training"
git commit -m "🐛 fix(data): handle missing values in preprocessor"

# Documentation
git commit -m "📝 docs: add tutorial for fine-tuning"

# Breaking change (Major version bump: 1.0.0 → 2.0.0)
git commit -m "✨ feat!: redesign model API for consistency"
```

**Important**: The conventional commit type (`feat:`, `fix:`, etc.) is **required** for the CI checks to pass.

See [`.github/COMMIT_CONVENTIONS.md`](.github/COMMIT_CONVENTIONS.md) for detailed guidelines.

## Pull Request Process

### 1. Create a Feature Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feat/my-feature
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write code following our style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure pre-commit hooks pass

### 3. Commit Changes

```bash
# Pre-commit hooks will run automatically
git add <files>
git commit -m "✨ feat: add my awesome feature"
```

If pre-commit hooks fail:
- Fix the issues automatically made by the hooks
- Review the changes
- Stage the fixes and try committing again

### 4. Push to Your Fork

```bash
git push origin feat/my-feature
```

### 5. Open a Pull Request

1. Go to https://github.com/tahoebio/tahoe-x1
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill out the PR template:

**PR Title**: Must follow conventional commits (e.g., `✨ feat: add new feature`)

**PR Description** should include:
- Summary of changes
- Motivation and context
- Related issues (e.g., "Fixes #123")
- Testing done
- Breaking changes (if any)

### 6. Address Review Feedback

- Respond to reviewer comments
- Make requested changes
- Push additional commits (they'll be squashed on merge)
- Use conventional commit messages for all commits

### 7. PR Approval and Merge

Once approved:
- **Maintainers will squash and merge** your PR
- The merge commit will use your PR title (must be conventional!)
- Your changes will be included in the next release

### CI Checks

All PRs must pass:
- ✅ **Pre-commit hooks**: Code quality checks
- ✅ **Conventional commits**: PR title and commits follow format
- ✅ **Tests**: All tests pass (when applicable)

## Release Process

Tahoe-x1 uses **Release Please** for automated releases.

### How Releases Work

1. **You merge PRs to main** with conventional commit messages
2. **Release Please creates/updates a "Release PR"** containing:
   - Version bump in `tahoex/_version.py`
   - Auto-generated `CHANGELOG.md`
   - All changes since last release
3. **Maintainers review and merge the Release PR**
4. **Automatic release triggered**:
   - Git tag created (e.g., `v1.1.0`)
   - GitHub Release published
   - Package built and published to PyPI

### Version Bumping

Release Please determines version bumps from commit messages:

- `feat:` → **Minor** version bump (1.0.0 → 1.1.0)
- `fix:`, `perf:` → **Patch** version bump (1.0.0 → 1.0.1)
- `feat!:` or `BREAKING CHANGE:` → **Major** version bump (1.0.0 → 2.0.0)
- `docs:`, `chore:`, etc. → No version bump

### Release Timeline

Releases happen when:
1. Sufficient changes have accumulated
2. A Release PR is merged
3. Typically every 1-2 months, or as needed

### Checking Release Status

- **Release PRs**: Look for PR titled `chore(main): release X.Y.Z`
- **Published releases**: https://github.com/tahoebio/tahoe-x1/releases
- **PyPI**: https://pypi.org/project/tahoex/

See [`.github/RELEASE.md`](.github/RELEASE.md) for detailed release documentation.

## Getting Help

### Resources

- **Documentation**: [README.md](README.md)
- **Commit conventions**: [.github/COMMIT_CONVENTIONS.md](.github/COMMIT_CONVENTIONS.md)
- **Release process**: [.github/RELEASE.md](.github/RELEASE.md)
- **Issues**: https://github.com/tahoebio/tahoe-x1/issues
- **Discussions**: https://github.com/tahoebio/tahoe-x1/discussions

### Questions?

- Open a [GitHub Discussion](https://github.com/tahoebio/tahoe-x1/discussions) for questions
- Open an [Issue](https://github.com/tahoebio/tahoe-x1/issues) for bugs or feature requests
- Check existing issues and discussions first

### Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help others when you can
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md) (if we add one)

## Thank You!

Every contribution, no matter how small, makes Tahoe-x1 better. We appreciate your time and effort in helping advance single-cell foundation models! 🙏

---

**External contributors are very welcome!** Don't hesitate to open issues or submit pull requests. We're here to help you contribute successfully.
