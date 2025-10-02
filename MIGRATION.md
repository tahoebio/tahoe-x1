# Migration from Docker to uv (State-style installation)

## What changed?

### Old way (Docker-based):
```bash
# Using docker
pip install -e .
```

### New way (uv-based, like arc-state):
```bash
# Development install
uv pip install -e .

# OR install as a tool
uv tool install mosaicfm
```

## Installation Steps

### Prerequisites

1. **Install uv**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```



### Option 1: Development Installation (Recommended for contributors)

```bash
# Clone the repo
git clone <your-repo-url>
cd tahoex

# Create virtual environment with uv
uv venv

source .venv/bin/activate  # Linux/Mac


# Install in editable mode
uv pip install -e .
```

### Option 2: User Installation (For using as a tool)

```bash
# Install from PyPI (when published)
uv tool install tahoex

# OR install from git
uv tool install git+https://github.com/your-org/tahoex.git
```


## FAQ

**Q: How do I publish to PyPI?**
A: 
```bash
uv build
uv publish
```
