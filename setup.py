# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
"""TahoeX package setup."""

import os
import re

import setuptools
from setuptools import setup

_PACKAGE_NAME = "tahoex"
_PACKAGE_DIR = "tahoex"
_REPO_REAL_PATH = os.path.dirname(os.path.realpath(__file__))
_PACKAGE_REAL_PATH = os.path.join(_REPO_REAL_PATH, _PACKAGE_DIR)

# Read the repo version
# We can't use `.__version__` from the library since it's not installed yet
with open(os.path.join(_PACKAGE_REAL_PATH, "__init__.py")) as f:
    content = f.read()
# regex: '__version__', whitespace?, '=', whitespace, quote, version, quote
# we put parens around the version so that it becomes elem 1 of the match
expr = re.compile(
    r"""^__version__\s*=\s*['"]([0-9]+\.[0-9]+\.[0-9]+(?:\.\w+)?)['"]""",
    re.MULTILINE,
)
repo_version = expr.findall(content)[0]

install_requires = [
    "torchtext>=0.17.1",
    "awscli>=1.32",
    "llm-foundry[all-flash2]>=0.17.0",
]

extra_deps = {}

extra_deps["dev"] = [
    "pre-commit>=3.4.0,<4",
    "toml>=0.10.2,<0.11",
    "ipykernel",
    "packaging>=21,<23",
]

extra_deps["gpu"] = [
    "transformer-engine@git+https://github.com/NVIDIA/TransformerEngine.git@stable",
]

extra_deps["all"] = {
    dep
    for key, deps in extra_deps.items()
    for dep in deps
    if key not in {"gpu-flash2", "all-cpu"}
}

setup(
    name=_PACKAGE_NAME,
    version=repo_version,
    author="Tahoe Therapeutics",
    description="tahoex",
    package_data={
        "tahoex": ["py.typed"],
    },
    packages=setuptools.find_packages(
        exclude=[
            ".github*",
            "envs*",
            "tutorials*",
            "tests*",
            "scripts*",
            "mcli*",
            "runai*",
        ],
    ),
    install_requires=install_requires,
    extras_require=extra_deps,
    python_requires=">=3.10",
)
