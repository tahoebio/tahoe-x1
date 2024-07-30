# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
"""Mosaicfm package setup."""

import os
import re

import setuptools
from setuptools import setup

_PACKAGE_NAME = "mosaicfm"
_PACKAGE_DIR = "mosaicfm"
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
    "llm-foundry==0.6.0",
]

setup(
    name=_PACKAGE_NAME,
    version=repo_version,
    description="mosaicfm",
    package_data={
        "mosaicfm": ["py.typed"],
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
    python_requires=">=3.10",
)
