# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tahoe-x1 package setup."""

import os
from typing import Any, Mapping

import setuptools
from setuptools import setup

_PACKAGE_NAME = "tahoex"
_PACKAGE_DIR = "tahoex"
_REPO_REAL_PATH = os.path.dirname(os.path.realpath(__file__))
_PACKAGE_REAL_PATH = os.path.join(_REPO_REAL_PATH, _PACKAGE_DIR)

# Read the tahoex version
version_path = os.path.join(_PACKAGE_REAL_PATH, '_version.py')
with open(version_path, encoding='utf-8') as f:
    version_globals: dict[str, Any] = {}
    version_locals: Mapping[str, object] = {}
    content = f.read()
    exec(content, version_globals, version_locals)
    repo_version = str(version_locals['__version__'])

# Use repo README for PyPI description
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

# Hide the content between <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN --> and
# <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END --> tags in the README
while True:
    start_tag = '<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->'
    end_tag = '<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->'
    start = long_description.find(start_tag)
    end = long_description.find(end_tag)
    if start == -1:
        assert end == -1, 'there should be a balanced number of start and ends'
        break
    else:
        assert end != -1, 'there should be a balanced number of start and ends'
        long_description = long_description[:start] + long_description[
            end + len(end_tag):]

classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
]

install_requires = [
    "awscli>=1.32",
    "llm-foundry[gpu]==0.17.1",
    "mosaicml-streaming",
    "ninja",
    "cmake",
    "packaging",
    "torch==2.5.*",
    "llvmlite",
    "numba",
    "natsort",
    "scanpy",
    "pynndescent",
    "umap-learn",
    "anndata",
    "h5py",
]

extra_deps = {}

extra_deps["dev"] = [
    "pre-commit>=3.4.0,<4",
    "toml>=0.10.2,<0.11",
    "ipykernel",
    "packaging>=21,<23",
]

# Note: transformer-engine is an optional dependency that must be installed separately
# Users can install it with: pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
# It's not included here because PyPI doesn't allow direct git dependencies

extra_deps["all"] = {
    dep
    for key, deps in extra_deps.items()
    for dep in deps
}

setup(
    name=_PACKAGE_NAME,
    version=repo_version,
    author="Tahoe Therapeutics",
    author_email="admin@tahoebio.ai",
    description="Tahoe-x1: Perturbation trained single-cell foundation models with up to 3 billion parameters",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/tahoebio/tahoe-x1/",
    project_urls={
        'Homepage': 'https://github.com/tahoebio/tahoe-x1',
        'Documentation': 'https://github.com/tahoebio/tahoe-x1#readme',
        'Repository': 'https://github.com/tahoebio/tahoe-x1',
        'Bug Tracker': 'https://github.com/tahoebio/tahoe-x1/issues',
        'Model Card': 'https://huggingface.co/tahoebio/Tahoe-x1',
        'Paper': 'http://www.tahoebio.ai/news/tahoe-x1',
    },
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
            "gcloud*",
            "configs*",
        ],
    ),
    classifiers=classifiers,
    install_requires=install_requires,
    extras_require=extra_deps,
    python_requires=">=3.10",
    license='Apache-2.0',
    keywords=[
        'single-cell',
        'foundation-model',
        'genomics',
        'transcriptomics',
        'perturbation',
        'machine-learning',
        'deep-learning',
    ],
)
