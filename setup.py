# Copyright 2024 vevo-scGPT  authors

"""MosaicML LLM Foundry package setup."""

import os

import setuptools
from setuptools import setup

_PACKAGE_NAME = 'vevo-scgpt'
_PACKAGE_DIR = 'scgpt'
_REPO_REAL_PATH = os.path.dirname(os.path.realpath(__file__))
_PACKAGE_REAL_PATH = os.path.join(_REPO_REAL_PATH, _PACKAGE_DIR)


install_requires = [
    "torchtext>=0.17.1",
    "awscli>=1.32"
]

setup(
    name=_PACKAGE_NAME,
    description='vevo-scgpt',
    package_data={
        'vevo-scgpt': ['py.typed'],
    },
    packages=setuptools.find_packages(
        exclude=['.github*', 'envs*', 'tutorials*', 'tests*','examples*','mcli*']),
    install_requires=install_requires,
    python_requires='>=3.9',
)