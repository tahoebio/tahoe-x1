# Copyright (C) Tahoe Therapeutics 2025-2026. All rights reserved.
from . import _flash_attn_compat

_flash_attn_compat.patch()

from . import data, model, tokenizer, utils
from ._version import __version__

__all__ = ["__version__", "data", "model", "tokenizer", "utils"]
