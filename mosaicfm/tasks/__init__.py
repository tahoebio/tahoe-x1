# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
from .emb_extractor import get_batch_embeddings
from .marginal_essentiality import MarginalEssentiality

__all__ = ["MarginalEssentiality", "get_batch_embeddings"]
