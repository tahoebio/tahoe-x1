# Copyright (C) Tahoe Therapeutics 2025-2026. All rights reserved.
"""Generate cell and gene embeddings using ``composer.Trainer.predict``.

This script loads a trained :class:`~tahoe_x1.model.ComposerTX` and
produces embeddings for an input AnnData file. Configuration is provided via a
YAML file.

Example usage:

.. code-block:: bash

    python scripts/inference/predict_embeddings.py configs/predict.yaml [--key=value ...]
"""

import logging
import sys

from omegaconf import OmegaConf as om

from tahoe_x1.inference import predict_embeddings

log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":

    num_mand_args = 2
    if len(sys.argv) < num_mand_args:
        raise SystemExit("Usage: predict_embeddings.py <config.yaml> [--key=value ...]")

    # Load base config from YAML file
    cfg = om.load(sys.argv[1])

    # Merge with command line arguments
    cli_args = []
    for arg in sys.argv[num_mand_args:]:
        # Convert --key=value to key=value format for OmegaConf
        if arg.startswith("--"):
            cli_args.append(arg[2:])
        else:
            cli_args.append(arg)

    cli_cfg = om.from_cli(cli_args)
    cfg = om.merge(cfg, cli_cfg)

    om.resolve(cfg)
    predict_embeddings(cfg)
