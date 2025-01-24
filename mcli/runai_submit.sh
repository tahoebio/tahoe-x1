# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
cd /src/mosaicfm
pip install -e .
cd scripts
composer train.py /src/mosaicfm/runai/scgpt-50m-train.yaml
```
