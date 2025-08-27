# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
cd /src/mosaicfm
pip install -e . --no-deps
cd scripts
composer train.py /src/mosaicfm/runai/mosaicfm-70m-tahoe.yaml
```
