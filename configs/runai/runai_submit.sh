# Copyright (C) Tahoe Therapeutics 2025-2026. All rights reserved.
cd /src/tahoe-x1
pip install -e . --no-deps
cd scripts
composer train.py /src/tahoe-x1/configs/test_run.yaml
```
