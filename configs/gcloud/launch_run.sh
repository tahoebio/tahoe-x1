# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
cd /src/tahoe-x1
pip install -e . --no-deps
cd scripts
composer train.py /src/tahoe-x1/gcloud/tahoe_x1-70m-merged.yaml
```
