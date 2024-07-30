# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
cd /src/vevo-scgpt-private
pip install -e .
cd scripts
composer train.py /src/vevo-scgpt-private/runai/scgpt-50m-train.yaml
```
