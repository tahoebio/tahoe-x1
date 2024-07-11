cd /src/vevo-scGPT
pip install -e .
cd scripts
composer train.py /src/vevo-scGPT/runai/scgpt-50m-train.yaml
```