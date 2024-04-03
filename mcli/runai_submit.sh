pip install llm-foundry==0.6.0 mosaicml[deepspeed]
cd /src/vevo-scGPT
pip install -e .
cd scripts
composer train.py /src/vevo-scGPT/runai/scgpt-50m-train.yaml
```