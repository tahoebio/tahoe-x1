# Submitting a Training Job

## On Run AI:

runai submit train-scgpt -i docker.io/mosaicml/llm-foundry:2.2.1_cu121_flash2-latest -g 2 --git-sync source=https://github.com/vevotx/vevo-scGPT,branch=dev-temp,target=/src -- /src/vevo-scGPT/mcli/runai_submit.sh

## On MCLI:

mcli run -f scgpt-1_3b-train.yaml --follow