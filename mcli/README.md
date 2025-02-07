# Submitting a Training Job

## On Run AI:

runai submit train-mosaicfm -i docker.io/vevotx/ml-scgpt:shreshth -g 4 --git-sync source=https://github.com/vevotx/mosaicfm,branch=14-jan-2025-training-dataset-update,target=/src -- /src/mosaicfm/mcli/runai_submit.sh

## On MCLI:

mcli run -f scgpt-1_3b-train.yaml --follow

runai training submit train-mosaicfm-2 -i docker.io/vevotx/ml-scgpt:shreshth -g 2 --large-shm --git-sync repository=https://github.com/vevotx/mosaicfm/tree/14-jan-2025-training-dataset-update,path=/src --command -- /src/mosaicfm/mcli/runai_submit.sh