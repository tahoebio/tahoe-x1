# Submitting a Training Job

## On Run AI:

runai submit train-mosaicfm -i docker.io/vevotx/mosaicfm:1.1.0 -g 4 --git-sync source=https://github.com/vevotx/mosaicfm,branch=14-jan-2025-training-dataset-update,target=/src -- /src/mosaicfm/mcli/runai_submit.sh

## On MCLI:

mcli run -f mosaicfm-3b-v2.yaml --follow

runai training submit train-mosaicfm-2 -i docker.io/vevotx/mosaicfm:1.1.0 -g 2 --large-shm --git-sync repository=https://github.com/vevotx/mosaicfm/tree/14-jan-2025-training-dataset-update,path=/src --command -- /src/mosaicfm/mcli/runai_submit.sh