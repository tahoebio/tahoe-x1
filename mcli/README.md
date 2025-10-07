# Submitting a Training Job

## On Run AI:

runai submit train-tahoex -i docker.io/vevotx/mosaicfm:1.1.0 -g 4 --git-sync source=https://github.com/tahoebio/tahoe-x1,branch=14-jan-2025-training-dataset-update,target=/src -- /src/tahoe-x1/mcli/runai_submit.sh

## On MCLI:

mcli run -f tahoex-3b-v2.yaml --follow

runai training submit train-tahoex-2 -i docker.io/vevotx/mosaicfm:1.1.0 -g 2 --large-shm --git-sync repository=https://github.com/tahoebio/tahoe-x1/tree/14-jan-2025-training-dataset-update,path=/src --command -- /src/tahoe-x1/mcli/runai_submit.sh