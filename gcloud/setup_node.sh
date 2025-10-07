# Copyright (C) Vevo Therapeutics 2025. All rights reserved.

# Update system and install mdadm
sudo apt update && sudo apt install mdadm --no-install-recommends

# Create RAID 0 array
sudo mdadm --create /dev/md0 --chunk=256 --raid-devices=16 --level=raid0 \
/dev/disk/by-id/google-local-nvme-ssd-0 \
/dev/disk/by-id/google-local-nvme-ssd-1 \
/dev/disk/by-id/google-local-nvme-ssd-2 \
/dev/disk/by-id/google-local-nvme-ssd-3 \
/dev/disk/by-id/google-local-nvme-ssd-4 \
/dev/disk/by-id/google-local-nvme-ssd-5 \
/dev/disk/by-id/google-local-nvme-ssd-6 \
/dev/disk/by-id/google-local-nvme-ssd-7 \
/dev/disk/by-id/google-local-nvme-ssd-8 \
/dev/disk/by-id/google-local-nvme-ssd-9 \
/dev/disk/by-id/google-local-nvme-ssd-10 \
/dev/disk/by-id/google-local-nvme-ssd-11 \
/dev/disk/by-id/google-local-nvme-ssd-12 \
/dev/disk/by-id/google-local-nvme-ssd-13 \
/dev/disk/by-id/google-local-nvme-ssd-14 \
/dev/disk/by-id/google-local-nvme-ssd-15

# Format the RAID array
sudo mkfs.ext4 -F /dev/md0

# Create mount directory and set permissions
sudo mkdir -p /mnt/disks/ssd

# Mount the RAID array
sudo mount /dev/md0 /mnt/disks/ssd

sudo chmod a+w /mnt/disks/ssd


# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

docker run --network host -it --gpus all \
  --shm-size 50gb \
  -e AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> \
  -e AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> \
  -e AWS_DEFAULT_REGION="us-east-2" \
  -e GITHUB_TOKEN=<GITHUB_TOKEN> \
  -e WANDB_API_KEY=<WANDB_API_KEY> \
  -e WANDB_ENTITY="vevotx" \
  -e WANDB_PROJECT="tahoex" \
  --mount type=bind,source=/mnt/disks/ssd,target=/src \
  --entrypoint /bin/bash \
  vevotx/mosaicfm:1.1.0 -c "\
  mkdir -p /src && \
  cd /src && \
  git clone -b 32-train-13b-model-with-full-dataset https://oauth2:\${GITHUB_TOKEN}@github.com/tahoebio/tahoe-x1.git && \
  cd mosaicfm && \
  pip install -e . --no-deps && \
  cd scripts && \
  /bin/bash"


