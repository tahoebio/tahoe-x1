# Launching a MosaicFM run on Google Cloud

DWS can be used either through Google Kubernetes Engine (GKE) or through MIG resize requests. Although the process with GKE would be more automated and require fewer manual steps (such as launching the launcher individually on each node) 
I was not able to get it to work. The initial steps for setting up GKE can be found [here](https://cloud.google.com/kubernetes-engine/docs/how-to/provisioningrequest). The provisioning-request and job-spec created for this process are included in this folder. 
To launch a run using a managed instance group please follow the instructions below:

1) Create an instance template following the guide [here](https://cloud.google.com/compute/docs/instance-groups/create-resize-requests-mig)
Settings used:
 - Machine type: a3-highgpu-8g
 - GPUs: 8 x NVIDIA H100 80GB
 - Firewall: Allow HTTP traffic, Allow HTTPS traffic.
 - Boot disk: 
   - c0-deeplearning-common-cu124-v20241224-debian-11
   - 200 GB attached SSD for boot drive
 - Reservations: Don't use any reservations
 - Location: Regional (us-central-1)

2) Enable egress on port 29500 for the instance group. This is required for the nodes to communicate with each other.
IP ranges: 0.0.0.0/0
ports: tcp:29500
See stackoverflow [link](https://stackoverflow.com/a/21068402) for steps to do this.

3) Create a managed instance group using the instance template created in step 1
Follow the guide linked in step-1 and use a resize request to create a MIG with 8 nodes.

4) SSH into the nodes individually and run the commands in setup_node.sh
Once you SSH you will be prompted to install NVIDIA drivers, select Y and then follow the rest of the steps in 
These are required to install NVIDIA docker and mount the SSD. You could also add the steps to the instance 
template to make this faster but I did not try this. 

5) Install screen to run the training script in a screen session on each node
```shell
sudo apt-get update
sudo apt-get install screen
```

```shell
screen -S trainingSession
```

6) Launch the run on each node
The master node address can be found using the guide [here](https://cloud.google.com/compute/docs/networking/using-internal-dns)
It follows the format `VM_NAME.ZONE.c.PROJECT_ID.internal`. You can check that it is accessible using a ping command.
In my case the value was `MASTER_ADDR=h100-cluster-401h.us-central1-a.c.vevo-ml.internal`
The Node rank is a number from 0-7 and must be changed for each node. The world size is 64 for the 8 nodes.
The final launch command inside the docker statement should look something like:
> composer --world_size 64 --node_rank 7 --master_addr h100-cluster-401h.us-central1-a.c.vevo-ml.internal --master_port 29500 train.py ../gcloud/mosaicfm-1_3b-merged.yaml

The keys for AWS, GITHUB, WANDB need to be populated as well.

```shell
docker run --network host --gpus all \
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
    if [ -d mosaicfm ]; then \
      cd tahoe-x1 && \
      git fetch --all && \
      git reset --hard origin/32-train-13b-model-with-full-dataset; \
    else \
      git clone -b 32-train-13b-model-with-full-dataset https://oauth2:\${GITHUB_TOKEN}@github.com/tahoebio/tahoe-x1.git && \
      cd tahoe-x1; \
    fi && \
    pip install -e . --no-deps && \
    cd scripts && \
    composer --world_size 64 --node_rank <NODE_RANK_0-7> --master_addr <MASTER_ADDR> --master_port 29500 train.py ../gcloud/tahoex-1_3b-merged.yaml"
```

I launched the runs in node order, ie master node first, followed by 1-7. 
Also, I used 127.0.0.0 as the MASTER_ADDR for rank_0 and `h100-cluster-401h.us-central1-a.c.vevo-ml.internal` for the rest.
Once the commands have been launched on each node, the training should begin.


