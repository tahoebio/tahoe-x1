# Google Cloud DWS Launch Instructions

Follow the guide [here](https://cloud.google.com/kubernetes-engine/docs/how-to/provisioningrequest?authuser=1#create-provisioningrequest)

1) Create an autopilot GKE cluster. Using Autopilot mode is recommended since you can skip the steps requiring node-pool creation.
2) Use the provisioning request form to request a DWS instance. Access the cloud console first and then use the following commands.
```shell
kubectl apply -f provisioning-request.yaml
```
3) Wait for the cluster to be provisioned. You can check the status of the provisioning request by running:
```shell
kubectl get provisioningrequest
```
4) Submtit your job using the job-spec template. Provide the image and commands as required.
```shell
kubectl apply -f job-spec.yml
```

