# Setting up KubeRay

This guide will walk you through setting up a Ray cluster on Kubernetes using the KubeRay project. KubeRay simplifies the deployment and management of Ray clusters on Kubernetes, making it easier to leverage Ray’s distributed computing capabilities.

## Prerequisites
Before you begin, ensure that you have the following tools installed and configured:

- **Kubernetes cluster**: A running Kubernetes cluster (e.g., Minikube, GKE, EKS, or AKS).
- **kubectl**: Command-line tool for interacting with Kubernetes.
- **Helm**: Kubernetes package manager.

For MacOS, you can install these tools using Homebrew:
```
brew install kubectl
brew install helm
```

---

## Step 1: Install the KubeRay Operator

The KubeRay operator is responsible for managing the lifecycle of the Ray cluster on Kubernetes. Install the operator using Helm:

1. Add the KubeRay Helm repository:
   ```bash
   helm repo add kuberay https://ray-project.github.io/kuberay-helm/
   ```

2. Install the KubeRay operator:
   ```bash
   helm install kuberay-operator kuberay/kuberay-operator \
       --version 1.1.1 \
       --skip-crds \
       --namespace ray-cluster \
       --create-namespace \
       --set singleNamespaceInstall=true \
       --set rbacEnable=true
   ```
   
   - The `nnamespace` flag specifies the namespace it's installed in. Here we use the `ray-cluster` namespace.
   - The `singleNamespaceInstall` flag limits the operator’s scope to the namespace it’s installed in (useful for testing and development).
   - The `rbacEnable` flag ensures the operator has the necessary permissions.

---

## Step 2: Install the Ray Cluster

Once the KubeRay operator is installed, you can create a Ray cluster using Helm. This uses a configuration file (`values.yaml`) specifying the cluster settings.

1. Install the Ray cluster:
   ```bash
   helm install raycluster kuberay/ray-cluster \
       --version 1.1.1 \
       --namespace ray-cluster \
       -f values.yaml
   ```

2. To upgrade the Ray cluster later, use the following command:
   ```bash
   helm upgrade raycluster kuberay/ray-cluster \
       --version 1.1.1 \
       --namespace ray-cluster \
       -f values.yaml
   ```

### Verify the Installation
After installing the cluster, check its status:

1. Verify that the Ray cluster is created:
   ```bash
   kubectl get rayclusters -namespace ray-cluster
   ```

2. Check the status of the Ray cluster pods:
   ```bash
   kubectl describe pods --selector=ray.io/cluster=raycluster-kuberay -namespace ray-cluster
   ```

---

## Step 3: Connect to the Ray Cluster

To use the Ray cluster, connect to it programmatically using Ray’s Python API:

```python
import ray

ray.init(address='ray://raycluster-kuberay-head-svc.ray-cluster.svc.cluster.local:10001')
```

Ensure the DNS resolution within your cluster is correctly configured for the above address to work. Replace `raycluster-kuberay-head-svc` with the name of your Ray cluster service if customized.

---

## Step 4: Enable the Ray Dashboard

The Ray dashboard provides a web-based interface to monitor the cluster’s status and tasks.

1. Apply the Ingress configuration (`ingress.yaml`):
   ```bash
   kubectl apply -f ingress.yaml
   ```

2. Access the Ray dashboard via the specified host (e.g., `ray-dashboard.your-domain.com`).

If you’re using a local Kubernetes cluster, consider using port-forwarding instead:
```bash
kubectl port-forward service/raycluster-kuberay-head-svc -n ray-cluster 8265:8265
```
Then access the dashboard at `http://localhost:8265`.

---

## Tips and Troubleshooting

1. **Debugging Issues**:
   - Check the logs of the Ray operator:
     ```bash
     kubectl logs deployment/kuberay-operator -n ray-cluster
     ```
   - Check the logs of the Ray head and worker pods:
     ```bash
     kubectl logs pod/<ray-head-or-worker-pod> -n ray-cluster
     ```

2. **Scaling the Cluster**:
   Modify the `values.yaml` file to adjust the number of workers or the resources allocated to the Ray head/worker pods. Then upgrade the deployment:
   ```bash
   helm upgrade raycluster kuberay/ray-cluster -n ray-cluster -f values.yaml
   ```

3. **Namespace Isolation**:
   If you want to manage multiple Ray clusters in different namespaces, disable the `singleNamespaceInstall` flag during the operator installation.

---

You now have a fully functional Ray cluster running on Kubernetes. You can scale it, monitor it via the dashboard, and use it for distributed computing tasks.

