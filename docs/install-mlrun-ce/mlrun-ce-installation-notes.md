# MLRun CE installation notes

This page lists additional steps or configuration options you may need to follow for non-default MLRun CE installations.

**In this section**

- [Advanced chart configuration](#advanced-chart-configuration)
- [Opt out of components](#opt-out-of-components)
- [Using NFS storage](#using-nfs-storage)
- [Configuring the online feature store](#configuring-the-online-feature-store)
- [Installing Spark Operator on non-mlrun namespace](#installing-spark-operator-on-non-mlrun-namespace)

## Advanced chart configuration

Configurable values are documented in the `values.yaml`, and the `values.yaml` of all sub charts. Override those [in the normal methods](https://helm.sh/docs/chart_template_guide/values_files/).

See also the [MLRun CE values file reference](https://github.com/mlrun/ce/blob/development/charts/mlrun-ce/values.yaml)
## Opt out of components
The chart installs many components. You may not need them all in your deployment depending on your use cases.
To opt out of some of the components, use the following helm values:
```bash
--set pipelines.enabled=false \
--set kube-prometheus-stack.enabled=false \
--set spark-operator.enabled=false \
```


## Using NFS storage
If you are using NFS storage in your Kubernetes cluster, add these flags to the chart deployment command:
```
  --set kube-prometheus-stack.grafana.securityContext.runAsUser=1000 
  --set kube-prometheus-stack.grafana.securityContext.runAsGroup=1000 
  --set kube-prometheus-stack.grafana.securityContext.fsGroup=1000 
  --set kube-prometheus-stack.grafana.securityContext.fsGroupChangePolicy=OnRootMismatch 
  --set kube-prometheus-stack.grafana.initChownData.enabled
```

## Configuring the online feature store
The MLRun Community Edition supports the online feature store. To enable it, you need to first deploy a Redis service that is accessible to your MLRun CE cluster.
To deploy a Redis service, refer to the [Redis documentation](https://redis.io/learn/howtos/quick-start).

When you have a Redis service deployed, you can configure MLRun CE to use it by adding the following helm value configuration to your helm install command:
```bash
--set mlrun.api.extraEnvKeyValue.MLRUN_REDIS__URL=<redis-address>
```
## Installing Spark Operator on non-mlrun namespace
By default Spark Operator jobNamespaces is set to "mlrun" namespace. If you are installing Spark Operator on a different namespace you need to set the jobNamespaces value accordingly
```bash
--set spark-operator.jobNamespaces={your-namespace}
```