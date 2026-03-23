# MLRun CE development notes

This page contains notes for configuring your development system (after installation).

**In this section**

- [Change the deployment and jobs default PVC](#change-the-deployment-and-jobs-default-pvc)
- [Configuring the user Jupyter conda environment](#configuring-the-user-jupyter-conda-environment)
- [Configuring TimescaleDB and Kafka for model monitoring](#configuring-timescaledb-and-kafka-for-model-monitoring)

## Change the deployment and jobs default PVC
A default PVC is created during the MLRun installation. If you modified the env vars before importing MLRun (to change the PVC), those values are overwritten. Change the PVC, after importing MLRun, by running this code:

```
import mlrun
mlrun.mlconf.storage.auto_mount_type = "pvc"
pvc_params = {
    "pvc_name": <your-pvc-name>,
    "volume_name": <volume-name>,
    "volume_mount_path": <container mount path>,
}
mlrun.mlconf.storage.auto_mount_params = ",".join(
    [f"{key}={value}" for key, value in pvc_params.items()]
)
```

## Configuring the user Jupyter conda environment

The default Jupyter comes with a conda env named `mlrun`. This conda is not persistent.
If you install any packages on this conda env, and then the Jupyter pod gets restarted or deleted, those packages will be deleted.

To create a new, persistent, environment, run this in your Jupyter terminal, where `myenv` is the name of your environment:

```bash
# Create the virtual environment
conda create -n <myenv> python==3.11 -y

# Activate the virtual environment
conda activate <myenv>

# Make sure that ipykernel is installed
pip install --user ipykernel

# Add the new virtual environment to Jupyter
python -m ipykernel install --user --name <myenv> --display-name "Python (<myenv>)"
```

## Configuring TimescaleDB and Kafka for model monitoring
TimescaleDB and Kafka are part of the default CE installations for model monitoring.

  [TimescaleDB](https://docs.timescale.com/self-hosted/latest/install/) is a PostgreSQL-based time-series database used as the TSDB backend for model monitoring. 
  Default connection values for CE:
  - `Host`: timescaledb.<namespace>.svc.cluster.local
  - `Port`: 5432
  - `Database`: postgres
  - `User`: postgres
  - `Password`: postgres
  
  [Kafka](https://github.com/bitnami/charts/tree/main/bitnami/kafka) is the streaming platform used for data flow between model monitoring components.
  Default connection values for CE:
  - `Brokers`: kafka-stream.<namespace>.svc.cluster.local:9092
  
  ### Configuring data store profiles
  The connections are managed by using [data store profiles](../store/datastore.md#datastore-profiles). Data store profiles manage the connection credentials securely.
  ```python
  from mlrun.datastore.datastore_profile import (
      DatastoreProfileKafkaStream,
      DatastoreProfilePostgreSQL,
  )
  # Create and register TSDB profile
  tsdb_profile = DatastoreProfilePostgreSQL(
      name=tsdb_profile_name,
      user="postgres",
      password="postgres",
      host="timescaledb",
      port=5432,
      database="postgres",
  )
  project.register_datastore_profile(tsdb_profile)
  # Create and register stream profile
  stream_profile = DatastoreProfileKafkaStream(
      name=stream_profile_name,
      brokers="kafka-stream:9092",
      topics=[],
  )
  project.register_datastore_profile(stream_profile)
  # Set model monitoring credentials and enable the infrastructure
  project.set_model_monitoring_credentials(
      tsdb_profile_name=tsdb_profile.name,
      stream_profile_name=stream_profile.name,
  )
```
See more details, including additional configuration options, in {py:class}`~mlrun.projects.MlrunProject.set_model_monitoring_credentials`.
