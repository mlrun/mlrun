# MLRun CE development notes

This page contains notes for configuring your development system (after installation).

**In this section**

- [Change the deployment and jobs default PVC](#change-the-deployment-and-jobs-default-pvc)
- [Configuring the user Jupyter conda environment](#configuring-the-user-jupyter-conda-environment)
- [Configuring TDengine and Kafka for model monitoring](#configuring-tdengine-and-kafka-for-model-monitoring)

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

## Configuring TDengine and Kafka for model monitoring
TDengine and Kafka are part of the default CE installations. These are the default TDengine and Kafka installation values.

The connections are managed by using [MLRun datastore profiles](https://docs.mlrun.org/en/stable/store/datastore.html#data-store-profiles). datastore profiles manage the connection credentials securely.
```py
# Create and register TSDB profile
tsdb_profile = DatastoreProfileTDEngine(
    name=tsdb_profile_name,
    user="root",
    password="taosdata",
    host=f"tdengine-tsdb",
    port="6041",
)
project.register_datastore_profile(tsdb_profile)

# Create and register stream profile
stream_profile = DatastoreProfileKafkaSource(
    name=stream_profile_name,
    brokers=f"kafka-stream:9092",
    topics=[],
)

# Set model monitoring credentials and enable the infrastructure
project.set_model_monitoring_credentials(
    tsdb_profile_name=tsdb_profile.name,
    stream_profile_name=stream_profile.name,
)
```

See more details, including additional configuration options, in {py:class}`~mlrun.projects.MlrunProject.set_model_monitoring_credentials`.
