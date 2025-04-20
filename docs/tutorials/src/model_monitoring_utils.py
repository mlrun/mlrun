import os

import mlrun
from mlrun.datastore.datastore_profile import (
    DatastoreProfileKafkaSource,
    DatastoreProfileTDEngine,
    DatastoreProfileV3io,
)


def enable_model_monitoring(
    project: mlrun.projects.MlrunProject = None,
    tsdb_profile_name: str = "tsdb-profile",
    stream_profile_name: str = "stream-profile",
    base_period: int = 10,
    wait_for_deployment: bool = False,
    deploy_histogram_data_drift_app: bool = True,
) -> mlrun.projects.MlrunProject:
    # Setting model monitoring creds
    tsdb_profile = DatastoreProfileV3io(name=tsdb_profile_name)
    stream_profile = DatastoreProfileV3io(
        name=stream_profile_name, v3io_access_key=mlrun.mlconf.get_v3io_access_key()
    )

    if mlrun.mlconf.is_ce_mode():
        mlrun_namespace = os.environ.get("MLRUN_NAMESPACE", "mlrun")
        tsdb_profile = DatastoreProfileTDEngine(
            name=tsdb_profile_name,
            user="root",
            password="taosdata",
            host=f"tdengine-tsdb.{mlrun_namespace}.svc.cluster.local",
            port="6041",
        )

        stream_profile = DatastoreProfileKafkaSource(
            name=stream_profile_name,
            brokers=f"kafka-stream.{mlrun_namespace}.svc.cluster.local:9092",
            topics=[],
        )

    project.register_datastore_profile(stream_profile)
    project.register_datastore_profile(tsdb_profile)

    project.set_model_monitoring_credentials(
        replace_creds=True,
        tsdb_profile_name=tsdb_profile.name,
        stream_profile_name=stream_profile.name,
    )

    project.enable_model_monitoring(
        base_period=base_period,
        wait_for_deployment=wait_for_deployment,
        deploy_histogram_data_drift_app=deploy_histogram_data_drift_app,
    )
    return project
