import os

import mlrun
from mlrun.datastore.datastore_profile import (
    DatastoreProfileKafkaStream,
    DatastoreProfilePostgreSQL,
    DatastoreProfileV3io,
)


def enable_model_monitoring(
    project: mlrun.projects.MlrunProject = None,
    tsdb_profile_name: str = "tsdb-profile",
    stream_profile_name: str = "stream-profile",
    base_period: int = 10,
    wait_for_deployment: bool = False,
    deploy_histogram_data_drift_app: bool = True,
    lag_threshold: int | None = None,
    lag_event_cooldown: int | None = None,
) -> mlrun.projects.MlrunProject:
    # Setting model monitoring creds
    tsdb_profile = DatastoreProfileV3io(name=tsdb_profile_name)
    stream_profile = DatastoreProfileV3io(
        name=stream_profile_name, v3io_access_key=mlrun.mlconf.get_v3io_access_key()
    )

    if not mlrun.mlconf.is_using_v3io():
        mlrun_namespace = os.environ.get("MLRUN_NAMESPACE", "mlrun")

        kafka_broker = ""
        if "KAFKA_BROKER" in os.environ:
            kafka_broker = os.environ.get("KAFKA_BROKER")
        else:
            kafka_broker = f"kafka-stream.{mlrun_namespace}.svc.cluster.local:9092"

        pg_sql = ""
        if "PG_SQL" in os.environ:
            pg_sql = os.environ.get("PG_SQL")
        else:
            pg_sql = f"timescaledb.{mlrun_namespace}.svc.cluster.local"

        tsdb_profile = DatastoreProfilePostgreSQL(
            name=tsdb_profile_name,
            user="postgres",
            password="postgres",
            host=pg_sql,
            port=5432,
            database="mlrun",
        )

        stream_profile = DatastoreProfileKafkaStream(
            name=stream_profile_name,
            brokers=kafka_broker,
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
        lag_threshold=lag_threshold,
        lag_event_cooldown=lag_event_cooldown,
    )
    return project
