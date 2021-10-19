import typing

import pydantic


class ClientSpec(pydantic.BaseModel):
    version: typing.Optional[str]
    namespace: typing.Optional[str]
    docker_registry: typing.Optional[str]
    remote_host: typing.Optional[str]
    mpijob_crd_version: typing.Optional[str]
    ui_url: typing.Optional[str]
    artifact_path: typing.Optional[str]
    spark_app_image: typing.Optional[str]
    spark_app_image_tag: typing.Optional[str]
    spark_history_server_path: typing.Optional[str]
    spark_operator_version: typing.Optional[str]
    kfp_image: typing.Optional[str]
    dask_kfp_image: typing.Optional[str]
    api_url: typing.Optional[str]
    nuclio_version: typing.Optional[str]
    ui_projects_prefix: typing.Optional[str]
    scrape_metrics: typing.Optional[str]
    hub_url: typing.Optional[str]
    default_function_node_selector: typing.Optional[str]
    igz_version: typing.Optional[str]
    auto_mount_type: typing.Optional[str]
    auto_mount_params: typing.Optional[str]
    default_function_priority_class_name: typing.Optional[str]
    valid_function_priority_class_names: typing.Optional[str]
    default_tensorboard_logs_path: typing.Optional[str]
