import typing

from mlrun.config import config


def new_pipe_metadata(
    artifact_path: str = None,
    cleanup_ttl: int = None,
    op_transformers: typing.List[typing.Callable] = None,
):
    from kfp.dsl import PipelineConf

    def _set_artifact_path(task):
        from kubernetes import client as k8s_client

        task.add_env_variable(
            k8s_client.V1EnvVar(name="MLRUN_ARTIFACT_PATH", value=artifact_path)
        )
        return task

    conf = PipelineConf()
    cleanup_ttl = cleanup_ttl or int(config.kfp_ttl)

    if cleanup_ttl:
        conf.set_ttl_seconds_after_finished(cleanup_ttl)
    if artifact_path:
        conf.add_op_transformer(_set_artifact_path)
    if op_transformers:
        for op_transformer in op_transformers:
            conf.add_op_transformer(op_transformer)
    return conf
