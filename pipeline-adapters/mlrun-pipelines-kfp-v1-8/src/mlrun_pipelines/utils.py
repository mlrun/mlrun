# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import tempfile
import typing

from kubernetes import client

from mlrun_pipelines.helpers import new_pipe_metadata
from mlrun_pipelines.imports import compiler, kfp  # noqa: F401

if typing.TYPE_CHECKING:
    from mlrun.runtimes import BaseRuntime


def apply_kfp(
    modify: typing.Callable,
    cop: kfp.dsl.ContainerOp,
    runtime: "BaseRuntime",
) -> kfp.dsl.ContainerOp:
    modify(cop)

    # Have to do it here to avoid circular dependencies
    from mlrun.runtimes.pod import AutoMountType

    if AutoMountType.is_auto_modifier(modify):
        runtime.spec.disable_auto_mount = True

    api = client.ApiClient()
    for k, v in cop.pod_labels.items():
        runtime.metadata.labels[k] = v
    for k, v in cop.pod_annotations.items():
        runtime.metadata.annotations[k] = v
    if cop.container.env:
        env_names = [
            e.name if hasattr(e, "name") else e["name"] for e in runtime.spec.env
        ]
        for e in api.sanitize_for_serialization(cop.container.env):
            name = e["name"]
            if name in env_names:
                runtime.spec.env[env_names.index(name)] = e
            else:
                runtime.spec.env.append(e)
                env_names.append(name)
        cop.container.env.clear()

    if cop.volumes and cop.container.volume_mounts:
        vols = api.sanitize_for_serialization(cop.volumes)
        mounts = api.sanitize_for_serialization(cop.container.volume_mounts)
        runtime.spec.update_vols_and_mounts(vols, mounts)
        cop.volumes.clear()
        cop.container.volume_mounts.clear()

    return runtime


def compile_pipeline(
    artifact_path,
    cleanup_ttl,
    ops,
    pipeline,
    pipe_file: typing.Optional[str] = None,
    type_check: bool = False,
):
    if not pipe_file:
        pipe_file = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False).name
    conf = new_pipe_metadata(
        artifact_path=artifact_path,
        cleanup_ttl=cleanup_ttl,
        op_transformers=ops,
    )
    compiler.Compiler().compile(
        pipeline, pipe_file, type_check=type_check, pipeline_conf=conf
    )
    return pipe_file


def get_client(
    url: typing.Optional[str] = None, namespace: typing.Optional[str] = None
) -> kfp.Client:
    if url or namespace:
        return kfp.Client(host=url, namespace=namespace)
    return kfp.Client()
