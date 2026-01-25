# Copyright 2026 Iguazio
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


from kubernetes import client

import mlrun

from services.api.utils.image_builder.buildah import BuildahImageBuilder


def test_buildah_builder_basic_pod_with_secret(mocked_k8s_helper):
    runtime_spec = _make_runtime_spec()

    builder = BuildahImageBuilder()
    kpod = builder.make_build_pod(
        project="test",
        context="/context",
        dest="example.com/repo/image:tag",
        dockerfile="/context/Dockerfile",
        secret_name="reg-secret",
        runtime_spec=runtime_spec,
        builder_env=[client.V1EnvVar(name="A", value="b")],
        project_secrets=[],
    )

    assert kpod.image
    assert kpod.command == ["/bin/sh", "-c"]
    assert isinstance(kpod.args, list)
    # buildah command includes flags between "buildah" and "bud"/"push"
    assert any(" bud " in arg for arg in kpod.args)
    assert any(" push " in arg for arg in kpod.args)

    pod = kpod.pod
    mounts = pod.spec.containers[0].volume_mounts
    mount_paths = {m.mount_path for m in mounts}
    assert "/var/lib/containers" in mount_paths
    assert "/tmp/.docker" in mount_paths


def test_buildah_builder_remote_git_context_adds_init_container(mocked_k8s_helper):
    runtime_spec = _make_runtime_spec()
    builder = BuildahImageBuilder()
    kpod = builder.make_build_pod(
        project="test",
        context="git://github.com/mlrun/mlrun#refs/heads/development",
        dest="example.com/repo/image:tag",
        dockerfile="/context/Dockerfile",
        runtime_spec=runtime_spec,
    )

    init_names = [c.name for c in kpod.init_containers]
    assert "clone-context" in init_names
    pod = kpod.pod
    volume_names = {v.name for v in pod.spec.volumes}
    assert "context" in volume_names


def _make_runtime_spec():
    function = mlrun.new_function("test", kind="job")
    return function.spec
