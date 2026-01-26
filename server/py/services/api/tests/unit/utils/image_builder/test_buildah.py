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


import kubernetes.client as k8s_client

import mlrun

import services.api.utils.image_builder.buildah as buildah_image_builder


def test_buildah_builder_basic_pod_with_secret(mocked_k8s_helper):
    runtime_spec = _make_runtime_spec()

    builder = buildah_image_builder.BuildahImageBuilder()
    kpod = builder.make_build_pod(
        project="test",
        context="/context",
        dest="example.com/repo/image:tag",
        dockerfile="/context/Dockerfile",
        secret_name="reg-secret",
        runtime_spec=runtime_spec,
        builder_env=[k8s_client.V1EnvVar(name="A", value="b")],
        project_secrets=[],
    )

    assert kpod.image
    assert kpod.command == ["/bin/sh", "-c"]
    assert isinstance(kpod.args, list)
    assert len(kpod.args) == 1
    build_and_push_cmd = kpod.args[0]
    parts = [part.strip() for part in build_and_push_cmd.split(";") if part.strip()]
    assert parts[0] == "set -e"
    build_cmd = parts[1]
    push_cmd = parts[2]

    # Ensure buildah global args appear before subcommand
    assert build_cmd.startswith("buildah --storage-driver=vfs --log-level=info build ")
    assert push_cmd.startswith("buildah --storage-driver=vfs --log-level=info push ")

    # Ensure tls verify is a subcommand arg (after build/push)
    assert " build --tls-verify=" in build_cmd
    assert " push --tls-verify=" in push_cmd
    assert "--retry=" in push_cmd

    pod = kpod.pod
    mounts = pod.spec.containers[0].volume_mounts
    mount_paths = {m.mount_path for m in mounts}
    assert "/var/lib/containers" in mount_paths
    assert "/tmp/.docker" in mount_paths


def test_buildah_builder_tls_verify_pull_and_push_modes(mocked_k8s_helper, monkeypatch):
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.builder, "insecure_pull_registry_mode", "enabled"
    )
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.builder, "insecure_push_registry_mode", "disabled"
    )

    runtime_spec = _make_runtime_spec()
    builder = buildah_image_builder.BuildahImageBuilder()
    kpod = builder.make_build_pod(
        project="test",
        context="/context",
        dest="example.com/repo/image:tag",
        dockerfile="/context/Dockerfile",
        runtime_spec=runtime_spec,
        secret_name=None,
    )

    build_and_push_cmd = kpod.args[0]
    parts = [part.strip() for part in build_and_push_cmd.split(";") if part.strip()]
    build_cmd = parts[1]
    push_cmd = parts[2]

    # enabled => allow insecure => --tls-verify=false
    assert " build --tls-verify=false " in f" {build_cmd} "
    # disabled => disallow insecure => --tls-verify=true
    assert " push --tls-verify=true " in f" {push_cmd} "


def test_buildah_builder_remote_git_context(mocked_k8s_helper):
    runtime_spec = _make_runtime_spec()
    builder = buildah_image_builder.BuildahImageBuilder()
    kpod = builder.make_build_pod(
        project="test",
        context="git://github.com/mlrun/mlrun#refs/heads/development",
        dest="example.com/repo/image:tag",
        dockerfile="/context/Dockerfile",
        runtime_spec=runtime_spec,
    )

    init_names = [c.name for c in kpod.init_containers]
    assert "clone-context" in init_names
    clone_container = next(c for c in kpod.init_containers if c.name == "clone-context")
    assert clone_container.args, "Expected clone-context to have args"
    clone_cmd = " ".join(clone_container.args)
    assert "git clone" in clone_cmd
    assert "https://github.com/mlrun/mlrun.git" in clone_cmd
    pod = kpod.pod
    volume_names = {v.name for v in pod.spec.volumes}
    assert "context" in volume_names


def _make_runtime_spec():
    function = mlrun.new_function("test", kind="job")
    return function.spec
