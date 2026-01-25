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

from services.api.utils.image_builder.kaniko import KanikoImageBuilder


def test_kaniko_image_builder_adds_build_args_from_builder_env(mocked_k8s_helper):
    function = mlrun.new_function("test", kind="job")

    builder = KanikoImageBuilder()
    kpod = builder.make_build_pod(
        project="test",
        context="/context",
        dest="docker-hub/repo:image",
        dockerfile="./Dockerfile",
        builder_env=[client.V1EnvVar(name="GIT_TOKEN", value="token")],
        runtime_spec=function.spec,
    )

    args = kpod.args
    build_args = [args[i + 1] for i in range(len(args)) if args[i] == "--build-arg"]
    assert "GIT_TOKEN=token" in build_args
