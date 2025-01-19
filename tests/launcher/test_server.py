# Copyright 2025 Iguazio
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
import pathlib

import mlrun.config
import mlrun.launcher.remote

import services.api.launcher

assets_path = pathlib.Path(__file__).parent / "assets"
func_path = assets_path / "sample_function.py"
handler = "hello_word"


def test_new_function_args_with_default_image_pull_secret(rundb_mock):
    mlrun.mlconf.function.spec.image_pull_secret = "adam-docker-registry-auth"
    launcher = services.api.launcher.ServerSideLauncher(
        auth_info=mlrun.common.schemas.AuthInfo()
    )
    runtime = mlrun.code_to_function(
        name="test",
        kind="job",
        filename=str(func_path),
        handler=handler,
        image="mlrun/mlrun",
    )
    uid = "123"
    run = mlrun.run.RunObject(
        metadata=mlrun.model.RunMetadata(uid=uid),
    )
    rundb_mock.store_run(run, uid)
    run = launcher._create_run_object(run)

    run = launcher._enrich_run(
        runtime,
        run=run,
    )
    assert run.spec.image_pull_secret == mlrun.mlconf.function.spec.image_pull_secret
