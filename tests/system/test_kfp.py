# Copyright 2018 Iguazio
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
import os

import pytest
from kfp import dsl

import mlrun
import tests.system.base
from mlrun import mlconf


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestKFP(tests.system.base.TestMLRunSystem):
    project_name = "kfp-system-test"

    @pytest.mark.enterprise
    def test_kfp_with_mount(self):
        code_path = str(self.assets_path / "kfp_with_mount.py")
        kfp_with_v3io_mount = mlrun.code_to_function(
            name="my-kfp-with-mount",
            kind="job",
            filename=code_path,
            project=self.project_name,
            image="mlrun/mlrun",
        )
        kfp_with_v3io_mount.apply(mlrun.mount_v3io())

        @dsl.pipeline(name="job test", description="demonstrating mlrun usage")
        def job_pipeline(p1=9):
            kfp_with_v3io_mount.as_step(
                handler="handler", params={"p1": p1}, outputs=["mymodel"]
            )

        out = mlconf.artifact_path or os.path.abspath("./data")
        artifact_path = os.path.join(out, "{{run.uid}}")
        arguments = {"p1": 8}
        run_id = mlrun.run_pipeline(
            job_pipeline,
            arguments,
            experiment="my-job",
            artifact_path=artifact_path,
            project=self.project_name,
        )

        mlrun.wait_for_pipeline_completion(run_id, project=self.project_name)
