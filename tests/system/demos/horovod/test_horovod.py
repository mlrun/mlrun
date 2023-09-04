# Copyright 2023 Iguazio
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

import pytest

import mlrun
from tests.system.base import TestMLRunSystem
from tests.system.demos.base import TestDemo


# Marked as enterprise because of v3io mount and pipelines
@pytest.mark.skip("not up to date demos needs to run demos from mlrun/demos repo")
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestHorovodTFv2(TestDemo):

    project_name = "horovod-project"

    def create_demo_project(self) -> mlrun.projects.MlrunProject:
        self._logger.debug("Creating horovod project")
        demo_project = mlrun.new_project(
            self.project_name, str(self.assets_path), init_git=True
        )

        mlrun.mount_v3io()

        self._logger.debug("Uploading training file")
        trainer_src_path = str(self.assets_path / "horovod_training.py")
        trainer_dest_path = pathlib.Path("/assets/horovod_training.py")
        stores = mlrun.datastore.store_manager.set()
        datastore, subpath = stores.get_or_create_store(
            self._get_v3io_user_store_path(trainer_dest_path)
        )
        datastore.upload(subpath, trainer_src_path)

        self._logger.debug("Creating iris-generator function")
        function_path = str(self.assets_path / "utils_functions.py")
        utils = mlrun.code_to_function(
            name="utils",
            kind="job",
            filename=function_path,
            image="mlrun/mlrun",
        )

        utils.spec.remote = True
        utils.spec.replicas = 1
        utils.spec.service_type = "NodePort"

        self._logger.debug("Setting project functions")
        demo_project.set_function(utils)

        trainer = mlrun.new_function(
            name="trainer",
            kind="mpijob",
            command=self._get_v3io_user_store_path(trainer_dest_path, remote=False),
            image="mlrun/mlrun",
        )
        trainer.spec.remote = True
        trainer.spec.replicas = 4
        trainer.spec.service_type = "NodePort"

        demo_project.set_function(trainer)
        demo_project.set_function("hub://tf2-serving", "serving")

        demo_project.log_artifact(
            "images",
            target_path="http://iguazio-sample-data.s3.amazonaws.com/catsndogs.zip",
            artifact_path=mlrun.mlconf.artifact_path,
        )

        self._logger.debug("Setting project workflow")
        demo_project.set_workflow(
            "main", str(self.assets_path / "workflow.py"), embed=True
        )

        return demo_project

    # FIXME: test is not working for long time, commenting out, should be changed to be like the mask detection demo
    # def test_demo(self):
    #     self.run_and_verify_project(
    #         runs_amount=3,
    #         arguments={
    #             "model_name": "cat_vs_dog_tfv2",
    #             "images_dir": self._workflow_artifact_path + "/images",
    #         },
    #     )
