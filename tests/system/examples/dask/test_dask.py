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
import datetime
import os

import kfp
import kfp.compiler
import pytest

import mlrun.utils
from mlrun import (
    _run_pipeline,
    code_to_function,
    mount_v3io,
    new_task,
    wait_for_pipeline_completion,
)
from mlrun.run import RunStatuses
from tests.system.base import TestMLRunSystem


# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestDask(TestMLRunSystem):
    def custom_setup(self):
        self._logger.debug("Creating dask function")
        self.dask_function = code_to_function(
            "mydask",
            kind="dask",
            filename=str(self.assets_path / "dask_function.py"),
        ).apply(mount_v3io())

        self.dask_function.spec.image = "mlrun/ml-base"
        self.dask_function.spec.remote = True
        self.dask_function.spec.replicas = 1
        self.dask_function.spec.service_type = "NodePort"
        self.dask_function.spec.command = str(self.assets_path / "dask_function.py")

    def test_dask(self):
        run_object = self.dask_function.run(handler="main", params={"x": 12})
        self._logger.debug("Finished running task", run_object=run_object.to_dict())

        run_uid = run_object.uid()

        assert run_uid is not None
        self._verify_run_metadata(
            run_object.to_dict()["metadata"],
            uid=run_uid,
            name="mydask-main",
            project=self.project_name,
            labels={
                "v3io_user": self._test_env["V3IO_USERNAME"],
                "owner": self._test_env["V3IO_USERNAME"],
            },
        )
        self._verify_run_spec(
            run_object.to_dict()["spec"],
            parameters={"x": 12},
            outputs=[],
            output_path="",
            secret_sources=[],
            data_stores=[],
        )

        assert run_object.state() == "completed"

    def test_run_pipeline(self):
        @kfp.dsl.pipeline(name="dask_pipeline")
        def dask_pipe(x=1, y=10):

            # use_db option will use a function (DB) pointer instead of adding the function spec to the YAML
            self.dask_function.as_step(
                new_task(handler="main", name="dask_pipeline", params={"x": x, "y": y}),
                use_db=True,
            )

        kfp.compiler.Compiler().compile(dask_pipe, "daskpipe.yaml", type_check=False)
        arguments = {"x": 4, "y": -5}
        artifact_path = "/User/test"
        workflow_run_id = _run_pipeline(
            dask_pipe,
            arguments,
            project=self.project_name,
            artifact_path=artifact_path,
            run="DaskExamplePipeline",
            experiment="dask pipe",
        )

        wait_for_pipeline_completion(workflow_run_id)

        runs = self._run_db.list_runs(
            project=self.project_name, labels=f"workflow={workflow_run_id}"
        )
        assert len(runs) == 1

        run = runs[0]
        run_uid = run["metadata"]["uid"]
        self._verify_run_metadata(
            run["metadata"],
            uid=run_uid,
            name="mydask-main",
            project=self.project_name,
            labels={
                "v3io_user": self._test_env["V3IO_USERNAME"],
                "owner": self._test_env["V3IO_USERNAME"],
            },
        )
        self._verify_run_spec(
            run["spec"],
            parameters={"x": 4, "y": -5},
            outputs=["run_id"],
            output_path="/User/test",
            data_stores=[],
        )

        # remove compiled dask.yaml file
        os.remove("daskpipe.yaml")

    def test_dask_close(self):
        self._logger.info("Initializing dask cluster")
        cluster_start_time = datetime.datetime.now()

        # initialize the dask cluster and get its dashboard url
        client = self.dask_function.client
        time_took = (datetime.datetime.now() - cluster_start_time).seconds
        self._logger.info(
            "Dask cluster initialization completed", took_in_seconds=time_took
        )

        worker_start_time = datetime.datetime.now()
        client.wait_for_workers(self.dask_function.spec.replicas)
        time_took = (datetime.datetime.now() - worker_start_time).seconds
        self._logger.info("Workers initialization completed", took_in_seconds=time_took)

        self._logger.info("Shutting Down Cluster")
        self.dask_function.close()

        # wait for the dask cluster to completely shut down
        mlrun.utils.retry_until_successful(
            5,
            60,
            self._logger,
            True,
            self._wait_for_dask_cluster_to_shutdown,
            "mydask",
        )

        # Client supposed to be closed
        with pytest.raises(AttributeError):
            client.list_datasets()

        # Cluster supposed to be decommissioned
        with pytest.raises(RuntimeError):
            client.restart()

    def _wait_for_dask_cluster_to_shutdown(self, dask_cluster_name):
        runtime_resources = mlrun.get_run_db().list_runtime_resources(
            project=self.project_name,
            kind="dask",
            object_id=dask_cluster_name,
        )
        resources = runtime_resources[0].resources
        # Waiting for workers to be removed and scheduler status to completed
        if len(resources.pod_resources) > 1:
            raise mlrun.errors.MLRunRuntimeError("Cluster did not completely clean up")

        for pod in resources.pod_resources:
            if pod.status.get("phase") != RunStatuses.succeeded:
                raise mlrun.errors.MLRunRuntimeError("Cluster still running")
        return True
