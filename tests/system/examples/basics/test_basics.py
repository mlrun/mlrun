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
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlrun import new_task, run_local
from mlrun.artifacts import PlotArtifact
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
class TestBasics(TestMLRunSystem):
    def custom_setup(self):
        self._logger.debug("Creating basics task")

        # {{run.uid}} will be substituted with the run id, so output will be written to different directories per run
        output_path = str(self.results_path / "{{run.uid}}")
        self._basics_task = (
            new_task(name="demo", params={"p1": 5}, artifact_path=output_path)
            .with_secrets("file", self.assets_path / "secrets.txt")
            .set_label("type", "demo")
        )

        self._logger.debug("Creating inline task")
        self._inline_task = new_task(
            name="demo2",
            handler=self._get_inline_handler(),
            artifact_path=str(self.results_path / "{{run.uid}}"),
        )

    def test_basics(self):
        run_object = run_local(
            self._basics_task, command="training.py", workdir=str(self.assets_path)
        )
        self._logger.debug("Finished running task", run_object=run_object.to_dict())

        run_uid = run_object.uid()

        assert run_uid is not None
        self._verify_run_metadata(
            run_object.to_dict()["metadata"],
            uid=run_uid,
            name="demo",
            project=self.project_name,
            labels={"kind": "", "framework": "sklearn"},
        )
        self._verify_run_spec(
            run_object.to_dict()["spec"],
            parameters={"p1": 5, "p2": "a-string"},
            inputs={"infile.txt": str(self.assets_path / "infile.txt")},
            outputs=[],
            output_path=str(self.results_path / run_uid),
            secret_sources=[],
            data_stores=[],
        )

        assert run_object.state() == "completed"

        self._verify_run_outputs(
            run_object.outputs,
            uid=run_uid,
            name="demo",
            project=self.project_name,
            output_path=self.results_path / run_uid,
            accuracy=10,
            loss=15,
        )

    def test_basics_hyper_parameters(self):
        run_object = run_local(
            self._basics_task.with_hyper_params({"p2": [5, 2, 3]}, "min.loss"),
            command="training.py",
            workdir=str(self.assets_path),
        )
        self._logger.debug("Finished running task", run_object=run_object.to_dict())

        run_uid = run_object.uid()

        assert run_uid is not None
        assert run_object.state() == "completed"

        self._verify_run_outputs(
            run_object.outputs,
            uid=run_uid,
            name="demo",
            project=self.project_name,
            output_path=self.results_path / run_uid,
            accuracy=10,
            loss=15,
            best_iteration=1,
            iteration_results=True,
        )

    def test_inline_code(self):
        run_object = run_local(self._inline_task.with_params(p1=7))
        self._logger.debug("Finished running task", run_object=run_object.to_dict())

        run_uid = run_object.uid()

        assert run_uid is not None
        assert run_object.state() == "completed"

    def test_inline_code_with_param_file(self):
        run_object = run_local(
            self._inline_task.with_param_file(
                str(self.assets_path / "params.csv"), "max.accuracy"
            )
        )
        self._logger.debug("Finished running task", run_object=run_object.to_dict())

        run_uid = run_object.uid()

        assert run_uid is not None
        assert run_object.state() == "completed"

    @staticmethod
    def _get_inline_handler():
        def handler(context, p1=1, p2="xx"):
            """this is a simple function

            :param context: handler context
            :param p1:  first param
            :param p2:  another param
            """
            # access input metadata, values, and inputs
            print(f"Run: {context.name} (uid={context.uid})")
            print(f"Params: p1={p1}, p2={p2}")

            time.sleep(1)

            # log the run results (scalar values)
            context.log_result("accuracy", p1 * 2)
            context.log_result("loss", p1 * 3)

            # add a lable/tag to this run
            context.set_label("category", "tests")

            # create a matplot figure and store as artifact
            fig, ax = plt.subplots()
            np.random.seed(0)
            x, y = np.random.normal(size=(2, 200))
            color, size = np.random.random((2, 200))
            ax.scatter(x, y, c=color, s=500 * size, alpha=0.3)
            ax.grid(color="lightgray", alpha=0.7)

            context.log_artifact(PlotArtifact("myfig", body=fig, title="my plot"))

            # create a dataframe artifact
            df = pd.DataFrame(
                [{"A": 10, "B": 100}, {"A": 11, "B": 110}, {"A": 12, "B": 120}]
            )
            context.log_dataset("mydf", df=df)

            # Log an ML Model artifact
            context.log_model(
                "mymodel",
                body=b"abc is 123",
                model_file="model.txt",
                model_dir="data",
                metrics={"accuracy": 0.85},
                parameters={"xx": "abc"},
                labels={"framework": "xgboost"},
            )

            return "my resp"

        return handler
