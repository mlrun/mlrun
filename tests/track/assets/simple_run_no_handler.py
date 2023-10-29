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

import tempfile
from random import randint, random

import mlflow
import mlflow.environment_variables

import mlrun


# simple general mlflow example of hand logging
def simple_run():
    with mlflow.start_run() as run:
        # Log some random params and metrics
        mlflow.log_param("param1", randint(0, 100))
        mlflow.log_metric("foo", random())
        mlflow.log_metric("foo", random() + 1)
        mlflow.log_metric("foo", random() + 2)
        # Create an artifact and log it
        with tempfile.TemporaryDirectory() as test_dir:
            with open(f"{test_dir}/test.txt", "w") as f:
                f.write("hello world!")
                mlflow.log_artifacts(test_dir)
    return run.info.run_id


if __name__ == "__main__":
    # need to set context in order to receive and send params to test
    context = mlrun.get_or_create_ctx("mlflow_test")
    mlflow.set_tracking_uri(context.parameters["tracking_uri"])
    run_id = simple_run()
    context.log_result("run_id", run_id)
