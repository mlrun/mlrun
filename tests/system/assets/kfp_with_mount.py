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
from mlrun.execution import MLClientCtx


def handler(context: MLClientCtx, p1: int = 1) -> None:
    print(f"Run: {context.name} (uid={context.uid})")
    context.logger.info("started training")
    context.log_result("accuracy", p1 * 2)
    context.log_result("loss", p1 * 3)
    context.set_label("category", "tests")
    context.log_artifact("somefile", body=b"abc is 123", local_path="myfile.txt")
    context.log_model(
        "mymodel",
        body=b"abc is 123",
        model_file="model.txt",
        metrics={"accuracy": 0.85},
        parameters={"xx": "abc"},
        labels={"framework": "xgboost"},
        artifact_path=context.artifact_subpath("models"),
    )
