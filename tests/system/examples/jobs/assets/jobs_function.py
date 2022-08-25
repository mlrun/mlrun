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
import pandas as pd

from mlrun.artifacts import get_model, update_model
from mlrun.datastore import DataItem
from mlrun.execution import MLClientCtx


def training(context: MLClientCtx, p1: int = 1, p2: int = 2) -> None:
    """Train a model.

    :param context: The runtime context object.
    :param p1: A model parameter.
    :param p2: Another model parameter.
    """
    # access input metadata, values, and inputs
    print(f"Run: {context.name} (uid={context.uid})")
    print(f"Params: p1={p1}, p2={p2}")
    context.logger.info("started training")

    # <insert training code here>

    # log the run results (scalar values)
    context.log_result("accuracy", p1 * 2)
    context.log_result("loss", p1 * 3)

    # add a lable/tag to this run
    context.set_label("category", "tests")

    # log a simple artifact + label the artifact
    # If you want to upload a local file to the artifact repo add src_path=<local-path>
    context.log_artifact("somefile", body=b"abc is 123", local_path="myfile.txt")

    # create a dataframe artifact
    df = pd.DataFrame([{"A": 10, "B": 100}, {"A": 11, "B": 110}, {"A": 12, "B": 120}])
    context.log_dataset("mydf", df=df)

    # Log an ML Model artifact, add metrics, params, and labels to it
    # and place it in a subdir ('models') under artifacts path
    context.log_model(
        "mymodel",
        body=b"abc is 123",
        model_file="model.txt",
        metrics={"accuracy": 0.85},
        parameters={"xx": "abc"},
        labels={"framework": "xgboost"},
        artifact_path=context.artifact_subpath("models"),
    )


def validation(context: MLClientCtx, model: DataItem) -> None:
    """Model validation.

    Dummy validation function.

    :param context: The runtime context object.
    :param model: The extimated model object.
    """
    # access input metadata, values, files, and secrets (passwords)
    print(f"Run: {context.name} (uid={context.uid})")
    context.logger.info("started validation")

    # get the model file, class (metadata), and extra_data (dict of key: DataItem)
    model_file, model_obj, _ = get_model(model)

    # update model object elements and data
    update_model(model_obj, parameters={"one_more": 5})

    print(f"path to local copy of model file - {model_file}")
    print("parameters:", model_obj.parameters)
    print("metrics:", model_obj.metrics)
    context.log_artifact("validation", body=b"<b> validated </b>", format="html")
