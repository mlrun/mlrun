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
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import mlrun
from mlrun import DataItem
from mlrun.artifacts import PlotlyArtifact


def log_dataset(context: mlrun.MLClientCtx, dataset_name: str):
    raw_data = {
        "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
    }
    df = pd.DataFrame(raw_data, columns=["first_name"])
    context.log_dataset(dataset_name, df=df, stats=True, format="parquet")


def api_backward_compatibility_tests_succeeding_function(
    context: mlrun.MLClientCtx, dataset_src: DataItem
):
    # Dataset loading
    df = dataset_src.as_df()

    # Dataset logging, this is for test purposes only, most of the times user will won't save the df with exactly the
    # same data but rather, do some transformation on the data, or use it for training.
    logged_dataset = context.log_dataset("mydf", df=df, stats=True)
    context.logger.info("Logged dataset", dataset_artifact=logged_dataset.base_dict())

    # Simple artifact logging
    logged_artifact = context.log_artifact(
        "model",
        body=b"abc is 123",
        local_path="model.txt",
        labels={"framework": "xgboost"},
    )
    context.logger.info("Logged artifact", artifact=logged_artifact.base_dict())

    # logging PlotlyArtifact
    x = np.arange(10)
    fig = go.Figure(data=go.Scatter(x=x, y=x**2))

    plotly = PlotlyArtifact(figure=fig, key="plotly")
    logged_plotly = context.log_artifact(plotly)
    context.logger.info(
        "Logged plotly artifact", plotly_artifact=logged_plotly.base_dict()
    )

    # Model logging
    logged_model = context.log_model(
        "model",
        body="{}",
        artifact_path=context.artifact_subpath("models"),
        model_file="model.pkl",
        labels={"type": "test"},
    )
    context.logger.info("Logged model", model_artifact=logged_model.base_dict())


def api_backward_compatibility_tests_failing_function():
    raise RuntimeError("Failing on purpose")
