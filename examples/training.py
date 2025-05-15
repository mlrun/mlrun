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

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mlrun import get_or_create_ctx
from mlrun.artifacts import PlotlyArtifact


def my_job(context, p1=1, p2="x"):
    # load MLRUN runtime context (will be set by the runtime framework e.g. KubeFlow)

    # get parameters from the runtime context (or use defaults)

    # access input metadata, values, files, and secrets (passwords)
    print(f"Run: {context.name} (uid={context.uid})")
    print(f"Params: p1={p1}, p2={p2}")
    access_key = context.get_secret("ACCESS_KEY")
    if access_key:
        print("Access key retrieved successfully.")
    input_file = context.get_input("infile.txt", "infile.txt").get()
    print(f"File\n{input_file}\n")

    # Run some useful code e.g. ML training, data prep, etc.

    # log scalar result values (job result metrics)
    context.log_result("accuracy", p1 * 2)
    context.log_result("loss", p1 * 3)
    context.set_label("framework", "sklearn")

    # log various types of artifacts (file, web page, table), will be versioned and visible in the UI
    context.log_artifact(
        "model",
        body=b"abc is 123",
        local_path="model.txt",
        labels={"framework": "xgboost"},
    )
    context.log_artifact(
        "html_result", body=b"<b> Some HTML <b>", local_path="result.html"
    )

    # create a plotly output (will show in the pipelines UI)
    x = np.arange(10)
    fig = go.Figure(data=go.Scatter(x=x, y=x**2))

    # Create a PlotlyArtifact using the figure and log it
    plotly_artifact = PlotlyArtifact(figure=fig, key="plotly")
    context.log_artifact(plotly_artifact)

    raw_data = {
        "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
        "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
        "age": [42, 52, 36, 24, 73],
        "testScore": [25, 94, 57, 62, 70],
    }
    df = pd.DataFrame(raw_data, columns=["first_name", "last_name", "age", "testScore"])
    context.log_dataset("mydf", df=df, stats=True)


if __name__ == "__main__":
    context = get_or_create_ctx("train")
    p1 = context.get_param("p1", 1)
    p2 = context.get_param("p2", "a-string")
    my_job(context, p1, p2)
