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
from kfp import dsl

from mlrun import mount_v3io

funcs = {}
DATASET = "iris_dataset"
LABELS = "label"


# init functions is used to configure function resources and local settings
def init_functions(functions: dict, project=None, secrets=None):
    for f in functions.values():
        f.apply(mount_v3io())

    # uncomment this line to collect the inference results into a stream
    # and specify a path in V3IO (<datacontainer>/<subpath>)
    # functions['serving'].set_env('INFERENCE_STREAM', 'users/admin/model_stream')


@dsl.pipeline(name="Demo training pipeline", description="Shows how to use mlrun.")
def kfpipeline():
    # build our ingestion function (container image)
    builder = funcs["gen-iris"].deploy_step(skip_deployed=True)

    # run the ingestion function with the new image and params
    ingest = funcs["gen-iris"].as_step(
        name="get-data",
        handler="iris_generator",
        image=builder.outputs["image"],
        params={"format": "pq"},
        outputs=[DATASET],
    )

    # analyze our dataset
    funcs["describe"].as_step(
        name="summary",
        params={"label_column": LABELS},
        inputs={"table": ingest.outputs[DATASET]},
    )

    # train with hyper-paremeters
    train = funcs["train"].as_step(
        name="train-skrf",
        params={"sample": -1, "label_column": LABELS, "test_size": 0.10},
        hyperparams={
            "model_pkg_class": [
                "sklearn.ensemble.RandomForestClassifier",
                "sklearn.linear_model.LogisticRegression",
                "sklearn.ensemble.AdaBoostClassifier",
            ]
        },
        selector="max.accuracy",
        inputs={"dataset": ingest.outputs[DATASET]},
        outputs=["model", "test_set"],
    )

    # test and visualize our model
    funcs["test"].as_step(
        name="test",
        params={"label_column": LABELS},
        inputs={
            "models_path": train.outputs["model"],
            "test_set": train.outputs["test_set"],
        },
    )

    # deploy our model as a serverless function
    deploy = funcs["serving"].deploy_step(
        models={f"{DATASET}_v1": train.outputs["model"]}, tag="v2"
    )

    # test out new model server (via REST API calls)
    funcs["live_tester"].as_step(
        name="model-tester",
        params={"addr": deploy.outputs["endpoint"], "model": f"{DATASET}_v1"},
        inputs={"table": train.outputs["test_set"]},
    )
