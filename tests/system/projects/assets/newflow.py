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

from mlrun import build_function, deploy_function, get_current_project, run_function
from mlrun.model import HyperParamOptions

funcs = {}
DATASET = "iris_dataset"
LABELS = "label"

in_kfp = True


@dsl.pipeline(name="Demo training pipeline", description="Shows how to use mlrun.")
def newpipe():

    project = get_current_project()

    # build our ingestion function (container image)
    builder = build_function("gen-iris")

    # run the ingestion function with the new image and params
    ingest = run_function(
        "gen-iris",
        name="get-data",
        handler="iris_generator",
        params={"format": "pq"},
        outputs=[DATASET],
    ).after(builder)
    print(ingest.outputs)

    # analyze our dataset
    run_function(
        "describe",
        name="summary",
        params={"label_column": project.get_param("label", "label")},
        inputs={"table": ingest.outputs[DATASET]},
    )

    # train with hyper-paremeters
    train = run_function(
        "auto_trainer",
        name="train",
        params={"label_columns": LABELS, "train_test_split_size": 0.10},
        hyperparams={
            "model_class": [
                "sklearn.ensemble.RandomForestClassifier",
                "sklearn.linear_model.LogisticRegression",
                "sklearn.ensemble.AdaBoostClassifier",
            ]
        },
        hyper_param_options=HyperParamOptions(selector="max.accuracy"),
        inputs={"dataset": ingest.outputs[DATASET]},
        outputs=["model", "test_set"],
    )
    print(train.outputs)

    # test and visualize our model
    run_function(
        "auto_trainer",
        name="test",
        handler="evaluate",
        params={"label_columns": LABELS, "model": train.outputs["model"]},
        inputs={
            "dataset": train.outputs["test_set"],
        },
    )

    # deploy our model as a serverless function, we can pass a list of models to serve
    deploy = deploy_function(
        "serving",
        models=[{"key": f"{DATASET}:v1", "model_path": train.outputs["model"]}],
    )

    # test out new model server (via REST API calls), use imported function
    run_function(
        "hub://v2_model_tester",
        name="model-tester",
        params={"addr": deploy.outputs["endpoint"], "model": f"{DATASET}:v1"},
        inputs={"table": train.outputs["test_set"]},
    )
