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

import mlrun

funcs = {}
project = mlrun.projects.pipeline_context.project
iris_data = "https://s3.wasabisys.com/iguazio/data/iris/iris.data.raw.csv"
default_pkg_class = "sklearn.linear_model.LogisticRegression"


@dsl.pipeline(name="Demo training pipeline", description="Shows how to use mlrun.")
def kfpipeline(model_class=default_pkg_class, build=0):

    # if build=True, build the function image before the run
    with dsl.Condition(build == 1) as build_cond:
        funcs["prep-data"].deploy_step()

    # run a local data prep function
    prep_data = (
        funcs["prep-data"]
        .as_step(
            name="prep_data",
            inputs={"source_url": project.get_artifact_uri("data")},
            outputs=["cleaned_data"],
        )
        .after(build_cond)
    )

    # train the model using a library (hub://) function and the generated data
    # no need to define handler in this step because the train function is the default handler
    train = funcs["auto_trainer"].as_step(
        name="train",
        inputs={"dataset": prep_data.outputs["cleaned_data"]},
        params={
            "model_class": model_class,
            "label_columns": project.get_param("label", "label"),
        },
        outputs=["model", "test_set"],
    )

    # test the model using a library (hub://) function and the generated model
    funcs["auto_trainer"].as_step(
        name="test",
        handler="evaluate",
        params={"label_columns": "label", "model": train.outputs["model"]},
        inputs={
            "dataset": train.outputs["test_set"],
        },
    )
