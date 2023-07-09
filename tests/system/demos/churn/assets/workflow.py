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

GPUS = False


# init functions is used to configure function resources and local settings
def init_functions(functions: dict, project=None, secrets=None):
    for f in functions.values():
        f.apply(mount_v3io())

    functions["server"].set_env(
        "INFERENCE_STREAM", "users/admin/artifacts/churn/model_stream"
    )


@dsl.pipeline(name="Demo training pipeline", description="Shows how to use mlrun.")
def kfpipeline():

    # encode the data
    clean = funcs["clean-data"].as_step(
        name="clean-data",
        handler="data_clean",
        params={"file_ext": "csv", "models_dest": "models/encoders"},
        inputs={"src": "store:///raw-data"},  # use an artifact from the feature store
        outputs=["cleaned-data", "encoded-data"],
    )

    # analyze our dataset
    funcs["describe"].as_step(
        name="summary",
        params={"label_column": "labels"},
        inputs={"table": clean.outputs["encoded-data"]},
    )

    # train with hyper-paremeters
    xgb = funcs["classify"].as_step(
        name="current-state",
        handler="train_model",
        params={
            "sample": -1,
            "label_column": "labels",
            "model_type": "classifier",
            # xgb class initializers (tuning candidates):
            "CLASS_tree_method": "gpu_hist" if GPUS else "hist",
            "CLASS_objective": "binary:logistic",
            "CLASS_n_estimators": 50,
            "CLASS_max_depth": 5,
            "CLASS_learning_rate": 0.15,
            "CLASS_colsample_bylevel": 0.7,
            "CLASS_colsample_bytree": 0.8,
            "CLASS_gamma": 1.0,
            "CLASS_max_delta_step": 3,
            "CLASS_min_child_weight": 1.0,
            "CLASS_reg_lambda": 10.0,
            "CLASS_scale_pos_weight": 1,
            "FIT_verbose": 0,
            "CLASS_subsample": 0.9,
            "CLASS_booster": "gbtree",
            "CLASS_random_state": 1,
            # encoding:
            "encode_cols": {
                "InternetService": "ISP",
                "Contract": "Contract",
                "PaymentMethod": "Payment",
            },
            # outputs
            "models_dest": "models",
            "plots_dest": "plots",
            "file_ext": "csv",
        },
        inputs={"dataset": clean.outputs["encoded-data"]},
        outputs=["model", "test_set"],
    )

    cox = funcs["survive"].as_step(
        name="survival-curves",
        params={
            "sample": -1,
            "event_column": "labels",
            "strata_cols": [
                "InternetService",
                "StreamingMovies",
                "StreamingTV",
                "PhoneService",
            ],
            "encode_cols": {"Contract": "Contract", "PaymentMethod": "Payment"},
            # outputs
            "models_dest": "models/cox",
            "plots_dest": "plots",
            "file_ext": "csv",
        },
        inputs={"dataset": clean.outputs["encoded-data"]},
        outputs=["cx-model", "tenured-test-set"],
    )

    funcs["xgbtest"].as_step(
        name="test-classifier",
        params={"label_column": "labels", "plots_dest": "churn/test/xgb"},
        inputs={
            "models_path": xgb.outputs["model"],
            "test_set": xgb.outputs["test_set"],
        },
    )

    funcs["coxtest"].as_step(
        name="test-regressor",
        params={"label_column": "labels", "plots_dest": "churn/test/cox"},
        inputs={
            "models_path": cox.outputs["cx-model"],
            "test_set": cox.outputs["tenured-test-set"],
        },
    )

    # deploy our model as a serverless function
    deploy_xgb = funcs["server"].deploy_step(
        models={"churn_server_v1": xgb.outputs["model"]}
    )
    deploy_xgb.after(cox)
