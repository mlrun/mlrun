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


def init_functions(functions: dict, project=None, secrets=None):
    """
    This function will run before running the project.
    It allows us to add our specific system configurations to the functions
    like mounts or secrets if needed.

    In this case we will add Iguazio's user mount to our functions using the
    `mount_v3io()` function to automatically set the mount with the needed
    variables taken from the environment.
    * mount_v3io can be replaced with mlrun.platforms.mount_pvc() for
    non-iguazio mount

    @param functions: <function_name: function_yaml> dict of functions in the
                        workflow
    @param project: project object
    @param secrets: secrets required for the functions for s3 connections and
                    such
    """
    for f in functions.values():
        f.apply(mount_v3io())  # On Iguazio (Auto-mount /User)
        # f.apply(mlrun.platforms.mount_pvc()) # Non-Iguazio mount

    functions["serving"].set_env("MODEL_CLASS", "TFModel")
    functions["serving"].set_env("IMAGE_HEIGHT", "128")
    functions["serving"].set_env("IMAGE_WIDTH", "128")
    functions["serving"].set_env("ENABLE_EXPLAINER", "False")
    functions["serving"].spec.min_replicas = 1


@dsl.pipeline(
    name="Image classification demo",
    description="Train an Image Classification TF Algorithm using MLRun",
)
def kfpipeline(
    image_archive="store:///images",
    images_dir="/User/artifacts/images",
    checkpoints_dir="/User/artifacts/models/checkpoints",
    model_name="cat_vs_dog_tfv1",
):
    # step 1: download images
    open_archive = funcs["utils"].as_step(
        name="download",
        handler="open_archive",
        params={"target_path": images_dir},
        inputs={"archive_url": image_archive},
        outputs=["content"],
    )

    # step 2: label images
    source_dir = str(open_archive.outputs["content"]) + "/cats_n_dogs"
    label = funcs["utils"].as_step(
        name="label",
        handler="categories_map_builder",
        params={"source_dir": source_dir},
        outputs=["categories_map", "file_categories"],
    )

    # step 3: train the model
    train = funcs["trainer"].as_step(
        name="train",
        params={
            "epochs": 4,
            "checkpoints_dir": checkpoints_dir,
            "data_path": source_dir,
            "model_dir": "tfmodels",
            "batch_size": 256,
        },
        inputs={
            "categories_map": label.outputs["categories_map"],
            "file_categories": label.outputs["file_categories"],
        },
        outputs=["model"],
    )

    # deploy the model using nuclio functions
    funcs["serving"].deploy_step(models={model_name: train.outputs["model"]})
