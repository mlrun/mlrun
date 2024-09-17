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
import os

import mlrun


def setup(
    project: mlrun.projects.MlrunProject,
) -> mlrun.projects.MlrunProject:
    """
    Creating the project for the demo. This function is expected to call automatically when calling the function
    `mlrun.get_or_create_project`.

    :param project: The project to set up.

    :returns: A fully prepared project for this demo.
    """

    # Adding secrets to the projects:
    project.set_secrets(
        {
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
            "OPENAI_API_BASE": os.environ["OPENAI_BASE_URL"],
            "HF_TOKEN": os.environ["HF_TOKEN"],
        }
    )

    # Unpack parameters:
    source = project.get_param(key="source")
    default_image = project.get_param(key="default_image")
    # gpus = project.get_param(key="gpus", default=0)
    # node_name = project.get_param(key="node_name", default=None)

    # Set the project git source:
    if source:
        print(f"Project Source: {source}")
        project.set_source(source=source, pull_at_runtime=True)

    # Set or build the default image:
    if default_image is None:
        print("Building default image for the demo:")
        _build_image(project=project)
    else:
        project.set_default_image(default_image)

    # Set functions
    _set_function(
        project=project,
        func="model_server.py",
        name="llm-server",
        kind="serving",
        image="gcr.io/iguazio/llm-serving:1.7.0",
        gpus=1,
    )
    _set_function(
        project=project,
        func="train.py",
        name="train",
        kind="job",
        image="gcr.io/iguazio/monitoring-demo-adapters:1.7.0",
        gpus=1,
    )
    _set_function(
        project=project,
        func="metric_sample.py",
        name="metric-sample",
        kind="job",
        image="mlrun/mlrun",
    )
    _set_function(
        project=project,
        func="generate_ds.py",
        name="generate-ds",
        kind="job",
        image="gcr.io/iguazio/llm-serving:1.7.0",
    )

    # Save and return the project:
    project.save()
    return project


def _build_image(project: mlrun.projects.MlrunProject):
    assert project.build_image(
        base_image="mlrun/mlrun-gpu",
        commands=[
            # Update apt-get to install ffmpeg (support audio file formats):
            "apt-get update -y",
            # Install demo requirements:
            "pip install torch --index-url https://download.pytorch.org/whl/cu118",
        ],
        set_as_default=True,
    )


def _set_function(
    project: mlrun.projects.MlrunProject,
    func: str,
    name: str,
    kind: str,
    gpus: int = 0,
    node_name: str = None,
    image: str = None,
):
    # Set the given function:
    mlrun_function = project.set_function(
        func=func,
        name=name,
        kind=kind,
        with_repo=False,
        image=image,
    ).apply(mlrun.auto_mount())

    # Configure GPUs according to the given kind:
    if gpus >= 1:
        mlrun_function.with_node_selection(
            node_selector={"app.iguazio.com/node-group": "added-t4x4"}
        )
        # All GPUs for the single job:
        mlrun_function.with_limits(gpus=gpus)

        mlrun_function.spec.min_replicas = 1
        mlrun_function.spec.max_replicas = 1

    # Set the node selection:
    elif node_name:
        mlrun_function.with_node_selection(node_name=node_name)
    # Save:
    mlrun_function.save()
