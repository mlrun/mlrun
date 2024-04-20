# Copyright 2024 Iguazio
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


def generate_kfp_dag_and_resolve_project(run, project=None):
    raise NotImplementedError


def add_default_function_resources(
    task: dsl.PipelineTask,
) -> dsl.PipelineTask:
    raise NotImplementedError


def add_function_node_selection_attributes(
    function, task: dsl.PipelineTask
) -> dsl.PipelineTask:
    raise NotImplementedError


def add_annotations(
    task: dsl.PipelineTask,
    kind: str,
    function,
    func_url: str = None,
    project: str = None,
):
    raise NotImplementedError


def add_labels(task, function, scrape_metrics=False):
    raise NotImplementedError


def add_default_env(task):
    raise NotImplementedError


def generate_pipeline_node(
    project_name: str,
    name: str,
    image: str,
    command: list,
    file_outputs: dict,
    function,
    func_url: str,
    scrape_metrics: bool,
    code_env: str,
    registry: str,
):
    raise NotImplementedError


def generate_image_builder_pipeline_node(
    name,
    function=None,
    func_url=None,
    cmd=None,
):
    raise NotImplementedError


def generate_deployer_pipeline_node(
    name,
    function,
    func_url=None,
    cmd=None,
):
    raise NotImplementedError
