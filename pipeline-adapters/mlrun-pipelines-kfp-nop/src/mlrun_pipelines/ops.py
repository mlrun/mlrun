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


def generate_kfp_dag_and_resolve_project(*args, **kwargs):
    raise NotImplementedError


def add_default_function_resources(*args, **kwargs):
    raise NotImplementedError


def add_function_node_selection_attributes(*args, **kwargs):
    raise NotImplementedError


def add_annotations(*args, **kwargs):
    raise NotImplementedError


def add_labels(*args, **kwargs):
    raise NotImplementedError


def add_default_env(task):
    raise NotImplementedError


def sync_environment_variables(function, task):
    raise NotImplementedError


def sync_mounts(function, task):
    raise NotImplementedError


def generate_pipeline_node(*args, **kwargs):
    raise NotImplementedError


def generate_image_builder_pipeline_node(*args, **kwargs):
    raise NotImplementedError


def generate_deployer_pipeline_node(*args, **kwargs):
    raise NotImplementedError
