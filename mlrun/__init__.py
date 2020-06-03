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
"""
MLRun is a generic and convenient mechanism for data scientists and software developers to describe and run tasks related to machine learning (ML) in various, scalable runtime environments and ML pipelines while automatically tracking executed code, metadata, inputs, and outputs.
MLRun integrates with the `Nuclio <https://nuclio.io/>`_ serverless project and with `Kubeflow Pipelines <https://github.com/kubeflow/pipelines>`_.

The MLRun package (``mlrun``) includes a Python API library and the ``mlrun`` command-line interface (CLI).
"""

__version__ = '0.4.8'

from .run import (get_or_create_ctx, new_function, code_to_function,
                  import_function, run_pipeline, run_local, function_to_module,
                  get_object, get_pipeline, wait_for_pipeline_completion)
from .db import get_run_db
from .model import RunTemplate, NewTask, RunObject
from .config import config as mlconf
from .runtimes import new_model_server
from .platforms import mount_v3io, v3io_cred
from .projects import load_project, new_project
from .datastore import DataItem
from .execution import MLClientCtx

from os import environ


def get_version():
    return __version__


if 'IGZ_NAMESPACE_DOMAIN' in environ:
    igz_domain = environ['IGZ_NAMESPACE_DOMAIN']
    kfp_ep = 'https://dashboard.{}/pipelines'.format(igz_domain)
    environ['KF_PIPELINES_UI_ENDPOINT'] = kfp_ep
    mlconf.remote_host = mlconf.remote_host or igz_domain
