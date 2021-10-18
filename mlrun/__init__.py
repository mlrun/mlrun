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

# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx

__all__ = ["get_version", "set_environment", "code_to_function", "import_function"]

import getpass
from os import environ, path

from .config import config as mlconf
from .datastore import DataItem, store_manager
from .db import get_run_db
from .errors import MLRunInvalidArgumentError
from .execution import MLClientCtx
from .model import NewTask, RunObject, RunTemplate, new_task
from .platforms import (
    VolumeMount,
    auto_mount,
    mount_v3io,
    mount_v3io_extended,
    mount_v3io_legacy,
    v3io_cred,
)
from .projects import (
    ProjectMetadata,
    build_function,
    deploy_function,
    get_or_create_project,
    load_project,
    new_project,
    pipeline_context,
    run_function,
)
from .projects.project import _add_username_to_project_name_if_needed
from .run import (
    code_to_function,
    function_to_module,
    get_dataitem,
    get_object,
    get_or_create_ctx,
    get_pipeline,
    import_function,
    new_function,
    run_local,
    run_pipeline,
    wait_for_pipeline_completion,
)
from .runtimes import new_model_server
from .utils.version import Version

__version__ = Version().get()["version"]


def get_version():
    """get current mlrun version"""
    return __version__


if "IGZ_NAMESPACE_DOMAIN" in environ:
    igz_domain = environ["IGZ_NAMESPACE_DOMAIN"]
    kfp_ep = f"https://dashboard.{igz_domain}/pipelines"
    environ["KF_PIPELINES_UI_ENDPOINT"] = kfp_ep
    mlconf.remote_host = mlconf.remote_host or igz_domain


def set_environment(
    api_path: str = None,
    artifact_path: str = "",
    project: str = "",
    access_key: str = None,
    user_project=False,
):
    """set and test default config for: api path, artifact_path and project

    this function will try and read the configuration from the environment/api
    and merge it with the user provided project name, artifacts path or api path/access_key.
    it returns the configured artifacts path, this can be used to define sub paths.

    Note: the artifact path is an mlrun data uri (e.g. `s3://bucket/path`) and can not be used with file utils.

    example::

        from os import path
        artifact_path = set_environment(project='my-project')
        data_subpath = path.join(artifact_path, 'data')

    :param api_path:       location/url of mlrun api service
    :param artifact_path:  path/url for storing experiment artifacts
    :param project:        default project name
    :param access_key:     set the remote cluster access key (V3IO_ACCESS_KEY)
    :param user_project:   add the current user name to the provided project name (making it unique per user)

    :returns:
        default project name
        actual artifact path/url, can be used to create subpaths per task or group of artifacts
    """
    mlconf.dbpath = mlconf.dbpath or api_path
    if not mlconf.dbpath:
        raise ValueError("DB/API path was not detected, please specify its address")

    # check connectivity and load remote defaults
    get_run_db()
    if api_path:
        environ["MLRUN_DBPATH"] = mlconf.dbpath

    if access_key:
        environ["V3IO_ACCESS_KEY"] = access_key

    project = _add_username_to_project_name_if_needed(project, user_project)
    if project:
        ProjectMetadata.validate_project_name(project)

    mlconf.default_project = project or mlconf.default_project
    # We want to ensure the project exists, and verify we're authorized to work on it
    # if it doesn't exist this will create it (and obviously if we created it, we're authorized to work on it)
    # if it does exist - this will get it, which will fail if we're not authorized to work on it
    get_or_create_project(mlconf.default_project, "./")

    if not mlconf.artifact_path and not artifact_path:
        raise ValueError("please specify a valid artifact_path")

    if artifact_path:
        if artifact_path.startswith("./"):
            artifact_path = path.abspath(artifact_path)
        elif not artifact_path.startswith("/") and "://" not in artifact_path:
            raise ValueError(
                "artifact_path must refer to an absolute path" " or a valid url"
            )
        mlconf.artifact_path = artifact_path
    return mlconf.default_project, mlconf.artifact_path


def get_current_project(silent=False):
    if not pipeline_context.project and not silent:
        raise MLRunInvalidArgumentError(
            "current project is not initialized, use new, get or load project methods first"
        )
    return pipeline_context.project


def get_sample_path(subpath=""):
    """
    return the url of a sample dataset or model
    """
    samples_path = environ.get(
        "SAMPLE_DATA_SOURCE_URL_PREFIX", mlconf.default_samples_path
    )
    if subpath:
        samples_path = path.join(samples_path, subpath.lstrip("/"))
    return samples_path
