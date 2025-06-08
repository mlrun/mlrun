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

import datetime
import getpass
import glob
import http
import importlib.util as imputil
import json
import os
import pathlib
import shutil
import tempfile
import typing
import uuid
import warnings
import zipfile
from copy import deepcopy
from os import environ, makedirs, path
from typing import Callable, Optional, Union, cast

import deprecated
import dotenv
import git
import git.exc
import nuclio.utils
import requests
import yaml

import mlrun.artifacts.model
import mlrun.common.formatters
import mlrun.common.helpers
import mlrun.common.runtimes.constants
import mlrun.common.schemas.alert
import mlrun.common.schemas.artifact
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.datastore.datastore_profile
import mlrun.db
import mlrun.errors
import mlrun.k8s_utils
import mlrun.lists
import mlrun.model_monitoring.applications as mm_app
import mlrun.runtimes
import mlrun.runtimes.mounts
import mlrun.runtimes.nuclio.api_gateway
import mlrun.runtimes.pod
import mlrun.runtimes.utils
import mlrun.serving
import mlrun.utils
import mlrun.utils.regex
import mlrun_pipelines.common.models
from mlrun.alerts.alert import AlertConfig
from mlrun.common.schemas import alert as alert_constants
from mlrun.datastore.datastore_profile import (
    DatastoreProfile,
    DatastoreProfile2Json,
    datastore_profile_read,
)
from mlrun.datastore.vectorstore import VectorStoreCollection
from mlrun.model_monitoring.helpers import (
    filter_results_by_regex,
    get_alert_name_from_result_fqn,
    get_result_instance_fqn,
)
from mlrun.runtimes.nuclio.function import RemoteRuntime
from mlrun_pipelines.models import PipelineNodeWrapper

from ..artifacts import (
    Artifact,
    ArtifactProducer,
    DatasetArtifact,
    DocumentArtifact,
    DocumentLoaderSpec,
    ModelArtifact,
)
from ..artifacts.manager import ArtifactManager, dict_to_artifact, extend_artifact_path
from ..common.runtimes.constants import RunStates
from ..datastore import store_manager
from ..features import Feature
from ..model import EntrypointParam, ImageBuilder, ModelObj
from ..run import code_to_function, get_object, import_function, new_function
from ..secrets import SecretsStore
from ..utils import is_jupyter, is_relative_path, is_yaml_path, logger, update_in
from ..utils.clones import (
    add_credentials_git_remote_url,
    clone_git,
    clone_tgz,
    clone_zip,
    get_repo_url,
)
from ..utils.helpers import (
    ensure_git_branch,
    resolve_git_reference_from_source,
)
from ..utils.notifications import CustomNotificationPusher, NotificationTypes
from .operations import (
    BuildStatus,
    DeployStatus,
    build_function,
    deploy_function,
    run_function,
)
from .pipelines import (
    FunctionsDict,
    WorkflowSpec,
    _PipelineRunStatus,
    _RemoteRunner,
    enrich_function_object,
    get_db_function,
    get_workflow_engine,
    pipeline_context,
)


class ProjectError(Exception):
    pass


def init_repo(context, url, init_git):
    repo = None
    context_path = pathlib.Path(context)
    if not context_path.exists():
        context_path.mkdir(parents=True, exist_ok=True)
    elif not context_path.is_dir():
        raise ValueError(f"Context {context} is not a dir path")
    try:
        repo = git.Repo(context)
        url = get_repo_url(repo)
    except Exception:
        if init_git:
            repo = git.Repo.init(context)
    return repo, url


def new_project(
    name,
    context: str = "./",
    init_git: bool = False,
    user_project: bool = False,
    remote: Optional[str] = None,
    from_template: Optional[str] = None,
    secrets: Optional[dict] = None,
    description: Optional[str] = None,
    subpath: Optional[str] = None,
    save: bool = True,
    overwrite: bool = False,
    parameters: Optional[dict] = None,
    default_function_node_selector: Optional[dict] = None,
) -> "MlrunProject":
    """Create a new MLRun project, optionally load it from a yaml/zip/git template

    A new project is created and returned, you can customize the project by placing a project_setup.py file
    in the project root dir, it will be executed upon project creation or loading.


    example::

        # create a project with local and hub functions, a workflow, and an artifact
        project = mlrun.new_project(
            "myproj", "./", init_git=True, description="my new project"
        )
        project.set_function(
            "prep_data.py", "prep-data", image="mlrun/mlrun", handler="prep_data"
        )
        project.set_function("hub://auto-trainer", "train")
        project.set_artifact("data", Artifact(target_path=data_url))
        project.set_workflow("main", "./myflow.py")
        project.save()

        # run the "main" workflow (watch=True to wait for run completion)
        project.run("main", watch=True)

    example (load from template)::

        # create a new project from a zip template (can also use yaml/git templates)
        # initialize a local git, and register the git remote path
        project = mlrun.new_project(
            "myproj",
            "./",
            init_git=True,
            remote="git://github.com/mlrun/project-demo.git",
            from_template="http://mysite/proj.zip",
        )
        project.run("main", watch=True)


    example using project_setup.py to init the project objects::

            def setup(project):
                project.set_function(
                    "prep_data.py", "prep-data", image="mlrun/mlrun", handler="prep_data"
                )
                project.set_function("hub://auto-trainer", "train")
                project.set_artifact("data", Artifact(target_path=data_url))
                project.set_workflow("main", "./myflow.py")
                return project


    :param name:         project name
    :param context:      project local directory path (default value = "./")
    :param init_git:     if True, will git init the context dir
    :param user_project: add the current username to the provided project name (making it unique per user)
    :param remote:       remote Git url
    :param from_template:     path to project YAML/zip file that will be used as a template
    :param secrets:      key:secret dict or SecretsStore used to download sources
    :param description:  text describing the project
    :param subpath:      project subpath (relative to the context dir)
    :param save:         whether to save the created project in the DB
    :param overwrite:    overwrite project using 'cascade' deletion strategy (deletes project resources)
                         if project with name exists
    :param parameters:   key/value pairs to add to the project.spec.params
    :param default_function_node_selector: defines the default node selector for scheduling functions within the project

    :returns: project object
    """
    context = context or "./"
    name = _add_username_to_project_name_if_needed(name, user_project)

    if from_template:
        if subpath:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Unsupported option, cannot use subpath argument with project templates"
            )
        if from_template.endswith(".yaml"):
            project = _load_project_file(
                from_template, name, secrets, allow_cross_project=True
            )
        elif from_template.startswith("git://"):
            clone_git(from_template, context, secrets, clone=True)
            shutil.rmtree(path.join(context, ".git"))
            project = _load_project_dir(context, name, allow_cross_project=True)
        elif from_template.endswith(".zip"):
            clone_zip(from_template, context, secrets)
            project = _load_project_dir(context, name, allow_cross_project=True)
        else:
            raise ValueError("template must be a path to .yaml or .zip file")
        project.metadata.name = name
        # Remove original owner name for avoiding possible conflicts
        project.spec.owner = None
    else:
        project = MlrunProject.from_dict(
            {
                "metadata": {
                    "name": name,
                }
            }
        )
    project.spec.context = context
    project.spec.subpath = subpath or project.spec.subpath

    repo, url = init_repo(context, remote, init_git or remote)
    project.spec.repo = repo
    if remote and url != remote:
        project.create_remote(remote)
    elif url:
        logger.info("Identified pre-initialized git repo, using it", url=url)
        project.spec._source = url
        project.spec.origin_url = url
    if description:
        project.spec.description = description

    if default_function_node_selector:
        project.spec.default_function_node_selector = default_function_node_selector

    if parameters:
        # Enable setting project parameters at load time, can be used to customize the project_setup
        for key, val in parameters.items():
            project.spec.params[key] = val

    _set_as_current_default_project(project)

    if save and mlrun.mlconf.dbpath:
        if overwrite:
            logger.info(
                "Overwriting project (by deleting and then creating)", name=name
            )
            _delete_project_from_db(
                name, secrets, mlrun.common.schemas.DeletionStrategy.cascade
            )

        try:
            project.save(store=False)
        except mlrun.errors.MLRunConflictError as exc:
            raise mlrun.errors.MLRunConflictError(
                f"Project with name {name} already exists. "
                "Use overwrite=True to overwrite the existing project."
            ) from exc
        logger.info(
            "Created and saved project",
            name=name,
            from_template=from_template,
            overwrite=overwrite,
            context=context,
            save=save,
        )

    # Hook for initializing the project using a project_setup script
    project = project.setup(save and mlrun.mlconf.dbpath)

    return project


def load_project(
    context: str = "./",
    url: Optional[str] = None,
    name: Optional[str] = None,
    secrets: Optional[dict] = None,
    init_git: bool = False,
    subpath: Optional[str] = None,
    clone: bool = False,
    user_project: bool = False,
    save: bool = True,
    sync_functions: bool = False,
    parameters: Optional[dict] = None,
    allow_cross_project: Optional[bool] = None,
) -> "MlrunProject":
    """Load an MLRun project from git or tar or dir

    MLRun looks for a project.yaml file with project definition and objects in the project root path
    and use it to initialize the project, in addition it runs the project_setup.py file (if it exists)
    for further customization.

    Usage example::

        # Load the project and run the 'main' workflow.
        # When using git as the url source the context directory must be an empty or
        # non-existent folder as the git repo will be cloned there
        project = load_project("./demo_proj", "git://github.com/mlrun/project-demo.git")
        project.run("main", arguments={"data": data_url})


    project_setup.py example::

        def setup(project):
            train_function = project.set_function(
                "src/trainer.py",
                name="mpi-training",
                kind="mpijob",
                image="mlrun/mlrun",
            )
            # Set the number of replicas for the training from the project parameter
            train_function.spec.replicas = project.spec.params.get("num_replicas", 1)
            return project


    :param context:         project local directory path (default value = "./")
    :param url:             name (in DB) or git or tar.gz or .zip sources archive path e.g.:
                            git://github.com/mlrun/demo-xgb-project.git
                            http://mysite/archived-project.zip
                            <project-name>
                            The git project should include the project yaml file.
                            If the project yaml file is in a sub-directory, must specify the sub-directory.
    :param name:            project name
    :param secrets:         key:secret dict or SecretsStore used to download sources
    :param init_git:        if True, will git init the context dir
    :param subpath:         project subpath (within the archive)
    :param clone:           if True, always clone (delete any existing content)
    :param user_project:    add the current username to the project name (for db:// prefixes)
    :param save:            whether to save the created project and artifact in the DB
    :param sync_functions:  sync the project's functions into the project object (will be saved to the DB if save=True)
    :param parameters:      key/value pairs to add to the project.spec.params
    :param allow_cross_project: if True, override the loaded project name. This flag ensures awareness of
                                loading an existing project yaml as a baseline for a new project with a different name

    :returns: project object
    """
    if not context:
        raise ValueError("valid context (local dir path) must be provided")

    secrets = secrets or {}
    repo = None
    project = None
    name = _add_username_to_project_name_if_needed(name, user_project)

    from_db = False
    if url:
        url = str(url)  # to support path objects
        if is_yaml_path(url):
            project = _load_project_file(url, name, secrets, allow_cross_project)
            project.spec.context = context
        elif url.startswith("git://"):
            url, repo = clone_git(url, context, secrets, clone)
            # Validate that git source includes branch and refs
            url = ensure_git_branch(url=url, repo=repo)
        elif url.endswith(".tar.gz"):
            clone_tgz(url, context, secrets, clone)
        elif url.endswith(".zip"):
            clone_zip(url, context, secrets, clone)
        elif url.startswith("db://") or "://" not in url:
            project = _load_project_from_db(url, secrets, user_project)
            project.spec.context = context
            if not path.isdir(context):
                makedirs(context)
            project.spec.subpath = subpath or project.spec.subpath
            from_db = True
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Unsupported url scheme, supported schemes are: git://, db:// or "
                ".zip/.tar.gz/.yaml file path (could be local or remote) or project name which will be loaded from DB"
            )

    if not repo:
        repo, url = init_repo(context, url, init_git)

    if not project:
        project = _load_project_dir(context, name, subpath, allow_cross_project)

    if not project.metadata.name:
        raise ValueError("Project name must be specified")

    if parameters:
        # Enable setting project parameters at load time, can be used to customize the project_setup
        for key, val in parameters.items():
            project.spec.params[key] = val

    if not from_db:
        project.spec.source = url or project.spec.source
        project.spec.origin_url = url or project.spec.origin_url
        # Remove original owner name for avoiding possible conflicts when loading project from remote
        project.spec.owner = None

    project.spec.repo = repo
    if repo:
        try:
            # handle cases where active_branch is not set (e.g. in Gitlab CI)
            project.spec.branch = repo.active_branch.name
        except Exception:
            pass

    to_save = bool(save and mlrun.mlconf.dbpath)
    if to_save:
        project.save()

    # Hook for initializing the project using a project_setup script
    project = project.setup(to_save)

    if to_save:
        project.register_artifacts()

    if sync_functions:
        project.sync_functions(save=to_save)

    _set_as_current_default_project(project)

    return project


def get_or_create_project(
    name: str,
    context: str = "./",
    url: Optional[str] = None,
    secrets: Optional[dict] = None,
    init_git=False,
    subpath: Optional[str] = None,
    clone: bool = False,
    user_project: bool = False,
    from_template: Optional[str] = None,
    save: bool = True,
    parameters: Optional[dict] = None,
    allow_cross_project: Optional[bool] = None,
) -> "MlrunProject":
    """Load a project from MLRun DB, or create/import if it does not exist.
    The project will become the default project for the current session.

    MLRun looks for a project.yaml file with project definition and objects in the project root path
    and use it to initialize the project, in addition it runs the project_setup.py file (if it exists)
    for further customization.

    Usage example::

        # load project from the DB (if exist) or the source repo
        project = get_or_create_project(
            "myproj", "./", "git://github.com/mlrun/demo-xgb-project.git"
        )
        project.pull("development")  # pull the latest code from git
        project.run("main", arguments={"data": data_url})  # run the workflow "main"


    project_setup.py example::

        def setup(project):
            train_function = project.set_function(
                "src/trainer.py",
                name="mpi-training",
                kind="mpijob",
                image="mlrun/mlrun",
            )
            # Set the number of replicas for the training from the project parameter
            train_function.spec.replicas = project.spec.params.get("num_replicas", 1)
            return project


    :param name:         project name
    :param context:      project local directory path (default value = "./")
    :param url:          name (in DB) or git or tar.gz or .zip sources archive path e.g.:
                         git://github.com/mlrun/demo-xgb-project.git
                         http://mysite/archived-project.zip
    :param secrets:      key:secret dict or SecretsStore used to download sources
    :param init_git:     if True, will execute `git init` on the context dir
    :param subpath:      project subpath (within the archive/context)
    :param clone:        if True, always clone (delete any existing content)
    :param user_project: add the current username to the project name (for db:// prefixes)
    :param from_template:     path to project YAML file that will be used as from_template (for new projects)
    :param save:         whether to save the created project in the DB
    :param parameters:   key/value pairs to add to the project.spec.params
    :param allow_cross_project: if True, override the loaded project name. This flag ensures awareness of
                                loading an existing project yaml as a baseline for a new project with a different name

    :returns: project object
    """
    context = context or "./"
    try:
        # load project from the DB.
        # use `name` as `url` as we load the project from the DB
        project = load_project(
            context,
            name,
            name,
            secrets=secrets,
            init_git=init_git,
            subpath=subpath,
            clone=clone,
            user_project=user_project,
            # only loading project from db so no need to save it
            save=False,
            parameters=parameters,
            allow_cross_project=allow_cross_project,
        )
        logger.info("Project loaded successfully", project_name=project.name)
        return project
    except mlrun.errors.MLRunNotFoundError:
        logger.debug(
            "Project not found in db", project_name=name, user_project=user_project
        )

    spec_path = path.join(context, subpath or "", "project.yaml")
    load_from_path = url or path.isfile(spec_path)
    # do not nest under "try" or else the exceptions raised below will be logged along with the "not found" message
    if load_from_path:
        # loads a project from archive or local project.yaml
        logger.info(
            "Loading project from path",
            project_name=name,
            user_project=user_project,
            path=url or context,
        )
        project = load_project(
            context,
            url,
            name,
            secrets=secrets,
            init_git=init_git,
            subpath=subpath,
            clone=clone,
            user_project=user_project,
            save=save,
            parameters=parameters,
            allow_cross_project=allow_cross_project,
        )

        logger.info(
            "Project loaded successfully",
            project_name=project.name,
            path=url or context,
            stored_in_db=save,
        )
        return project

    # create a new project
    project = new_project(
        name,
        context,
        init_git=init_git,
        user_project=user_project,
        from_template=from_template,
        secrets=secrets,
        subpath=subpath,
        save=save,
        parameters=parameters,
    )
    logger.info(
        "Project created successfully", project_name=project.name, stored_in_db=save
    )
    return project


def _run_project_setup(
    project: "MlrunProject", setup_file_path: str, save: bool = False
):
    """Run the project setup file if found

    When loading a project MLRun will look for a project_setup.py file, if it is found
    it will execute the setup(project) handler, which can enrich the project with additional
    objects, functions, artifacts, etc.

    Example::

        def setup(project):
            train_function = project.set_function(
                "src/trainer.py",
                name="mpi-training",
                kind="mpijob",
                image="mlrun/mlrun",
            )
            # Set the number of replicas for the training from the project parameter
            train_function.spec.replicas = project.spec.params.get("num_replicas", 1)
            return project

    """
    if not path.exists(setup_file_path):
        return project
    spec = imputil.spec_from_file_location("workflow", setup_file_path)
    if spec is None:
        raise ImportError(f"cannot import project setup file in {setup_file_path}")
    mod = imputil.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if hasattr(mod, "setup"):
        try:
            project = getattr(mod, "setup")(project)
            if not project or not isinstance(project, mlrun.projects.MlrunProject):
                raise ValueError(
                    "MLRun project_setup:setup() must return a project object"
                )
        except Exception as exc:
            logger.error(
                "Failed to run project_setup script",
                setup_file_path=setup_file_path,
                exc=mlrun.errors.err_to_str(exc),
            )
            raise exc
        if save:
            project.save()
    else:
        logger.warn(
            f"skipping setup, setup() handler was not found in {path.basename(setup_file_path)}"
        )
    return project


def _load_project_dir(context, name="", subpath="", allow_cross_project=None):
    subpath_str = subpath or ""

    # support both .yaml and .yml file extensions
    project_file_path = path.join(context, subpath_str, "project.y*ml")
    function_file_path = path.join(context, subpath_str, "function.y*ml")
    setup_file_path = path.join(context, subpath_str, "project_setup.py")

    if project_files := glob.glob(project_file_path):
        # if there are multiple project files, use the first one
        project_file_path = project_files[0]
        with open(project_file_path) as fp:
            data = fp.read()
            struct = yaml.load(data, Loader=yaml.FullLoader)
            project = _project_instance_from_struct(struct, name, allow_cross_project)
            project.spec.context = context
    elif function_files := glob.glob(function_file_path):
        function_path = function_files[0]
        func = import_function(function_path)
        function_file_name = path.basename(path.normpath(function_path))
        project = MlrunProject.from_dict(
            {
                "metadata": {
                    "name": func.metadata.project,
                },
                "spec": {
                    "functions": [
                        {
                            "url": function_file_name,
                            "name": func.metadata.name,
                        },
                    ],
                },
            }
        )
    elif path.exists(setup_file_path):
        # If there is a setup script do not force having project.yaml file
        project = MlrunProject()
    else:
        message = "Project or function YAML not found in path"
        logger.error(
            message,
            context=context,
            name=name,
            subpath=subpath,
        )
        raise mlrun.errors.MLRunNotFoundError(message)

    project.spec.context = context
    project.metadata.name = name or project.metadata.name
    project.spec.subpath = subpath
    return project


def _add_username_to_project_name_if_needed(name, user_project):
    if user_project:
        if not name:
            raise ValueError("user_project must be specified together with name")
        username = environ.get("V3IO_USERNAME") or getpass.getuser()
        normalized_username = mlrun.utils.normalize_project_username(username.lower())
        if username != normalized_username:
            logger.debug(
                "Username was normalized to match the required pattern for project name",
                username=username,
                normalized_username=normalized_username,
            )
        name = f"{name}-{normalized_username}"
    return name


def _load_project_from_db(url, secrets, user_project=False):
    db = mlrun.db.get_run_db(secrets=secrets)
    project_name = _add_username_to_project_name_if_needed(
        url.replace("db://", ""), user_project
    )
    project = db.get_project(project_name)
    if not project:
        raise mlrun.errors.MLRunNotFoundError(f"Project {project_name} not found")

    return project


def _delete_project_from_db(project_name, secrets, deletion_strategy):
    db = mlrun.db.get_run_db(secrets=secrets)
    db.delete_project(project_name, deletion_strategy=deletion_strategy)


def _load_project_file(url, name="", secrets=None, allow_cross_project=None):
    try:
        obj = get_object(url, secrets)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"cant find project file at {url}") from exc
    struct = yaml.load(obj, Loader=yaml.FullLoader)
    return _project_instance_from_struct(struct, name, allow_cross_project)


def _project_instance_from_struct(struct, name, allow_cross_project):
    name_from_struct = struct.get("metadata", {}).get("name", "")
    if name and name_from_struct and name_from_struct != name:
        error_message = (
            f"Project name mismatch, {name_from_struct} != {name}, project is loaded from {name_from_struct} "
            f"project yaml. To prevent/allow this, you can take one of the following actions:\n"
            "1. Set the `allow_cross_project=True` when loading the project.\n"
            f"2. Delete the existing project yaml, or ensure its name is equal to {name}.\n"
            "3. Use different project context dir."
        )

        if allow_cross_project is None:
            # TODO: Remove this warning in version 1.10.0 and also fix cli to support allow_cross_project
            warnings.warn(
                f"Project {name=} is different than specified on the context's project yaml. "
                "This behavior is deprecated and will not be supported from version 1.10.0."
            )
            logger.warn(error_message)
        elif allow_cross_project:
            logger.debug(
                "Project name is different than specified on the context's project yaml. Overriding.",
                existing_name=name_from_struct,
                overriding_name=name,
            )
        else:
            raise ValueError(error_message)
    struct.setdefault("metadata", {})["name"] = name or name_from_struct
    return MlrunProject.from_dict(struct)


class ProjectMetadata(ModelObj):
    def __init__(self, name=None, created=None, labels=None, annotations=None):
        self.name = name
        self.created = created
        self.labels = labels or {}
        self.annotations = annotations or {}

    @property
    def name(self) -> str:
        """Project name"""
        return self._name

    @name.setter
    def name(self, name):
        if name:
            self.validate_project_name(name)
        self._name = name

    @staticmethod
    def validate_project_name(name: str, raise_on_failure: bool = True) -> bool:
        try:
            mlrun.utils.helpers.verify_field_regex(
                "project.metadata.name", name, mlrun.utils.regex.project_name
            )
        except mlrun.errors.MLRunInvalidArgumentError:
            if raise_on_failure:
                raise
            return False
        return True

    @staticmethod
    def validate_project_labels(labels: dict, raise_on_failure: bool = True) -> bool:
        """
        This
        https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#syntax-and-character-set
        """

        # no labels is a valid case
        if not labels:
            return True
        if not isinstance(labels, dict):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Labels must be a dictionary of key-value pairs"
            )
        try:
            for key, value in labels.items():
                mlrun.k8s_utils.verify_label_key(key)
                mlrun.k8s_utils.verify_label_value(value, label_key=key)

        except mlrun.errors.MLRunInvalidArgumentError:
            if raise_on_failure:
                raise
            return False
        return True


class ProjectSpec(ModelObj):
    def __init__(
        self,
        description=None,
        params=None,
        functions=None,
        workflows=None,
        artifacts=None,
        artifact_path=None,
        conda=None,
        source=None,
        subpath=None,
        origin_url=None,
        goals=None,
        load_source_on_run=None,
        default_requirements: Optional[typing.Union[str, list[str]]] = None,
        desired_state=mlrun.common.schemas.ProjectState.online.value,
        owner=None,
        disable_auto_mount=None,
        workdir=None,
        default_image=None,
        build=None,
        custom_packagers: Optional[list[tuple[str, bool]]] = None,
        default_function_node_selector=None,
        notifications=None,
    ):
        self.repo = None

        self.description = description
        self.context = ""
        self._mountdir = None
        self._source = None
        self.source = source or ""
        self.load_source_on_run = load_source_on_run
        self.subpath = subpath
        self.origin_url = origin_url
        self.goals = goals
        self.desired_state = desired_state
        self.owner = owner
        self.branch = None
        self.tag = ""
        self.params = params or {}
        self.conda = conda or ""
        self.artifact_path = artifact_path
        self._artifacts = {}
        self.artifacts = artifacts or []
        self.default_requirements = default_requirements
        self.workdir = workdir

        self._workflows = {}
        self.workflows = workflows or []

        self._function_objects = {}
        self._function_definitions = {}
        self.functions = functions or []
        self.disable_auto_mount = disable_auto_mount
        self.default_image = default_image

        self.build = build

        # A list of custom packagers to include when running the functions of the project. A custom packager is stored
        # in a tuple where the first index is the packager module's path (str) and the second is a flag (bool) for
        # whether it is mandatory for a run (raise exception on collection error) or not.
        self.custom_packagers = custom_packagers or []
        self._default_function_node_selector = default_function_node_selector or None
        self.notifications = notifications or []

    @property
    def source(self) -> str:
        """source url or git repo"""
        if not self._source:
            if self.repo:
                url = get_repo_url(self.repo)
                if url:
                    self._source = url

        return self._source

    @source.setter
    def source(self, src):
        self._source = src

    @property
    def mountdir(self) -> str:
        """specify to mount the context dir inside the function container
        use '.' to use the same path as in the client e.g. Jupyter"""

        if self._mountdir and self._mountdir in [".", "./"]:
            return path.abspath(self.context)
        return self._mountdir

    @mountdir.setter
    def mountdir(self, mountdir):
        self._mountdir = mountdir

    @property
    def functions(self) -> list:
        """list of function object/specs used in this project"""
        functions = []
        for name, function in self._function_definitions.items():
            if hasattr(function, "to_dict"):
                spec = function.to_dict(strip=True)
                if function.spec.build.source and function.spec.build.source.startswith(
                    self._source_repo()
                ):
                    update_in(spec, "spec.build.source", "./")
                functions.append({"name": name, "spec": spec})
            else:
                functions.append(function)
        return functions

    @functions.setter
    def functions(self, functions):
        if not functions:
            functions = []
        if not isinstance(functions, list):
            raise ValueError("functions must be a list")

        function_definitions = {}
        for function in functions:
            if not isinstance(function, dict) and not hasattr(function, "to_dict"):
                raise ValueError("function must be an object or dict")
            if isinstance(function, dict):
                name = function.get("name", "")
                if not name:
                    raise ValueError("function name must be specified in dict")
            else:
                name = function.metadata.name
            function_definitions[name] = function

        self._function_definitions = function_definitions

    def set_function(self, name, function_object, function_dict):
        self._function_definitions[name] = function_dict
        self._function_objects[name] = function_object

    def remove_function(self, name):
        if name in self._function_objects:
            del self._function_objects[name]
        if name in self._function_definitions:
            del self._function_definitions[name]

    @property
    def workflows(self) -> list[dict]:
        """
        :returns: list of workflows specs dicts used in this project
        """
        return [workflow.to_dict() for workflow in self._workflows.values()]

    @workflows.setter
    def workflows(self, workflows: list[typing.Union[dict, WorkflowSpec]]):
        if not workflows:
            workflows = []
        if not isinstance(workflows, list):
            raise ValueError("workflows must be a list")

        workflows_dict = {}
        for workflow in workflows:
            if not isinstance(workflow, dict) and not isinstance(
                workflow, WorkflowSpec
            ):
                raise ValueError(
                    f"workflow must be a dict or `WorkflowSpec` type. Given: {type(workflow)}"
                )
            if isinstance(workflow, dict):
                workflow = WorkflowSpec.from_dict(workflow)
            name = workflow.name
            # todo: support steps dsl as code alternative
            if not name:
                raise ValueError('workflow "name" must be specified')
            if not workflow.path and not workflow.code:
                raise ValueError('workflow source "path" or "code" must be specified')
            workflows_dict[name] = workflow

        self._workflows = workflows_dict

    def set_workflow(self, name, workflow):
        self._workflows[name] = (
            workflow
            if isinstance(workflow, WorkflowSpec)
            else WorkflowSpec.from_dict(workflow)
        )

    def remove_workflow(self, name):
        if name in self._workflows:
            del self._workflows[name]

    @property
    def artifacts(self) -> list:
        """list of artifacts used in this project"""
        return [artifact for artifact in self._artifacts.values()]

    @artifacts.setter
    def artifacts(self, artifacts):
        if not artifacts:
            artifacts = []
        if not isinstance(artifacts, list):
            raise ValueError("artifacts must be a list")

        artifacts_dict = {}
        for artifact in artifacts:
            invalid_object_type = not isinstance(artifact, dict) and not hasattr(
                artifact, "to_dict"
            )
            is_artifact_model = not isinstance(artifact, dict) and hasattr(
                artifact, "to_dict"
            )

            if invalid_object_type:
                raise ValueError("artifacts must be a dict or class")
            elif is_artifact_model:
                key = artifact.key
                artifact = artifact.to_dict()
            else:  # artifact is a dict
                # imported/legacy artifacts don't have metadata,spec,status fields
                key_field = (
                    "key"
                    if _is_imported_artifact(artifact)
                    or mlrun.utils.is_legacy_artifact(artifact)
                    else "metadata.key"
                )
                key = mlrun.utils.get_in(artifact, key_field, "")
                if not key:
                    raise ValueError(f'artifacts "{key_field}" must be specified')

            artifacts_dict[key] = artifact

        self._artifacts = artifacts_dict

    def set_artifact(self, key, artifact):
        if hasattr(artifact, "base_dict"):
            artifact = artifact.base_dict()
        if not _is_imported_artifact(artifact):
            artifact["metadata"]["key"] = key
        self._artifacts[key] = artifact

    def remove_artifact(self, key):
        if key in self._artifacts:
            del self._artifacts[key]

    @property
    def default_function_node_selector(self):
        return self._default_function_node_selector

    @default_function_node_selector.setter
    def default_function_node_selector(self, node_selector: dict[str, str]):
        self._default_function_node_selector = deepcopy(node_selector)

    @property
    def build(self) -> ImageBuilder:
        return self._build

    @build.setter
    def build(self, build):
        self._build = self._verify_dict(build, "build", ImageBuilder)

    def add_custom_packager(self, packager: str, is_mandatory: bool):
        """
        Add a custom packager from the custom packagers list.

        :param packager:     The packager module path to add. For example, if a packager `MyPackager` is in the
                             project's source at my_module.py, then the module path is: "my_module.MyPackager".
        :param is_mandatory: Whether this packager must be collected during a run. If False, failing to collect it won't
                             raise an error during the packagers collection phase.
        """
        # TODO: enable importing packagers from the hub.
        if packager in [
            custom_packager[0] for custom_packager in self.custom_packagers
        ]:
            logger.warn(
                f"The packager's module path '{packager}' is already registered in the project."
            )
            return
        self.custom_packagers.append((packager, is_mandatory))

    def remove_custom_packager(self, packager: str):
        """
        Remove a custom packager from the custom packagers list.

        :param packager: The packager module path to remove.

        :raise MLRunInvalidArgumentError: In case the packager was not in the list.
        """
        # Look for the packager tuple in the list to remove it:
        packager_tuple: tuple[str, bool] = None
        for custom_packager in self.custom_packagers:
            if custom_packager[0] == packager:
                packager_tuple = custom_packager

        # If not found, raise an error, otherwise remove:
        if packager_tuple is None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"The packager module path '{packager}' is not registered in the project, hence it cannot be removed."
            )
        self.custom_packagers.remove(packager_tuple)

    def _source_repo(self):
        src = self.source
        if src:
            return src.split("#")[0]
        return ""

    def _need_repo(self):
        for f in self._function_objects.values():
            if f.spec.build.source in [".", "./"]:
                return True
        return False

    def get_code_path(self):
        """Get the path to the code root/workdir"""
        return path.join(self.context, self.workdir or self.subpath or "")

    def _replace_default_image_in_enriched_functions(self, previous_image, new_image):
        """
        Set a new project-default-image in functions that were already enriched.
        """
        if previous_image == new_image:
            return
        for key in self._function_objects:
            function = self._function_objects[key]
            if function._enriched_image:
                function.spec.image = new_image


class ProjectStatus(ModelObj):
    def __init__(self, state=None):
        self.state = state


class MlrunProject(ModelObj):
    kind = "project"
    _dict_fields = ["kind", "metadata", "spec", "status"]

    def __init__(
        self,
        metadata: Optional[Union[ProjectMetadata, dict]] = None,
        spec: Optional[Union[ProjectSpec, dict]] = None,
    ):
        self.metadata: ProjectMetadata = metadata
        self.spec: ProjectSpec = spec
        self.status = None

        self._initialized = False
        self._secrets = SecretsStore()
        self._artifact_manager = None
        self._notifiers = CustomNotificationPusher(
            [
                NotificationTypes.console,
                NotificationTypes.ipython,
            ]
        )

    @property
    def metadata(self) -> ProjectMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, "metadata", ProjectMetadata)

    @property
    def spec(self) -> ProjectSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", ProjectSpec)

    @property
    def status(self) -> ProjectStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", ProjectStatus)

    @property
    def notifiers(self):
        return self._notifiers

    @property
    def name(self) -> str:
        """Project name, this is a property of the project metadata"""
        return self.metadata.name

    @name.setter
    def name(self, name):
        self.metadata.name = name

    @property
    def artifact_path(self) -> str:
        return self.spec.artifact_path

    @artifact_path.setter
    def artifact_path(self, artifact_path):
        self.spec.artifact_path = artifact_path

    @property
    def source(self) -> str:
        return self.spec.source

    @source.setter
    def source(self, source):
        self.spec.source = source

    def set_source(
        self,
        source: str = "",
        pull_at_runtime: bool = False,
        workdir: Optional[str] = None,
    ):
        """set the project source code path(can be git/tar/zip archive)

        :param source:          valid absolute path or URL to git, zip, or tar file, (or None for current) e.g.
                                git://github.com/mlrun/something.git
                                http://some/url/file.zip
                                note path source must exist on the image or exist locally when run is local
                                (it is recommended to use 'workdir' when source is a filepath instead)
        :param pull_at_runtime: load the archive into the container at job runtime vs on build/deploy
        :param workdir:         workdir path relative to the context dir or absolute
        """
        mlrun.utils.helpers.validate_builder_source(source, pull_at_runtime, workdir)

        self.spec.load_source_on_run = pull_at_runtime

        source_has_changed = source != self.spec.source
        self.spec.source = source or self.spec.source

        # new source should not relay on old workdir
        if source_has_changed:
            self.spec.workdir = workdir

        if self.spec.source.startswith("git://"):
            source, reference, branch = resolve_git_reference_from_source(source)
            if not branch and not reference:
                logger.warn(
                    "Please add git branch or refs to the source e.g.: "
                    "'git://<url>/org/repo.git#<branch-name or refs/heads/..>'"
                )

        try:
            # reset function objects (to recalculate build attributes)
            self.sync_functions()
        except mlrun.errors.MLRunMissingDependencyError as exc:
            logger.error(
                "Failed to resolve all function related dependencies "
                "while working with the new project source. Aborting"
            )
            raise exc

    def get_artifact_uri(
        self,
        key: str,
        category: str = "artifact",
        tag: Optional[str] = None,
        iter: Optional[int] = None,
    ) -> str:
        """return the project artifact uri (store://..) from the artifact key

        example::

            uri = project.get_artifact_uri("my_model", category="model", tag="prod", iter=0)

        :param key:  artifact key/name
        :param category:  artifact category (artifact, model, feature-vector, ..)
        :param tag:  artifact version tag, default to latest version
        :param iter:  iteration number, default to no iteration
        """
        uri = f"store://{category}s/{self.metadata.name}/{key}"
        if iter is not None:
            uri = f"{uri}#{iter}"
        if tag is not None:
            uri = f"{uri}:{tag}"
        return uri

    def get_store_resource(self, uri):
        """get store resource object by uri"""
        return mlrun.datastore.get_store_resource(
            uri, secrets=self._secrets, project=self.metadata.name
        )

    @property
    def context(self) -> str:
        return self.spec.context

    @context.setter
    def context(self, context):
        self.spec.context = context

    @property
    def mountdir(self) -> str:
        return self.spec.mountdir

    @mountdir.setter
    def mountdir(self, mountdir):
        self.spec.mountdir = mountdir

    @property
    def params(self) -> dict:
        return self.spec.params

    @params.setter
    def params(self, params):
        self.spec.params = params

    @property
    def description(self) -> str:
        return self.spec.description

    @description.setter
    def description(self, description):
        self.spec.description = description

    @property
    def default_function_node_selector(self) -> dict:
        return self.spec.default_function_node_selector

    @default_function_node_selector.setter
    def default_function_node_selector(self, default_function_node_selector):
        self.spec.default_function_node_selector = default_function_node_selector

    @property
    def default_image(self) -> str:
        return self.spec.default_image

    def set_default_image(self, default_image: str):
        """
        Set the default image to be used for running runtimes (functions) in this project. This image will be used
        if an image was not provided for a runtime. In case the default image is replaced, functions already
        registered with the project that used the previous default image will have their image replaced on
        next execution.

        :param default_image: Default image to use
        """
        current_default_image = self.spec.default_image
        if current_default_image:
            self.spec._replace_default_image_in_enriched_functions(
                current_default_image, default_image
            )
        self.spec.default_image = default_image

    @property
    def workflows(self) -> list:
        return self.spec.workflows

    @workflows.setter
    def workflows(self, workflows):
        self.spec.workflows = workflows

    def set_workflow(
        self,
        name,
        workflow_path: str,
        embed: bool = False,
        engine: Optional[str] = None,
        args_schema: Optional[list[EntrypointParam]] = None,
        handler: Optional[str] = None,
        schedule: typing.Union[str, mlrun.common.schemas.ScheduleCronTrigger] = None,
        ttl: Optional[int] = None,
        image: Optional[str] = None,
        **args,
    ):
        """Add or update a workflow, specify a name and the code path

        :param name:          Name of the workflow
        :param workflow_path: URL (remote) / Path (absolute or relative to the project code path i.e.
            <project.spec.get_code_path()>/<workflow_path>) for the workflow file.
        :param embed:         Add the workflow code into the project.yaml
        :param engine:        Workflow processing engine ("kfp", "local", "remote" or "remote:local")
        :param args_schema:   List of arg schema definitions (:py:class`~mlrun.model.EntrypointParam`)
        :param handler:       Workflow function handler
        :param schedule:      ScheduleCronTrigger class instance or a standard crontab expression string
                              (which will be converted to the class using its `from_crontab` constructor),
                              see this link for help:
                              https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html#module-apscheduler.triggers.cron
                              Note that "local" engine does not support this argument
        :param ttl:           Pipeline ttl in secs (after that the pods will be removed)
        :param image:         Image for workflow runner job, only for scheduled and remote workflows.
                              The image must have mlrun[kfp] installed which requires python 3.9.
                              Therefore, the project default image will not be used for the workflow,
                              and the image must be specified explicitly.
        :param args:          Argument values (key=value, ..)
        """

        # validate the provided workflow_path
        self._validate_file_path(
            workflow_path, param_name="workflow_path", engine=engine
        )

        if engine and "local" in engine and schedule:
            raise ValueError("'schedule' argument is not supported for 'local' engine.")

        # engine could be "remote" or "remote:local"
        if image and ((engine and "remote" not in engine) and not schedule):
            logger.warning("Image is only relevant for 'remote' engine, ignoring it")

        if embed:
            if (
                self.context
                and not workflow_path.startswith("/")
                # since the user may provide a path the includes the context,
                # we need to make sure we don't add it twice
                and not workflow_path.startswith(self.context)
            ):
                workflow_path = path.join(self.context, workflow_path)
            with open(workflow_path) as fp:
                txt = fp.read()
            workflow = {"name": name, "code": txt}
        else:
            workflow = {"name": name, "path": workflow_path}
        if args:
            workflow["args"] = args
        if handler:
            workflow["handler"] = handler
        if args_schema:
            args_schema = [
                schema.to_dict() if hasattr(schema, "to_dict") else schema
                for schema in args_schema
            ]
            workflow["args_schema"] = args_schema
        workflow["engine"] = engine
        workflow["schedule"] = schedule
        if ttl:
            workflow["ttl"] = ttl
        if image:
            workflow["image"] = image
        self.spec.set_workflow(name, workflow)

    def set_artifact(
        self,
        key,
        artifact: typing.Union[str, dict, Artifact] = None,
        target_path: Optional[str] = None,
        tag: Optional[str] = None,
    ):
        """add/set an artifact in the project spec (will be registered on load)

        example::

            # register a simple file artifact
            project.set_artifact("data", target_path=data_url)
            # register a model artifact
            project.set_artifact(
                "model", ModelArtifact(model_file="model.pkl"), target_path=model_dir_url
            )

            # register a path to artifact package (will be imported on project load)
            # to generate such package use `artifact.export(target_path)`
            project.set_artifact("model", "https://mystuff.com/models/mymodel.zip")

        :param key:  artifact key/name
        :param artifact:  mlrun Artifact object/dict (or its subclasses) or path to artifact
                          file to import (yaml/json/zip), relative paths are relative to the context path
        :param target_path: absolute target path url (point to the artifact content location)
        :param tag:    artifact tag
        """
        if artifact and isinstance(artifact, str):
            artifact_path, _ = self.get_item_absolute_path(
                artifact, check_path_in_context=True
            )
            artifact = {
                "import_from": artifact_path,
                "key": key,
            }
            if tag:
                artifact["tag"] = tag
        else:
            if not artifact:
                artifact = Artifact()
            artifact.spec.target_path = target_path or artifact.spec.target_path
            if artifact.spec.target_path and "://" not in artifact.spec.target_path:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "target_path url must point to a shared/object storage path"
                )
            artifact.metadata.tag = tag or artifact.metadata.tag
        self.spec.set_artifact(key, artifact)

    def register_artifacts(self):
        """register the artifacts in the MLRun DB (under this project)"""
        artifact_manager = self._get_artifact_manager()
        artifact_path = mlrun.utils.helpers.template_artifact_path(
            self.spec.artifact_path or mlrun.mlconf.artifact_path, self.metadata.name
        )
        project_tag = self._get_project_tag()
        for artifact_dict in self.spec.artifacts:
            if _is_imported_artifact(artifact_dict):
                import_from = artifact_dict["import_from"]
                if is_relative_path(import_from):
                    # source path should be relative to the project context
                    import_from = path.join(self.spec.get_code_path(), import_from)

                self.import_artifact(
                    import_from,
                    artifact_dict["key"],
                    tag=artifact_dict.get("tag"),
                )
            else:
                artifact = dict_to_artifact(artifact_dict)
                if is_relative_path(artifact.src_path):
                    # source path should be relative to the project context
                    artifact.src_path = path.join(
                        self.spec.get_code_path(), artifact.src_path
                    )
                producer, is_retained_producer = self._resolve_artifact_producer(
                    artifact, project_tag
                )
                # log the artifact only if it doesn't already exist
                if (
                    producer.name != self.metadata.name
                    and self._resolve_existing_artifact(
                        artifact,
                    )
                ):
                    continue
                artifact_manager.log_artifact(
                    producer,
                    artifact,
                    artifact_path=artifact_path,
                    project=self.metadata.name,
                    is_retained_producer=is_retained_producer,
                )

    def update_artifact(self, artifact_object: Artifact):
        artifacts_manager = self._get_artifact_manager()
        project_tag = self._get_project_tag()
        producer, _ = self._resolve_artifact_producer(artifact_object, project_tag)
        artifacts_manager.update_artifact(producer, artifact_object)

    def _get_artifact_manager(self):
        if self._artifact_manager:
            return self._artifact_manager
        db = mlrun.db.get_run_db(secrets=self._secrets)
        store_manager.set(self._secrets, db)
        self._artifact_manager = ArtifactManager(db)
        return self._artifact_manager

    def _get_hexsha(self):
        try:
            if self.spec.repo:
                return self.spec.repo.head.commit.hexsha
        except Exception:
            pass
        return None

    def get_item_absolute_path(
        self,
        url: str,
        check_path_in_context: bool = False,
    ) -> tuple[str, bool]:
        """
        Get the absolute path of the artifact or function file
        :param url:                   remote url, absolute path or relative path
        :param check_path_in_context: if True, will check if the path exists when in the context
        (temporary parameter to allow for backwards compatibility)

        :returns:   absolute path / url, whether the path is in the project context
        """
        # If the URL is for a remote location, we do not want to change it
        if not url or "://" in url:
            return url, False

        # We don't want to change the url if the project has no context or if it is already absolute
        in_context = self.spec.context and not url.startswith("/")
        if in_context:
            url = path.normpath(path.join(self.spec.get_code_path(), url))

        if (not in_context or check_path_in_context) and not path.isfile(url):
            raise FileNotFoundError(f"{url} not found")

        return url, in_context

    def log_artifact(
        self,
        item,
        body=None,
        tag: str = "",
        local_path: str = "",
        artifact_path: Optional[str] = None,
        format: Optional[str] = None,
        upload: Optional[bool] = None,
        labels: Optional[dict[str, str]] = None,
        target_path: Optional[str] = None,
        **kwargs,
    ) -> Artifact:
        """Log an output artifact and optionally upload it to datastore

        If the artifact already exists with the same key and tag, it will be overwritten.

        example::

            project.log_artifact(
                "some-data",
                body=b"abc is 123",
                local_path="model.txt",
                labels={"framework": "xgboost"},
            )


        :param item:          artifact key or artifact object (can be any type, such as dataset, model, feature store)
        :param body:          will use the body as the artifact content
        :param local_path:    path to the local file we upload, will also be use
                              as the destination subpath (under "artifact_path")
        :param artifact_path: target artifact path (when not using the default)
                              to define a subpath under the default location use:
                              `artifact_path=context.artifact_subpath('data')`
        :param format:        artifact file format: csv, png, ..
        :param tag:           version tag
        :param target_path:   absolute target path (instead of using artifact_path + local_path)
        :param upload:        Whether to upload the artifact to the datastore. If not provided, and the `local_path`
                              is not a directory, upload occurs by default. Directories are uploaded only when this
                              flag is explicitly set to `True`.
        :param labels:        a set of key/value labels to tag the artifact with

        :returns: artifact object
        """
        am = self._get_artifact_manager()
        artifact_path = extend_artifact_path(
            artifact_path, self.spec.artifact_path or mlrun.mlconf.artifact_path
        )
        artifact_path = mlrun.utils.helpers.template_artifact_path(
            artifact_path, self.metadata.name
        )
        producer, is_retained_producer = self._resolve_artifact_producer(item)
        if producer.name != self.metadata.name:
            # the artifact producer is retained, log it only if it doesn't already exist
            if existing_artifact := self._resolve_existing_artifact(
                item,
                tag,
            ):
                artifact_key = item if isinstance(item, str) else item.key
                logger.info(
                    "Artifact already exists, skipping logging",
                    key=artifact_key,
                    tag=tag,
                )
                return existing_artifact
        item = am.log_artifact(
            producer,
            item,
            body,
            tag=tag,
            local_path=local_path,
            artifact_path=artifact_path,
            format=format,
            upload=upload,
            labels=labels,
            target_path=target_path,
            project=self.metadata.name,
            is_retained_producer=is_retained_producer,
            **kwargs,
        )
        return item

    def delete_artifact(
        self,
        item: Artifact,
        deletion_strategy: mlrun.common.schemas.artifact.ArtifactsDeletionStrategies = (
            mlrun.common.schemas.artifact.ArtifactsDeletionStrategies.metadata_only
        ),
        secrets: Optional[dict] = None,
    ):
        """Delete an artifact object in the DB and optionally delete the artifact data

        :param item: Artifact object (can be any type, such as dataset, model, feature store).
        :param deletion_strategy: The artifact deletion strategy types.
        :param secrets: Credentials needed to access the artifact data.
        """
        am = self._get_artifact_manager()
        am.delete_artifact(item, deletion_strategy, secrets)

    def log_dataset(
        self,
        key,
        df,
        tag="",
        local_path=None,
        artifact_path=None,
        upload=None,
        labels=None,
        format="",
        preview=None,
        stats=None,
        target_path="",
        extra_data=None,
        label_column: Optional[str] = None,
        **kwargs,
    ) -> DatasetArtifact:
        """
        Log a dataset artifact and optionally upload it to datastore.

        If the dataset already exists with the same key and tag, it will be overwritten.

        example::

            raw_data = {
                "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
                "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
                "age": [42, 52, 36, 24, 73],
                "testScore": [25, 94, 57, 62, 70],
            }
            df = pd.DataFrame(
                raw_data, columns=["first_name", "last_name", "age", "testScore"]
            )
            project.log_dataset("mydf", df=df, stats=True)

        :param key:           artifact key
        :param df:            dataframe object
        :param label_column:  name of the label column (the one holding the target (y) values)
        :param local_path:    path to the local dataframe file that exists locally.
                              The given file extension will be used to save the dataframe to a file
                              If the file exists, it will be uploaded to the datastore instead of the given df.
        :param artifact_path: target artifact path (when not using the default).
                              to define a subpath under the default location use:
                              `artifact_path=context.artifact_subpath('data')`
        :param tag:           version tag
        :param format:        optional, format to use (`csv`, `parquet`, `pq`, `tsdb`, `kv`)
        :param target_path:   absolute target path (instead of using artifact_path + local_path)
        :param preview:       number of lines to store as preview in the artifact metadata
        :param stats:         calculate and store dataset stats in the artifact metadata
        :param extra_data:    key/value list of extra files/charts to link with this dataset
        :param upload:        upload to datastore (default is True)
        :param labels:        a set of key/value labels to tag the artifact with

        :returns: dataset artifact object
        """
        ds = DatasetArtifact(
            key,
            df,
            preview=preview,
            extra_data=extra_data,
            format=format,
            stats=stats,
            label_column=label_column,
            **kwargs,
        )

        item = cast(
            DatasetArtifact,
            self.log_artifact(
                ds,
                local_path=local_path,
                artifact_path=artifact_path,
                target_path=target_path,
                tag=tag,
                upload=upload,
                labels=labels,
            ),
        )
        return item

    def log_model(
        self,
        key,
        body=None,
        framework="",
        tag="",
        model_dir=None,
        model_file=None,
        algorithm=None,
        metrics=None,
        parameters=None,
        artifact_path=None,
        upload=None,
        labels=None,
        inputs: Optional[list[Feature]] = None,
        outputs: Optional[list[Feature]] = None,
        feature_vector: Optional[str] = None,
        feature_weights: Optional[list] = None,
        training_set=None,
        label_column=None,
        extra_data=None,
        **kwargs,
    ) -> ModelArtifact:
        """Log a model artifact and optionally upload it to datastore

        If the model already exists with the same key and tag, it will be overwritten.

        example::

            project.log_model(
                "model",
                body=dumps(model),
                model_file="model.pkl",
                metrics=context.results,
                training_set=training_df,
                label_column="label",
                feature_vector=feature_vector_uri,
                labels={"app": "fraud"},
            )

        :param key:             artifact key or artifact class ()
        :param body:            will use the body as the artifact content
        :param model_file:      path to the local model file we upload (see also model_dir)
                                or to a model file data url (e.g. `http://host/path/model.pkl`)
        :param model_dir:       path to the local dir holding the model file and extra files
        :param artifact_path:   target artifact path (when not using the default)
                                to define a subpath under the default location use:
                                `artifact_path=context.artifact_subpath('data')`
        :param framework:       name of the ML framework
        :param algorithm:       training algorithm name
        :param tag:             version tag
        :param metrics:         key/value dict of model metrics
        :param parameters:      key/value dict of model parameters
        :param inputs:          ordered list of model input features (name, type, ..)
        :param outputs:         ordered list of model output/result elements (name, type, ..)
        :param upload:          upload to datastore (if not specified, defaults to True (uploads artifact))
        :param labels:          a set of key/value labels to tag the artifact with
        :param feature_vector:  feature store feature vector uri (store://feature-vectors/<project>/<name>[:tag])
        :param feature_weights: list of feature weights, one per input column
        :param training_set:    training set dataframe, used to infer inputs & outputs
        :param label_column:    which columns in the training set are the label (target) columns
        :param extra_data:      key/value list of extra files/charts to link with this dataset
                                value can be absolute path | relative path (to model dir) | bytes | artifact object

        :returns: model artifact object
        """

        if training_set is not None and inputs:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "cannot specify inputs and training set together"
            )

        model = ModelArtifact(
            key,
            body,
            model_file=model_file,
            model_dir=model_dir,
            metrics=metrics,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            framework=framework,
            algorithm=algorithm,
            feature_vector=feature_vector,
            feature_weights=feature_weights,
            extra_data=extra_data,
            **kwargs,
        )
        if training_set is not None:
            model.infer_from_df(training_set, label_column)

        item = cast(
            ModelArtifact,
            self.log_artifact(
                model,
                artifact_path=artifact_path,
                tag=tag,
                upload=upload,
                labels=labels,
            ),
        )
        return item

    def get_vector_store_collection(
        self,
        vector_store: "VectorStore",  # noqa: F821
        collection_name: Optional[str] = None,
    ) -> VectorStoreCollection:
        """
        Create a VectorStoreCollection wrapper for a given vector store instance.

        This method wraps a vector store implementation (like Milvus, Chroma) with MLRun
        integration capabilities. The wrapper provides access to the underlying vector
        store's functionality while adding MLRun-specific features like document and
        artifact management.

        Args:
            vector_store: The vector store instance to wrap (e.g., Milvus, Chroma).
                        This is the underlying implementation that will handle
                        vector storage and retrieval.
            collection_name: Optional name for the collection. If not provided,
                            will attempt to extract it from the vector_store object
                            by looking for 'collection_name', '_collection_name',
                            'index_name', or '_index_name' attributes.

        Returns:
            VectorStoreCollection: A wrapped vector store instance with MLRun integration.
                                This wrapper provides both access to the original vector
                                store's capabilities and additional MLRun functionality.

        Example:
            >>> vector_store = Chroma(embedding_function=embeddings)
            >>> collection = project.get_vector_store_collection(
            ...     vector_store, collection_name="my_collection"
            ... )
        """
        return VectorStoreCollection(
            self,
            vector_store,
            collection_name,
        )

    def log_document(
        self,
        key: str = "",
        tag: str = "",
        local_path: str = "",
        artifact_path: Optional[str] = None,
        document_loader_spec: Optional[DocumentLoaderSpec] = None,
        upload: Optional[bool] = False,
        labels: Optional[dict[str, str]] = None,
        target_path: Optional[str] = None,
        **kwargs,
    ) -> DocumentArtifact:
        """
        Log a document as an artifact.

        :param key: Optional artifact key. If not provided, will be derived from local_path
                or target_path using DocumentArtifact.key_from_source()
        :param tag: Version tag
        :param local_path:    path to the local file we upload, will also be use
                              as the destination subpath (under "artifact_path")
        :param artifact_path: Target path for artifact storage
        :param document_loader_spec: Spec to use to load the artifact as langchain document.

            By default, uses DocumentLoaderSpec() which initializes with:

            * loader_class_name="langchain_community.document_loaders.TextLoader"
            * src_name="file_path"
            * kwargs=None

            Can be customized for different document types, e.g.::

                DocumentLoaderSpec(
                    loader_class_name="langchain_community.document_loaders.PDFLoader",
                    src_name="file_path",
                    kwargs={"extract_images": True}
                )
        :param upload: Whether to upload the artifact
        :param labels:  Key-value labels. A 'source' label is automatically added using either
                        local_path or target_path to facilitate easier document searching.
        :param target_path: Target file path
        :param kwargs: Additional keyword arguments
        :return: DocumentArtifact object

        Example:
            >>> # Log a PDF document with custom loader
            >>> project.log_document(
            ...     local_path="path/to/doc.pdf",
            ...     document_loader=DocumentLoaderSpec(
            ...         loader_class_name="langchain_community.document_loaders.PDFLoader",
            ...         src_name="file_path",
            ...         kwargs={"extract_images": True},
            ...     ),
            ... )

        """
        if not key and not local_path and not target_path:
            raise ValueError(
                "Must provide either 'key' parameter or 'local_path'/'target_path' to derive the key from"
            )
        if not key:
            key = DocumentArtifact.key_from_source(local_path or target_path)

        document_loader_spec = document_loader_spec or DocumentLoaderSpec()
        if not document_loader_spec.download_object and upload:
            raise ValueError(
                "The document loader is configured to not support downloads but the upload flag is set to True."
                "Either set loader.download_object=True or set upload=False"
            )
        original_source = local_path or target_path
        doc_artifact = DocumentArtifact(
            key=key,
            original_source=original_source,
            document_loader_spec=document_loader_spec,
            collections=kwargs.pop("collections", None),
            **kwargs,
        )

        # limit label to a max of 255 characters (for db reasons)
        max_length = 255
        labels = labels or {}
        labels["source"] = (
            original_source[: max_length - 3] + "..."
            if len(original_source) > max_length
            else original_source
        )

        return self.log_artifact(
            item=doc_artifact,
            tag=tag,
            local_path=local_path,
            artifact_path=artifact_path,
            upload=upload,
            labels=labels,
            target_path=target_path,
        )

    def import_artifact(
        self, item_path: str, new_key=None, artifact_path=None, tag=None
    ):
        """Import an artifact object/package from .yaml, .json, or .zip file

        :param item_path:     dataitem url  or file path to the file/package
        :param new_key:       overwrite the artifact key/name
        :param artifact_path: target artifact path (when not using the default)
        :param tag:           artifact tag to set
        :return: artifact object
        """

        def get_artifact(spec):
            artifact = dict_to_artifact(spec)
            artifact.metadata.key = new_key or artifact.metadata.key
            artifact.metadata.project = self.metadata.name
            artifact.metadata.updated = None
            artifact.metadata.tag = tag or artifact.metadata.tag
            if new_key:
                if artifact.spec.db_key:
                    logger.warning(
                        f"Overwriting artifact old db_key '{artifact.spec.db_key}' with new key '{new_key}' - "
                        f"artifact can be fetched using the new key only"
                    )
                artifact.spec.db_key = new_key
            return artifact

        # Obtaining the item's absolute path from the project context, in case the user provided a relative path
        item_path, _ = self.get_item_absolute_path(item_path)
        dataitem = mlrun.get_dataitem(item_path)

        if is_yaml_path(item_path):
            artifact_dict = yaml.load(dataitem.get(), Loader=yaml.FullLoader)
            artifact = get_artifact(artifact_dict)
        elif item_path.endswith(".json"):
            artifact_dict = json.loads(dataitem.get())
            artifact = get_artifact(artifact_dict)
        elif item_path.endswith(".zip"):
            item_file = dataitem.local()
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(item_file, "r") as zf:
                    zf.extractall(temp_dir)
                with open(f"{temp_dir}/_spec.yaml") as fp:
                    data = fp.read()
                spec = yaml.load(data, Loader=yaml.FullLoader)
                artifact = get_artifact(spec)
                with open(f"{temp_dir}/_body", "rb") as fp:
                    artifact.spec._body = fp.read()

                # if the dataitem is not a file, it means we downloaded it from a remote source to a temp file,
                # so we need to remove it after we're done with it
                dataitem.remove_local()

                return self.log_artifact(
                    artifact,
                    local_path=temp_dir,
                    artifact_path=artifact_path,
                    upload=True,
                )

        else:
            raise ValueError("unsupported file suffix, use .yaml, .json, or .zip")

        return self.log_artifact(artifact, artifact_path=artifact_path, upload=False)

    def reload(self, sync=False, context=None) -> "MlrunProject":
        """reload the project and function objects from the project yaml/specs

        :param sync:    set to True to load functions objects
        :param context: context directory (where the yaml and code exist)

        :returns: project object
        """
        context = context or self.spec.context
        if context:
            project = _load_project_dir(
                context,
                self.metadata.name,
                self.spec.subpath,
                allow_cross_project=False,
            )
        else:
            project = _load_project_file(
                self.spec.origin_url,
                self.metadata.name,
                self._secrets,
                allow_cross_project=None,
            )
        project.spec.source = self.spec.source
        project.spec.repo = self.spec.repo
        project.spec.branch = self.spec.branch
        project.spec.origin_url = self.spec.origin_url
        if sync:
            project.sync_functions()
        self.__dict__.update(project.__dict__)
        return project

    def setup(self, save: bool = True) -> "MlrunProject":
        """Run the project setup file if found

        When loading a project MLRun will look for a project_setup.py file, if it is found
        it will execute the setup(project) handler, which can enrich the project with additional
        objects, functions, artifacts, etc.

        :param save: save the project after the setup
        """
        # Hook for initializing the project using a project_setup script
        setup_file_path = path.join(
            self.context, self.spec.subpath or "", "project_setup.py"
        )
        return _run_project_setup(self, setup_file_path, save)

    def create_model_monitoring_alert_configs(
        self,
        name: str,
        summary: str,
        endpoints: mlrun.common.schemas.ModelEndpointList,
        events: Union[list[alert_constants.EventKind], alert_constants.EventKind],
        notifications: list[alert_constants.AlertNotification],
        result_names: Optional[
            list[str]
        ] = None,  # can use wildcards - see below for explanation.
        severity: alert_constants.AlertSeverity = alert_constants.AlertSeverity.MEDIUM,
        criteria: alert_constants.AlertCriteria = alert_constants.AlertCriteria(
            count=1, period="10m"
        ),
        reset_policy: mlrun.common.schemas.alert.ResetPolicy = mlrun.common.schemas.alert.ResetPolicy.AUTO,
    ) -> list[mlrun.alerts.alert.AlertConfig]:
        """
        Generate alert configurations based on specified model endpoints and result names, which can be defined
        explicitly or using regex patterns.

                :param name:                   The name of the AlertConfig template. It will be combined with
                                               mep id, app name and result name to generate a unique name.
                :param summary:                Summary of the alert, will be sent in the generated notifications
                :param endpoints:              The endpoints from which metrics will be retrieved to configure
                                               the alerts.
                                               The ModelEndpointList object is obtained via the `list_model_endpoints`
                                               method or created manually using `ModelEndpoint` objects.
                :param events:                 AlertTrigger event types (EventKind).
                :param notifications:          List of notifications to invoke once the alert is triggered
                :param result_names:           Optional. Filters the result names used to create the alert
                                               configuration, constructed from the app and result_name regex.

                                               For example:
                                               [`app1.result-*`, `*.result1`]
                                               will match "mep_uid1.app1.result.result-1" and
                                               "mep_uid1.app2.result.result1".
                                               A specific result_name (not a wildcard) will always create a new alert
                                               config, regardless of whether the result name exists.
                :param severity:               Severity of the alert.
                :param criteria:               The threshold for triggering the alert based on the
                                               specified number of events within the defined time period.
                :param reset_policy:           When to clear the alert. Either "manual" for manual reset of the alert,
                                               or "auto" if the criteria contains a time period.
                :returns:                      List of AlertConfig according to endpoints results,
                                               filtered by result_names.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        matching_results = []
        specific_result_names = [
            result_name
            for result_name in result_names
            if result_name.count(".") == 3 and "*" not in result_name
        ]
        alerts = []
        endpoint_ids = [endpoint.metadata.uid for endpoint in endpoints.endpoints]
        # using separation to group by endpoint IDs:
        # {"mep_id1": [...], "mep_id2": [...]}
        results_by_endpoint = db.get_metrics_by_multiple_endpoints(
            project=self.name,
            endpoint_ids=endpoint_ids,
            type="results",
            events_format=mm_constants.GetEventsFormat.SEPARATION,
        )
        for endpoint_uid, results in results_by_endpoint.items():
            results_fqn_by_endpoint = [
                get_result_instance_fqn(
                    model_endpoint_id=endpoint_uid,
                    app_name=result.app,
                    result_name=result.name,
                )
                for result in results
            ]
            matching_results += filter_results_by_regex(
                existing_result_names=results_fqn_by_endpoint,
                result_name_filters=result_names,
            )
        for specific_result_name in specific_result_names:
            if specific_result_name not in matching_results:
                logger.warning(
                    f"The specific result name '{specific_result_name}' was"
                    f" not found in the existing endpoint results. Adding alert configuration anyway."
                )
        alert_result_names = list(set(specific_result_names + matching_results))
        for result_fqn in alert_result_names:
            result_fqn_name = get_alert_name_from_result_fqn(result_fqn)
            alerts.append(
                mlrun.alerts.alert.AlertConfig(
                    project=self.name,
                    name=f"{name}--{result_fqn_name}",
                    summary=summary,
                    severity=severity,
                    entities=alert_constants.EventEntities(
                        kind=alert_constants.EventEntityKind.MODEL_ENDPOINT_RESULT,
                        project=self.name,
                        ids=[result_fqn],
                    ),
                    trigger=alert_constants.AlertTrigger(
                        events=events if isinstance(events, list) else [events]
                    ),
                    criteria=criteria,
                    notifications=notifications,
                    reset_policy=reset_policy,
                )
            )
        if not alerts:
            warnings.warn(
                "No alert config has been created. "
                "Try specifying a result name explicitly or verifying that results are available"
            )

        return alerts

    def set_model_monitoring_function(
        self,
        name: str,
        func: typing.Union[str, mlrun.runtimes.RemoteRuntime, None] = None,
        application_class: typing.Union[
            str, mm_app.ModelMonitoringApplicationBase, None
        ] = None,
        image: Optional[str] = None,
        handler: Optional[str] = None,
        with_repo: Optional[bool] = None,
        tag: Optional[str] = None,
        requirements: Optional[typing.Union[str, list[str]]] = None,
        requirements_file: str = "",
        **application_kwargs,
    ) -> mlrun.runtimes.RemoteRuntime:
        """
        Update or add a monitoring function to the project.
        Note: to deploy the function after linking it to the project,
        call `fn.deploy()` where `fn` is the object returned by this method.

        Example::

            project.set_model_monitoring_function(
                name="myApp", application_class="MyApp", image="mlrun/mlrun"
            )

        :param func:                    Remote function object or spec/code URL. :code:`None` refers to the current
                                        notebook.
        :param name:                    Name of the function (under the project), can be specified with a tag to support
                                        versions (e.g. myfunc:v1).
        :param image:                   Docker image to be used, can also be specified in
                                        the function object/yaml
        :param handler:                 Default function handler to invoke (can only be set with .py/.ipynb files)
        :param with_repo:               Add (clone) the current repo to the build source
        :param tag:                     Function version tag (none for 'latest', can only be set with .py/.ipynb files)
                                        if tag is specified and name is empty, the function key (under the project)
                                        will be enriched with the tag value. (i.e. 'function-name:tag')
        :param requirements:            A list of python packages
        :param requirements_file:       Path to a python requirements file
        :param application_class:       Name or an Instance of a class that implements the monitoring application.
        :param application_kwargs:      Additional keyword arguments to be passed to the
                                        monitoring application's constructor.
        :returns:                       The model monitoring remote function object.
        """
        (
            resolved_function_name,
            function_object,
            func,
        ) = self._instantiate_model_monitoring_function(
            func,
            application_class,
            name,
            image,
            handler,
            with_repo,
            tag,
            requirements,
            requirements_file,
            **application_kwargs,
        )
        # save to project spec
        self.spec.set_function(resolved_function_name, function_object, func)

        return function_object

    def create_model_monitoring_function(
        self,
        name: str,
        func: Optional[str] = None,
        application_class: typing.Union[
            str,
            mm_app.ModelMonitoringApplicationBase,
            None,
        ] = None,
        image: Optional[str] = None,
        handler: Optional[str] = None,
        with_repo: Optional[bool] = None,
        tag: Optional[str] = None,
        requirements: Optional[typing.Union[str, list[str]]] = None,
        requirements_file: str = "",
        **application_kwargs,
    ) -> mlrun.runtimes.RemoteRuntime:
        """
        Create a monitoring function object without setting it to the project

        Example::

            project.create_model_monitoring_function(
                name="myApp", application_class="MyApp", image="mlrun/mlrun"
            )

        :param func:                    The function's code URL. :code:`None` refers to the current notebook.
        :param name:                    Name of the function, can be specified with a tag to support
                                        versions (e.g. myfunc:v1).
        :param image:                   Docker image to be used, can also be specified in
                                        the function object/yaml
        :param handler:                 Default function handler to invoke (can only be set with .py/.ipynb files)
        :param with_repo:               Add (clone) the current repo to the build source
        :param tag:                     Function version tag (none for 'latest', can only be set with .py/.ipynb files)
                                        if tag is specified and name is empty, the function key (under the project)
                                        will be enriched with the tag value. (i.e. 'function-name:tag')
        :param requirements:            A list of python packages
        :param requirements_file:       Path to a python requirements file
        :param application_class:       Name or an Instance of a class that implementing the monitoring application.
        :param application_kwargs:      Additional keyword arguments to be passed to the
                                        monitoring application's constructor.
        :returns:                       The model monitoring remote function object.
        """

        _, function_object, _ = self._instantiate_model_monitoring_function(
            func,
            application_class,
            name,
            image,
            handler,
            with_repo,
            tag,
            requirements,
            requirements_file,
            **application_kwargs,
        )
        return function_object

    def _instantiate_model_monitoring_function(
        self,
        func: typing.Union[str, mlrun.runtimes.BaseRuntime, None] = None,
        application_class: typing.Union[
            str,
            mm_app.ModelMonitoringApplicationBase,
            None,
        ] = None,
        name: typing.Optional[str] = None,
        image: typing.Optional[str] = None,
        handler: typing.Optional[str] = None,
        with_repo: typing.Optional[bool] = None,
        tag: typing.Optional[str] = None,
        requirements: typing.Union[str, list[str], None] = None,
        requirements_file: str = "",
        **application_kwargs,
    ) -> tuple[str, mlrun.runtimes.RemoteRuntime, dict]:
        import mlrun.model_monitoring.api

        kind = None
        if (isinstance(func, str) or func is None) and application_class is not None:
            kind = mlrun.run.RuntimeKinds.serving
            func = mlrun.model_monitoring.api._create_model_monitoring_function_base(
                project=self.name,
                func=func,
                application_class=application_class,
                name=name,
                image=image,
                tag=tag,
                requirements=requirements,
                requirements_file=requirements_file,
                **application_kwargs,
            )
        elif isinstance(func, str) and isinstance(handler, str):
            kind = mlrun.run.RuntimeKinds.nuclio

        (
            resolved_function_name,
            tag,
            function_object,
            func,
        ) = self._instantiate_function(
            func,
            name,
            kind,
            image,
            handler,
            with_repo,
            tag,
            requirements,
            requirements_file,
        )
        function_object.set_label(
            mm_constants.ModelMonitoringAppLabel.KEY,
            mm_constants.ModelMonitoringAppLabel.VAL,
        )

        return resolved_function_name, function_object, func

    def _wait_for_functions_deployment(self, function_names: list[str]) -> None:
        """
        Wait for the deployment of functions on the backend.

        :param function_names: A list of function names.
        """
        for fn_name in function_names:
            fn = typing.cast(RemoteRuntime, self.get_function(key=fn_name))
            fn._wait_for_function_deployment(db=fn._get_db())

    def enable_model_monitoring(
        self,
        default_controller_image: str = "mlrun/mlrun",
        base_period: int = 10,
        image: str = "mlrun/mlrun",
        *,
        deploy_histogram_data_drift_app: bool = True,
        wait_for_deployment: bool = False,
        fetch_credentials_from_sys_config: bool = False,
    ) -> None:
        """
        Deploy model monitoring application controller, writer and stream functions.
        While the main goal of the controller function is to handle the monitoring processing and triggering
        applications, the goal of the model monitoring writer function is to write all the monitoring
        application results to the databases.
        The stream function goal is to monitor the log of the data stream. It is triggered when a new log entry
        is detected. It processes the new events into statistics that are then written to statistics databases.

        :param default_controller_image:          Deprecated.
        :param base_period:                       The time period in minutes in which the model monitoring controller
                                                  function is triggered. By default, the base period is 10 minutes
                                                  (which is also the minimum value for production environments).
        :param image:                             The image of the model monitoring controller, writer, monitoring
                                                  stream & histogram data drift functions, which are real time nuclio
                                                  functions. By default, the image is mlrun/mlrun.
        :param deploy_histogram_data_drift_app:   If true, deploy the default histogram-based data drift application:
            :py:class:`~mlrun.model_monitoring.applications.histogram_data_drift.HistogramDataDriftApplication`.
            If false, and you want to deploy the histogram data drift application
            afterwards, you may use the
            :py:func:`~set_model_monitoring_function` method::

                import mlrun.model_monitoring.applications.histogram_data_drift as histogram_data_drift

                hist_app = project.set_model_monitoring_function(
                    name=histogram_data_drift.HistogramDataDriftApplicationConstants.NAME,  # keep the default name
                    func=histogram_data_drift.__file__,
                    application_class=histogram_data_drift.HistogramDataDriftApplication.__name__,
                )

                project.deploy_function(hist_app)

        :param wait_for_deployment:               If true, return only after the deployment is done on the backend.
                                                  Otherwise, deploy the model monitoring infrastructure on the
                                                  background, including the histogram data drift app if selected.
        :param fetch_credentials_from_sys_config: If true, fetch the credentials from the system configuration.
        """
        if default_controller_image != "mlrun/mlrun":
            # TODO: Remove this in 1.10.0
            warnings.warn(
                "'default_controller_image' is deprecated in 1.7.0 and will be removed in 1.10.0, "
                "use 'image' instead",
                FutureWarning,
            )
            image = default_controller_image
        if base_period < 10:
            logger.warn(
                "enable_model_monitoring: 'base_period' < 10 minutes is not supported in production environments",
                project=self.name,
            )

        db = mlrun.db.get_run_db(secrets=self._secrets)
        db.enable_model_monitoring(
            project=self.name,
            image=image,
            base_period=base_period,
            deploy_histogram_data_drift_app=deploy_histogram_data_drift_app,
            fetch_credentials_from_sys_config=fetch_credentials_from_sys_config,
        )

        if wait_for_deployment:
            deployment_functions = mm_constants.MonitoringFunctionNames.list()
            if deploy_histogram_data_drift_app:
                deployment_functions.append(
                    mm_constants.HistogramDataDriftApplicationConstants.NAME
                )
            self._wait_for_functions_deployment(deployment_functions)

    def update_model_monitoring_controller(
        self,
        base_period: int = 10,
        image: str = "mlrun/mlrun",
        *,
        wait_for_deployment: bool = False,
    ) -> None:
        """
        Redeploy model monitoring application controller functions.

        :param base_period:         The time period in minutes in which the model monitoring controller function
                                    is triggered. By default, the base period is 10 minutes.
        :param image:               The image of the model monitoring controller, writer & monitoring
                                    stream functions, which are real time nuclio functions.
                                    By default, the image is mlrun/mlrun.
        :param wait_for_deployment: If true, return only after the deployment is done on the backend.
                                    Otherwise, deploy the controller on the background.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        db.update_model_monitoring_controller(
            project=self.name,
            base_period=base_period,
            image=image,
        )

        if wait_for_deployment:
            self._wait_for_functions_deployment(
                [mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER]
            )

    def disable_model_monitoring(
        self,
        *,
        delete_resources: bool = True,
        delete_stream_function: bool = False,
        delete_histogram_data_drift_app: bool = True,
        delete_user_applications: bool = False,
        user_application_list: Optional[list[str]] = None,
    ) -> None:
        """
        Disable model monitoring application controller, writer, stream, histogram data drift application
        and the user's applications functions, according to the given params.

        :param delete_resources:                    If True, it would delete the model monitoring controller & writer
                                                    functions. Default True
        :param delete_stream_function:              If True, it would delete model monitoring stream function,
                                                    need to use wisely because if you're deleting this function
                                                    this can cause data loss in case you will want to
                                                    enable the model monitoring capability to the project.
                                                    Default False.
        :param delete_histogram_data_drift_app:     If True, it would delete the default histogram-based data drift
                                                    application. Default False.
        :param delete_user_applications:            If True, it would delete the user's model monitoring
                                                    application according to user_application_list, Default False.
        :param user_application_list:               List of the user's model monitoring application to disable.
                                                    Default all the applications.
                                                    Note: you have to set delete_user_applications to True
                                                    in order to delete the desired application.
        """
        if not delete_user_applications and user_application_list:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "user_application_list can be specified only if delete_user_applications is set to True"
            )

        db = mlrun.db.get_run_db(secrets=self._secrets)
        succeed = db.disable_model_monitoring(
            project=self.name,
            delete_resources=delete_resources,
            delete_stream_function=delete_stream_function,
            delete_histogram_data_drift_app=delete_histogram_data_drift_app,
            delete_user_applications=delete_user_applications,
            user_application_list=user_application_list,
        )
        if succeed and delete_resources:
            if delete_resources:
                logger.info("Model Monitoring disabled", project=self.name)
            if delete_user_applications:
                logger.info(
                    "All the desired monitoring application were deleted",
                    project=self.name,
                )
        else:
            if delete_resources:
                logger.info(
                    "Model Monitoring was not disabled properly", project=self.name
                )
            if delete_user_applications:
                logger.info(
                    "Some of the desired monitoring application were not deleted",
                    project=self.name,
                )

    def set_function(
        self,
        func: typing.Union[str, mlrun.runtimes.BaseRuntime, None] = None,
        name: str = "",
        kind: str = "job",
        image: Optional[str] = None,
        handler: Optional[str] = None,
        with_repo: Optional[bool] = None,
        tag: Optional[str] = None,
        requirements: Optional[typing.Union[str, list[str]]] = None,
        requirements_file: str = "",
    ) -> mlrun.runtimes.BaseRuntime:
        """
        | Update or add a function object to the project.
        | Function can be provided as an object (func) or a .py/.ipynb/.yaml URL.

        | Creating a function from a single file is done by specifying ``func`` and disabling ``with_repo``.
        | Creating a function with project source (specify ``with_repo=True``):
        |   1. Specify a relative ``func`` path.
        |   2. Specify a module ``handler`` (e.g. ``handler=package.package.func``) without ``func``.
        | Creating a function with non project source is done by specifying a module ``handler`` and on the
         returned function set the source with ``function.with_source_archive(<source>)``.

        Support URL prefixes:

            | Object (s3://, v3io://, ..)
            | MLRun DB e.g. db://project/func:ver
            | Functions hub/market: e.g. hub://auto-trainer:master

        Examples::

            proj.set_function(func_object)
            proj.set_function("http://.../mynb.ipynb", "train")
            proj.set_function("./func.yaml")
            proj.set_function("hub://get_toy_data", "getdata")

            # Create a function from a single file
            proj.set_function("./src/mycode.py", "ingest")

            # Creating a function with project source
            proj.set_function(
                "./src/mycode.py", "ingest", image="myrepo/ing:latest", with_repo=True
            )
            proj.set_function("ingest", handler="package.package.func", with_repo=True)

            # Creating a function with non project source
            func = proj.set_function(
                "ingest", handler="package.package.func", with_repo=False
            )
            func.with_source_archive("git://github.com/mlrun/something.git")

            # Set function requirements

            # By providing a list of packages
            proj.set_function("my.py", requirements=["requests", "pandas"])

            # By providing a path to a pip requirements file
            proj.set_function("my.py", requirements="requirements.txt")

        :param func:                Function object or spec/code url, None refers to current Notebook
        :param name:                Name of the function (under the project), can be specified with a tag to support
                                    Versions (e.g. myfunc:v1). If the `tag` parameter is provided, the tag in the name
                                    must match the tag parameter.
                                    Specifying a tag in the name will update the project's tagged function (myfunc:v1)
        :param kind:                Runtime kind e.g. job, nuclio, spark, dask, mpijob
                                    Default: job
        :param image:               Docker image to be used, can also be specified in the function object/yaml
        :param handler:             Default function handler to invoke (can only be set with .py/.ipynb files)
        :param with_repo:           Add (clone) the current repo to the build source - use when the function code is in
                                    the project repo (project.spec.source).
        :param tag:                 Function version tag to set (none for current or 'latest')
                                    Specifying a tag as a parameter will update the project's tagged function
                                    (myfunc:v1) and the untagged function (myfunc)
        :param requirements:        A list of python packages
        :param requirements_file:   Path to a python requirements file

        :returns: :py:class:`~mlrun.runtimes.BaseRuntime`
        """
        (
            resolved_function_name,
            tag,
            function_object,
            func,
        ) = self._instantiate_function(
            func,
            name,
            kind,
            image,
            handler,
            with_repo,
            tag,
            requirements,
            requirements_file,
        )

        self._set_function(resolved_function_name, tag, function_object, func)
        return function_object

    def push_run_notifications(
        self,
        uid,
        timeout=45,
    ):
        """
        Push notifications for a run.

        :param uid: Unique ID of the run.
        :returns: :py:class:`~mlrun.common.schemas.BackgroundTask`.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        return db.push_run_notifications(
            project=self.name,
            uid=uid,
            timeout=timeout,
        )

    def push_pipeline_notification_kfp_runner(
        self,
        pipeline_id: str,
        current_run_state: mlrun_pipelines.common.models.RunStatuses,
        message: str,
        notifications: Optional[list] = None,
    ):
        """
        Push notifications for a pipeline run(KFP).

        :param pipeline_id:         Unique ID of the pipeline run.
        :param current_run_state:   Current run state of the pipeline.
        :param message:             Message to send in the notification.
        :param notifications:       List of notifications to send.
        """
        current_run_state = RunStates.pipeline_run_status_to_run_state(
            current_run_state
        )
        db = mlrun.get_run_db()
        notifications = notifications or self.spec.notifications
        notifications_to_send = []
        for notification in notifications:
            if current_run_state in notification.when:
                notification_copy = notification.copy()
                notification_copy.message = message
                notifications_to_send.append(notification_copy)
        db.push_pipeline_notifications(
            pipeline_id, self.metadata.name, notifications_to_send
        )

    def _instantiate_function(
        self,
        func: typing.Union[str, mlrun.runtimes.BaseRuntime] = None,
        name: str = "",
        kind: str = "",
        image: Optional[str] = None,
        handler: Optional[str] = None,
        with_repo: Optional[bool] = None,
        tag: Optional[str] = None,
        requirements: Optional[typing.Union[str, list[str]]] = None,
        requirements_file: str = "",
    ) -> tuple[str, str, mlrun.runtimes.BaseRuntime, dict]:
        if (
            func is None
            and not _has_module(handler, kind)
            and mlrun.runtimes.RuntimeKinds.supports_from_notebook(kind)
        ):
            # if function path is not provided and it is not a module (no ".")
            # use the current notebook as default
            if is_jupyter:
                from IPython import get_ipython

                kernel = get_ipython()
                func = nuclio.utils.notebook_file_name(kernel)
                if func.startswith(path.abspath(self.spec.context)):
                    func = path.relpath(func, self.spec.context)

        func = func or ""

        name = mlrun.utils.normalize_name(name) if name else name
        untagged_name = name
        # validate tag in name if specified
        if len(split_name := name.split(":")) == 2:
            untagged_name, name_tag = split_name
            if tag and name_tag and tag != name_tag:
                raise ValueError(
                    f"Tag parameter ({tag}) and tag in function name ({name}) must match"
                )

            tag = tag or name_tag
        elif len(split_name) > 2:
            raise ValueError(
                f"Function name ({name}) must be in the format <name>:<tag> or <name>"
            )

        if isinstance(func, str):
            # in hub or db functions name defaults to the function name
            if not name and not (func.startswith("db://") or func.startswith("hub://")):
                raise ValueError("Function name must be specified")
            function_dict = {
                "url": func,
                "name": untagged_name,
                "kind": kind,
                "image": image,
                "handler": handler,
                "with_repo": with_repo,
                "tag": tag,
                "requirements": requirements,
                "requirements_file": requirements_file,
            }
            func = {k: v for k, v in function_dict.items() if v}
            resolved_function_name, function_object = _init_function_from_dict(
                func, self
            )
            func["name"] = resolved_function_name

        elif hasattr(func, "to_dict"):
            resolved_function_name, function_object = _init_function_from_obj(
                func, self, name=untagged_name
            )
            if handler:
                raise ValueError(
                    "Default handler cannot be set for existing function object"
                )
            if image:
                function_object.spec.image = image
            if with_repo:
                # mark source to be enriched before run with project source (enrich_function_object)
                function_object.spec.build.source = "./"
            if requirements or requirements_file:
                function_object.with_requirements(
                    requirements, requirements_file=requirements_file, overwrite=True
                )
            if not resolved_function_name:
                raise ValueError("Function name must be specified")
        else:
            raise ValueError("'func' parameter must be a function url or object")

        function_object.metadata.tag = tag or function_object.metadata.tag or "latest"
        # resolved_function_name is the name without the tag or the actual function name if it was not specified
        name = name or resolved_function_name

        return (
            name,
            tag,
            function_object,
            func,
        )

    def _set_function(
        self,
        name: str,
        tag: str,
        function_object: mlrun.runtimes.BaseRuntime,
        func: dict,
    ):
        # if the name contains the tag we only update the tagged entry
        # if the name doesn't contain the tag (or was not specified) we update both the tagged and untagged entries
        # for consistency
        if tag and not name.endswith(f":{tag}"):
            self.spec.set_function(f"{name}:{tag}", function_object, func)

        self.spec.set_function(name, function_object, func)

    # TODO: Remove this in 1.11.0
    @deprecated.deprecated(
        version="1.8.0",
        reason="'remove_function' is deprecated and will be removed in 1.11.0. "
        "Please use `delete_function` instead.",
        category=FutureWarning,
    )
    def remove_function(self, name):
        """remove the specified function from the project

        :param name:    name of the function (under the project)
        """
        self.spec.remove_function(name)

    def delete_function(self, name, delete_from_db=False):
        """deletes the specified function from the project

        :param name: name of the function (under the project)
        :param delete_from_db: default is False. If False, the function is removed
                               only from the project's cache and spec.
                               If True, the function is also removed from the database.
        """
        if delete_from_db:
            mlrun.db.get_run_db().delete_function(name=name, project=self.metadata.name)
        self.spec.remove_function(name)

    def remove_model_monitoring_function(self, name: Union[str, list[str]]):
        """delete the specified model-monitoring-app function/s

        :param name: name of the model-monitoring-function/s (under the project)
        """
        # TODO: Remove this in 1.10.0
        warnings.warn(
            "'remove_model_monitoring_function' is deprecated in 1.7.0 and will be removed in 1.10.0. "
            "Please use `delete_model_monitoring_function` instead.",
            FutureWarning,
        )
        self.delete_model_monitoring_function(name)

    def delete_model_monitoring_function(self, name: Union[str, list[str]]):
        """delete the specified model-monitoring-app function/s

        :param name: name of the model-monitoring-function/s (under the project)
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        succeed = db.delete_model_monitoring_function(
            project=self.name,
            functions=name if isinstance(name, list) else [name],
        )
        if succeed:
            logger.info(
                "All the desired monitoring functions were deleted",
                project=self.name,
                functions=name,
            )
        else:
            logger.info(
                "Some of the desired monitoring functions were not deleted",
                project=self.name,
            )

    def get_function(
        self,
        key,
        sync=False,
        enrich=False,
        ignore_cache=False,
        copy_function=True,
        tag: str = "",
    ) -> mlrun.runtimes.BaseRuntime:
        """get function object by name

        :param key:             name of key for search
        :param sync:            will reload/reinit the function from the project spec
        :param enrich:          add project info/config/source info to the function object
        :param ignore_cache:    read the function object from the DB (ignore the local cache)
        :param copy_function:   return a copy of the function object
        :param tag:             provide if the function key is tagged under the project (function was set with a tag)

        :returns: function object
        """
        if tag and ":" not in key:
            key = f"{key}:{tag}"

        function, err = self._get_function(
            mlrun.utils.normalize_name(key), sync, ignore_cache
        )
        if not function and "_" in key:
            function, err = self._get_function(key, sync, ignore_cache)

        if not function:
            raise err

        if enrich:
            function = enrich_function_object(
                self, function, copy_function=copy_function
            )
            self.spec._function_objects[key] = function

        return function

    def _get_function(self, key, sync, ignore_cache):
        """
        Function can be retrieved from the project spec (cache) or from the database.
        In sync mode, we first perform a sync of the function_objects from the function_definitions,
        and then returning it from the function_objects (if exists).
        When not in sync mode, we verify and return from the function objects directly.
        In ignore_cache mode, we query the function from the database rather than from the project spec.
        """
        if key in self.spec._function_objects and not sync and not ignore_cache:
            function = self.spec._function_objects[key]

        elif key in self.spec._function_definitions and not ignore_cache:
            self.sync_functions([key])
            function = self.spec._function_objects[key]
        else:
            try:
                function = get_db_function(self, key)
                self.spec._function_objects[key] = function
            except requests.HTTPError as exc:
                if exc.response.status_code != http.HTTPStatus.NOT_FOUND.value:
                    raise exc
                return None, exc

        return function, None

    def get_function_objects(self) -> FunctionsDict:
        """ "get a virtual dict with all the project functions ready for use in a pipeline"""
        self.sync_functions()
        return FunctionsDict(self)

    def get_function_names(self) -> list[str]:
        """get a list of all the project function names"""
        return [func["name"] for func in self.spec.functions]

    def pull(
        self,
        branch: Optional[str] = None,
        remote: Optional[str] = None,
        secrets: Union[SecretsStore, dict] = None,
    ):
        """pull/update sources from git or tar into the context dir

        :param branch:  git branch, if not the current one
        :param remote:  git remote, if other than origin
        :param secrets: dict or SecretsStore with Git credentials e.g. secrets={"GIT_TOKEN": token}
        """
        url = self.spec.origin_url
        if url and url.startswith("git://"):
            if not self.spec.repo:
                raise ValueError("repo was not initialized, use load_project()")
            remote = remote or "origin"
            self._run_authenticated_git_action(
                action=self.spec.repo.git.pull,
                remote=remote,
                args=[remote, branch or self.spec.repo.active_branch.name],
                secrets=secrets or {},
            )
        elif url and url.endswith(".tar.gz"):
            clone_tgz(url, self.spec.context, self._secrets)
        elif url and url.endswith(".zip"):
            clone_zip(url, self.spec.context, self._secrets)

    def create_remote(self, url, name="origin", branch=None):
        """Create remote for the project git

        This method creates a new remote repository associated with the project's Git repository.
        If a remote with the specified name already exists, it will not be overwritten.

        If you wish to update the URL of an existing remote, use the `set_remote` method instead.

        :param url:    remote git url
        :param name:   name for the remote (default is 'origin')
        :param branch: Git branch to use as source
        """
        self.set_remote(url, name=name, branch=branch, overwrite=False)

    def set_remote(self, url, name="origin", branch=None, overwrite=True):
        """Create or update a remote for the project git repository.

        This method allows you to manage remote repositories associated with the project.
        It checks if a remote with the specified name already exists.

        If a remote with the same name does not exist, it will be created.
        If a remote with the same name already exists,
        the behavior depends on the value of the 'overwrite' flag.

        :param url: remote git url
        :param name: name for the remote (default is 'origin')
        :param branch: Git branch to use as source
        :param overwrite: if True (default), updates the existing remote with the given URL if it already exists.
                          if False, raises an error when attempting to create a remote with a name that already exists.
        :raises MLRunConflictError: If a remote with the same name already exists and overwrite
                                     is set to False.
        """
        self._ensure_git_repo()
        if self._remote_exists(name):
            if overwrite:
                self.spec.repo.delete_remote(name)
            else:
                raise mlrun.errors.MLRunConflictError(
                    f"Remote '{name}' already exists in the project, "
                    f"each remote in the project must have a unique name."
                    "Use 'set_remote' with 'override=True' inorder to update the remote, or choose a different name."
                )
        self.spec.repo.create_remote(name, url=url)
        url = url.replace("https://", "git://")
        if not branch:
            try:
                branch = self.spec.repo.active_branch.name
            except Exception:
                pass
        if branch:
            url = f"{url}#{branch}"
        self.spec._source = self.spec.source or url
        self.spec.origin_url = self.spec.origin_url or url

    def remove_remote(self, name):
        """Remove a remote from the project's Git repository.

        This method removes the remote repository associated with the specified name from the project's Git repository.

        :param name: Name of the remote to remove.
        """
        if self._remote_exists(name):
            self.spec.repo.delete_remote(name)
        else:
            logger.warning(f"The remote '{name}' does not exist. Nothing to remove.")

    def _remote_exists(self, name):
        """Check if a remote with the given name already exists"""
        return any(remote.name == name for remote in self.spec.repo.remotes)

    def _ensure_git_repo(self):
        if self.spec.repo:
            return
        context = self.context
        git_dir_path = path.join(context, ".git")

        if not path.exists(git_dir_path):
            logger.warning("Git repository not initialized. initializing now")
            self.spec.repo = git.Repo.init(context)
        else:
            # git already initialized
            self.spec.repo = git.Repo(context)

    def push(
        self,
        branch,
        message=None,
        update=True,
        remote: Optional[str] = None,
        add: Optional[list] = None,
        author_name: Optional[str] = None,
        author_email: Optional[str] = None,
        secrets: Union[SecretsStore, dict] = None,
    ):
        """update spec and push updates to remote git repo

        :param branch:       target git branch
        :param message:      git commit message
        :param update:       update files (git add update=True)
        :param remote:       git remote, default to origin
        :param add:          list of files to add
        :param author_name:  author's git user name to be used on this commit
        :param author_email: author's git user email to be used on this commit
        :param secrets:      dict or SecretsStore with Git credentials e.g. secrets={"GIT_TOKEN": token}
        """
        repo = self.spec.repo
        if not repo:
            raise ValueError("git repo is not set/defined")
        self.save()

        with repo.config_writer() as config:
            if author_name:
                config.set_value("user", "name", author_name)
            if author_email:
                config.set_value("user", "email", author_email)

        add = add or []
        add.append("project.yaml")
        repo.index.add(add)
        if update:
            repo.git.add(update=True)
        if repo.is_dirty():
            if not message:
                raise ValueError("please specify the commit message")
            try:
                repo.git.commit(m=message)
            except git.exc.GitCommandError as exc:
                if "Please tell me who you are" in str(exc):
                    warning_message = (
                        'Git is not configured. Either use "author_name", "author_email" and "secrets" parameters or '
                        "run the following commands from the terminal and run git push once to store "
                        "your credentials:\n"
                        '\tgit config --global user.email "<my@email.com>"\n'
                        '\tgit config --global user.name "<name>"\n'
                        "\tgit config --global credential.helper store\n"
                    )
                    raise mlrun.errors.MLRunPreconditionFailedError(
                        warning_message
                    ) from exc
                raise exc

        if not branch:
            raise ValueError("please specify the remote branch")

        remote = remote or "origin"
        self._run_authenticated_git_action(
            action=repo.git.push,
            remote=remote,
            args=[remote, branch],
            secrets=secrets or {},
        )

    def sync_functions(
        self,
        names: Optional[list] = None,
        always: bool = True,
        save: bool = False,
        silent: bool = False,
    ):
        """
        Reload function objects from specs and files.
        The function objects are synced against the definitions spec in `self.spec._function_definitions`.
        Referenced files/URLs in the function spec will be reloaded.
        Function definitions are parsed by the following precedence:

        1. Contains runtime spec.
        2. Contains module in the project's context.
        3. Contains path to function definition (yaml, DB, Hub).
        4. Contains path to .ipynb or .py files.
        5. Contains a Nuclio/Serving function image / an 'Application' kind definition.

        If function definition is already an object, some project metadata updates will apply however,
        it will not be reloaded.

        :param names:   Names of functions to reload, defaults to `self.spec._function_definitions.keys()`.
        :param always:  Force reloading the functions.
        :param save:    Whether to save the loaded functions or not.
        :param silent:  Whether to raise an exception when a function fails to load.

        :returns: Dictionary of function objects
        """
        if self._initialized and not always:
            return self.spec._function_objects

        functions = self.spec._function_objects
        if not names:
            names = self.spec._function_definitions.keys()
            functions = {}

        origin = mlrun.runtimes.utils.add_code_metadata(self.spec.context)
        for name in names:
            function_definition = self.spec._function_definitions.get(name)
            if not function_definition:
                if silent:
                    logger.warn(
                        "Function definition was not found, skipping reload", name=name
                    )
                    continue

                raise ValueError(f"Function named {name} not found")

            function_object = self.spec._function_objects.get(name, None)
            is_base_runtime = isinstance(
                function_object, mlrun.runtimes.base.BaseRuntime
            )
            # If this function is already available locally, don't recreate it unless always=True
            if is_base_runtime and not always:
                functions[name] = function_object
                continue

            # Reload the function
            if hasattr(function_definition, "to_dict"):
                name, func = _init_function_from_obj(function_definition, self, name)
            elif isinstance(function_definition, dict):
                try:
                    name, func = _init_function_from_dict(
                        function_definition, self, name
                    )
                except FileNotFoundError as exc:
                    message = f"File {exc.filename} not found while syncing project functions."
                    if silent:
                        message += " Skipping function reload"
                        logger.warn(message, name=name)
                        continue

                    raise mlrun.errors.MLRunMissingDependencyError(message) from exc

                except Exception as exc:
                    if silent:
                        logger.warn(
                            "Failed to instantiate function",
                            name=name,
                            error=mlrun.utils.err_to_str(exc),
                        )
                        continue
                    raise exc
            else:
                message = f"Function {name} must be an object or dict."
                if silent:
                    message += " Skipping function reload"
                    logger.warn(message, name=name)
                    continue
                raise ValueError(message)

            func.spec.build.code_origin = origin
            functions[name] = func
            if save:
                func.save(versioned=False)

        self.spec._function_objects = functions
        self._initialized = True
        return self.spec._function_objects

    def with_secrets(self, kind, source, prefix=""):
        """register a secrets source (file, env or dict)

        read secrets from a source provider to be used in workflows, example::

            proj.with_secrets("file", "file.txt")
            proj.with_secrets("inline", {"key": "val"})
            proj.with_secrets("env", "ENV1,ENV2", prefix="PFX_")

        Vault secret source has several options::

            proj.with_secrets('vault', {'user': <user name>, 'secrets': ['secret1', 'secret2' ...]})
            proj.with_secrets('vault', {'project': <proj.name>, 'secrets': ['secret1', 'secret2' ...]})
            proj.with_secrets('vault', ['secret1', 'secret2' ...])

        The 2nd option uses the current project name as context.
        Can also use empty secret list::

            proj.with_secrets("vault", [])

        This will enable access to all secrets in vault registered to the current project.

        :param kind:   secret type (file, inline, env, vault)
        :param source: secret data or link (see example)
        :param prefix: add a prefix to the keys in this source

        :returns: project object
        """

        if kind == "vault" and isinstance(source, list):
            source = {"project": self.metadata.name, "secrets": source}

        self._secrets.add_source(kind, source, prefix)
        return self

    def get_secret(self, key: str):
        """get a key based secret e.g. DB password from the context
        secrets can be specified when invoking a run through files, env, ..
        """
        if self._secrets:
            return self._secrets.get(key)
        return None

    def set_secrets(
        self,
        secrets: Optional[dict] = None,
        file_path: Optional[str] = None,
        provider: typing.Union[str, mlrun.common.schemas.SecretProviderName] = None,
    ):
        """
        Set project secrets from dict or secrets env file
        when using a secrets file it should have lines in the form KEY=VALUE, comment line start with "#"
        V3IO paths/credentials and MLrun service API address are dropped from the secrets

        example secrets file:

        .. code-block:: shell

            # this is an env file
            AWS_ACCESS_KEY_ID=XXXX
            AWS_SECRET_ACCESS_KEY=YYYY

        usage:

            # read env vars from dict or file and set as project secrets
            project.set_secrets({"SECRET1": "value"})
            project.set_secrets(file_path="secrets.env")

        :param secrets:   dict with secrets key/value
        :param file_path: path to secrets file
        :param provider:  MLRun secrets provider
        """
        if (not secrets and not file_path) or (secrets and file_path):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "must specify secrets OR file_path"
            )
        if file_path:
            if path.isfile(file_path):
                secrets = dotenv.dotenv_values(file_path)
                if None in secrets.values():
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        "env file lines must be in the form key=value"
                    )
            else:
                raise mlrun.errors.MLRunNotFoundError(f"{file_path} does not exist")
        # drop V3IO paths/credentials and MLrun service API address
        env_vars = {
            key: val
            for key, val in secrets.items()
            if key != "MLRUN_DBPATH" and not key.startswith("V3IO_")
        }
        provider = provider or mlrun.common.schemas.SecretProviderName.kubernetes
        mlrun.db.get_run_db().create_project_secrets(
            self.metadata.name, provider=provider, secrets=env_vars
        )

    def get_param(self, key: str, default=None):
        """get project param by key"""
        if self.spec.params:
            return self.spec.params.get(key, default)
        return default

    def _enrich_artifact_path_with_workflow_uid(self):
        artifact_path = self.spec.artifact_path or mlrun.mlconf.artifact_path

        workflow_uid_string = "{{workflow.uid}}"
        if (
            not mlrun.mlconf.enrich_artifact_path_with_workflow_id
            # no need to add workflow.uid to the artifact path for uniqueness,
            # this is already being handled by generating
            # the artifact target path from the artifact content hash ( body / file etc...)
            or mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash
            # if the artifact path already contains workflow.uid, no need to add it again
            or workflow_uid_string in artifact_path
        ):
            return artifact_path

        # join paths and replace "\" with "/" (in case of windows clients)
        artifact_path = path.join(artifact_path, workflow_uid_string).replace("\\", "/")
        return artifact_path

    def run(
        self,
        name: Optional[str] = None,
        workflow_path: Optional[str] = None,
        arguments: Optional[dict[str, typing.Any]] = None,
        artifact_path: Optional[str] = None,
        workflow_handler: Optional[typing.Union[str, typing.Callable]] = None,
        namespace: Optional[str] = None,
        sync: bool = False,
        watch: bool = False,
        dirty: bool = False,
        engine: Optional[str] = None,
        local: Optional[bool] = None,
        schedule: typing.Union[
            str, mlrun.common.schemas.ScheduleCronTrigger, bool
        ] = None,
        timeout: Optional[int] = None,
        source: Optional[str] = None,
        cleanup_ttl: Optional[int] = None,
        notifications: Optional[list[mlrun.model.Notification]] = None,
        workflow_runner_node_selector: typing.Optional[dict[str, str]] = None,
        context: typing.Optional[mlrun.execution.MLClientCtx] = None,
    ) -> _PipelineRunStatus:
        """Run a workflow using kubeflow pipelines

        :param name:                Name of the workflow
        :param workflow_path:       URL to a workflow file, if not a project workflow
        :param arguments:           Kubeflow pipelines arguments (parameters)
        :param artifact_path:       Target path/URL for workflow artifacts, the string '{{workflow.uid}}' will be
                                    replaced by workflow id.
        :param workflow_handler:    Workflow function handler (for running workflow function directly)
        :param namespace: Kubernetes namespace if other than default
        :param sync:      Force functions sync before run
        :param watch:     Wait for pipeline completion
        :param dirty:     Allow running the workflow when the git repo is dirty
        :param engine:    Workflow engine running the workflow.
                          Supported values are 'kfp' (default), 'local' or 'remote'.
                          For setting engine for remote running use 'remote:local' or 'remote:kfp'.
        :param local:     Run local pipeline with local functions (set local=True in function.run())
        :param schedule:  ScheduleCronTrigger class instance or a standard crontab expression string
                          (which will be converted to the class using its `from_crontab` constructor),
                          see this link for help:
                          https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html#module-apscheduler.triggers.cron
                          For using the pre-defined workflow's schedule, set `schedule=True`
        :param timeout:   Timeout in seconds to wait for pipeline completion (watch will be activated)
        :param source:    Source to use instead of the actual `project.spec.source` (used when engine is remote).
                          Can be one of:

                          * Remote URL which is loaded dynamically to the workflow runner.
                          * A path to the project's context on the workflow runner's image.
                            Path can be absolute or relative to `project.spec.build.source_code_target_dir` if defined
                            (enriched when building a project image with source, see `MlrunProject.build_image`).
                            For other engines the source is used to validate that the code is up-to-date.

        :param cleanup_ttl:
                          Pipeline cleanup ttl in secs (time to wait after workflow completion, at which point the
                          workflow and all its resources are deleted)
        :param notifications:
                          List of notifications to send for workflow completion
        :param workflow_runner_node_selector:
                          Defines the node selector for the workflow runner pod when using a remote engine.
                          This allows you to control and specify where the workflow runner pod will be scheduled.
                          This setting is only relevant when the engine is set to 'remote' or for scheduled workflows,
                          and it will be ignored if the workflow is not run on a remote engine.
        :param context:             mlrun context.
        :returns: ~py:class:`~mlrun.projects.pipelines._PipelineRunStatus` instance
        """

        arguments = arguments or {}
        need_repo = self.spec._need_repo()
        if not dirty:
            if self.spec.repo and self.spec.repo.is_dirty():
                msg = "You seem to have uncommitted git changes, use .push()"
                if not need_repo:
                    logger.warning("WARNING!, " + msg)
                else:
                    raise ProjectError(msg + " or dirty=True")

        if need_repo and self.spec.repo and not self.spec.source:
            raise ProjectError(
                "Remote repo is not defined, use .create_remote() + push()"
            )

        if (engine is None or not engine.startswith("remote")) and not schedule:
            # For remote/scheduled runs there is no need to sync functions as they can be loaded dynamically during run
            self.sync_functions(always=sync, silent=True)
            if not self.spec._function_objects:
                logger.warn(
                    "There are no functions in the project."
                    " Make sure you've set your functions with project.set_function()."
                )

        if not name and not workflow_path and not workflow_handler:
            raise ValueError("Workflow name, path, or handler must be specified")

        if workflow_path or (workflow_handler and callable(workflow_handler)):
            workflow_spec = WorkflowSpec(path=workflow_path, args=arguments)
        else:
            if name not in self.spec._workflows.keys():
                raise mlrun.errors.MLRunNotFoundError(f"Workflow {name} does not exist")
            workflow_spec = self.spec._workflows[name].copy()
            workflow_spec.merge_args(arguments)
        workflow_spec.cleanup_ttl = cleanup_ttl or workflow_spec.cleanup_ttl
        workflow_spec.run_local = local

        name = f"{self.metadata.name}-{name}" if name else self.metadata.name
        artifact_path = artifact_path or self._enrich_artifact_path_with_workflow_uid()

        if not schedule:
            workflow_spec.schedule = None
        elif not isinstance(schedule, bool):
            # Schedule = True -> use workflow_spec.schedule
            workflow_spec.schedule = schedule

        inner_engine = None
        if engine and engine.startswith("remote"):
            if ":" in engine:
                # inner could be either kfp or local
                engine, inner_engine = engine.split(":")
        elif workflow_spec.schedule:
            inner_engine = engine
            engine = "remote"
        # The default engine is kfp if not given:
        workflow_engine = get_workflow_engine(engine or workflow_spec.engine, local)
        if not inner_engine and workflow_engine.engine == "remote":
            # if inner engine is set to remote, assume kfp as the default inner engine with remote as the runner
            engine_kind = (
                workflow_spec.engine if workflow_spec.engine != "remote" else "kfp"
            )
            inner_engine = get_workflow_engine(engine_kind, local).engine
        workflow_spec.engine = inner_engine or workflow_engine.engine
        if workflow_runner_node_selector:
            if workflow_engine.engine == "remote":
                workflow_spec.workflow_runner_node_selector = (
                    workflow_runner_node_selector
                )
            else:
                logger.warn(
                    "'workflow_runner_node_selector' applies only to remote engines"
                    " and is ignored for non-remote runs."
                )

        run = workflow_engine.run(
            self,
            workflow_spec,
            name,
            workflow_handler=workflow_handler,
            secrets=self._secrets,
            artifact_path=artifact_path,
            namespace=namespace,
            source=source,
            notifications=notifications,
            context=context,
        )
        # run is None when scheduling
        if run and run.state == mlrun_pipelines.common.models.RunStatuses.failed:
            return run
        if not workflow_spec.schedule:
            # Failure and schedule messages already logged
            logger.info(
                f"Started run workflow {name} with run id = '{run.run_id}' by {workflow_engine.engine} engine"
            )
        workflow_spec.clear_tmp()
        if (timeout or watch) and not workflow_spec.schedule:
            run_status_kwargs = {}
            status_engine = run._engine
            # run's engine gets replaced with inner engine if engine is remote,
            # so in that case we need to get the status from the remote engine manually
            if workflow_engine.engine == "remote":
                status_engine = _RemoteRunner
                run_status_kwargs["inner_engine"] = run._engine

            status_engine.get_run_status(
                project=self, run=run, timeout=timeout, **run_status_kwargs
            )
        return run

    def save_workflow(self, name, target, artifact_path=None, ttl=None):
        """create and save a workflow as a yaml or archive file

        :param name:   workflow name
        :param target: target file path (can end with .yaml or .zip)
        :param artifact_path:
                       target path/url for workflow artifacts, the string
                       '{{workflow.uid}}' will be replaced by workflow id
        :param ttl:    pipeline ttl (time to live) in secs (after that the pods will be removed)
        """
        if not name or name not in self.spec._workflows:
            raise ValueError(f"workflow {name} not found")

        workflow_spec = self.spec._workflows[name]
        self.sync_functions()
        workflow_engine = get_workflow_engine(workflow_spec.engine)
        workflow_engine.save(self, workflow_spec, target, artifact_path=artifact_path)

    def get_run_status(
        self,
        run,
        timeout=None,
        expected_statuses=None,
        notifiers: CustomNotificationPusher = None,
    ):
        return run._engine.get_run_status(
            project=self,
            run=run,
            timeout=timeout,
            expected_statuses=expected_statuses,
            notifiers=notifiers,
        )

    def save(self, filepath=None, store=True):
        """export project to yaml file and save project in database

        :store: if True, allow updating in case project already exists
        """
        self.export(filepath)
        self.save_to_db(store)
        return self

    def save_to_db(self, store=True):
        """save project to database

        :store: if True, allow updating in case project already exists
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        if store:
            return db.store_project(self.metadata.name, self.to_dict())

        return db.create_project(self.to_dict())

    def export(self, filepath=None, include_files: Optional[str] = None):
        """save the project object into a yaml file or zip archive (default to project.yaml)

        By default, the project object is exported to a yaml file, when the filepath suffix is '.zip'
        the project context dir (code files) are also copied into the zip, the archive path can include
        DataItem urls (for remote object storage, e.g. s3://<bucket>/<path>).

        :param filepath: path to store project .yaml or .zip (with the project dir content)
        :param include_files: glob filter string for selecting files to include in the zip archive
        """
        project_file_path = filepath
        archive_code = filepath and str(filepath).endswith(".zip")
        if not filepath or archive_code:
            project_file_path = path.join(
                self.spec.context, self.spec.subpath or "", "project.yaml"
            )
        if filepath and "://" in str(filepath) and not archive_code:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "URLs are only applicable to archives"
            )

        project_dir = pathlib.Path(project_file_path).parent
        project_dir.mkdir(parents=True, exist_ok=True)
        with open(project_file_path, "w") as fp:
            fp.write(self.to_yaml())

        if archive_code:
            files_filter = include_files or "**"
            with tempfile.NamedTemporaryFile(suffix=".zip") as f:
                remote_file = "://" in filepath
                fpath = f.name if remote_file else filepath
                with zipfile.ZipFile(fpath, "w") as zipf:
                    for file_path in glob.iglob(
                        f"{project_dir}/{files_filter}", recursive=True
                    ):
                        write_path = pathlib.Path(file_path)
                        zipf.write(
                            write_path, arcname=write_path.relative_to(project_dir)
                        )
                if remote_file:
                    mlrun.get_dataitem(filepath).upload(zipf.filename)

    def set_model_monitoring_credentials(
        self,
        *,
        tsdb_profile_name: str,
        stream_profile_name: str,
        replace_creds: bool = False,
    ) -> None:
        """
        Set the credentials that will be used by the project's model monitoring infrastructure functions.
        Please note that you have to set the credentials before deploying any model monitoring application
        or a tracked serving function.

        For example, the full flow for enabling model monitoring infrastructure with **TDEngine** and **Kafka**, is:

        .. code-block:: python

            import mlrun
            from mlrun.datastore.datastore_profile import (
                DatastoreProfileKafkaSource,
                DatastoreProfileTDEngine,
            )

            project = mlrun.get_or_create_project("mm-infra-setup")

            # Create and register TSDB profile
            tsdb_profile = DatastoreProfileTDEngine(
                name="my-tdengine",
                host="<tdengine-server-ip-address>",
                port=6041,
                user="username",
                password="<tdengine-password>",
            )
            project.register_datastore_profile(tsdb_profile)

            # Create and register stream profile
            stream_profile = DatastoreProfileKafkaSource(
                name="my-kafka",
                brokers=["<kafka-broker-ip-address>:9094"],
                topics=[],  # Keep the topics list empty
                ## SASL is supported
                # sasl_user="user1",
                # sasl_pass="<kafka-sasl-password>",
            )
            project.register_datastore_profile(stream_profile)

            # Set model monitoring credentials and enable the infrastructure
            project.set_model_monitoring_credentials(
                tsdb_profile_name=tsdb_profile.name,
                stream_profile_name=stream_profile.name,
            )
            project.enable_model_monitoring()

        Note that you will need to change the profiles if you want to use **V3IO** TSDB and stream:

        .. code-block:: python

            from mlrun.datastore.datastore_profile import DatastoreProfileV3io

            # Create and register TSDB profile
            tsdb_profile = DatastoreProfileV3io(
                name="my-v3io-tsdb",
            )
            project.register_datastore_profile(tsdb_profile)

            # Create and register stream profile
            stream_profile = DatastoreProfileV3io(
                name="my-v3io-stream",
                v3io_access_key=mlrun.mlconf.get_v3io_access_key(),
            )
            project.register_datastore_profile(stream_profile)

        In the V3IO datastore, you must provide an explicit access key to the stream, but not to the TSDB.

        :param tsdb_profile_name:         The datastore profile name of the time-series database to be used in model
                                          monitoring. The supported profiles are:

                                          * :py:class:`~mlrun.datastore.datastore_profile.DatastoreProfileV3io`
                                          * :py:class:`~mlrun.datastore.datastore_profile.DatastoreProfileTDEngine`

                                          You need to register one of them, and pass the profile's name.
        :param stream_profile_name:       The datastore profile name of the stream to be used in model monitoring.
                                          The supported profiles are:

                                          * :py:class:`~mlrun.datastore.datastore_profile.DatastoreProfileV3io`
                                          * :py:class:`~mlrun.datastore.datastore_profile.DatastoreProfileKafkaSource`

                                          You need to register one of them, and pass the profile's name.
        :param replace_creds:             If ``True`` - override the existing credentials.
                                          Please keep in mind that if you have already enabled model monitoring
                                          on your project, replacing the credentials can cause data loss, and will
                                          require redeploying all the model monitoring functions, model monitoring
                                          infrastructure, and tracked model servers.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)

        db.set_model_monitoring_credentials(
            project=self.name,
            credentials={
                "tsdb_profile_name": tsdb_profile_name,
                "stream_profile_name": stream_profile_name,
            },
            replace_creds=replace_creds,
        )
        if replace_creds:
            logger.info(
                "Model monitoring credentials were set successfully. "
                "Please keep in mind that if you already had model monitoring functions "
                "/ model monitoring infra / tracked model server "
                "deployed on your project, you will need to redeploy them. "
                "For redeploying the model monitoring infra, first disable it using "
                "`project.disable_model_monitoring()` and then enable it using `project.enable_model_monitoring()`."
            )

    def list_model_endpoints(
        self,
        names: Optional[Union[str, list[str]]] = None,
        model_name: Optional[str] = None,
        model_tag: Optional[str] = None,
        function_name: Optional[str] = None,
        function_tag: Optional[str] = None,
        labels: Optional[list[str]] = None,
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
        top_level: bool = False,
        uids: Optional[list[str]] = None,
        latest_only: bool = False,
        tsdb_metrics: bool = False,
        metric_list: Optional[list[str]] = None,
    ) -> mlrun.common.schemas.ModelEndpointList:
        """
        Returns a list of `ModelEndpoint` objects. Each `ModelEndpoint` object represents the current state of a
        model endpoint. This functions supports filtering by the following parameters:
        1) name
        2) model_name
        3) model_tag
        4) function_name
        5) function_tag
        6) labels
        7) top level
        8) uids
        9) start and end time, corresponding to the `created` field.
        By default, when no filters are applied, all available endpoints for the given project will be listed.

        In addition, this functions provides a facade for listing endpoint related metrics. This facade is time-based
        and depends on the 'start' and 'end' parameters.

        :param names: The name of the model to filter by
        :param model_name: The name of the model to filter by
        :param function_name: The name of the function to filter by
        :param function_tag: The tag of the function to filter by
        :param labels: Filter model endpoints by label key-value pairs or key existence. This can be provided as:
            - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
            or `{"label": None}` to check for key existence.
            - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
            or just `"label"` for key existence.
            - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
            the specified key-value pairs or key existence.
        :param start:           The start time to filter by.Corresponding to the `created` field.
        :param end:             The end time to filter by. Corresponding to the `created` field.
        :param top_level:       If true will return only routers and endpoint that are NOT children of any router.
        :param uids:            If passed will return a list `ModelEndpoint` object with uid in uids.
        :param tsdb_metrics:    When True, the time series metrics will be added to the output
                                of the resulting.
        :param metric_list:     List of metrics to include from the time series DB. Defaults to all metrics.
                                If tsdb_metrics=False, this parameter will be ignored and no tsdb metrics
                                will be included.

        :returns: Returns a list of `ModelEndpoint` objects.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        return db.list_model_endpoints(
            project=self.name,
            names=names,
            model_name=model_name,
            model_tag=model_tag,
            function_name=function_name,
            function_tag=function_tag,
            labels=labels,
            start=start,
            end=end,
            top_level=top_level,
            uids=uids,
            latest_only=latest_only,
            tsdb_metrics=tsdb_metrics,
            metric_list=metric_list,
        )

    def run_function(
        self,
        function: typing.Union[str, mlrun.runtimes.BaseRuntime],
        handler: Optional[str] = None,
        name: str = "",
        params: Optional[dict] = None,
        hyperparams: Optional[dict] = None,
        hyper_param_options: mlrun.model.HyperParamOptions = None,
        inputs: Optional[dict] = None,
        outputs: Optional[list[str]] = None,
        workdir: str = "",
        labels: Optional[dict] = None,
        base_task: mlrun.model.RunTemplate = None,
        watch: bool = True,
        local: Optional[bool] = None,
        verbose: Optional[bool] = None,
        selector: Optional[str] = None,
        auto_build: Optional[bool] = None,
        schedule: typing.Union[str, mlrun.common.schemas.ScheduleCronTrigger] = None,
        artifact_path: Optional[str] = None,
        notifications: Optional[list[mlrun.model.Notification]] = None,
        returns: Optional[list[Union[str, dict[str, str]]]] = None,
        builder_env: Optional[dict] = None,
        reset_on_run: Optional[bool] = None,
    ) -> typing.Union[mlrun.model.RunObject, PipelineNodeWrapper]:
        """Run a local or remote task as part of a local/kubeflow pipeline

        example (use with project)::

            # create a project with two functions (local and from hub)
            project = mlrun.new_project(project_name, "./proj")
            project.set_function("mycode.py", "myfunc", image="mlrun/mlrun")
            project.set_function("hub://auto-trainer", "train")

            # run functions (refer to them by name)
            run1 = project.run_function("myfunc", params={"x": 7})
            run2 = project.run_function(
                "train",
                params={"label_columns": LABELS},
                inputs={"dataset": run1.outputs["data"]},
            )

        :param function:        name of the function (in the project) or function object
        :param handler:         name of the function handler
        :param name:            execution name
        :param params:          input parameters (dict)
        :param hyperparams:     hyper parameters
        :param selector:        selection criteria for hyper params e.g. "max.accuracy"
        :param hyper_param_options:  hyper param options (selector, early stop, strategy, ..)
                                see: :py:class:`~mlrun.model.HyperParamOptions`
        :param inputs:          Input objects to pass to the handler. Type hints can be given so the input will be
                                parsed during runtime from `mlrun.DataItem` to the given type hint. The type hint can be
                                given in the key field of the dictionary after a colon, e.g: "<key> : <type_hint>".
        :param outputs:         list of outputs which can pass in the workflow
        :param workdir:         default input artifacts path
        :param labels:          labels to tag the job/run with ({key:val, ..})
        :param base_task:       task object to use as base
        :param watch:           watch/follow run log, True by default
        :param local:           run the function locally vs on the runtime/cluster
        :param verbose:         add verbose prints/logs
        :param auto_build:      when set to True and the function require build it will be built on the first
                                function run, use only if you dont plan on changing the build config between runs
        :param schedule:        ScheduleCronTrigger class instance or a standard crontab expression string
                                (which will be converted to the class using its `from_crontab` constructor),
                                see this link for help:
                                https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html#module-apscheduler.triggers.cron
        :param artifact_path:   path to store artifacts, when running in a workflow this will be set automatically
        :param notifications:   list of notifications to push when the run is completed
        :param returns:         List of log hints - configurations for how to log the returning values from the
                                handler's run (as artifacts or results). The list's length must be equal to the amount
                                of returning objects. A log hint may be given as:

                                * A string of the key to use to log the returning value as result or as an artifact. To
                                  specify The artifact type, it is possible to pass a string in the following structure:
                                  "<key> : <type>". Available artifact types can be seen in `mlrun.ArtifactType`. If no
                                  artifact type is specified, the object's default artifact type will be used.
                                * A dictionary of configurations to use when logging. Further info per object type and
                                  artifact type can be given there. The artifact key must appear in the dictionary as
                                  "key": "the_key".
        :param builder_env:     env vars dict for source archive config/credentials e.g. builder_env={"GIT_TOKEN":
                                token}
        :param reset_on_run:    When True, function python modules would reload prior to code execution.
                                This ensures latest code changes are executed. This argument must be used in
                                conjunction with the local=True argument.

        :return: MLRun RunObject or PipelineNodeWrapper
        """
        return run_function(
            function,
            handler=handler,
            name=name,
            params=params,
            hyperparams=hyperparams,
            hyper_param_options=hyper_param_options,
            inputs=inputs,
            outputs=outputs,
            workdir=workdir,
            labels=labels,
            base_task=base_task,
            watch=watch,
            local=local,
            verbose=verbose,
            selector=selector,
            project_object=self,
            auto_build=auto_build,
            schedule=schedule,
            artifact_path=artifact_path,
            notifications=notifications,
            returns=returns,
            builder_env=builder_env,
            reset_on_run=reset_on_run,
        )

    def build_function(
        self,
        function: typing.Union[str, mlrun.runtimes.BaseRuntime],
        with_mlrun: Optional[bool] = None,
        skip_deployed: bool = False,
        image: Optional[str] = None,
        base_image: Optional[str] = None,
        commands: Optional[list] = None,
        secret_name: Optional[str] = None,
        requirements: Optional[typing.Union[str, list[str]]] = None,
        mlrun_version_specifier: Optional[str] = None,
        builder_env: Optional[dict] = None,
        overwrite_build_params: bool = False,
        requirements_file: Optional[str] = None,
        extra_args: Optional[str] = None,
        force_build: bool = False,
    ) -> typing.Union[BuildStatus, PipelineNodeWrapper]:
        """deploy ML function, build container with its dependencies

        :param function:            name of the function (in the project) or function object
        :param with_mlrun:          add the current mlrun package to the container build
        :param skip_deployed:       skip the build if we already have an image for the function
        :param image:               target image name/path
        :param base_image:          base image name/path (commands and source code will be added to it)
        :param commands:            list of docker build (RUN) commands e.g. ['pip install pandas']
        :param secret_name:         k8s secret for accessing the docker registry
        :param requirements:        list of python packages, defaults to None
        :param requirements_file:   pip requirements file path, defaults to None
        :param mlrun_version_specifier:  which mlrun package version to include (if not current)
        :param builder_env:         Kaniko builder pod env vars dict (for config/credentials)
            e.g. builder_env={"GIT_TOKEN": token}, does not work yet in KFP
        :param overwrite_build_params:  Overwrite existing build configuration (currently applies to
            requirements and commands)

            * False: The new params are merged with the existing
            * True: The existing params are replaced by the new ones

        :param extra_args:  A string containing additional builder arguments in the format of command-line options,
            e.g. extra_args="--skip-tls-verify --build-arg A=val"
        :param force_build:  force building the image, even when no changes were made
        """
        return build_function(
            function,
            with_mlrun=with_mlrun,
            skip_deployed=skip_deployed,
            image=image,
            base_image=base_image,
            commands=commands,
            secret_name=secret_name,
            requirements=requirements,
            requirements_file=requirements_file,
            mlrun_version_specifier=mlrun_version_specifier,
            builder_env=builder_env,
            project_object=self,
            overwrite_build_params=overwrite_build_params,
            extra_args=extra_args,
            force_build=force_build,
        )

    def build_config(
        self,
        image: Optional[str] = None,
        set_as_default: bool = False,
        with_mlrun: Optional[bool] = None,
        base_image: Optional[str] = None,
        commands: Optional[list] = None,
        secret_name: Optional[str] = None,
        requirements: Optional[typing.Union[str, list[str]]] = None,
        overwrite_build_params: bool = False,
        requirements_file: Optional[str] = None,
        builder_env: Optional[dict] = None,
        extra_args: Optional[str] = None,
        source_code_target_dir: Optional[str] = None,
    ):
        """specify builder configuration for the project

        :param image: target image name/path. If not specified the project's existing `default_image` name will be
            used. If not set, the `mlconf.default_project_image_name` value will be used
        :param set_as_default: set `image` to be the project's default image (default False)
        :param with_mlrun: add the current mlrun package to the container build
        :param base_image: base image name/path
        :param commands:   list of docker build (RUN) commands e.g. ['pip install pandas']
        :param secret_name:     k8s secret for accessing the docker registry
        :param requirements: a list of packages to install on the built image
        :param requirements_file: requirements file to install on the built image
        :param overwrite_build_params:  Overwrite existing build configuration (currently applies to
            requirements and commands)

            * False: The new params are merged with the existing
            * True: The existing params are replaced by the new ones

        :param builder_env: Kaniko builder pod env vars dict (for config/credentials)
            e.g. builder_env={"GIT_TOKEN": token}, does not work yet in KFP
        :param extra_args:  A string containing additional builder arguments in the format of command-line options,
            e.g. extra_args="--skip-tls-verify --build-arg A=val"
        :param source_code_target_dir: Path on the image where source code would be extracted
            (by default `/home/mlrun_code`)
        """
        if not overwrite_build_params:
            # TODO: change overwrite_build_params default to True in 1.10.0
            warnings.warn(
                "The `overwrite_build_params` parameter default will change from 'False' to 'True' in 1.10.0.",
                mlrun.utils.OverwriteBuildParamsWarning,
            )
        default_image_name = mlrun.mlconf.default_project_image_name.format(
            name=self.name
        )
        image = image or self.default_image or default_image_name

        self.spec.build.build_config(
            image=image,
            base_image=base_image,
            commands=commands,
            secret=secret_name,
            with_mlrun=with_mlrun,
            requirements=requirements,
            requirements_file=requirements_file,
            overwrite=overwrite_build_params,
            builder_env=builder_env,
            extra_args=extra_args,
            source_code_target_dir=source_code_target_dir,
        )

        if set_as_default and image != self.default_image:
            self.set_default_image(image)

    def build_image(
        self,
        image: Optional[str] = None,
        set_as_default: bool = True,
        with_mlrun: Optional[bool] = None,
        base_image: Optional[str] = None,
        commands: Optional[list] = None,
        secret_name: Optional[str] = None,
        requirements: Optional[typing.Union[str, list[str]]] = None,
        mlrun_version_specifier: Optional[str] = None,
        builder_env: Optional[dict] = None,
        overwrite_build_params: bool = False,
        requirements_file: Optional[str] = None,
        extra_args: Optional[str] = None,
        target_dir: Optional[str] = None,
    ) -> typing.Union[BuildStatus, PipelineNodeWrapper]:
        """Builder docker image for the project, based on the project's build config. Parameters allow to override
        the build config.
        If the project has a source configured and pull_at_runtime is not configured, this source will be cloned to the
        image built. The `target_dir` parameter allows specifying the target path where the code will be extracted.

        :param image: target image name/path. If not specified the project's existing `default_image` name will be
                        used. If not set, the `mlconf.default_project_image_name` value will be used
        :param set_as_default: set `image` to be the project's default image (default True)
        :param with_mlrun:      add the current mlrun package to the container build
        :param base_image:      base image name/path (commands and source code will be added to it) defaults to
                                mlrun.mlconf.default_base_image
        :param commands:        list of docker build (RUN) commands e.g. ['pip install pandas']
        :param secret_name:     k8s secret for accessing the docker registry
        :param requirements:    list of python packages, defaults to None
        :param requirements_file:  pip requirements file path, defaults to None
        :param mlrun_version_specifier:  which mlrun package version to include (if not current)
        :param builder_env:     Kaniko builder pod env vars dict (for config/credentials)
            e.g. builder_env={"GIT_TOKEN": token}, does not work yet in KFP
        :param overwrite_build_params:  Overwrite existing build configuration (currently applies to
            requirements and commands)

            * False: The new params are merged with the existing
            * True: The existing params are replaced by the new ones

        :param extra_args:  A string containing additional builder arguments in the format of command-line options,
            e.g. extra_args="--skip-tls-verify --build-arg A=val"
        :param target_dir: Path on the image where source code would be extracted (by default `/home/mlrun_code`)
        """
        if not base_image:
            base_image = mlrun.mlconf.default_base_image
            logger.info(
                "Base image not specified, using default base image",
                base_image=base_image,
            )

        if not overwrite_build_params:
            # TODO: change overwrite_build_params default to True in 1.10.0
            warnings.warn(
                "The `overwrite_build_params` parameter default will change from 'False' to 'True' in 1.10.0.",
                mlrun.utils.OverwriteBuildParamsWarning,
            )

        # TODO: remove filter once overwrite_build_params default is changed to True in 1.8.0
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", category=mlrun.utils.OverwriteBuildParamsWarning
            )

            self.build_config(
                image=image,
                set_as_default=set_as_default,
                base_image=base_image,
                commands=commands,
                secret_name=secret_name,
                with_mlrun=with_mlrun,
                requirements=requirements,
                requirements_file=requirements_file,
                overwrite_build_params=overwrite_build_params,
            )

            function = mlrun.new_function("mlrun--project--image--builder", kind="job")

            if self.spec.source and not self.spec.load_source_on_run:
                function.with_source_archive(
                    source=self.spec.source,
                    target_dir=target_dir,
                    pull_at_runtime=False,
                )

            build = self.spec.build
            result = self.build_function(
                function=function,
                with_mlrun=build.with_mlrun,
                image=build.image,
                base_image=build.base_image,
                commands=build.commands,
                secret_name=build.secret,
                requirements=build.requirements,
                overwrite_build_params=overwrite_build_params,
                mlrun_version_specifier=mlrun_version_specifier,
                builder_env=builder_env,
                extra_args=extra_args,
                force_build=True,
            )

            # Get the enriched target dir from the function
            self.spec.build.source_code_target_dir = (
                function.spec.build.source_code_target_dir
            )

        try:
            mlrun.db.get_run_db(secrets=self._secrets).delete_function(
                name=function.metadata.name
            )
        except Exception as exc:
            logger.warning(
                f"Image was successfully built, but failed to delete temporary function {function.metadata.name}."
                " To remove the function, attempt to manually delete it.",
                exc=mlrun.errors.err_to_str(exc),
            )

        return result

    def deploy_function(
        self,
        function: typing.Union[str, mlrun.runtimes.BaseRuntime],
        models: Optional[list] = None,
        env: Optional[dict] = None,
        tag: Optional[str] = None,
        verbose: Optional[bool] = None,
        builder_env: Optional[dict] = None,
        mock: Optional[bool] = None,
    ) -> typing.Union[DeployStatus, PipelineNodeWrapper]:
        """deploy real-time (nuclio based) functions

        :param function:    name of the function (in the project) or function object
        :param models:      list of model items
        :param env:         dict of extra environment variables
        :param tag:         extra version tag
        :param verbose:     add verbose prints/logs
        :param builder_env: env vars dict for source archive config/credentials e.g. `builder_env={"GIT_TOKEN": token}`
        :param mock:        deploy mock server vs a real Nuclio function (for local simulations)
        """
        return deploy_function(
            function,
            models=models,
            env=env,
            tag=tag,
            verbose=verbose,
            builder_env=builder_env,
            project_object=self,
            mock=mock,
        )

    def get_artifact(
        self, key, tag=None, iter=None, tree=None, uid=None
    ) -> typing.Optional[Artifact]:
        """Return an artifact object

        :param key: Artifact key
        :param tag: Version tag
        :param iter: Iteration number (for hyper-param tasks)
        :param tree: The producer id (tree)
        :param uid: The artifact uid
        :return: Artifact object
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        artifact = db.read_artifact(
            key, tag, iter=iter, project=self.metadata.name, tree=tree, uid=uid
        )

        # in tests, if an artifact is not found, the db returns None
        # in real usage, the db should raise an exception
        if artifact:
            return dict_to_artifact(artifact)
        return None

    def list_artifacts(
        self,
        name=None,
        tag=None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
        since=None,
        until=None,
        iter: Optional[int] = None,
        best_iteration: bool = False,
        kind: Optional[str] = None,
        category: typing.Union[str, mlrun.common.schemas.ArtifactCategories] = None,
        tree: Optional[str] = None,
        limit: Optional[int] = None,
        format_: Optional[
            mlrun.common.formatters.ArtifactFormat
        ] = mlrun.common.formatters.ArtifactFormat.full,
        partition_by: Optional[
            Union[mlrun.common.schemas.ArtifactPartitionByField, str]
        ] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Optional[
            Union[mlrun.common.schemas.SortField, str]
        ] = mlrun.common.schemas.SortField.updated,
        partition_order: Union[
            mlrun.common.schemas.OrderType, str
        ] = mlrun.common.schemas.OrderType.desc,
    ) -> mlrun.lists.ArtifactList:
        """List artifacts filtered by various parameters.

        The returned result is an `ArtifactList` (list of dict), use `.to_objects()` to convert it to a list of
        RunObjects, `.show()` to view graphically in Jupyter, and `.to_df()` to convert to a DataFrame.

        Examples::

            # Get latest version of all artifacts in project
            latest_artifacts = project.list_artifacts(tag="latest")
            # check different artifact versions for a specific artifact, return as objects list
            result_versions = project.list_artifacts("results", tag="*").to_objects()

        :param name: Name of artifacts to retrieve. Name with '~' prefix is used as a like query, and is not
            case-sensitive. This means that querying for ``~name`` may return artifacts named
            ``my_Name_1`` or ``surname``.
        :param tag: Return artifacts assigned this tag.
        :param labels: Filter artifacts by label key-value pairs or key existence. This can be provided as:

                       - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
                         or `{"label": None}` to check for key existence.
                       - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
                         or just `"label"` for key existence.
                       - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
                         the specified key-value pairs or key existence.

        :param since: Not in use in :py:class:`HTTPRunDB`.
        :param until: Not in use in :py:class:`HTTPRunDB`.
        :param iter: Return artifacts from a specific iteration (where ``iter=0`` means the root iteration). If
            ``None`` (default) return artifacts from all iterations.
        :param best_iteration: Returns the artifact which belongs to the best iteration of a given run, in the case of
            artifacts generated from a hyper-param run. If only a single iteration exists, will return the artifact
            from that iteration. If using ``best_iter``, the ``iter`` parameter must not be used.
        :param kind: Return artifacts of the requested kind.
        :param category: Return artifacts of the requested category.
        :param tree: Return artifacts of the requested tree.
        :param limit: Deprecated - Maximum number of artifacts to return (will be removed in 1.11.0).
        :param format_: The format in which to return the artifacts. Default is 'full'.
        :param partition_by: Field to group results by. When `partition_by` is specified, the `partition_sort_by`
            parameter must be provided as well.
        :param rows_per_partition: How many top rows (per sorting defined by `partition_sort_by` and `partition_order`)
            to return per group. Default value is 1.
        :param partition_sort_by: What field to sort the results by, within each partition defined by `partition_by`.
            Currently the only allowed values are `created` and `updated`.
        :param partition_order: Order of sorting within partitions - `asc` or `desc`. Default is `desc`.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)

        if limit:
            # TODO: Remove this in 1.11.0
            warnings.warn(
                "'limit' is deprecated and will be removed in 1.11.0. Use 'page' and 'page_size' instead.",
                FutureWarning,
            )

        return db.list_artifacts(
            name,
            self.metadata.name,
            tag,
            labels=labels,
            since=since,
            until=until,
            iter=iter,
            best_iteration=best_iteration,
            kind=kind,
            category=category,
            tree=tree,
            format_=format_,
            limit=limit,
            partition_by=partition_by,
            rows_per_partition=rows_per_partition,
            partition_sort_by=partition_sort_by,
            partition_order=partition_order,
        )

    def paginated_list_artifacts(
        self,
        *args,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        **kwargs,
    ) -> tuple[mlrun.lists.ArtifactList, Optional[str]]:
        """List artifacts with support for pagination and various filtering options.

        This method retrieves a paginated list of artifacts based on the specified filter parameters.
        Pagination is controlled using the `page`, `page_size`, and `page_token` parameters. The method
        will return a list of artifacts that match the filtering criteria provided.

        The returned result is an `ArtifactList` (list of dict), use `.to_objects()` to convert it to a list of
        RunObjects, `.show()` to view graphically in Jupyter, and `.to_df()` to convert to a DataFrame.

        For detailed information about the parameters, refer to the list_artifacts method:
            See :py:func:`~list_artifacts` for more details.

        Examples::

            # Fetch first page of artifacts with page size of 5
            artifacts, token = project.paginated_list_artifacts("results", page_size=5)
            # Fetch next page using the pagination token from the previous response
            artifacts, token = project.paginated_list_artifacts("results", page_token=token)
            # Fetch artifacts for a specific page (e.g., page 3)
            artifacts, token = project.paginated_list_artifacts(
                "results", page=3, page_size=5
            )

            # Automatically iterate over all pages without explicitly specifying the page number
            artifacts = []
            token = None
            while True:
                page_artifacts, token = project.paginated_list_artifacts(
                    page_token=token, page_size=5
                )
                artifacts.extend(page_artifacts)

                # If token is None and page_artifacts is empty, we've reached the end (no more artifacts).
                # If token is None and page_artifacts is not empty, we've fetched the last page of artifacts.
                if not token:
                    break
            print(f"Total artifacts retrieved: {len(artifacts)}")

        :param page: The page number to retrieve. If not provided, the next page will be retrieved.
        :param page_size: The number of items per page to retrieve. Up to `page_size` responses are expected.
            Defaults to `mlrun.mlconf.httpdb.pagination.default_page_size` if not provided.
        :param page_token: A pagination token used to retrieve the next page of results. Should not be provided
            for the first request.

        :returns: A tuple containing the list of artifacts and an optional `page_token` for pagination.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        return db.paginated_list_artifacts(
            *args,
            project=self.metadata.name,
            page=page,
            page_size=page_size,
            page_token=page_token,
            **kwargs,
        )

    def list_models(
        self,
        name=None,
        tag=None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
        since=None,
        until=None,
        iter: Optional[int] = None,
        best_iteration: bool = False,
        tree: Optional[str] = None,
        limit: Optional[int] = None,
        format_: Optional[
            mlrun.common.formatters.ArtifactFormat
        ] = mlrun.common.formatters.ArtifactFormat.full,
    ):
        """List models in project, filtered by various parameters.

        Examples::

            # Get latest version of all models in project
            latest_models = project.list_models(tag="latest")


        :param name: Name of artifacts to retrieve. Name with '~' prefix is used as a like query, and is not
            case-sensitive. This means that querying for ``~name`` may return artifacts named
            ``my_Name_1`` or ``surname``.
        :param tag: Return artifacts assigned this tag.
        :param labels: Filter model artifacts by label key-value pairs or key existence. This can be provided as:

                       - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
                         or `{"label": None}` to check for key existence.
                       - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
                         or just `"label"` for key existence.
                       - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
                         the specified key-value pairs or key existence.

        :param since: Not in use in :py:class:`HTTPRunDB`.
        :param until: Not in use in :py:class:`HTTPRunDB`.
        :param iter: Return artifacts from a specific iteration (where ``iter=0`` means the root iteration). If
            ``None`` (default) return artifacts from all iterations.
        :param best_iteration: Returns the artifact which belongs to the best iteration of a given run, in the case of
            artifacts generated from a hyper-param run. If only a single iteration exists, will return the artifact
            from that iteration. If using ``best_iter``, the ``iter`` parameter must not be used.
        :param tree: Return artifacts of the requested tree.
        :param limit: Deprecated - Maximum number of artifacts to return (will be removed in 1.11.0).
        :param format_: The format in which to return the artifacts. Default is 'full'.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        return db.list_artifacts(
            name,
            self.metadata.name,
            tag,
            labels=labels,
            since=since,
            until=until,
            iter=iter,
            best_iteration=best_iteration,
            kind=mlrun.artifacts.model.ModelArtifact.kind,
            tree=tree,
            limit=limit,
            format_=format_,
        ).to_objects()

    def paginated_list_models(
        self,
        *args,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        **kwargs,
    ) -> tuple[mlrun.lists.ArtifactList, Optional[str]]:
        """List models in project with support for pagination and various filtering options.

        This method retrieves a paginated list of artifacts based on the specified filter parameters.
        Pagination is controlled using the `page`, `page_size`, and `page_token` parameters. The method
        will return a list of artifacts that match the filtering criteria provided.

        For detailed information about the parameters, refer to the list_models method:
            See :py:func:`~list_models` for more details.

        Examples::

            # Fetch first page of artifacts with page size of 5
            artifacts, token = project.paginated_list_models("results", page_size=5)
            # Fetch next page using the pagination token from the previous response
            artifacts, token = project.paginated_list_models("results", page_token=token)
            # Fetch artifacts for a specific page (e.g., page 3)
            artifacts, token = project.paginated_list_models("results", page=3, page_size=5)

            # Automatically iterate over all pages without explicitly specifying the page number
            artifacts = []
            token = None
            while True:
                page_artifacts, token = project.paginated_list_models(
                    page_token=token, page_size=5
                )
                artifacts.extend(page_artifacts)

                # If token is None and page_artifacts is empty, we've reached the end (no more artifacts).
                # If token is None and page_artifacts is not empty, we've fetched the last page of artifacts.
                if not token:
                    break
            print(f"Total artifacts retrieved: {len(artifacts)}")

        :param page: The page number to retrieve. If not provided, the next page will be retrieved.
        :param page_size: The number of items per page to retrieve. Up to `page_size` responses are expected.
            Defaults to `mlrun.mlconf.httpdb.pagination.default_page_size` if not provided.
        :param page_token: A pagination token used to retrieve the next page of results. Should not be provided
            for the first request.

        :returns: A tuple containing the list of artifacts and an optional `page_token` for pagination.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        return db.paginated_list_artifacts(
            *args,
            project=self.metadata.name,
            kind=mlrun.artifacts.model.ModelArtifact.kind,
            page=page,
            page_size=page_size,
            page_token=page_token,
            **kwargs,
        )

    def list_functions(
        self,
        name: Optional[str] = None,
        tag: Optional[str] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
        kind: Optional[str] = None,
        format_: Optional[str] = None,
    ):
        """Retrieve a list of functions, filtered by specific criteria.

        example::

            functions = project.list_functions(tag="latest")


        :param name: Return only functions with a specific name.
        :param tag: Return function versions with specific tags. To return only tagged functions, set tag to ``"*"``.
        :param labels: Filter functions by label key-value pairs or key existence. This can be provided as:

                       - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
                         or `{"label": None}` to check for key existence.
                       - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
                         or just `"label"` for key existence.
                       - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
                         the specified key-value pairs or key existence.

        :param kind: Return functions of the specified kind. If not provided, all function kinds will be returned.
        :param format_: The format in which to return the functions. Default is 'full'.
        :returns: List of function objects.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        functions = db.list_functions(
            name,
            project=self.metadata.name,
            tag=tag,
            kind=kind,
            labels=labels,
            format_=format_,
        )
        if functions:
            # convert dict to function objects
            return [mlrun.new_function(runtime=func) for func in functions]

    def paginated_list_functions(
        self,
        *args,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        **kwargs,
    ) -> tuple[list, Optional[str]]:
        """List functions with support for pagination and various filtering options.

        This method retrieves a paginated list of functions based on the specified filter parameters.
        Pagination is controlled using the `page`, `page_size`, and `page_token` parameters. The method
        will return a list of functions that match the filtering criteria provided.

        For detailed information about the parameters, refer to the list_functions method:
            See :py:func:`~list_functions` for more details.

        Examples::

            # Fetch first page of functions with page size of 5
            functions, token = project.paginated_list_functions(page_size=5)
            # Fetch next page using the pagination token from the previous response
            functions, token = project.paginated_list_functions(page_token=token)
            # Fetch functions for a specific page (e.g., page 3)
            functions, token = project.paginated_list_functions(page=3, page_size=5)

            # Automatically iterate over all pages without explicitly specifying the page number
            functions = []
            token = None
            while True:
                page_functions, token = project.paginated_list_functions(
                    page_token=token, page_size=5
                )
                functions.extend(page_functions)

                # If token is None and page_functions is empty, we've reached the end (no more functions).
                # If token is None and page_functions is not empty, we've fetched the last page of functions.
                if not token:
                    break
            print(f"Total functions retrieved: {len(functions)}")

        :param page: The page number to retrieve. If not provided, the next page will be retrieved.
        :param page_size: The number of items per page to retrieve. Up to `page_size` responses are expected.
            Defaults to `mlrun.mlconf.httpdb.pagination.default_page_size` if not provided.
        :param page_token: A pagination token used to retrieve the next page of results. Should not be provided
            for the first request.

        :returns: A tuple containing the list of functions and an optional `page_token` for pagination.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        functions, token = db.paginated_list_functions(
            *args,
            project=self.metadata.name,
            page=page,
            page_size=page_size,
            page_token=page_token,
            **kwargs,
        )
        # convert dict to function objects
        return [mlrun.new_function(runtime=func) for func in functions], token

    def list_model_monitoring_functions(
        self,
        name: Optional[str] = None,
        tag: Optional[str] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
    ) -> Optional[list]:
        """
        Retrieve a list of all the model monitoring functions.
        Example::

            functions = project.list_model_monitoring_functions()

        :param name:    Return only functions with a specific name.
        :param tag:     Return function versions with specific tags.
        :param labels: Filter functions by label key-value pairs or key existence. This can be provided as:

                       - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
                         or `{"label": None}` to check for key existence.
                       - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
                         or just `"label"` for key existence.
                       - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
                         the specified key-value pairs or key existence.

        :returns: List of function objects.
        """

        model_monitoring_labels_list = [str(mm_constants.ModelMonitoringAppLabel())]
        if labels:
            model_monitoring_labels_list += labels
        return self.list_functions(
            name=name,
            tag=tag,
            labels=model_monitoring_labels_list,
        )

    def list_runs(
        self,
        name: Optional[str] = None,
        uid: Optional[Union[str, list[str]]] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
        state: Optional[
            mlrun.common.runtimes.constants.RunStates
        ] = None,  # Backward compatibility
        states: typing.Optional[list[mlrun.common.runtimes.constants.RunStates]] = None,
        sort: bool = True,
        iter: bool = False,
        start_time_from: Optional[datetime.datetime] = None,
        start_time_to: Optional[datetime.datetime] = None,
        last_update_time_from: Optional[datetime.datetime] = None,
        last_update_time_to: Optional[datetime.datetime] = None,
        end_time_from: Optional[datetime.datetime] = None,
        end_time_to: Optional[datetime.datetime] = None,
        **kwargs,
    ) -> mlrun.lists.RunList:
        """Retrieve a list of runs.
        The default returns the runs from the last week, partitioned by name.
        To override the default, specify any filter.

        The returned result is a `` (list of dict), use `.to_objects()` to convert it to a list of RunObjects,
        `.show()` to view graphically in Jupyter, `.to_df()` to convert to a DataFrame, and `compare()` to
        generate comparison table and PCP plot.

        Example::

            # return a list of runs matching the name and label and compare
            runs = project.list_runs(name="download", labels="owner=admin")
            runs.compare()

            # multi-label filter can also be provided
            runs = project.list_runs(name="download", labels=["kind=job", "owner=admin"])

            # If running in Jupyter, can use the .show() function to display the results
            project.list_runs(name="").show()


        :param name: Name of the run to retrieve.
        :param uid: Unique ID of the run.
        :param labels: Filter runs by label key-value pairs or key existence. This can be provided as:

                       - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
                         or `{"label": None}` to check for key existence.
                       - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
                         or just `"label"` for key existence.
                       - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
                         the specified key-value pairs or key existence.

        :param state: Deprecated - List only runs whose state is specified.
        :param states: List only runs whose state is one of the provided states.
        :param sort: Whether to sort the result according to their start time. Otherwise, results will be
            returned by their internal order in the DB (order will not be guaranteed).
        :param last: Deprecated - currently not used (will be removed in 1.10.0).
        :param iter: If ``True`` return runs from all iterations. Otherwise, return only runs whose ``iter`` is 0.
        :param start_time_from: Filter by run start time in ``[start_time_from, start_time_to]``.
        :param start_time_to: Filter by run start time in ``[start_time_from, start_time_to]``.
        :param last_update_time_from: Filter by run last update time in ``(last_update_time_from,
            last_update_time_to)``.
        :param last_update_time_to: Filter by run last update time in ``(last_update_time_from, last_update_time_to)``.
        :param end_time_from: Filter by run end time in ``[end_time_from, end_time_to]``.
        :param end_time_to: Filter by run end time in ``[end_time_from, end_time_to]``.
        """
        if state:
            # TODO: Remove this in 1.10.0
            warnings.warn(
                "'state' is deprecated in 1.7.0 and will be removed in 1.10.0. Use 'states' instead.",
                FutureWarning,
            )

        db = mlrun.db.get_run_db(secrets=self._secrets)
        return db.list_runs(
            name,
            uid,
            self.metadata.name,
            labels=labels,
            states=(
                mlrun.utils.helpers.as_list(state)
                if state is not None
                else states or None
            ),
            sort=sort,
            iter=iter,
            start_time_from=start_time_from,
            start_time_to=start_time_to,
            last_update_time_from=last_update_time_from,
            last_update_time_to=last_update_time_to,
            end_time_from=end_time_from,
            end_time_to=end_time_to,
            **kwargs,
        )

    def paginated_list_runs(
        self,
        *args,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        **kwargs,
    ) -> tuple[mlrun.lists.RunList, Optional[str]]:
        """List runs with support for pagination and various filtering options.

        This method retrieves a paginated list of runs based on the specified filter parameters.
        Pagination is controlled using the `page`, `page_size`, and `page_token` parameters. The method
        will return a list of runs that match the filtering criteria provided.

        The returned result is a `` (list of dict), use `.to_objects()` to convert it to a list of RunObjects,
        `.show()` to view graphically in Jupyter, `.to_df()` to convert to a DataFrame, and `compare()` to
        generate comparison table and PCP plot.

        For detailed information about the parameters, refer to the list_runs method:
            See :py:func:`~list_runs` for more details.

        Examples::

            # Fetch first page of runs with page size of 5
            runs, token = project.paginated_list_runs(page_size=5)
            # Fetch next page using the pagination token from the previous response
            runs, token = project.paginated_list_runs(page_token=token)
            # Fetch runs for a specific page (e.g., page 3)
            runs, token = project.paginated_list_runs(page=3, page_size=5)

            # Automatically iterate over all pages without explicitly specifying the page number
            runs = []
            token = None
            while True:
                page_runs, token = project.paginated_list_runs(
                    page_token=token, page_size=5
                )
                runs.extend(page_runs)

                # If token is None and page_runs is empty, we've reached the end (no more runs).
                # If token is None and page_runs is not empty, we've fetched the last page of runs.
                if not token:
                    break
            print(f"Total runs retrieved: {len(runs)}")

        :param page: The page number to retrieve. If not provided, the next page will be retrieved.
        :param page_size: The number of items per page to retrieve. Up to `page_size` responses are expected.
            Defaults to `mlrun.mlconf.httpdb.pagination.default_page_size` if not provided.
        :param page_token: A pagination token used to retrieve the next page of results. Should not be provided
            for the first request.

        :returns: A tuple containing the list of runs and an optional `page_token` for pagination.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        return db.paginated_list_runs(
            *args,
            project=self.metadata.name,
            page=page,
            page_size=page_size,
            page_token=page_token,
            **kwargs,
        )

    def register_datastore_profile(self, profile: DatastoreProfile):
        private_body = DatastoreProfile2Json.get_json_private(profile)
        public_body = DatastoreProfile2Json.get_json_public(profile)
        # send project data to DB
        profile = mlrun.common.schemas.DatastoreProfile(
            name=profile.name,
            type=profile.type,
            object=public_body,
            private=private_body,
            project=self.name,
        )
        mlrun.db.get_run_db(secrets=self._secrets).store_datastore_profile(
            profile, self.name
        )

    def get_config_profile_attributes(self, name: str) -> dict:
        """
        Get the merged attributes from a named configuration profile.

        Retrieves a profile from the datastore using the provided name and returns its
        merged public and private attributes as a dictionary.

        Args:
            name (str): Name of the configuration profile to retrieve. Will be prefixed
                with "ds://" to form the full profile path.

        Returns:
            dict: The merged attributes dictionary containing both public and private
                configuration settings from the profile. Returns nested dictionaries if
                the profile contains nested configurations.
        """
        profile = datastore_profile_read(f"ds://{name}", self.name)
        return profile.attributes()

    def delete_datastore_profile(self, profile: str):
        mlrun.db.get_run_db(secrets=self._secrets).delete_datastore_profile(
            profile, self.name
        )

    def get_datastore_profile(self, profile: str) -> DatastoreProfile:
        return mlrun.db.get_run_db(secrets=self._secrets).get_datastore_profile(
            profile, self.name
        )

    def list_datastore_profiles(self) -> list[DatastoreProfile]:
        """
        Returns a list of datastore profiles associated with the project.
        The information excludes private details, showcasing only public data.
        """
        return mlrun.db.get_run_db(secrets=self._secrets).list_datastore_profiles(
            self.name
        )

    def get_custom_packagers(self) -> list[tuple[str, bool]]:
        """
        Get the custom packagers registered in the project.

        :return: A list of the custom packagers module paths.
        """
        # Return a copy so the user won't be able to edit the list by the reference returned (no need for deep copy as
        # tuples do not support item assignment):
        return self.spec.custom_packagers.copy()

    def add_custom_packager(self, packager: str, is_mandatory: bool):
        """
        Add a custom packager from the custom packagers list. All project's custom packagers are added to each project
        function.

        **Notice** that in order to run a function with the custom packagers included, you must set a source for the
        project (using the `project.set_source` method) with the parameter `pull_at_runtime=True` so the source code of
        the packagers will be able to be imported.

        :param packager:     The packager module path to add. For example, if a packager `MyPackager` is in the
                             project's source at my_module.py, then the module path is: "my_module.MyPackager".
        :param is_mandatory: Whether this packager must be collected during a run. If False, failing to collect it won't
                             raise an error during the packagers collection phase.
        """
        self.spec.add_custom_packager(packager=packager, is_mandatory=is_mandatory)

    def remove_custom_packager(self, packager: str):
        """
        Remove a custom packager from the custom packagers list.

        :param packager: The packager module path to remove.

        :raise MLRunInvalidArgumentError: In case the packager was not in the list.
        """
        self.spec.remove_custom_packager(packager=packager)

    def store_api_gateway(
        self,
        api_gateway: mlrun.runtimes.nuclio.api_gateway.APIGateway,
        wait_for_readiness=True,
        max_wait_time=90,
    ) -> mlrun.runtimes.nuclio.api_gateway.APIGateway:
        """
        Creates or updates a Nuclio API Gateway using the provided APIGateway object.

        This method interacts with the MLRun service to create/update a Nuclio API Gateway based on the provided
        APIGateway object. Once done, it returns the updated APIGateway object containing all fields propagated
        on MLRun and Nuclio sides, such as the 'host' attribute.
        Nuclio docs here: https://docs.nuclio.io/en/latest/reference/api-gateway/http.html

        :param api_gateway:        An instance of :py:class:`~mlrun.runtimes.nuclio.APIGateway` representing the
            configuration of the API Gateway to be created or updated.
        :param wait_for_readiness: (Optional) A boolean indicating whether to wait for the API Gateway to become
            ready after creation or update (default is True).
        :param max_wait_time:      (Optional) Maximum time to wait for API Gateway readiness in seconds (default is 90s)


        :returns: An instance of :py:class:`~mlrun.runtimes.nuclio.APIGateway` with all fields populated based on the
            information retrieved from the Nuclio API
        """

        api_gateway_json = mlrun.db.get_run_db().store_api_gateway(
            api_gateway=api_gateway,
            project=self.name,
        )

        if api_gateway_json:
            # fill in all the fields in the user's api_gateway object
            api_gateway = mlrun.runtimes.nuclio.api_gateway.APIGateway.from_scheme(
                api_gateway_json
            )
            if wait_for_readiness:
                api_gateway.wait_for_readiness(max_wait_time=max_wait_time)

        return api_gateway

    def list_api_gateways(self) -> list[mlrun.runtimes.nuclio.api_gateway.APIGateway]:
        """
        Retrieves a list of Nuclio API gateways associated with the project.

        :returns: List of :py:class:`~mlrun.runtimes.nuclio.api_gateway.APIGateway` objects representing
           the Nuclio API gateways associated with the project.
        """
        gateways_list = mlrun.db.get_run_db().list_api_gateways(self.name)
        return [
            mlrun.runtimes.nuclio.api_gateway.APIGateway.from_scheme(gateway_dict)
            for gateway_dict in gateways_list.api_gateways.values()
        ]

    def get_api_gateway(
        self,
        name: str,
    ) -> mlrun.runtimes.nuclio.api_gateway.APIGateway:
        """
        Retrieves an API gateway by name instance.

        :param name: The name of the API gateway to retrieve.

        Returns:
            mlrun.runtimes.nuclio.APIGateway: An instance of APIGateway.
        """

        gateway = mlrun.db.get_run_db().get_api_gateway(name=name, project=self.name)
        return mlrun.runtimes.nuclio.api_gateway.APIGateway.from_scheme(gateway)

    def delete_api_gateway(
        self,
        name: str,
    ):
        """
        Deletes an API gateway by name.

        :param name: The name of the API gateway to delete.
        """

        mlrun.db.get_run_db().delete_api_gateway(name=name, project=self.name)

    def store_alert_config(
        self,
        alert_data: AlertConfig,
        alert_name: typing.Optional[str] = None,
        force_reset: bool = False,
    ) -> AlertConfig:
        """
        Create/modify an alert.

        :param alert_data: The data of the alert.
        :param alert_name: The name of the alert.
        :param force_reset: If True and the alert already exists, the alert would be reset.
        :return: the created/modified alert.
        """
        if not alert_data:
            raise mlrun.errors.MLRunInvalidArgumentError("Alert data must be provided")

        db = mlrun.db.get_run_db(secrets=self._secrets)
        alert_name = alert_name or alert_data.name
        if alert_data.project is not None and alert_data.project != self.metadata.name:
            logger.warn(
                "Project in alert does not match project in operation",
                project=alert_data.project,
            )
        alert_data.project = self.metadata.name
        return db.store_alert_config(
            alert_name, alert_data, project=self.metadata.name, force_reset=force_reset
        )

    def get_alert_config(self, alert_name: str) -> AlertConfig:
        """
        Retrieve an alert.

        :param alert_name: The name of the alert to retrieve.
        :return: The alert object.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        return db.get_alert_config(alert_name, self.metadata.name)

    def list_alerts_configs(
        self, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> list[AlertConfig]:
        """
        Retrieve list of alerts of a project.

        :param limit: The maximum number of alerts to return.
            Defaults to `mlconf.alerts.default_list_alert_configs_limit` if not provided.
        :param offset: The number of alerts to skip before starting to collect alerts.

        :return: All the alerts objects of the project.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        return db.list_alerts_configs(self.metadata.name, limit=limit, offset=offset)

    def delete_alert_config(
        self, alert_data: AlertConfig = None, alert_name: Optional[str] = None
    ):
        """
        Delete an alert.

        :param alert_data: The data of the alert.
        :param alert_name: The name of the alert to delete.
        """
        if alert_data is None and alert_name is None:
            raise ValueError(
                "At least one of alert_data or alert_name must be provided"
            )
        if alert_data and alert_name and alert_data.name != alert_name:
            raise ValueError("Alert_data name does not match the provided alert_name")
        db = mlrun.db.get_run_db(secrets=self._secrets)
        if alert_data:
            alert_name = alert_data.name
        db.delete_alert_config(alert_name, self.metadata.name)

    def reset_alert_config(
        self, alert_data: AlertConfig = None, alert_name: Optional[str] = None
    ):
        """
        Reset an alert.

        :param alert_data: The data of the alert.
        :param alert_name: The name of the alert to reset.
        """
        if alert_data is None and alert_name is None:
            raise ValueError(
                "At least one of alert_data or alert_name must be provided"
            )
        if alert_data and alert_name and alert_data.name != alert_name:
            raise ValueError("Alert_data name does not match the provided alert_name")
        db = mlrun.db.get_run_db(secrets=self._secrets)
        if alert_data:
            alert_name = alert_data.name
        db.reset_alert_config(alert_name, self.metadata.name)

    def get_alert_template(
        self, template_name: str
    ) -> mlrun.common.schemas.alert.AlertTemplate:
        """
        Retrieve a specific alert template.

        :param template_name: The name of the template to retrieve.
        :return: The template object.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        return db.get_alert_template(template_name)

    def list_alert_templates(self) -> list[mlrun.common.schemas.alert.AlertTemplate]:
        """
        Retrieve list of all alert templates.

        :return: All the alert template objects in the database.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        return db.list_alert_templates()

    def list_alert_activations(
        self,
        name: Optional[str] = None,
        since: Optional[datetime.datetime] = None,
        until: Optional[datetime.datetime] = None,
        entity: Optional[str] = None,
        severity: Optional[
            list[Union[mlrun.common.schemas.alert.AlertSeverity, str]]
        ] = None,
        entity_kind: Optional[
            Union[mlrun.common.schemas.alert.EventEntityKind, str]
        ] = None,
        event_kind: Optional[Union[mlrun.common.schemas.alert.EventKind, str]] = None,
    ) -> list[mlrun.common.schemas.alert.AlertActivation]:
        """
        Retrieve a list of alert activations for a project.

        :param name: The alert name to filter by. Supports exact matching or partial matching if prefixed with `~`.
        :param since: Filters for alert activations occurring after this timestamp.
        :param until: Filters for alert activations occurring before this timestamp.
        :param entity: The entity ID to filter by. Supports wildcard matching if prefixed with `~`.
        :param severity: A list of severity levels to filter by (e.g., ["high", "low"]).
        :param entity_kind: The kind of entity (e.g., "job", "endpoint") to filter by.
        :param event_kind: The kind of event (e.g., ""data-drift-detected"", "failed") to filter by.

        :returns: A list of alert activations matching the provided filters.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        return db.list_alert_activations(
            project=self.metadata.name,
            name=name,
            since=since,
            until=until,
            entity=entity,
            severity=severity,
            entity_kind=entity_kind,
            event_kind=event_kind,
        )

    def paginated_list_alert_activations(
        self,
        *args,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        **kwargs,
    ) -> tuple[mlrun.common.schemas.alert.AlertActivation, Optional[str]]:
        """
        List alerts activations with support for pagination and various filtering options.

        This method retrieves a paginated list of alert activations based on the specified filter parameters.
        Pagination is controlled using the `page`, `page_size`, and `page_token` parameters. The method
        will return a list of alert activations that match the filtering criteria provided.

        For detailed information about the parameters, refer to the list_alert_activations method:
            See :py:func:`~list_alert_activations` for more details.

        Examples::

            # Fetch first page of alert activations with page size of 5
            alert_activations, token = project.paginated_list_alert_activations(page_size=5)
            # Fetch next page using the pagination token from the previous response
            alert_activations, token = project.paginated_list_alert_activations(
                page_token=token
            )
            # Fetch alert activations for a specific page (e.g., page 3)
            alert_activations, token = project.paginated_list_alert_activations(
                page=3, page_size=5
            )

            # Automatically iterate over all pages without explicitly specifying the page number
            alert_activations = []
            token = None
            while True:
                page_alert_activations, token = project.paginated_list_alert_activations(
                    page_token=token, page_size=5
                )
                alert_activations.extend(page_alert_activations)

                # If token is None and page_alert_activations is empty, we've reached the end (no more activations).
                # If token is None and page_alert_activations is not empty, we've fetched the last page of activations.
                if not token:
                    break
            print(f"Total alert activations retrieved: {len(alert_activations)}")

        :param page: The page number to retrieve. If not provided, the next page will be retrieved.
        :param page_size: The number of items per page to retrieve. Up to `page_size` responses are expected.
            Defaults to `mlrun.mlconf.httpdb.pagination.default_page_size` if not provided.
        :param page_token: A pagination token used to retrieve the next page of results. Should not be provided
            for the first request.

        :returns: A tuple containing the list of alert activations and an optional `page_token` for pagination.
        """
        db = mlrun.db.get_run_db(secrets=self._secrets)
        return db.paginated_list_alert_activations(
            *args,
            project=self.metadata.name,
            page=page,
            page_size=page_size,
            page_token=page_token,
            **kwargs,
        )

    def _run_authenticated_git_action(
        self,
        action: Callable,
        remote: str,
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
        secrets: Union[SecretsStore, dict] = None,
    ):
        """Run an arbitrary Git routine while the remote is enriched with secrets
        Enrichment of the remote URL is undone before this method returns
        If no secrets are provided, remote remains untouched

        :param action:  git callback that may require authentication
        :param remote:  git remote to be temporarily enriched with secrets
        :param args:    positional arguments to be passed along to action
        :param kwargs:  keyword arguments to be passed along to action
        :param secrets: dict or SecretsStore with Git credentials e.g. secrets={"GIT_TOKEN": token}
        """
        clean_remote = self.spec.repo.remotes[remote].url
        enriched_remote, is_remote_enriched = add_credentials_git_remote_url(
            clean_remote, secrets=secrets or {}
        )
        try:
            if is_remote_enriched:
                self.spec.repo.remotes[remote].set_url(enriched_remote, clean_remote)
            args = args or []
            kwargs = kwargs or {}
            action(*args, **kwargs)
        except RuntimeError as e:
            raise mlrun.errors.MLRunRuntimeError(
                f"Failed to run Git action: {action}"
            ) from e
        finally:
            if is_remote_enriched:
                self.spec.repo.remotes[remote].set_url(clean_remote, enriched_remote)

    def _validate_file_path(self, file_path: str, param_name: str, engine: str):
        """
        The function checks if the given file_path is a valid path.
        If the file_path is a relative path, it is completed by joining it with the self.spec.get_code_path()
        Otherwise, the file_path is used as is.
        Additionally, it checks if the resulting path exists as a file, unless the file_path is a remote URL.
        If the file_path has no suffix, it is considered invalid.

        :param file_path:   The file path to be validated
        :param param_name:  The name of the parameter that holds the file path
        """
        if not file_path:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"{param_name} must be provided."
            )

        # If file path is remote, verify it is a file URL
        if "://" in file_path:
            if pathlib.Path(file_path).suffix:
                return

            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Invalid '{param_name}': '{file_path}'. Got a remote URL without a file suffix."
            )

        # if engine is remote then skip the local file validation
        if engine and engine.startswith("remote"):
            return

        code_path = self.spec.get_code_path()

        # If the file path is a relative path, it is completed by joining it with the code_path.
        code_path_relative = not path.isabs(file_path) and not file_path.startswith(
            code_path
        )
        if code_path_relative:
            abs_path = path.join(code_path, file_path.lstrip("./"))
        else:
            abs_path = file_path

        if not path.isfile(abs_path):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Invalid '{param_name}': '{file_path}'. Got a path to a non-existing file. "
                f"Path must be absolute or relative to the project code path i.e. "
                f"<project.spec.get_code_path()>/<{param_name}>)."
            )

    def _resolve_artifact_producer(
        self,
        artifact: typing.Union[str, Artifact],
        project_producer_tag: Optional[str] = None,
    ) -> tuple[ArtifactProducer, bool]:
        """
        Resolve the artifact producer of the given artifact.
        If the artifact's producer is a run, the artifact is registered with the original producer.
        Otherwise, the artifact is registered with the current project as the producer.

        :param artifact:                The artifact to resolve its producer.
        :param project_producer_tag:    The tag to use for the project as the producer. If not provided, a tag will be
        generated for the project.
        :return:                        A tuple of the resolved producer and whether it is retained or not.
        """

        if not isinstance(artifact, str) and artifact.spec.producer:
            # if the artifact was imported from a yaml file, the producer can be a dict
            if isinstance(artifact.spec.producer, ArtifactProducer):
                producer_dict = artifact.spec.producer.get_meta()
            else:
                producer_dict = artifact.spec.producer

            producer_tag = producer_dict.get("tag", None)
            producer_project = producer_dict.get("project", None)
            if not producer_tag or not producer_project:
                # try resolving the producer tag from the uri
                producer_uri = artifact.spec.producer.get("uri", "")
                producer_project, producer_tag, _ = ArtifactProducer.parse_uri(
                    producer_uri
                )

            if producer_dict.get("kind", "") == "run":
                return (
                    ArtifactProducer(
                        name=producer_dict.get("name", ""),
                        kind=producer_dict.get("kind", ""),
                        project=producer_project,
                        tag=producer_tag,
                        owner=producer_dict.get("owner", ""),
                    ),
                    True,
                )

        # do not retain the artifact's producer, replace it with the project as the producer
        project_producer_tag = project_producer_tag or self._get_project_tag()
        return (
            ArtifactProducer(
                kind="project",
                name=self.metadata.name,
                project=self.metadata.name,
                tag=project_producer_tag,
                owner=self._resolve_artifact_owner(),
            ),
            False,
        )

    def _resolve_existing_artifact(
        self,
        item: typing.Union[str, Artifact],
        tag: Optional[str] = None,
    ) -> typing.Optional[Artifact]:
        """
        Check if there is and existing artifact with the given item and tag.
        If there is, return the existing artifact. Otherwise, return None.

        :param item:    The item (or key) to check if there is an existing artifact for.
        :param tag:     The tag to check if there is an existing artifact for.
        :return:        The existing artifact if there is one, otherwise None.
        """
        try:
            if isinstance(item, str):
                existing_artifact = self.get_artifact(key=item, tag=tag)
            else:
                existing_artifact = self.get_artifact(
                    key=item.key,
                    tag=item.tag,
                    iter=item.iter,
                    tree=item.tree,
                )
            if existing_artifact is not None:
                return existing_artifact.from_dict(existing_artifact)
        except mlrun.errors.MLRunNotFoundError:
            logger.debug(
                "No existing artifact was found",
                key=item if isinstance(item, str) else item.key,
                tag=tag if isinstance(item, str) else item.tag,
                tree=None if isinstance(item, str) else item.tree,
            )
            return None

    def _get_project_tag(self):
        return self._get_hexsha() or str(uuid.uuid4())

    def _resolve_artifact_owner(self):
        return os.getenv("V3IO_USERNAME") or self.spec.owner


def _set_as_current_default_project(project: MlrunProject):
    mlrun.mlconf.default_project = project.metadata.name
    pipeline_context.set(project)


def _init_function_from_dict(
    f: dict,
    project: MlrunProject,
    name: typing.Optional[str] = None,
) -> tuple[str, mlrun.runtimes.BaseRuntime]:
    name = name or f.get("name", "")
    url = f.get("url", "")
    kind = f.get("kind", "")
    image = f.get("image", None)
    handler = f.get("handler", None)
    with_repo = f.get("with_repo", False)
    requirements = f.get("requirements", None)
    requirements_file = f.get("requirements_file", None)
    tag = f.get("tag", None)

    has_module = _has_module(handler, kind)
    relative_url = url
    url, in_context = project.get_item_absolute_path(url)

    if "spec" in f:
        if "spec" in f["spec"]:
            # Functions are stored in the project yaml as a dict with a spec key where the spec is the function
            func = new_function(name, runtime=f["spec"])
        else:
            func = new_function(name, runtime=f, tag=tag)

    elif not url and has_module:
        func = new_function(
            name, image=image, kind=kind or "job", handler=handler, tag=tag
        )

    elif is_yaml_path(url) or url.startswith("db://") or url.startswith("hub://"):
        func = import_function(url, new_name=name)
        if image:
            func.spec.image = image
        if tag:
            func.spec.tag = tag

    elif url.endswith(".ipynb"):
        # not defaulting kind to job here cause kind might come from magic annotations in the notebook
        func = code_to_function(
            name, filename=url, image=image, kind=kind, handler=handler, tag=tag
        )

    elif url.endswith(".py"):
        if in_context and with_repo:
            # when load_source_on_run is used we allow not providing image as code will be loaded pre-run. ML-4994
            if (
                not image
                and not project.default_image
                and kind != "local"
                and not project.spec.load_source_on_run
            ):
                raise ValueError(
                    "image must be provided with py code files which do not run on 'local' engine kind"
                )
            func = new_function(
                name,
                command=relative_url,
                image=image,
                kind=kind or "job",
                handler=handler,
                tag=tag,
            )
        else:
            func = code_to_function(
                name,
                filename=url,
                image=image,
                kind=kind or "job",
                handler=handler,
                tag=tag,
            )

    elif kind in mlrun.runtimes.RuntimeKinds.nuclio_runtimes():
        func = new_function(
            name,
            image=image,
            kind=kind,
            handler=handler,
            tag=tag,
        )
        if image and kind != mlrun.runtimes.RuntimeKinds.application:
            logger.info("Function code not specified, setting entry point to image")
            func.from_image(image)
    else:
        raise ValueError(f"Unsupported function url:handler {url}:{handler} or no spec")

    if with_repo:
        # mark source to be enriched before run with project source (enrich_function_object)
        func.spec.build.source = "./"
    if requirements or requirements_file:
        func.with_requirements(
            requirements=requirements,
            requirements_file=requirements_file,
            overwrite=True,
        )

    return _init_function_from_obj(func, project)


def _init_function_from_obj(
    func: mlrun.runtimes.BaseRuntime,
    project: MlrunProject,
    name: typing.Optional[str] = None,
) -> tuple[str, mlrun.runtimes.BaseRuntime]:
    build = func.spec.build
    if project.spec.origin_url:
        origin = project.spec.origin_url
        try:
            if project.spec.repo:
                origin += "#" + project.spec.repo.head.commit.hexsha
        except Exception:
            pass
        build.code_origin = origin
    if project.metadata.name:
        func.metadata.project = project.metadata.name

    # TODO: deprecate project tag
    if project.spec.tag:
        func.metadata.tag = project.spec.tag

    if name:
        func.metadata.name = name
    return func.metadata.name, func


def _has_module(handler, kind):
    if not handler:
        return False

    if (
        kind in mlrun.runtimes.RuntimeKinds.pure_nuclio_deployed_runtimes()
        and ":" in handler
    ):
        return True

    if "." in handler:
        return True

    return False


def _is_imported_artifact(artifact):
    return artifact and isinstance(artifact, dict) and "import_from" in artifact
