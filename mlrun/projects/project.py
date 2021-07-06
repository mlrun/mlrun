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
import getpass
import importlib.util as imputil
import pathlib
import shutil
import typing
import warnings
from os import environ, path, remove
from tempfile import mktemp

import yaml
from git import Repo
from kfp import compiler

import mlrun.api.schemas
import mlrun.errors
import mlrun.utils.regex

from ..artifacts import (
    ArtifactManager,
    ArtifactProducer,
    DatasetArtifact,
    ModelArtifact,
    dict_to_artifact,
)
from ..config import config
from ..datastore import store_manager
from ..db import get_run_db
from ..features import Feature
from ..model import ModelObj
from ..run import (
    code_to_function,
    get_object,
    import_function,
    new_function,
    run_pipeline,
    wait_for_pipeline_completion,
)
from ..runtimes.utils import add_code_metadata
from ..secrets import SecretsStore
from ..utils import RunNotifications, logger, new_pipe_meta, update_in
from ..utils.clones import clone_git, clone_tgz, clone_zip, get_repo_url


class ProjectError(Exception):
    pass


def new_project(name, context=None, init_git=False, user_project=False):
    """Create a new MLRun project

    :param name:         project name
    :param context:      project local directory path
    :param init_git:     if True, will git init the context dir
    :param user_project: add the current user name to the provided project name (making it unique per user)

    :returns: project object
    """
    if user_project:
        user = environ.get("V3IO_USERNAME") or getpass.getuser()
        name = f"{name}-{user}"

    project = MlrunProject(name=name)
    project.spec.context = context

    if init_git:
        repo = Repo.init(context)
        project.spec.repo = repo

    return project


def load_project(
    context,
    url=None,
    name=None,
    secrets=None,
    init_git=False,
    subpath="",
    clone=False,
    user_project=False,
):
    """Load an MLRun project from git or tar or dir

    :param context:      project local directory path
    :param url:          git or tar.gz sources archive path e.g.:
                         git://github.com/mlrun/demo-xgb-project.git
                         db://<project-name>
    :param name:         project name
    :param secrets:      key:secret dict or SecretsStore used to download sources
    :param init_git:     if True, will git init the context dir
    :param subpath:      project subpath (within the archive)
    :param clone:        if True, always clone (delete any existing content)
    :param user_project: add the current user name to the project name (for db:// prefixes)

    :returns: project object
    """

    secrets = secrets or {}
    repo = None
    project = None
    if url:
        if url.endswith(".yaml"):
            project = _load_project_file(url, name, secrets)
        elif url.startswith("git://"):
            url, repo = clone_git(url, context, secrets, clone)
        elif url.endswith(".tar.gz"):
            clone_tgz(url, context, secrets)
        elif url.endswith(".zip"):
            clone_zip(url, context, secrets)
        elif url.startswith("db://"):
            if user_project:
                user = environ.get("V3IO_USERNAME") or getpass.getuser()
                url = f"{url}-{user}"
            project = _load_project_from_db(url, secrets)
        else:
            raise ValueError(f"unsupported code archive {url}")

    else:
        if not path.isdir(context):
            raise ValueError(f"context {context} is not an existing dir path")
        try:
            repo = Repo(context)
            url = get_repo_url(repo)
        except Exception:
            if init_git:
                repo = Repo.init(context)

    if not project:
        project = _load_project_dir(context, name, subpath)
    project.spec.source = url
    project.spec.repo = repo
    if repo:
        project.spec.branch = repo.active_branch.name
    project.spec.origin_url = url
    project.register_artifacts()
    return project


def _load_project_dir(context, name="", subpath=""):
    fpath = path.join(context, subpath, "project.yaml")
    if path.isfile(fpath):
        with open(fpath) as fp:
            data = fp.read()
            struct = yaml.load(data, Loader=yaml.FullLoader)
            project = _project_instance_from_struct(struct, name)
            project.spec.context = context

    elif path.isfile(path.join(context, subpath, "function.yaml")):
        func = import_function(path.join(context, subpath, "function.yaml"))
        project = MlrunProject(
            name=func.metadata.project,
            functions=[{"url": "function.yaml", "name": func.metadata.name}],
        )
    else:
        raise ValueError("project or function YAML not found in path")

    project.spec.context = context
    project.metadata.name = name or project.metadata.name
    project.spec.subpath = subpath
    return project


def _load_project_from_db(url, secrets):
    db = get_run_db(secrets=secrets)
    project_name = url.replace("db://", "")
    return db.get_project(project_name)


def _load_project_file(url, name="", secrets=None):
    try:
        obj = get_object(url, secrets)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"cant find project file at {url}, {exc}")
    struct = yaml.load(obj, Loader=yaml.FullLoader)
    return _project_instance_from_struct(struct, name)


def _project_instance_from_struct(struct, name):
    # Name is in the root level only in the legacy project structure
    if "name" in struct:
        legacy_project = MlrunProjectLegacy.from_dict(struct)
        project = MlrunProject(
            legacy_project.name,
            legacy_project.description,
            legacy_project.params,
            [],
            legacy_project.workflows,
            legacy_project.artifacts,
            legacy_project.artifact_path,
            legacy_project.conda,
        )
        # other attributes that not passed on initialization
        project._initialized = legacy_project._initialized
        project._secrets = legacy_project._secrets
        project._artifact_manager = legacy_project._artifact_mngr

        project.spec.source = legacy_project.source
        project.spec.context = legacy_project.context
        project.spec.mountdir = legacy_project.mountdir
        project.spec.subpath = legacy_project.subpath
        project.spec.origin_url = legacy_project.origin_url
        project.spec.branch = legacy_project.branch
        project.spec.tag = legacy_project.tag
        project.spec._function_definitions = legacy_project._function_defs
        project.spec._function_objects = legacy_project._function_objects
        project.spec.functions = legacy_project.functions
    else:
        struct.setdefault("metadata", {})["name"] = name or struct.get(
            "metadata", {}
        ).get("name", "")
        project = MlrunProject.from_dict(struct)
    return project


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
        desired_state=mlrun.api.schemas.ProjectState.online.value,
    ):
        self.repo = None

        self.description = description
        self.context = ""
        self._mountdir = None
        self._source = None
        self.source = source or ""
        self.subpath = subpath or ""
        self.origin_url = origin_url or ""
        self.goals = goals
        self.desired_state = desired_state
        self.branch = None
        self.tag = ""
        self.params = params or {}
        self.conda = conda or {}
        self.artifact_path = artifact_path or config.artifact_path
        self._artifacts = {}
        self.artifacts = artifacts or []

        self._workflows = {}
        self.workflows = workflows or []

        self._function_objects = {}
        self._function_definitions = {}
        self.functions = functions or []

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
                if (
                    function.spec.build.source
                    and function.spec.build.source.startswith(self._source_repo())
                ):
                    update_in(spec, "spec.build.source", "./")
                functions.append({"name": name, "spec": spec})
            else:
                functions.append(function)
        return functions

    @functions.setter
    def functions(self, functions):
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
    def workflows(self) -> list:
        """list of workflows specs used in this project"""
        return [workflow for workflow in self._workflows.values()]

    @workflows.setter
    def workflows(self, workflows):
        if not isinstance(workflows, list):
            raise ValueError("workflows must be a list")

        workflows_dict = {}
        for workflow in workflows:
            if not isinstance(workflow, dict):
                raise ValueError("workflow must be a dict")
            name = workflow.get("name", "")
            # todo: support steps dsl as code alternative
            if not name:
                raise ValueError('workflow "name" must be specified')
            if "path" not in workflow and "code" not in workflow:
                raise ValueError('workflow source "path" or "code" must be specified')
            workflows_dict[name] = workflow

        self._workflows = workflows_dict

    def set_workflow(self, name, workflow):
        self._workflows[name] = workflow

    def remove_workflow(self, name):
        if name in self._workflows:
            del self._workflows[name]

    @property
    def artifacts(self) -> list:
        """list of artifacts used in this project"""
        return [artifact for artifact in self._artifacts.values()]

    @artifacts.setter
    def artifacts(self, artifacts):
        if not isinstance(artifacts, list):
            raise ValueError("artifacts must be a list")

        artifacts_dict = {}
        for artifact in artifacts:
            if not isinstance(artifact, dict) and not hasattr(artifact, "to_dict"):
                raise ValueError("artifacts must be a dict or class")
            if isinstance(artifact, dict):
                key = artifact.get("key", "")
                if not key:
                    raise ValueError('artifacts "key" must be specified')
            else:
                key = artifact.key
                artifact = artifact.to_dict()

            artifacts_dict[key] = artifact

        self._artifacts = artifacts_dict

    def set_artifact(self, key, artifact):
        self._artifacts[key] = artifact

    def remove_artifact(self, key):
        if key in self._artifacts:
            del self._artifacts[key]

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

    def _get_wf_cfg(self, name, arguments=None):
        workflow = self._workflows.get(name)
        code = workflow.get("code")
        if code:
            workflow_path = mktemp(".py")
            with open(workflow_path, "w") as wf:
                wf.write(code)
        else:
            workflow_path = workflow.get("path", "")
            if self.context and not workflow_path.startswith("/"):
                workflow_path = path.join(self.context, workflow_path)

        wf_args = workflow.get("args", {})
        if arguments:
            for k, v in arguments.items():
                wf_args[k] = v

        return workflow_path, code, wf_args


class ProjectStatus(ModelObj):
    def __init__(self, state=None):
        self.state = state


class MlrunProject(ModelObj):
    kind = "project"
    _dict_fields = ["kind", "metadata", "spec", "status"]

    def __init__(
        self,
        name=None,
        description=None,
        params=None,
        functions=None,
        workflows=None,
        artifacts=None,
        artifact_path=None,
        conda=None,
        # all except these 2 are for backwards compatibility with MlrunProjectLegacy
        metadata=None,
        spec=None,
    ):
        self._metadata = None
        self.metadata = metadata
        self._spec = None
        self.spec = spec
        self._status = None
        self.status = None

        # Handling the fields given in the legacy way
        self.metadata.name = name or self.metadata.name
        self.spec.description = description or self.spec.description
        self.spec.params = params or self.spec.params
        self.spec.functions = functions or self.spec.functions
        self.spec.workflows = workflows or self.spec.workflows
        self.spec.artifacts = artifacts or self.spec.artifacts
        self.spec.artifact_path = artifact_path or self.spec.artifact_path
        self.spec.conda = conda or self.spec.conda

        self._initialized = False
        self._secrets = SecretsStore()
        self._artifact_manager = None

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
    def name(self) -> str:
        """This is a property of the metadata, look there for documentation
        leaving here for backwards compatibility with users code that used MlrunProjectLegacy"""
        warnings.warn(
            "This is a property of the metadata, use project.metadata.name instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        return self.metadata.name

    @name.setter
    def name(self, name):
        warnings.warn(
            "This is a property of the metadata, use project.metadata.name instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        self.metadata.name = name

    @property
    def artifact_path(self) -> str:
        """This is a property of the spec, look there for documentation
        leaving here for backwards compatibility with users code that used MlrunProjectLegacy"""
        warnings.warn(
            "This is a property of the spec, use project.spec.artifact_path instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        return self.spec.artifact_path

    @artifact_path.setter
    def artifact_path(self, artifact_path):
        warnings.warn(
            "This is a property of the spec, use project.spec.artifact_path instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        self.spec.artifact_path = artifact_path

    @property
    def source(self) -> str:
        """This is a property of the spec, look there for documentation
        leaving here for backwards compatibility with users code that used MlrunProjectLegacy"""
        warnings.warn(
            "This is a property of the spec, use project.spec.source instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        return self.spec.source

    @source.setter
    def source(self, source):
        warnings.warn(
            "This is a property of the spec, use project.spec.source instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        self.spec.source = source

    @property
    def context(self) -> str:
        """This is a property of the spec, look there for documentation
        leaving here for backwards compatibility with users code that used MlrunProjectLegacy"""
        warnings.warn(
            "This is a property of the spec, use project.spec.context instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        return self.spec.context

    @context.setter
    def context(self, context):
        warnings.warn(
            "This is a property of the spec, use project.spec.context instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        self.spec.context = context

    @property
    def mountdir(self) -> str:
        """This is a property of the spec, look there for documentation
        leaving here for backwards compatibility with users code that used MlrunProjectLegacy"""
        warnings.warn(
            "This is a property of the spec, use project.spec.mountdir instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        return self.spec.mountdir

    @mountdir.setter
    def mountdir(self, mountdir):
        warnings.warn(
            "This is a property of the spec, use project.spec.mountdir instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        self.spec.mountdir = mountdir

    @property
    def params(self) -> str:
        """This is a property of the spec, look there for documentation
        leaving here for backwards compatibility with users code that used MlrunProjectLegacy"""
        warnings.warn(
            "This is a property of the spec, use project.spec.params instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        return self.spec.params

    @params.setter
    def params(self, params):
        warnings.warn(
            "This is a property of the spec, use project.spec.params instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        self.spec.params = params

    @property
    def description(self) -> str:
        """This is a property of the spec, look there for documentation
        leaving here for backwards compatibility with users code that used MlrunProjectLegacy"""
        warnings.warn(
            "This is a property of the spec, use project.spec.description instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        return self.spec.description

    @description.setter
    def description(self, description):
        warnings.warn(
            "This is a property of the spec, use project.spec.description instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        self.spec.description = description

    @property
    def functions(self) -> list:
        """This is a property of the spec, look there for documentation
        leaving here for backwards compatibility with users code that used MlrunProjectLegacy"""
        warnings.warn(
            "This is a property of the spec, use project.spec.functions instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        return self.spec.functions

    @functions.setter
    def functions(self, functions):
        warnings.warn(
            "This is a property of the spec, use project.spec.functions instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        self.spec.functions = functions

    @property
    def workflows(self) -> list:
        """This is a property of the spec, look there for documentation
        leaving here for backwards compatibility with users code that used MlrunProjectLegacy"""
        warnings.warn(
            "This is a property of the spec, use project.spec.workflows instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        return self.spec.workflows

    @workflows.setter
    def workflows(self, workflows):
        warnings.warn(
            "This is a property of the spec, use project.spec.workflows instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        self.spec.workflows = workflows

    def set_workflow(self, name, workflow_path: str, embed=False, **args):
        """add or update a workflow, specify a name and the code path"""
        if not workflow_path:
            raise ValueError("valid workflow_path must be specified")
        if embed:
            if self.spec.context and not workflow_path.startswith("/"):
                workflow_path = path.join(self.spec.context, workflow_path)
            with open(workflow_path, "r") as fp:
                txt = fp.read()
            workflow = {"name": name, "code": txt}
        else:
            workflow = {"name": name, "path": workflow_path}
        if args:
            workflow["args"] = args
        self.spec.set_workflow(name, workflow)

    @property
    def artifacts(self) -> list:
        """This is a property of the spec, look there for documentation
        leaving here for backwards compatibility with users code that used MlrunProjectLegacy"""
        warnings.warn(
            "This is a property of the spec, use project.spec.artifacts instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        return self.spec.artifacts

    @artifacts.setter
    def artifacts(self, artifacts):
        warnings.warn(
            "This is a property of the spec, use project.spec.artifacts instead"
            "This will be deprecated in 0.7.0, and will be removed in 0.9.0",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        self.spec.artifacts = artifacts

    def register_artifacts(self):
        """register the artifacts in the MLRun DB (under this project)"""
        artifact_manager = self._get_artifact_manager()
        producer = ArtifactProducer(
            "project",
            self.metadata.name,
            self.metadata.name,
            tag=self._get_hexsha() or "latest",
        )
        for artifact_dict in self.spec.artifacts:
            artifact = dict_to_artifact(artifact_dict)
            artifact_manager.log_artifact(producer, artifact, upload=False)

    def _get_artifact_manager(self):
        if self._artifact_manager:
            return self._artifact_manager
        db = get_run_db(secrets=self._secrets)
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

    def log_artifact(
        self,
        item,
        body=None,
        tag="",
        local_path="",
        artifact_path=None,
        format=None,
        upload=True,
        labels=None,
        target_path=None,
    ):
        am = self._get_artifact_manager()
        artifact_path = artifact_path or self.spec.artifact_path
        artifact_path = mlrun.utils.helpers.fill_artifact_path_template(
            artifact_path, self.metadata.name
        )
        producer = ArtifactProducer(
            "project",
            self.metadata.name,
            self.metadata.name,
            tag=self._get_hexsha() or "latest",
        )
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
        )
        self.spec.set_artifact(item.key, item.base_dict())
        return item

    def log_dataset(
        self,
        key,
        df,
        tag="",
        local_path=None,
        artifact_path=None,
        upload=True,
        labels=None,
        format="",
        preview=None,
        stats=False,
        target_path="",
        extra_data=None,
        **kwargs,
    ):
        """
        log a dataset artifact and optionally upload it to datastore

        example::

            raw_data = {
                "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
                "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
                "age": [42, 52, 36, 24, 73],
                "testScore": [25, 94, 57, 62, 70],
            }
            df = pd.DataFrame(raw_data, columns=["first_name", "last_name", "age", "testScore"])
            context.log_dataset("mydf", df=df, stats=True)

        :param key:           artifact key
        :param df:            dataframe object
        :param local_path:    path to the local file we upload, will also be use
                              as the destination subpath (under "artifact_path")
        :param artifact_path: target artifact path (when not using the default)
                              to define a subpath under the default location use:
                              `artifact_path=context.artifact_subpath('data')`
        :param tag:           version tag
        :param format:        optional, format to use (e.g. csv, parquet, ..)
        :param target_path:   absolute target path (instead of using artifact_path + local_path)
        :param preview:       number of lines to store as preview in the artifact metadata
        :param stats:         calculate and store dataset stats in the artifact metadata
        :param extra_data:    key/value list of extra files/charts to link with this dataset
        :param upload:        upload to datastore (default is True)
        :param labels:        a set of key/value labels to tag the artifact with

        :returns: artifact object
        """
        ds = DatasetArtifact(
            key,
            df,
            preview=preview,
            extra_data=extra_data,
            format=format,
            stats=stats,
            **kwargs,
        )

        item = self.log_artifact(
            ds,
            local_path=local_path,
            artifact_path=artifact_path or self.spec.artifact_path,
            target_path=target_path,
            tag=tag,
            upload=upload,
            labels=labels,
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
        upload=True,
        labels=None,
        inputs: typing.List[Feature] = None,
        outputs: typing.List[Feature] = None,
        feature_vector: str = None,
        feature_weights: list = None,
        training_set=None,
        label_column=None,
        extra_data=None,
    ):
        """log a model artifact and optionally upload it to datastore

        example::

            context.log_model("model", body=dumps(model),
                              model_file="model.pkl",
                              metrics=context.results,
                              training_set=training_df,
                              label_column='label',
                              feature_vector=feature_vector_uri,
                              labels={"app": "fraud"})

        :param key:             artifact key or artifact class ()
        :param body:            will use the body as the artifact content
        :param model_file:      path to the local model file we upload (see also model_dir)
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
        :param upload:          upload to datastore (default is True)
        :param labels:          a set of key/value labels to tag the artifact with
        :param feature_vector:  feature store feature vector uri (store://feature-vectors/<project>/<name>[:tag])
        :param feature_weights: list of feature weights, one per input column
        :param training_set:    training set dataframe, used to infer inputs & outputs
        :param label_column:    which columns in the training set are the label (target) columns
        :param extra_data:      key/value list of extra files/charts to link with this dataset
                                value can be abs/relative path string | bytes | artifact object

        :returns: artifact object
        """

        if training_set is not None and inputs:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "cannot specify inputs and training set together"
            )

        model = ModelArtifact(
            key,
            body,
            model_file=model_file,
            metrics=metrics,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            framework=framework,
            algorithm=algorithm,
            feature_vector=feature_vector,
            feature_weights=feature_weights,
            extra_data=extra_data,
        )
        if training_set is not None:
            model.infer_from_df(training_set, label_column)

        item = self.log_artifact(
            model,
            local_path=model_dir,
            artifact_path=artifact_path or self.spec.artifact_path,
            tag=tag,
            upload=upload,
            labels=labels,
        )
        return item

    def reload(self, sync=False):
        """reload the project and function objects from yaml/specs

        :param sync:  set to True to load functions objects

        :returns: project object
        """
        if self.spec.context:
            project = _load_project_dir(
                self.spec.context, self.metadata.name, self.spec.subpath
            )
        else:
            project = _load_project_file(
                self.spec.origin_url, self.metadata.name, self._secrets
            )
        project.spec.source = self.spec.source
        project.spec.repo = self.spec.repo
        project.spec.branch = self.spec.branch
        project.spec.origin_url = self.spec.origin_url
        if sync:
            project.sync_functions()
        self.__dict__.update(project.__dict__)
        return project

    def set_function(self, func, name="", kind="", image=None, with_repo=None):
        """update or add a function object to the project

        function can be provided as an object (func) or a .py/.ipynb/.yaml url
        support url prefixes::

            object (s3://, v3io://, ..)
            MLRun DB e.g. db://project/func:ver
            functions hub/market: e.g. hub://sklearn_classifier:master

        examples::

            proj.set_function(func_object)
            proj.set_function('./src/mycode.py', 'ingest',
                              image='myrepo/ing:latest', with_repo=True)
            proj.set_function('http://.../mynb.ipynb', 'train')
            proj.set_function('./func.yaml')
            proj.set_function('hub://get_toy_data', 'getdata')

        :param func:      function object or spec/code url
        :param name:      name of the function (under the project)
        :param kind:      runtime kind e.g. job, nuclio, spark, dask, mpijob
                          default: job
        :param image:     docker image to be used, can also be specified in
                          the function object/yaml
        :param with_repo: add (clone) the current repo to the build source

        :returns: project object
        """
        if isinstance(func, str):
            # in hub or db functions name defaults to the function name
            if not name and not (func.startswith("db://") or func.startswith("hub://")):
                raise ValueError("function name must be specified")
            function_dict = {
                "url": func,
                "name": name,
                "kind": kind,
                "image": image,
                "with_repo": with_repo,
            }
            func = {k: v for k, v in function_dict.items() if v}
            name, function_object = _init_function_from_dict(func, self)
            func["name"] = name
        elif hasattr(func, "to_dict"):
            name, function_object = _init_function_from_obj(func, self, name=name)
            if image:
                function_object.spec.image = image
            if with_repo:
                function_object.spec.build.source = "./"

            if not name:
                raise ValueError("function name must be specified")
        else:
            raise ValueError("func must be a function url or object")

        self.spec.set_function(name, function_object, func)
        return function_object

    def remove_function(self, name):
        """remove a function from a project

        :param name:    name of the function (under the project)
        """
        self.spec.remove_function(name)

    def func(self, key, sync=False):
        """get function object by name

        :param sync:  will reload/reinit the function

        :returns: function object
        """
        if key not in self.spec._function_definitions:
            raise KeyError(f"function {key} not found")
        if sync or not self._initialized or key not in self.spec._function_objects:
            self.sync_functions()
        return self.spec._function_objects[key]

    def pull(self, branch=None, remote=None):
        """pull/update sources from git or tar into the context dir

        :param branch:  git branch, if not the current one
        :param remote:  git remote, if other than origin
        """
        url = self.spec.origin_url
        if url and url.startswith("git://"):
            if not self.spec.repo:
                raise ValueError("repo was not initialized, use load_project()")
            branch = branch or self.spec.repo.active_branch.name
            remote = remote or "origin"
            self.spec.repo.git.pull(remote, branch)
        elif url and url.endswith(".tar.gz"):
            clone_tgz(url, self.spec.context, self._secrets)
        elif url and url.endswith(".zip"):
            clone_zip(url, self.spec.context, self._secrets)

    def create_remote(self, url, name="origin"):
        """create remote for the project git

        :param url:   remote git url
        :param name:  name for the remote (default is 'origin')
        """
        if not self.spec.repo:
            raise ValueError("git repo is not set/defined")
        self.spec.repo.create_remote(name, url=url)
        url = url.replace("https://", "git://")
        try:
            url = f"{url}#refs/heads/{self.spec.repo.active_branch.name}"
        except Exception:
            pass
        self.spec._source = self.spec.source or url
        self.spec.origin_url = self.spec.origin_url or url

    def push(self, branch, message=None, update=True, remote=None, add: list = None):
        """update spec and push updates to remote git repo

        :param branch:  target git branch
        :param message: git commit message
        :param update:  update files (git add update=True)
        :param remote:  git remote, default to origin
        :param add:     list of files to add
        """
        repo = self.spec.repo
        if not repo:
            raise ValueError("git repo is not set/defined")
        self.save()

        add = add or []
        add.append("project.yaml")
        repo.index.add(add)
        if update:
            repo.git.add(update=True)
        if repo.is_dirty():
            if not message:
                raise ValueError("please specify the commit message")
            repo.git.commit(m=message)

        if not branch:
            raise ValueError("please specify the remote branch")
        repo.git.push(remote or "origin", branch)

    def sync_functions(self, names: list = None, always=True, save=False):
        """reload function objects from specs and files"""
        if self._initialized and not always:
            return

        funcs = {}
        if not names:
            names = self.spec._function_definitions.keys()
        origin = add_code_metadata(self.spec.context)
        for name in names:
            f = self.spec._function_definitions.get(name)
            if not f:
                raise ValueError(f"function named {name} not found")
            if hasattr(f, "to_dict"):
                name, func = _init_function_from_obj(f, self)
            else:
                if not isinstance(f, dict):
                    raise ValueError("function must be an object or dict")
                name, func = _init_function_from_dict(f, self)
            func.spec.build.code_origin = origin
            funcs[name] = func
            if save:
                func.save(versioned=False)

        self.spec._function_objects = funcs
        self._initialized = True

    def with_secrets(self, kind, source, prefix=""):
        """register a secrets source (file, env or dict)

        read secrets from a source provider to be used in workflows,example::

            proj.with_secrets('file', 'file.txt')
            proj.with_secrets('inline', {'key': 'val'})
            proj.with_secrets('env', 'ENV1,ENV2', prefix='PFX_')

        Vault secret source has several options::

            proj.with_secrets('vault', {'user': <user name>, 'secrets': ['secret1', 'secret2' ...]})
            proj.with_secrets('vault', {'project': <proj. name>, 'secrets': ['secret1', 'secret2' ...]})
            proj.with_secrets('vault', ['secret1', 'secret2' ...])

        The 2nd option uses the current project name as context.
        Can also use empty secret list::

            proj.with_secrets('vault', [])

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

    def create_vault_secrets(self, secrets):
        run_db = get_run_db(secrets=self._secrets)
        run_db.create_project_secrets(
            self.metadata.name, mlrun.api.schemas.SecretProviderName.vault, secrets
        )

    def get_vault_secrets(self, secrets=None, local=False):
        if local:
            logger.warning(
                "get_vault_secrets executed locally. This is not recommended and may become deprecated soon"
            )
            return self._secrets.vault.get_secrets(secrets, project=self.metadata.name)

        run_db = get_run_db(secrets=self._secrets)
        project_secrets = run_db.list_project_secrets(
            self.metadata.name,
            self._secrets.vault.token,
            mlrun.api.schemas.SecretProviderName.vault,
            secrets,
        )
        return project_secrets.secrets

    def get_param(self, key: str, default=None):
        """get project param by key"""
        if self.spec.params:
            return self.spec.params.get(key, default)
        return default

    def run(
        self,
        name=None,
        workflow_path=None,
        arguments=None,
        artifact_path=None,
        namespace=None,
        sync=False,
        watch=False,
        dirty=False,
        ttl=None,
    ):
        """run a workflow using kubeflow pipelines

        :param name:      name of the workflow
        :param workflow_path:
                          url to a workflow file, if not a project workflow
        :param arguments:
                          kubeflow pipelines arguments (parameters)
        :param artifact_path:
                          target path/url for workflow artifacts, the string
                          '{{workflow.uid}}' will be replaced by workflow id
        :param namespace: kubernetes namespace if other than default
        :param sync:      force functions sync before run
        :param watch:     wait for pipeline completion
        :param dirty:     allow running the workflow when the git repo is dirty
        :param ttl:       pipeline ttl in secs (after that the pods will be removed)

        :returns: run id
        """

        need_repo = self.spec._need_repo()
        if self.spec.repo and self.spec.repo.is_dirty():
            msg = "you seem to have uncommitted git changes, use .push()"
            if dirty or not need_repo:
                logger.warning("WARNING!, " + msg)
            else:
                raise ProjectError(msg + " or dirty=True")

        if need_repo and self.spec.repo and not self.spec.source:
            raise ProjectError(
                "remote repo is not defined, use .create_remote() + push()"
            )

        self.sync_functions(always=sync)
        if not self.spec._function_objects:
            raise ValueError("no functions in the project")

        if not name and not workflow_path:
            if self.spec.workflows:
                name = list(self.spec._workflows.keys())[0]
            else:
                raise ValueError("workflow name or path must be specified")

        code = None
        if not workflow_path:
            if name not in self.spec._workflows:
                raise ValueError(f"workflow {name} not found")
            workflow_path, code, arguments = self.spec._get_wf_cfg(name, arguments)

        name = f"{self.metadata.name}-{name}" if name else self.metadata.name
        artifact_path = artifact_path or self.spec.artifact_path
        run = _run_pipeline(
            self,
            name,
            workflow_path,
            self.spec._function_objects,
            secrets=self._secrets,
            arguments=arguments,
            artifact_path=artifact_path,
            namespace=namespace,
            ttl=ttl,
        )
        if code:
            remove(workflow_path)
        if watch:
            self.get_run_status(run, notifiers=RunNotifications(with_slack=True))
        return run

    def save_workflow(self, name, target, artifact_path=None, ttl=None):
        """create and save a workflow as a yaml or archive file

        :param name:   workflow name
        :param target: target file path (can end with .yaml or .zip)
        :param artifact_path:
                       target path/url for workflow artifacts, the string
                       '{{workflow.uid}}' will be replaced by workflow id
        :param ttl:    pipeline ttl in secs (after that the pods will be removed)
        """
        if not name or name not in self.workflows:
            raise ValueError(f"workflow {name} not found")

        workflow_path, code, _ = self.spec._get_wf_cfg(name)
        pipeline = _create_pipeline(
            self, workflow_path, self.spec._function_objects, secrets=self._secrets
        )

        artifact_path = artifact_path or self.spec.artifact_path
        conf = new_pipe_meta(artifact_path, ttl=ttl)
        compiler.Compiler().compile(pipeline, target, pipeline_conf=conf)
        if code:
            remove(workflow_path)

    def get_run_status(
        self,
        workflow_id,
        timeout=60 * 60,
        expected_statuses=None,
        notifiers: RunNotifications = None,
    ):
        status = ""
        if timeout:
            logger.info("waiting for pipeline run completion")
            run_info = wait_for_pipeline_completion(
                workflow_id, timeout=timeout, expected_statuses=expected_statuses
            )
            if run_info:
                status = run_info["run"].get("status")

        mldb = get_run_db(secrets=self._secrets)
        runs = mldb.list_runs(
            project=self.metadata.name, labels=f"workflow={workflow_id}"
        )

        had_errors = 0
        for r in runs:
            if r["status"].get("state", "") == "error":
                had_errors += 1

        text = f"Workflow {workflow_id} finished"
        if had_errors:
            text += f" with {had_errors} errors"
        if status:
            text += f", status={status}"

        if notifiers:
            notifiers.push(text, runs)
        return status, had_errors, text

    def clear_context(self):
        """delete all files and clear the context dir"""
        if (
            self.spec.context
            and path.exists(self.spec.context)
            and path.isdir(self.spec.context)
        ):
            shutil.rmtree(self.spec.context)

    def save(self, filepath=None):
        self.export(filepath)
        self.save_to_db()

    def save_to_db(self):
        db = get_run_db(secrets=self._secrets)
        db.store_project(self.metadata.name, self.to_dict())

    def export(self, filepath=None):
        """save the project object into a file (default to project.yaml)"""
        filepath = filepath or path.join(
            self.spec.context, self.spec.subpath, "project.yaml"
        )
        project_dir = pathlib.Path(filepath).parent
        if not project_dir.exists():
            project_dir.mkdir(parents=True)
        with open(filepath, "w") as fp:
            fp.write(self.to_yaml())


class MlrunProjectLegacy(ModelObj):
    kind = "project"

    def __init__(
        self,
        name=None,
        description=None,
        params=None,
        functions=None,
        workflows=None,
        artifacts=None,
        artifact_path=None,
        conda=None,
    ):

        self._initialized = False
        self.name = name
        self.description = description
        self.tag = ""
        self.origin_url = ""
        self._source = ""
        self.context = None
        self.subpath = ""
        self.branch = None
        self.repo = None
        self._secrets = SecretsStore()
        self.params = params or {}
        self.conda = conda or {}
        self._mountdir = None
        self._artifact_mngr = None
        self.artifact_path = artifact_path or config.artifact_path

        self.workflows = workflows or []
        self.artifacts = artifacts or []

        self._function_objects = {}
        self._function_defs = {}
        self.functions = functions or []

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

    def _source_repo(self):
        src = self.source
        if src:
            return src.split("#")[0]
        return ""

    def _get_hexsha(self):
        try:
            if self.repo:
                return self.repo.head.commit.hexsha
        except Exception:
            pass
        return None

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
        funcs = []
        for name, f in self._function_defs.items():
            if hasattr(f, "to_dict"):
                spec = f.to_dict(strip=True)
                if f.spec.build.source and f.spec.build.source.startswith(
                    self._source_repo()
                ):
                    update_in(spec, "spec.build.source", "./")
                funcs.append({"name": name, "spec": spec})
            else:
                funcs.append(f)
        return funcs

    @functions.setter
    def functions(self, funcs):
        if not isinstance(funcs, list):
            raise ValueError("functions must be a list")

        func_defs = {}
        for f in funcs:
            if not isinstance(f, dict) and not hasattr(f, "to_dict"):
                raise ValueError("functions must be an objects or dict")
            if isinstance(f, dict):
                name = f.get("name", "")
                if not name:
                    raise ValueError("function name must be specified in dict")
            else:
                name = f.metadata.name
            func_defs[name] = f

        self._function_defs = func_defs

    @property
    def workflows(self) -> list:
        """list of workflows specs used in this project"""
        return [w for w in self._workflows.values()]

    @workflows.setter
    def workflows(self, workflows):
        if not isinstance(workflows, list):
            raise ValueError("workflows must be a list")

        wfdict = {}
        for w in workflows:
            if not isinstance(w, dict):
                raise ValueError("workflow must be a dict")
            name = w.get("name", "")
            # todo: support steps dsl as code alternative
            if not name:
                raise ValueError('workflow "name" must be specified')
            if "path" not in w and "code" not in w:
                raise ValueError('workflow source "path" or "code" must be specified')
            wfdict[name] = w

        self._workflows = wfdict

    @property
    def artifacts(self) -> list:
        """list of artifacts used in this project"""
        return [a for a in self._artifacts.values()]

    @artifacts.setter
    def artifacts(self, artifacts):
        if not isinstance(artifacts, list):
            raise ValueError("artifacts must be a list")

        afdict = {}
        for a in artifacts:
            if not isinstance(a, dict) and not hasattr(a, "to_dict"):
                raise ValueError("artifacts must be a dict or class")
            if isinstance(a, dict):
                key = a.get("key", "")
                if not key:
                    raise ValueError('artifacts "key" must be specified')
            else:
                key = a.key
                a = a.to_dict()

            afdict[key] = a

        self._artifacts = afdict

    # needed for tests
    def set_workflow(self, name, workflow_path: str, embed=False, **args):
        """add or update a workflow, specify a name and the code path"""
        if not workflow_path:
            raise ValueError("valid workflow_path must be specified")
        if embed:
            if self.context and not workflow_path.startswith("/"):
                workflow_path = path.join(self.context, workflow_path)
            with open(workflow_path, "r") as fp:
                txt = fp.read()
            workflow = {"name": name, "code": txt}
        else:
            workflow = {"name": name, "path": workflow_path}
        if args:
            workflow["args"] = args
        self._workflows[name] = workflow

    # needed for tests
    def set_function(self, func, name="", kind="", image=None, with_repo=None):
        """update or add a function object to the project

        function can be provided as an object (func) or a .py/.ipynb/.yaml url

        supported url prefixes::

            object (s3://, v3io://, ..)
            MLRun DB e.g. db://project/func:ver
            functions hub/market: e.g. hub://sklearn_classifier:master

        examples::

            proj.set_function(func_object)
            proj.set_function('./src/mycode.py', 'ingest',
                              image='myrepo/ing:latest', with_repo=True)
            proj.set_function('http://.../mynb.ipynb', 'train')
            proj.set_function('./func.yaml')
            proj.set_function('hub://get_toy_data', 'getdata')

        :param func:      function object or spec/code url
        :param name:      name of the function (under the project)
        :param kind:      runtime kind e.g. job, nuclio, spark, dask, mpijob
                          default: job
        :param image:     docker image to be used, can also be specified in
                          the function object/yaml
        :param with_repo: add (clone) the current repo to the build source

        :returns: project object
        """
        if isinstance(func, str):
            if not name:
                raise ValueError("function name must be specified")
            fdict = {
                "url": func,
                "name": name,
                "kind": kind,
                "image": image,
                "with_repo": with_repo,
            }
            func = {k: v for k, v in fdict.items() if v}
            name, f = _init_function_from_dict_legacy(func, self)
        elif hasattr(func, "to_dict"):
            name, f = _init_function_from_obj_legacy(func, self, name=name)
            if image:
                f.spec.image = image
            if with_repo:
                f.spec.build.source = "./"

            if not name:
                raise ValueError("function name must be specified")
        else:
            raise ValueError("func must be a function url or object")

        self._function_defs[name] = func
        self._function_objects[name] = f
        return f

    # needed for tests
    def save(self, filepath=None):
        """save the project object into a file (default to project.yaml)"""
        filepath = filepath or path.join(self.context, self.subpath, "project.yaml")
        with open(filepath, "w") as fp:
            fp.write(self.to_yaml())


def _init_function_from_dict(f, project):
    name = f.get("name", "")
    url = f.get("url", "")
    kind = f.get("kind", "")
    image = f.get("image", None)
    with_repo = f.get("with_repo", False)

    if with_repo and not project.spec.source:
        raise ValueError("project source must be specified when cloning context")

    in_context = False
    if not url and "spec" not in f:
        raise ValueError("function missing a url or a spec")

    if url and "://" not in url:
        if project.spec.context and not url.startswith("/"):
            url = path.join(project.spec.context, url)
            in_context = True
        if not path.isfile(url):
            raise OSError(f"{url} not found")

    if "spec" in f:
        func = new_function(name, runtime=f["spec"])
    elif url.endswith(".yaml") or url.startswith("db://") or url.startswith("hub://"):
        func = import_function(url)
        if image:
            func.spec.image = image
    elif url.endswith(".ipynb"):
        func = code_to_function(name, filename=url, image=image, kind=kind)
    elif url.endswith(".py"):
        if not image:
            raise ValueError(
                "image must be provided with py code files, "
                "use function object for more control/settings"
            )
        if in_context and with_repo:
            func = new_function(name, command=url, image=image, kind=kind or "job")
        else:
            func = code_to_function(name, filename=url, image=image, kind=kind or "job")
    else:
        raise ValueError(f"unsupported function url {url} or no spec")

    if with_repo:
        func.spec.build.source = "./"

    return _init_function_from_obj(func, project, name)


def _init_function_from_obj(func, project, name=None):
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
    if project.spec.tag:
        func.metadata.tag = project.spec.tag
    return name or func.metadata.name, func


def _init_function_from_dict_legacy(f, project):
    name = f.get("name", "")
    url = f.get("url", "")
    kind = f.get("kind", "")
    image = f.get("image", None)
    with_repo = f.get("with_repo", False)

    if with_repo and not project.source:
        raise ValueError("project source must be specified when cloning context")

    in_context = False
    if not url and "spec" not in f:
        raise ValueError("function missing a url or a spec")

    if url and "://" not in url:
        if project.context and not url.startswith("/"):
            url = path.join(project.context, url)
            in_context = True
        if not path.isfile(url):
            raise OSError(f"{url} not found")

    if "spec" in f:
        func = new_function(name, runtime=f["spec"])
    elif url.endswith(".yaml") or url.startswith("db://") or url.startswith("hub://"):
        func = import_function(url)
        if image:
            func.spec.image = image
    elif url.endswith(".ipynb"):
        func = code_to_function(name, filename=url, image=image, kind=kind)
    elif url.endswith(".py"):
        if not image:
            raise ValueError(
                "image must be provided with py code files, "
                "use function object for more control/settings"
            )
        if in_context and with_repo:
            func = new_function(name, command=url, image=image, kind=kind or "job")
        else:
            func = code_to_function(name, filename=url, image=image, kind=kind or "job")
    else:
        raise ValueError(f"unsupported function url {url} or no spec")

    if with_repo:
        func.spec.build.source = "./"

    return _init_function_from_obj_legacy(func, project, name)


def _init_function_from_obj_legacy(func, project, name=None):
    build = func.spec.build
    if project.origin_url:
        origin = project.origin_url
        try:
            if project.repo:
                origin += "#" + project.repo.head.commit.hexsha
        except Exception:
            pass
        build.code_origin = origin
    if project.name:
        func.metadata.project = project.name
    if project.tag:
        func.metadata.tag = project.tag
    return name or func.metadata.name, func


def _create_pipeline(project, pipeline, funcs, secrets=None):
    functions = {}
    for name, func in funcs.items():
        f = func.copy()
        src = f.spec.build.source
        if project.spec.source and src and src in [".", "./"]:
            if project.spec.mountdir:
                f.spec.workdir = project.spec.mountdir
                f.spec.build.source = ""
            else:
                f.spec.build.source = project.spec.source

        functions[name] = f

    spec = imputil.spec_from_file_location("workflow", pipeline)
    if spec is None:
        raise ImportError(f"cannot import workflow {pipeline}")
    mod = imputil.module_from_spec(spec)
    spec.loader.exec_module(mod)

    setattr(mod, "funcs", functions)
    setattr(mod, "this_project", project)

    if hasattr(mod, "init_functions"):
        getattr(mod, "init_functions")(functions, project, secrets)

    # verify all functions are in this project
    for f in functions.values():
        f.metadata.project = project.metadata.name

    if not hasattr(mod, "kfpipeline"):
        raise ValueError("pipeline function (kfpipeline) not found")

    kfpipeline = getattr(mod, "kfpipeline")
    return kfpipeline


def _run_pipeline(
    project,
    name,
    pipeline,
    functions,
    secrets=None,
    arguments=None,
    artifact_path=None,
    namespace=None,
    ttl=None,
):
    kfpipeline = _create_pipeline(project, pipeline, functions, secrets)

    namespace = namespace or config.namespace
    id = run_pipeline(
        kfpipeline,
        project=project.metadata.name,
        arguments=arguments,
        experiment=name,
        namespace=namespace,
        artifact_path=artifact_path,
        ttl=ttl,
    )
    return id


def github_webhook(request):
    signature = request.headers.get("X-Hub-Signature")
    data = request.data
    print("sig:", signature)
    print("headers:", request.headers)
    print("data:", data)
    print("json:", request.get_json())

    if request.headers.get("X-GitHub-Event") == "ping":
        return {"msg": "Ok"}

    return {"msg": "pushed"}
