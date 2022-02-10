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
import pathlib
import shutil
import typing
import warnings
from os import environ, makedirs, path

import dotenv
import inflection
import kfp
import yaml
from git import Repo

import mlrun.api.schemas
import mlrun.errors
import mlrun.utils.regex

from ..artifacts import Artifact, ArtifactProducer, DatasetArtifact, ModelArtifact
from ..artifacts.manager import ArtifactManager, dict_to_artifact, extend_artifact_path
from ..datastore import store_manager
from ..db import get_run_db
from ..features import Feature
from ..model import EntrypointParam, ModelObj
from ..run import code_to_function, get_object, import_function, new_function
from ..runtimes.utils import add_code_metadata
from ..secrets import SecretsStore
from ..utils import RunNotifications, logger, update_in
from ..utils.clones import clone_git, clone_tgz, clone_zip, get_repo_url
from ..utils.model_monitoring import set_project_model_monitoring_credentials
from .operations import build_function, deploy_function, run_function
from .pipelines import (
    FunctionsDict,
    WorkflowSpec,
    _PipelineRunStatus,
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
        context_path.mkdir(parents=True)
    elif not context_path.is_dir():
        raise ValueError(f"context {context} is not a dir path")
    try:
        repo = Repo(context)
        url = get_repo_url(repo)
    except Exception:
        if init_git:
            repo = Repo.init(context)
    return repo, url


def new_project(
    name,
    context=None,
    init_git=False,
    user_project=False,
    remote=None,
    from_template=None,
    secrets=None,
    description=None,
) -> "MlrunProject":
    """Create a new MLRun project, optionally load it from a yaml/zip/git template

    example::

        # create a project with local and marketplace functions, a workflow, and an artifact
        project = mlrun.new_project("myproj", "./", init_git=True, description="my new project")
        project.set_function('prep_data.py', 'prep-data', image='mlrun/mlrun', handler='prep_data')
        project.set_function('hub://sklearn_classifier', 'train')
        project.set_artifact('data', Artifact(target_path=data_url))
        project.set_workflow('main', "./myflow.py")
        project.save()

        # run the "main" workflow (watch=True to wait for run completion)
        project.run("main", watch=True)

    example (load from template)::

        # create a new project from a zip template (can also use yaml/git templates)
        # initialize a local git, and register the git remote path
        project = mlrun.new_project("myproj", "./", init_git=True,
                                    remote="git://github.com/mlrun/project-demo.git",
                                    from_template="http://mysite/proj.zip")
        project.run("main", watch=True)


    :param name:         project name
    :param context:      project local directory path
    :param init_git:     if True, will git init the context dir
    :param user_project: add the current user name to the provided project name (making it unique per user)
    :param remote:       remote Git url
    :param from_template:     path to project YAML/zip file that will be used as a template
    :param secrets:      key:secret dict or SecretsStore used to download sources
    :param description:  text describing the project

    :returns: project object
    """
    context = context or "./"
    name = _add_username_to_project_name_if_needed(name, user_project)

    if from_template:
        if from_template.endswith(".yaml"):
            project = _load_project_file(from_template, name, secrets)
        elif from_template.startswith("git://"):
            clone_git(from_template, context, secrets, clone=True)
            shutil.rmtree(path.join(context, ".git"))
            project = _load_project_dir(context, name)
        elif from_template.endswith(".zip"):
            clone_zip(from_template, context, secrets)
            project = _load_project_dir(context, name)
        else:
            raise ValueError("template must be a path to .yaml or .zip file")
        project.metadata.name = name
    else:
        project = MlrunProject(name=name)
    project.spec.context = context

    repo, url = init_repo(context, remote, init_git or remote)
    project.spec.repo = repo
    if remote and url != remote:
        project.create_remote(remote)
    elif url:
        project.spec._source = url
        project.spec.origin_url = url
    if description:
        project.spec.description = description
    mlrun.mlconf.default_project = project.metadata.name
    pipeline_context.set(project)
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
) -> "MlrunProject":
    """Load an MLRun project from git or tar or dir

    example::

        # load the project and run the 'main' workflow
        project = load_project("./", "git://github.com/mlrun/project-demo.git")
        project.run("main", arguments={'data': data_url})

    :param context:      project local directory path
    :param url:          name (in DB) or git or tar.gz or .zip sources archive path e.g.:
                         git://github.com/mlrun/demo-xgb-project.git
                         http://mysite/archived-project.zip
                         <project-name>
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
    name = _add_username_to_project_name_if_needed(name, user_project)
    if not context:
        raise ValueError("valid context (local dir path) must be provided")

    from_db = False
    if url:
        if url.endswith(".yaml"):
            project = _load_project_file(url, name, secrets)
        elif url.startswith("git://"):
            url, repo = clone_git(url, context, secrets, clone)
        elif url.endswith(".tar.gz"):
            clone_tgz(url, context, secrets)
        elif url.endswith(".zip"):
            clone_zip(url, context, secrets)
        else:
            project = _load_project_from_db(url, secrets, user_project)
            project.spec.context = context
            if not path.isdir(context):
                makedirs(context)
            project.spec.subpath = subpath or project.spec.subpath
            from_db = True

    if not repo:
        repo, url = init_repo(context, url, init_git)

    if not project:
        project = _load_project_dir(context, name, subpath)
    if not project.metadata.name:
        raise ValueError("project name must be specified")
    if not from_db or (url and url.startswith("git://")):
        project.spec.source = url or project.spec.source
        project.spec.origin_url = url or project.spec.origin_url
    project.spec.repo = repo
    if repo:
        try:
            # handle cases where active_branch is not set (e.g. in Gitlab CI)
            project.spec.branch = repo.active_branch.name
        except Exception:
            pass
    project.register_artifacts()
    mlrun.mlconf.default_project = project.metadata.name
    pipeline_context.set(project)
    return project


def get_or_create_project(
    name,
    context,
    url=None,
    secrets=None,
    init_git=False,
    subpath="",
    clone=False,
    user_project=False,
    from_template=None,
) -> "MlrunProject":
    """Load a project from MLRun DB, or create/import if doesnt exist

    example::

        # load project from the DB (if exist) or the source repo
        project = get_or_create_project("myproj", "./", "git://github.com/mlrun/demo-xgb-project.git")
        project.pull("development")  # pull the latest code from git
        project.run("main", arguments={'data': data_url})  # run the workflow "main"

    :param context:      project local directory path
    :param url:          name (in DB) or git or tar.gz or .zip sources archive path e.g.:
                         git://github.com/mlrun/demo-xgb-project.git
                         http://mysite/archived-project.zip
    :param name:         project name
    :param secrets:      key:secret dict or SecretsStore used to download sources
    :param init_git:     if True, will git init the context dir
    :param subpath:      project subpath (within the archive)
    :param clone:        if True, always clone (delete any existing content)
    :param user_project: add the current user name to the project name (for db:// prefixes)
    :param from_template:     path to project YAML file that will be used as from_template (for new projects)

    :returns: project object
    """

    try:
        project = load_project(
            context,
            name,
            name,
            secrets=secrets,
            init_git=init_git,
            subpath=subpath,
            clone=clone,
            user_project=user_project,
        )
        logger.info(f"loaded project {name} from MLRun DB")
        return project

    except mlrun.errors.MLRunNotFoundError:
        if url:
            project = load_project(
                context,
                url,
                name,
                secrets=secrets,
                init_git=init_git,
                subpath=subpath,
                clone=clone,
                user_project=user_project,
            )
            logger.info(f"loaded project {name} from {url} or context")
        else:
            project = new_project(
                name,
                context,
                init_git=init_git,
                user_project=user_project,
                from_template=from_template,
                secrets=secrets,
            )
            logger.info(f"created and saved project {name}")
        project.save_to_db()
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


def _add_username_to_project_name_if_needed(name, user_project):
    if user_project:
        if not name:
            raise ValueError("user_project must be specified together with name")
        username = environ.get("V3IO_USERNAME") or getpass.getuser()
        normalized_username = inflection.dasherize(username.lower())
        if username != normalized_username:
            logger.info(
                "Username was normalized to match the required pattern for project name",
                username=username,
                normalized_username=normalized_username,
            )
        name = f"{name}-{normalized_username}"
    return name


def _load_project_from_db(url, secrets, user_project=False):
    db = get_run_db(secrets=secrets)
    project_name = _add_username_to_project_name_if_needed(
        url.replace("db://", ""), user_project
    )
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
        load_source_on_run=None,
        default_requirements: typing.Union[str, typing.List[str]] = None,
        desired_state=mlrun.api.schemas.ProjectState.online.value,
        owner=None,
        disable_auto_mount=False,
    ):
        self.repo = None

        self.description = description
        self.context = ""
        self._mountdir = None
        self._source = None
        self.source = source or ""
        self.load_source_on_run = load_source_on_run
        self.subpath = subpath or ""
        self.origin_url = origin_url or ""
        self.goals = goals
        self.desired_state = desired_state
        self.owner = owner
        self.branch = None
        self.tag = ""
        self.params = params or {}
        self.conda = conda or {}
        self.artifact_path = artifact_path
        self._artifacts = {}
        self.artifacts = artifacts or []
        self.default_requirements = default_requirements

        self._workflows = {}
        self.workflows = workflows or []

        self._function_objects = {}
        self._function_definitions = {}
        self.functions = functions or []
        self.disable_auto_mount = disable_auto_mount

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
    def workflows(self) -> list:
        """list of workflows specs used in this project"""
        return [workflow for workflow in self._workflows.values()]

    @workflows.setter
    def workflows(self, workflows):
        if not workflows:
            workflows = []
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
        if not artifacts:
            artifacts = []
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
        if hasattr(artifact, "base_dict"):
            artifact = artifact.base_dict()
        artifact["key"] = key
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
        default_requirements: typing.Union[str, typing.List[str]] = None,
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
        self.spec.default_requirements = (
            default_requirements or self.spec.default_requirements
        )

        self._initialized = False
        self._secrets = SecretsStore()
        self._artifact_manager = None
        self._notifiers = RunNotifications(with_slack=True)

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

    def set_source(self, source, pull_at_runtime=False):
        """set the project source code path(can be git/tar/zip archive)

        :param source:     valid path to git, zip, or tar file, (or None for current) e.g.
                           git://github.com/mlrun/something.git
                           http://some/url/file.zip
        :param pull_at_runtime: load the archive into the container at job runtime vs on build/deploy
        """
        self.spec.load_source_on_run = pull_at_runtime
        self.spec.source = source or self.spec.source

    def get_artifact_uri(self, key, category="artifact") -> str:
        """return the project artifact uri (store://..) from the artifact key

        :param key:  artifact key/name
        :param category:  artifact category (artifact, model, feature-vector, ..)
        """
        return f"store://{category}s/{self.metadata.name}/{key}"

    def get_store_resource(self, uri):
        """get store resource object by uri"""
        return mlrun.datastore.get_store_resource(
            uri, secrets=self._secrets, project=self.metadata.name
        )

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

    def set_workflow(
        self,
        name,
        workflow_path: str,
        embed=False,
        engine=None,
        args_schema: typing.List[EntrypointParam] = None,
        handler=None,
        **args,
    ):
        """add or update a workflow, specify a name and the code path

        :param name:          name of the workflow
        :param workflow_path: url/path for the workflow file
        :param embed:         add the workflow code into the project.yaml
        :param engine:        workflow processing engine ("kfp" or "local")
        :param args_schema:   list of arg schema definitions (:py:class`~mlrun.model.EntrypointParam`)
        :param handler:       workflow function handler
        :param args:          argument values (key=value, ..)
        """
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
        if handler:
            workflow["handler"] = handler
        if args_schema:
            args_schema = [
                schema.to_dict() if hasattr(schema, "to_dict") else schema
                for schema in args_schema
            ]
            workflow["args_schema"] = args_schema
        workflow["engine"] = engine
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

    def set_artifact(self, key, artifact=None, target_path=None):
        """add/set an artifact in the project spec (will be registered on load)

        example::

            # register a simple file artifact
            project.set_artifact('data', target_path=data_url)
            # register a model artifact
            project.set_artifact('model', ModelArtifact(model_file="model.pkl"), target_path=model_dir_url)

        :param key:  artifact key/name
        :param artifact:  mlrun Artifact object (or its subclasses)
        :param target_path: absolute target path url (point to the artifact content location)
        """
        if not artifact:
            artifact = Artifact()
        artifact.target_path = target_path or artifact.target_path
        if not artifact.target_path or "://" not in artifact.target_path:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "absolute target_path url to a shared/object storage must be specified"
            )
        self.spec.set_artifact(key, artifact)

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
        """log an output artifact and optionally upload it to datastore

        example::

            project.log_artifact(
                "some-data",
                body=b"abc is 123",
                local_path="model.txt",
                labels={"framework": "xgboost"},
            )


        :param item:          artifact key or artifact class ()
        :param body:          will use the body as the artifact content
        :param local_path:    path to the local file we upload, will also be use
                              as the destination subpath (under "artifact_path")
        :param artifact_path: target artifact path (when not using the default)
                              to define a subpath under the default location use:
                              `artifact_path=context.artifact_subpath('data')`
        :param format:        artifact file format: csv, png, ..
        :param tag:           version tag
        :param target_path:   absolute target path (instead of using artifact_path + local_path)
        :param upload:        upload to datastore (default is True)
        :param labels:        a set of key/value labels to tag the artifact with

        :returns: artifact object
        """
        am = self._get_artifact_manager()
        artifact_path = extend_artifact_path(
            artifact_path, self.spec.artifact_path or mlrun.mlconf.artifact_path
        )
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
    ) -> DatasetArtifact:
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
            project.log_dataset("mydf", df=df, stats=True)

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
            artifact_path=artifact_path,
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
        **kwargs,
    ):
        """log a model artifact and optionally upload it to datastore

        example::

            project.log_model("model", body=dumps(model),
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
            **kwargs,
        )
        if training_set is not None:
            model.infer_from_df(training_set, label_column)

        item = self.log_artifact(
            model,
            local_path=model_dir,
            artifact_path=artifact_path,
            tag=tag,
            upload=upload,
            labels=labels,
        )
        return item

    def reload(self, sync=False, context=None) -> "MlrunProject":
        """reload the project and function objects from the project yaml/specs

        :param sync:    set to True to load functions objects
        :param context: context directory (where the yaml and code exist)

        :returns: project object
        """
        context = context or self.spec.context
        if context:
            project = _load_project_dir(context, self.metadata.name, self.spec.subpath)
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

    def set_function(
        self,
        func: typing.Union[str, mlrun.runtimes.BaseRuntime],
        name: str = "",
        kind: str = "",
        image: str = None,
        handler=None,
        with_repo: bool = None,
        requirements: typing.Union[str, typing.List[str]] = None,
    ) -> mlrun.runtimes.BaseRuntime:
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
        :param handler:   default function handler to invoke (can only be set with .py/.ipynb files)
        :param with_repo: add (clone) the current repo to the build source
        :param requirements:    list of python packages or pip requirements file path

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
                "handler": handler,
                "with_repo": with_repo,
                "requirements": requirements,
            }
            func = {k: v for k, v in function_dict.items() if v}
            name, function_object = _init_function_from_dict(func, self)
            func["name"] = name
        elif hasattr(func, "to_dict"):
            name, function_object = _init_function_from_obj(func, self, name=name)
            if handler:
                raise ValueError(
                    "default handler cannot be set for existing function object"
                )
            if image:
                function_object.spec.image = image
            if with_repo:
                function_object.spec.build.source = "./"
            if requirements:
                function_object.with_requirements(requirements)
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

    def func(self, key, sync=False) -> mlrun.runtimes.BaseRuntime:
        """get function object by name

        :param sync:  will reload/reinit the function

        :returns: function object
        """
        warnings.warn(
            "This will be deprecated in future releases, use  get_function() instead",
            # TODO: do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        return self.get_function(key, sync)

    def get_function(self, key, sync=False, enrich=False) -> mlrun.runtimes.BaseRuntime:
        """get function object by name

        :param sync:  will reload/reinit the function
        :param enrich: add project info/config/source info to the function object

        :returns: function object
        """
        if key in self.spec._function_objects and not sync:
            function = self.spec._function_objects[key]
        elif key in self.spec._function_definitions:
            self.sync_functions()
            function = self.spec._function_objects[key]
        else:
            function = get_db_function(self, key)
            self.spec._function_objects[key] = function
        if enrich:
            return enrich_function_object(self, function)
        return function

    def get_function_objects(self) -> typing.Dict[str, mlrun.runtimes.BaseRuntime]:
        """"get a virtual dict with all the project functions ready for use in a pipeline"""
        self.sync_functions()
        return FunctionsDict(self)

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
            return self.spec._function_objects

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
        return self.spec._function_objects

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

    def set_secrets(
        self,
        secrets: dict = None,
        file_path: str = None,
        provider: typing.Union[str, mlrun.api.schemas.SecretProviderName] = None,
    ):
        """set project secrets from dict or secrets env file
        when using a secrets file it should have lines in the form KEY=VALUE, comment line start with "#"
        V3IO paths/credentials and MLrun service API address are dropped from the secrets

        example secrets file::

            # this is an env file
            AWS_ACCESS_KEY_ID-XXXX
            AWS_SECRET_ACCESS_KEY=YYYY

        usage::

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
            secrets = dotenv.dotenv_values(file_path)
            if None in secrets.values():
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "env file lines must be in the form key=value"
                )
        # drop V3IO paths/credentials and MLrun service API address
        env_vars = {
            key: val
            for key, val in secrets.items()
            if key != "MLRUN_DBPATH" and not key.startswith("V3IO_")
        }
        provider = provider or mlrun.api.schemas.SecretProviderName.kubernetes
        get_run_db().create_project_secrets(
            self.metadata.name, provider=provider, secrets=env_vars
        )

    def create_vault_secrets(self, secrets):
        warnings.warn(
            "This method is obsolete, use project.set_secrets() instead"
            "This will be deprecated and removed in 1.0.0",
            # TODO: In 1.0 remove
            PendingDeprecationWarning,
        )
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

    def _enrich_artifact_path_with_workflow_uid(self):
        artifact_path = self.spec.artifact_path or mlrun.mlconf.artifact_path
        if not mlrun.mlconf.enrich_artifact_path_with_workflow_id:
            return artifact_path
        workflow_uid_string = "{{workflow.uid}}"
        if workflow_uid_string in artifact_path:
            return artifact_path

        # join paths and replace "\" with "/" (in case of windows clients)
        artifact_path = path.join(artifact_path, workflow_uid_string).replace("\\", "/")
        return artifact_path

    def run(
        self,
        name=None,
        workflow_path=None,
        arguments=None,
        artifact_path=None,
        workflow_handler=None,
        namespace=None,
        sync=False,
        watch=False,
        dirty=False,
        ttl=None,
        engine=None,
        local=False,
    ) -> _PipelineRunStatus:
        """run a workflow using kubeflow pipelines

        :param name:      name of the workflow
        :param workflow_path:
                          url to a workflow file, if not a project workflow
        :param arguments:
                          kubeflow pipelines arguments (parameters)
        :param artifact_path:
                          target path/url for workflow artifacts, the string
                          '{{workflow.uid}}' will be replaced by workflow id
        :param workflow_handler:
                          workflow function handler (for running workflow function directly)
        :param namespace: kubernetes namespace if other than default
        :param sync:      force functions sync before run
        :param watch:     wait for pipeline completion
        :param dirty:     allow running the workflow when the git repo is dirty
        :param ttl:       pipeline ttl in secs (after that the pods will be removed)
        :param engine:    workflow engine running the workflow. supported values are 'kfp' (default) or 'local'
        :param local:     run local pipeline with local functions (set local=True in function.run())

        :returns: run id
        """

        arguments = arguments or {}
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

        if not name and not workflow_path and not workflow_handler:
            if self.spec.workflows:
                name = list(self.spec._workflows.keys())[0]
            else:
                raise ValueError("workflow name or path must be specified")

        if workflow_path or (workflow_handler and callable(workflow_handler)):
            workflow_spec = WorkflowSpec(path=workflow_path, args=arguments)
        else:
            workflow_spec = WorkflowSpec.from_dict(self.spec._workflows[name])
            workflow_spec.merge_args(arguments)
            workflow_spec.ttl = ttl or workflow_spec.ttl
        workflow_spec.run_local = local

        name = f"{self.metadata.name}-{name}" if name else self.metadata.name
        artifact_path = artifact_path or self._enrich_artifact_path_with_workflow_uid()
        workflow_engine = get_workflow_engine(engine or workflow_spec.engine, local)
        workflow_spec.engine = workflow_engine.engine

        run = workflow_engine.run(
            self,
            workflow_spec,
            name,
            workflow_handler=workflow_handler,
            secrets=self._secrets,
            artifact_path=artifact_path,
            namespace=namespace,
        )
        workflow_spec.clear_tmp()
        if watch and workflow_engine.engine == "kfp":
            self.get_run_status(run)
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
        if not name or name not in self.spec._workflows:
            raise ValueError(f"workflow {name} not found")

        workflow_spec = WorkflowSpec.from_dict(self.spec._workflows[name])
        self.sync_functions()
        workflow_engine = get_workflow_engine(workflow_spec.engine)
        workflow_engine.save(self, workflow_spec, target, artifact_path=artifact_path)

    def get_run_status(
        self,
        run,
        timeout=60 * 60,
        expected_statuses=None,
        notifiers: RunNotifications = None,
    ):
        state = ""
        raise_error = None
        try:
            if timeout:
                logger.info("waiting for pipeline run completion")
                state = run.wait_for_completion(
                    timeout=timeout, expected_statuses=expected_statuses
                )
        except RuntimeError as exc:
            # push runs table also when we have errors
            raise_error = exc

        mldb = get_run_db(secrets=self._secrets)
        runs = mldb.list_runs(
            project=self.metadata.name, labels=f"workflow={run.run_id}"
        )

        had_errors = 0
        for r in runs:
            if r["status"].get("state", "") == "error":
                had_errors += 1

        text = f"Workflow {run.run_id} finished"
        if had_errors:
            text += f" with {had_errors} errors"
        if state:
            text += f", state={state}"

        notifiers = notifiers or self._notifiers
        notifiers.push(text, runs)

        if raise_error:
            raise raise_error
        return state, had_errors, text

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
            self.spec.context, self.spec.subpath or "", "project.yaml"
        )
        project_dir = pathlib.Path(filepath).parent
        if not project_dir.exists():
            project_dir.mkdir(parents=True)
        with open(filepath, "w") as fp:
            fp.write(self.to_yaml())

    def set_model_monitoring_credentials(self, access_key: str):
        """ Set the credentials that will be used by the project's model monitoring
        infrastructure functions.
        The supplied credentials must have data access

        :param access_key: Model Monitoring access key for managing user permissions.
        """
        set_project_model_monitoring_credentials(
            access_key=access_key, project=self.metadata.name
        )

    def run_function(
        self,
        function: typing.Union[str, mlrun.runtimes.BaseRuntime],
        handler: str = None,
        name: str = "",
        params: dict = None,
        hyperparams: dict = None,
        hyper_param_options: mlrun.model.HyperParamOptions = None,
        inputs: dict = None,
        outputs: typing.List[str] = None,
        workdir: str = "",
        labels: dict = None,
        base_task: mlrun.model.RunTemplate = None,
        watch: bool = True,
        local: bool = False,
        verbose: bool = None,
        selector: str = None,
        auto_build: bool = None,
    ) -> typing.Union[mlrun.model.RunObject, kfp.dsl.ContainerOp]:
        """Run a local or remote task as part of a local/kubeflow pipeline

        example (use with project)::

            # create a project with two functions (local and from marketplace)
            project = mlrun.new_project(project_name, "./proj")
            project.set_function("mycode.py", "myfunc", image="mlrun/mlrun")
            project.set_function("hub://sklearn_classifier", "train")

            # run functions (refer to them by name)
            run1 = project.run_function("myfunc", params={"x": 7})
            run2 = project.run_function("train", params={"data": run1.outputs["data"]})

        :param function:        name of the function (in the project) or function object
        :param handler:         name of the function handler
        :param name:            execution name
        :param params:          input parameters (dict)
        :param hyperparams:     hyper parameters
        :param selector:        selection criteria for hyper params e.g. "max.accuracy"
        :param hyper_param_options:  hyper param options (selector, early stop, strategy, ..)
                                see: :py:class:`~mlrun.model.HyperParamOptions`
        :param inputs:          input objects (dict of key: path)
        :param outputs:         list of outputs which can pass in the workflow
        :param workdir:         default input artifacts path
        :param labels:          labels to tag the job/run with ({key:val, ..})
        :param base_task:       task object to use as base
        :param watch:           watch/follow run log, True by default
        :param local:           run the function locally vs on the runtime/cluster
        :param verbose:         add verbose prints/logs
        :param auto_build:      when set to True and the function require build it will be built on the first
                                function run, use only if you dont plan on changing the build config between runs

        :return: MLRun RunObject or KubeFlow containerOp
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
        )

    def build_function(
        self,
        function: typing.Union[str, mlrun.runtimes.BaseRuntime],
        with_mlrun: bool = None,
        skip_deployed: bool = False,
        image=None,
        base_image=None,
        commands: list = None,
        secret_name="",
        mlrun_version_specifier=None,
        builder_env: dict = None,
    ):
        """deploy ML function, build container with its dependencies

        :param function:        name of the function (in the project) or function object
        :param with_mlrun:      add the current mlrun package to the container build
        :param skip_deployed:   skip the build if we already have an image for the function
        :param image:           target image name/path
        :param base_image:      base image name/path (commands and source code will be added to it)
        :param commands:        list of docker build (RUN) commands e.g. ['pip install pandas']
        :param secret_name:     k8s secret for accessing the docker registry
        :param mlrun_version_specifier:  which mlrun package version to include (if not current)
        :param builder_env:     Kaniko builder pod env vars dict (for config/credentials)
                                e.g. builder_env={"GIT_TOKEN": token}, does not work yet in KFP
        """
        return build_function(
            function,
            with_mlrun=with_mlrun,
            skip_deployed=skip_deployed,
            image=image,
            base_image=base_image,
            commands=commands,
            secret_name=secret_name,
            mlrun_version_specifier=mlrun_version_specifier,
            builder_env=builder_env,
            project_object=self,
        )

    def deploy_function(
        self,
        function: typing.Union[str, mlrun.runtimes.BaseRuntime],
        dashboard: str = "",
        models: list = None,
        env: dict = None,
        tag: str = None,
        verbose: bool = None,
    ):
        """deploy real-time (nuclio based) functions

        :param function:   name of the function (in the project) or function object
        :param dashboard:  url of the remote Nuclio dashboard (when not local)
        :param models:     list of model items
        :param env:        dict of extra environment variables
        :param tag:        extra version tag
        :param verbose     add verbose prints/logs
        """
        return deploy_function(
            function,
            dashboard=dashboard,
            models=models,
            env=env,
            tag=tag,
            verbose=verbose,
            project_object=self,
        )


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
        self.artifact_path = artifact_path

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
    def set_function(
        self,
        func: typing.Union[str, mlrun.runtimes.BaseRuntime],
        name: str = "",
        kind: str = "",
        image: str = None,
        with_repo: bool = None,
    ):
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
    handler = f.get("handler", None)
    with_repo = f.get("with_repo", False)
    requirements = f.get("requirements", None)

    in_context = False
    if not url and "spec" not in f:
        raise ValueError("function missing a url or a spec")

    relative_url = url
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
        # not defaulting kind to job here cause kind might come from magic annotations in the notebook
        func = code_to_function(
            name, filename=url, image=image, kind=kind, handler=handler
        )
    elif url.endswith(".py"):
        if not image and kind != "local":
            raise ValueError(
                "image must be provided with py code files which do not "
                "run on 'local' engine kind"
            )
        if in_context and with_repo:
            func = new_function(
                name, command=relative_url, image=image, kind=kind or "job"
            )
            if handler:
                func.spec.default_handler = handler
        else:
            func = code_to_function(
                name, filename=url, image=image, kind=kind or "job", handler=handler
            )
    else:
        raise ValueError(f"unsupported function url {url} or no spec")

    if with_repo:
        func.spec.build.source = "./"
    if requirements:
        func.with_requirements(requirements)

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
