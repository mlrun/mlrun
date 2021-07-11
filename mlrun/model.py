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

import inspect
import re
import time
import warnings
from collections import OrderedDict
from copy import deepcopy
from os import environ
from typing import Dict, List, Optional, Tuple, Union

import mlrun

from .config import config
from .utils import dict_to_json, dict_to_yaml, get_artifact_target


class ModelObj:
    _dict_fields = []

    @staticmethod
    def _verify_list(param, name):
        if not isinstance(param, list):
            raise ValueError(f"parameter {name} must be a list")

    @staticmethod
    def _verify_dict(param, name, new_type=None):
        if (
            param is not None
            and not isinstance(param, dict)
            and not hasattr(param, "to_dict")
        ):
            raise ValueError(f"parameter {name} must be a dict or object")
        if new_type and (isinstance(param, dict) or param is None):
            return new_type.from_dict(param)
        return param

    def to_dict(self, fields=None, exclude=None):
        """convert the object to a python dictionary"""
        struct = {}
        fields = fields or self._dict_fields
        if not fields:
            fields = list(inspect.signature(self.__init__).parameters.keys())
        for t in fields:
            if not exclude or t not in exclude:
                val = getattr(self, t, None)
                if val is not None and not (isinstance(val, dict) and not val):
                    if hasattr(val, "to_dict"):
                        val = val.to_dict()
                        if val:
                            struct[t] = val
                    else:
                        struct[t] = val
        return struct

    @classmethod
    def from_dict(cls, struct=None, fields=None, deprecated_fields: dict = None):
        """create an object from a python dictionary"""
        struct = {} if struct is None else struct
        deprecated_fields = deprecated_fields or {}
        fields = fields or cls._dict_fields
        if not fields:
            fields = list(inspect.signature(cls.__init__).parameters.keys())
        new_obj = cls()
        if struct:
            for key, val in struct.items():
                if key in fields and key not in deprecated_fields:
                    setattr(new_obj, key, val)
            for deprecated_field, new_field in deprecated_fields.items():
                field_value = struct.get(new_field) or struct.get(deprecated_field)
                if field_value:
                    setattr(new_obj, new_field, field_value)

        return new_obj

    def to_yaml(self):
        """convert the object to yaml"""
        return dict_to_yaml(self.to_dict())

    def to_json(self):
        """convert the object to json"""
        return dict_to_json(self.to_dict())

    def to_str(self):
        """convert the object to string (with dict layout)"""
        return self.__str__()

    def __str__(self):
        return str(self.to_dict())

    def copy(self):
        """create a copy of the object"""
        return deepcopy(self)


# model class for building ModelObj dictionaries
class ObjectDict:
    def __init__(self, classes_map, default_kind=""):
        self._children = OrderedDict()
        self._default_kind = default_kind
        self._classes_map = classes_map

    def values(self):
        return self._children.values()

    def keys(self):
        return self._children.keys()

    def items(self):
        return self._children.items()

    def __len__(self):
        return len(self._children)

    def __iter__(self):
        yield from self._children.keys()

    def __getitem__(self, name):
        return self._children[name]

    def __setitem__(self, key, item):
        self._children[key] = self._get_child_object(item, key)

    def __delitem__(self, key):
        del self._children[key]

    def update(self, key, item):
        child = self._get_child_object(item, key)
        self._children[key] = child
        return child

    def to_dict(self):
        return {k: v.to_dict() for k, v in self._children.items()}

    @classmethod
    def from_dict(cls, classes_map: dict, children=None, default_kind=""):
        if children is None:
            return cls(classes_map, default_kind)
        if not isinstance(children, dict):
            raise ValueError("children must be a dict")

        new_obj = cls(classes_map, default_kind)
        for name, child in children.items():
            child_obj = new_obj._get_child_object(child, name)
            new_obj._children[name] = child_obj

        return new_obj

    def _get_child_object(self, child, name):
        if hasattr(child, "kind") and child.kind in self._classes_map.keys():
            child.name = name
            return child
        elif isinstance(child, dict):
            kind = child.get("kind", self._default_kind)
            if kind not in self._classes_map.keys():
                raise ValueError(f"illegal object kind {kind}")
            child_obj = self._classes_map[kind].from_dict(child)
            child_obj.name = name
            return child_obj
        else:
            raise ValueError(f"illegal child (should be dict or child kind), {child}")

    def to_yaml(self):
        return dict_to_yaml(self.to_dict())

    def to_json(self):
        return dict_to_json(self.to_dict())

    def to_str(self):
        return self.__str__()

    def __str__(self):
        return str(self.to_dict())

    def copy(self):
        return deepcopy(self)


class ObjectList:
    def __init__(self, child_class):
        self._children = OrderedDict()
        self._child_class = child_class

    def values(self):
        return self._children.values()

    def keys(self):
        return self._children.keys()

    def items(self):
        return self._children.items()

    def __len__(self):
        return len(self._children)

    def __iter__(self):
        yield from self._children.values()

    def __getitem__(self, name):
        if isinstance(name, int):
            return list(self._children.values())[name]
        return self._children[name]

    def __setitem__(self, key, item):
        self.update(item, key)

    def __delitem__(self, key):
        del self._children[key]

    def to_dict(self):
        # method used by ModelObj class to serialize the object to nested dict
        return [t.to_dict() for t in self._children.values()]

    @classmethod
    def from_list(cls, child_class, children=None):
        if children is None:
            return cls(child_class)
        if not isinstance(children, list):
            raise ValueError("states must be a list")

        new_obj = cls(child_class)
        for child in children:
            name, child_obj = new_obj._get_child_object(child)
            new_obj._children[name] = child_obj
        return new_obj

    def _get_child_object(self, child):
        if isinstance(child, self._child_class):
            return child.name, child
        elif isinstance(child, dict):
            if "name" not in child.keys():
                raise ValueError("illegal object no 'name' field")
            child_obj = self._child_class.from_dict(child)
            return child_obj.name, child_obj
        else:
            raise ValueError(f"illegal child (should be dict or child kind), {child}")

    def update(self, child, name=None):
        object_name, child_obj = self._get_child_object(child)
        child_obj.name = name or object_name
        self._children[child_obj.name] = child_obj
        return child_obj


class BaseMetadata(ModelObj):
    def __init__(
        self,
        name=None,
        tag=None,
        hash=None,
        namespace=None,
        project=None,
        labels=None,
        annotations=None,
        categories=None,
        updated=None,
    ):
        self.name = name
        self.tag = tag
        self.hash = hash
        self.namespace = namespace
        self.project = project or config.default_project
        self.labels = labels or {}
        self.categories = categories or []
        self.annotations = annotations or {}
        self.updated = updated


class ImageBuilder(ModelObj):
    """An Image builder"""

    def __init__(
        self,
        functionSourceCode=None,
        source=None,
        image=None,
        base_image=None,
        commands=None,
        extra=None,
        secret=None,
        code_origin=None,
        registry=None,
        load_source_on_run=None,
    ):
        self.functionSourceCode = functionSourceCode  #: functionSourceCode
        self.codeEntryType = ""  #: codeEntryType
        self.codeEntryAttributes = ""  #: codeEntryAttributes
        self.source = source  #: source
        self.code_origin = code_origin  #: code_origin
        self.image = image  #: image
        self.base_image = base_image  #: base_image
        self.commands = commands or []  #: commands
        self.extra = extra  #: extra
        self.secret = secret  #: secret
        self.registry = registry  #: registry
        self.load_source_on_run = load_source_on_run  #: load_source_on_run
        self.build_pod = None


class RunMetadata(ModelObj):
    """Run metadata"""

    def __init__(
        self,
        uid=None,
        name=None,
        project=None,
        labels=None,
        annotations=None,
        iteration=None,
    ):
        self.uid = uid
        self._iteration = iteration
        self.name = name
        self.project = project
        self.labels = labels or {}
        self.annotations = annotations or {}

    @property
    def iteration(self):
        return self._iteration or 0

    @iteration.setter
    def iteration(self, iteration):
        self._iteration = iteration


class HyperParamStrategies:
    grid = "grid"
    list = "list"
    random = "random"
    custom = "custom"

    @staticmethod
    def all():
        return [
            HyperParamStrategies.grid,
            HyperParamStrategies.list,
            HyperParamStrategies.random,
            HyperParamStrategies.custom,
        ]


class HyperParamOptions(ModelObj):
    """Hyper Parameter Options

    Parameters:
        param_file (str):       hyper params input file path/url, instead of inline
        strategy (str):         hyper param strategy - grid, list or random
        selector (str):         selection criteria for best result ([min|max.]<result>), e.g. max.accuracy
        stop_condition (str):   early stop condition e.g. "accuracy > 0.9"
        parallel_runs (int):    number of param combinations to run in parallel (over Dask)
        dask_cluster_uri (str): db uri for a deployed dask cluster function, e.g. db://myproject/dask
        max_iterations (int):   max number of runs (in random strategy)
        max_errors (int):       max number of child runs errors for the overall job to fail
        teardown_dask (bool):   kill the dask cluster pods after the runs
    """

    def __init__(
        self,
        param_file=None,
        strategy=None,
        selector: HyperParamStrategies = None,
        stop_condition=None,
        parallel_runs=None,
        dask_cluster_uri=None,
        max_iterations=None,
        max_errors=None,
        teardown_dask=None,
    ):
        self.param_file = param_file
        self.strategy = strategy
        self.selector = selector
        self.stop_condition = stop_condition
        self.max_iterations = max_iterations
        self.max_errors = max_errors
        self.parallel_runs = parallel_runs
        self.dask_cluster_uri = dask_cluster_uri
        self.teardown_dask = teardown_dask

    def validate(self):
        if self.strategy and self.strategy not in HyperParamStrategies.all():
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"illegal hyper param strategy, use {','.join(HyperParamStrategies.all())}"
            )
        if self.max_iterations and self.strategy != HyperParamStrategies.random:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "max_iterations is only valid in random strategy"
            )


class RunSpec(ModelObj):
    """Run specification"""

    def __init__(
        self,
        parameters=None,
        hyperparams=None,
        param_file=None,
        selector=None,
        handler=None,
        inputs=None,
        outputs=None,
        input_path=None,
        output_path=None,
        function=None,
        secret_sources=None,
        data_stores=None,
        strategy=None,
        verbose=None,
        scrape_metrics=None,
        hyper_param_options=None,
    ):

        self._hyper_param_options = None
        self._inputs = inputs
        self._outputs = outputs

        self.hyper_param_options = hyper_param_options
        self.parameters = parameters or {}
        self.hyperparams = hyperparams or {}
        self.param_file = param_file
        self.strategy = strategy
        self.selector = selector
        self.handler = handler
        self.input_path = input_path
        self.output_path = output_path
        self.function = function
        self._secret_sources = secret_sources or []
        self._data_stores = data_stores
        self.verbose = verbose
        self.scrape_metrics = scrape_metrics

    def to_dict(self, fields=None, exclude=None):
        struct = super().to_dict(fields, exclude=["handler"])
        if self.handler and isinstance(self.handler, str):
            struct["handler"] = self.handler
        return struct

    def is_hyper_job(self):
        param_file = self.param_file or self.hyper_param_options.param_file
        return param_file or self.hyperparams

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = self._verify_dict(inputs, "inputs")

    @property
    def hyper_param_options(self) -> HyperParamOptions:
        return self._hyper_param_options

    @hyper_param_options.setter
    def hyper_param_options(self, hyper_param_options):
        self._hyper_param_options = self._verify_dict(
            hyper_param_options, "hyper_param_options", HyperParamOptions
        )

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        self._verify_list(outputs, "outputs")
        self._outputs = outputs

    @property
    def secret_sources(self):
        return self._secret_sources

    @secret_sources.setter
    def secret_sources(self, secret_sources):
        self._verify_list(secret_sources, "secret_sources")
        self._secret_sources = secret_sources

    @property
    def data_stores(self):
        return self._data_stores

    @data_stores.setter
    def data_stores(self, data_stores):
        self._verify_list(data_stores, "data_stores")
        self._data_stores = data_stores

    @property
    def handler_name(self):
        if self.handler:
            if inspect.isfunction(self.handler):
                return self.handler.__name__
            else:
                return str(self.handler)
        return ""


class RunStatus(ModelObj):
    """Run status"""

    def __init__(
        self,
        state=None,
        error=None,
        host=None,
        commit=None,
        status_text=None,
        results=None,
        artifacts=None,
        start_time=None,
        last_update=None,
        iterations=None,
        ui_url=None,
    ):
        self.state = state or "created"
        self.status_text = status_text
        self.error = error
        self.host = host
        self.commit = commit
        self.results = results
        self.artifacts = artifacts
        self.start_time = start_time
        self.last_update = last_update
        self.iterations = iterations
        self.ui_url = ui_url


class RunTemplate(ModelObj):
    """Run template"""

    def __init__(self, spec: RunSpec = None, metadata: RunMetadata = None):
        self._spec = None
        self._metadata = None
        self.spec = spec
        self.metadata = metadata

    @property
    def spec(self) -> RunSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", RunSpec)

    @property
    def metadata(self) -> RunMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, "metadata", RunMetadata)

    def with_params(self, **kwargs):
        """set task parameters using key=value, key2=value2, .."""
        self.spec.parameters = kwargs
        return self

    def with_input(self, key, path):
        """set task data input, path is an Mlrun global DataItem uri

        examples::

            task.with_input("data", "/file-dir/path/to/file")
            task.with_input("data", "s3://<bucket>/path/to/file")
            task.with_input("data", "v3io://[<remote-host>]/<data-container>/path/to/file")
        """
        if not self.spec.inputs:
            self.spec.inputs = {}
        self.spec.inputs[key] = path
        return self

    def with_hyper_params(
        self,
        hyperparams,
        selector=None,
        strategy: HyperParamStrategies = None,
        **options,
    ):
        """set hyper param values and configurations,
        see parameters in: :py:class:`HyperParamOptions`

        example::

            grid_params = {"p1": [2,4,1], "p2": [10,20]}
            task = mlrun.new_task("grid-search")
            task.with_hyper_params(grid_params, selector="max.accuracy")
        """
        self.spec.hyperparams = hyperparams
        self.spec.hyper_param_options = options
        self.spec.hyper_param_options.selector = selector
        self.spec.hyper_param_options.strategy = strategy
        self.spec.hyper_param_options.validate()
        return self

    def with_param_file(
        self,
        param_file,
        selector=None,
        strategy: HyperParamStrategies = None,
        **options,
    ):
        """set hyper param values (from a file url) and configurations,
        see parameters in: :py:class:`HyperParamOptions`

        example::

            grid_params = "s3://<my-bucket>/path/to/params.json"
            task = mlrun.new_task("grid-search")
            task.with_param_file(grid_params, selector="max.accuracy")
        """
        self.spec.hyper_param_options = options
        self.spec.hyper_param_options.param_file = param_file
        self.spec.hyper_param_options.selector = selector
        self.spec.hyper_param_options.strategy = strategy
        self.spec.hyper_param_options.validate()
        return self

    def with_secrets(self, kind, source):
        """register a secrets source (file, env or dict)

        read secrets from a source provider to be used in workflows, example::

            task.with_secrets('file', 'file.txt')
            task.with_secrets('inline', {'key': 'val'})
            task.with_secrets('env', 'ENV1,ENV2')

            task.with_secrets('vault', ['secret1', 'secret2'...])

            # If using with k8s secrets, the k8s secret is managed by MLRun, through the project-secrets
            # mechanism. The secrets will be attached to the running pod as environment variables.
            task.with_secrets('kubernetes', ['secret1', 'secret2'])

            # If using an empty secrets list [] then all accessible secrets will be available.
            task.with_secrets('vault', [])

            # To use with Azure key vault, a k8s secret must be created with the following keys:
            # kubectl -n <namespace> create secret generic azure-key-vault-secret \\
            #     --from-literal=tenant_id=<service principal tenant ID> \\
            #     --from-literal=client_id=<service principal client ID> \\
            #     --from-literal=secret=<service principal secret key>

            task.with_secrets('azure_vault', {
                'name': 'my-vault-name',
                'k8s_secret': 'azure-key-vault-secret',
                # An empty secrets list may be passed ('secrets': []) to access all vault secrets.
                'secrets': ['secret1', 'secret2'...]
            })

        :param kind:   secret type (file, inline, env)
        :param source: secret data or link (see example)

        :returns: The RunTemplate object
        """

        if kind == "vault" and isinstance(source, list):
            source = {"project": self.metadata.project, "secrets": source}

        self.spec.secret_sources.append({"kind": kind, "source": source})
        return self

    def set_label(self, key, value):
        """set a key/value label for the task"""
        self.metadata.labels[key] = str(value)
        return self

    def to_env(self):
        environ["MLRUN_EXEC_CONFIG"] = self.to_json()


class RunObject(RunTemplate):
    """A run"""

    def __init__(
        self,
        spec: RunSpec = None,
        metadata: RunMetadata = None,
        status: RunStatus = None,
    ):
        super().__init__(spec, metadata)
        self._status = None
        self.status = status
        self.outputs_wait_for_completion = True

    @classmethod
    def from_template(cls, template: RunTemplate):
        return cls(template.spec, template.metadata)

    @property
    def status(self) -> RunStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", RunStatus)

    def output(self, key):
        """return the value of a specific result or artifact by key"""
        if self.outputs_wait_for_completion:
            self.wait_for_completion()
        if self.status.results and key in self.status.results:
            return self.status.results.get(key)
        artifact = self._artifact(key)
        if artifact:
            return get_artifact_target(artifact, self.metadata.project)
        return None

    @property
    def ui_url(self) -> str:
        """UI URL (for relevant runtimes)"""
        self.refresh()
        if not self._status.ui_url:
            print("UI currently not available (status={})".format(self._status.state))
        return self._status.ui_url

    @property
    def outputs(self):
        """return a dict of outputs, result values and artifact uris"""
        outputs = {}
        if self.outputs_wait_for_completion:
            self.wait_for_completion()
        if self.status.results:
            outputs = {k: v for k, v in self.status.results.items()}
        if self.status.artifacts:
            for a in self.status.artifacts:
                outputs[a["key"]] = get_artifact_target(a, self.metadata.project)
        return outputs

    def artifact(self, key) -> "mlrun.DataItem":
        """return artifact DataItem by key"""
        if self.outputs_wait_for_completion:
            self.wait_for_completion()
        artifact = self._artifact(key)
        if artifact:
            uri = get_artifact_target(artifact, self.metadata.project)
            if uri:
                return mlrun.get_dataitem(uri)
        return None

    def _artifact(self, key):
        """return artifact DataItem by key"""
        if self.status.artifacts:
            for a in self.status.artifacts:
                if a["key"] == key:
                    return a
        return None

    def uid(self):
        """run unique id"""
        return self.metadata.uid

    def state(self):
        """current run state"""
        if self.status.state in mlrun.runtimes.constants.RunStates.terminal_states():
            return self.status.state
        self.refresh()
        return self.status.state or "unknown"

    def refresh(self):
        """refresh run state from the db"""
        db = mlrun.get_run_db()
        run = db.read_run(
            uid=self.metadata.uid,
            project=self.metadata.project,
            iter=self.metadata.iteration,
        )
        if run:
            self.status = RunStatus.from_dict(run.get("status", {}))
            self.status.from_dict(run.get("status", {}))
            return self

    def show(self):
        """show the current status widget, in jupyter notebook"""
        db = mlrun.get_run_db()
        db.list_runs(uid=self.metadata.uid, project=self.metadata.project).show()

    def logs(self, watch=True, db=None):
        """return or watch on the run logs"""
        if not db:
            db = mlrun.get_run_db()
        if not db:
            print("DB is not configured, cannot show logs")
            return None

        if db.kind == "http":
            state = db.watch_log(self.metadata.uid, self.metadata.project, watch=watch)
        else:
            state, text = db.get_log(self.metadata.uid, self.metadata.project)
            if text:
                print(text.decode())

        if state:
            print(f"final state: {state}")
        return state

    def wait_for_completion(self, sleep=3, timeout=0, raise_on_failure=True):
        """wait for async run to complete"""
        total_time = 0
        while True:
            state = self.state()
            if state in mlrun.runtimes.constants.RunStates.terminal_states():
                break
            time.sleep(sleep)
            total_time += sleep
            if timeout and total_time > timeout:
                raise mlrun.errors.MLRunTimeoutError(
                    "Run did not reach terminal state on time"
                )
        if raise_on_failure and state != mlrun.runtimes.constants.RunStates.completed:
            self.logs(watch=False)
            raise mlrun.errors.MLRunRuntimeError(
                f"task {self.metadata.name} did not complete (state={state})"
            )
        return state

    @staticmethod
    def create_uri(project: str, uid: str, iteration: Union[int, str], tag: str = ""):
        if tag:
            tag = f":{tag}"
        iteration = str(iteration)
        return f"{project}@{uid}#{iteration}{tag}"

    @staticmethod
    def parse_uri(uri: str) -> Tuple[str, str, str, str]:
        uri_pattern = (
            r"^(?P<project>.*)@(?P<uid>.*)\#(?P<iteration>.*?)(:(?P<tag>.*))?$"
        )
        match = re.match(uri_pattern, uri)
        if not match:
            raise ValueError(
                "Uri not in supported format <project>@<uid>#<iteration>[:tag]"
            )
        group_dict = match.groupdict()
        return (
            group_dict["project"],
            group_dict["uid"],
            group_dict["iteration"],
            group_dict["tag"],
        )


class EntrypointParam(ModelObj):
    def __init__(self, name="", type=None, default=None, doc=""):
        self.name = name
        self.type = type
        self.default = default
        self.doc = doc


class FunctionEntrypoint(ModelObj):
    def __init__(self, name="", doc="", parameters=None, outputs=None, lineno=-1):
        self.name = name
        self.doc = doc
        self.parameters = [] if parameters is None else parameters
        self.outputs = [] if outputs is None else outputs
        self.lineno = lineno


# TODO: remove in 0.9.0
def NewTask(
    name=None,
    project=None,
    handler=None,
    params=None,
    hyper_params=None,
    param_file=None,
    selector=None,
    strategy=None,
    inputs=None,
    outputs=None,
    in_path=None,
    out_path=None,
    artifact_path=None,
    secrets=None,
    base=None,
):
    """Creates a new task - see new_task
    """
    warnings.warn(
        "NewTask will be deprecated in 0.7.0, and will be removed in 0.9.0, use new_task instead",
        # TODO: In 0.7.0 and replace NewTask to new_task in examples & demos
        PendingDeprecationWarning,
    )
    return new_task(
        name,
        project,
        handler,
        params,
        hyper_params,
        param_file,
        selector,
        strategy,
        inputs,
        outputs,
        in_path,
        out_path,
        artifact_path,
        secrets,
        base,
    )


def new_task(
    name=None,
    project=None,
    handler=None,
    params=None,
    hyper_params=None,
    param_file=None,
    selector=None,
    hyper_param_options=None,
    inputs=None,
    outputs=None,
    in_path=None,
    out_path=None,
    artifact_path=None,
    secrets=None,
    base=None,
):
    """Creates a new task

    :param name:            task name
    :param project:         task project
    :param handler:         code entry-point/handler name
    :param params:          input parameters (dict)
    :param hyper_params:    dictionary of hyper parameters and list values, each
                            hyper param holds a list of values, the run will be
                            executed for every parameter combination (GridSearch)
    :param param_file:      a csv file with parameter combinations, first row hold
                            the parameter names, following rows hold param values
    :param selector:        selection criteria for hyper params e.g. "max.accuracy"
    :param hyper_param_options:   hyper parameter options, see: :py:class:`HyperParamOptions`
    :param inputs:          dictionary of input objects + optional paths (if path is
                            omitted the path will be the in_path/key.
    :param outputs:         dictionary of input objects + optional paths (if path is
                            omitted the path will be the out_path/key.
    :param in_path:         default input path/url (prefix) for inputs
    :param out_path:        default output path/url (prefix) for artifacts
    :param artifact_path:   default artifact output path
    :param secrets:         extra secrets specs, will be injected into the runtime
                            e.g. ['file=<filename>', 'env=ENV_KEY1,ENV_KEY2']
    :param base:            task instance to use as a base instead of a fresh new task instance
    """

    if base:
        run = deepcopy(base)
    else:
        run = RunTemplate()
    run.metadata.name = name or run.metadata.name
    run.metadata.project = project or run.metadata.project
    run.spec.handler = handler or run.spec.handler
    run.spec.parameters = params or run.spec.parameters
    run.spec.inputs = inputs or run.spec.inputs
    run.spec.outputs = outputs or run.spec.outputs or []
    run.spec.input_path = in_path or run.spec.input_path
    run.spec.output_path = artifact_path or out_path or run.spec.output_path
    run.spec.secret_sources = secrets or run.spec.secret_sources or []

    run.spec.hyperparams = hyper_params or run.spec.hyperparams
    run.spec.hyper_param_options = hyper_param_options or run.spec.hyper_param_options
    run.spec.hyper_param_options.param_file = (
        param_file or run.spec.hyper_param_options.param_file
    )
    run.spec.hyper_param_options.selector = (
        selector or run.spec.hyper_param_options.selector
    )
    return run


class DataSource(ModelObj):
    """online or offline data source spec"""

    _dict_fields = [
        "kind",
        "name",
        "path",
        "attributes",
        "key_field",
        "time_field",
        "schedule",
        "online",
        "workers",
        "max_age",
    ]
    kind = None

    def __init__(
        self,
        name: str = None,
        path: str = None,
        attributes: Dict[str, str] = None,
        key_field: str = None,
        time_field: str = None,
        schedule: str = None,
    ):
        self.name = name
        self.path = str(path)
        self.attributes = attributes
        self.schedule = schedule
        self.key_field = key_field
        self.time_field = time_field

        self.online = None
        self.max_age = None
        self.workers = None
        self._secrets = {}

    def set_secrets(self, secrets):
        self._secrets = secrets


class DataTargetBase(ModelObj):
    """data target spec, specify a destination for the feature set data"""

    _dict_fields = [
        "name",
        "kind",
        "path",
        "after_step",
        "attributes",
        "partitioned",
        "key_bucketing_number",
        "partition_cols",
        "time_partitioning_granularity",
        "max_events",
        "flush_after_seconds",
    ]

    # TODO - remove once "after_state" is fully deprecated
    @classmethod
    def from_dict(cls, struct=None, fields=None):
        return super().from_dict(
            struct, fields=fields, deprecated_fields={"after_state": "after_step"}
        )

    def __init__(
        self,
        kind: str = None,
        name: str = "",
        path=None,
        attributes: Dict[str, str] = None,
        after_step=None,
        partitioned: bool = False,
        key_bucketing_number: Optional[int] = None,
        partition_cols: Optional[List[str]] = None,
        time_partitioning_granularity: Optional[str] = None,
        max_events: Optional[int] = None,
        flush_after_seconds: Optional[int] = None,
        after_state=None,
    ):
        if after_state:
            warnings.warn(
                "The after_state parameter is deprecated. Use after_step instead",
                # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
                PendingDeprecationWarning,
            )
            after_step = after_step or after_state

        self.name = name
        self.kind: str = kind
        self.path = path
        self.after_step = after_step
        self.attributes = attributes or {}
        self.partitioned = partitioned
        self.key_bucketing_number = key_bucketing_number
        self.partition_cols = partition_cols
        self.time_partitioning_granularity = time_partitioning_granularity
        self.max_events = max_events
        self.flush_after_seconds = flush_after_seconds


class FeatureSetProducer(ModelObj):
    """information about the task/job which produced the feature set data"""

    def __init__(self, kind=None, name=None, uri=None, owner=None, sources=None):
        self.kind = kind
        self.name = name
        self.owner = owner
        self.uri = uri
        self.sources = sources or {}


class DataTarget(DataTargetBase):
    """data target with extra status information (used in the feature-set/vector status)"""

    _dict_fields = [
        "name",
        "kind",
        "path",
        "start_time",
        "online",
        "status",
        "updated",
        "size",
    ]

    def __init__(
        self, kind: str = None, name: str = "", path=None, online=None,
    ):
        super().__init__(kind, name, path)
        self.status = ""
        self.updated = None
        self.size = None
        self.online = online
        self.max_age = None
        self.start_time = None
        self._producer = None
        self.producer = {}

    @property
    def producer(self) -> FeatureSetProducer:
        return self._producer

    @producer.setter
    def producer(self, producer):
        self._producer = self._verify_dict(producer, "producer", FeatureSetProducer)


class VersionedObjMetadata(ModelObj):
    def __init__(
        self,
        name: str = None,
        tag: str = None,
        uid: str = None,
        project: str = None,
        labels: Dict[str, str] = None,
        annotations: Dict[str, str] = None,
        updated=None,
    ):
        self.name = name
        self.tag = tag
        self.uid = uid
        self.project = project
        self.labels = labels or {}
        self.annotations = annotations or {}
        self.updated = updated
