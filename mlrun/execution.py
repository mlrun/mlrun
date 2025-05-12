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

import logging
import os
import uuid
import warnings
from copy import deepcopy
from typing import Optional, Union, cast

import numpy as np
import yaml
from dateutil import parser

import mlrun
import mlrun.common.constants as mlrun_constants
import mlrun.common.formatters
from mlrun.artifacts import (
    Artifact,
    DatasetArtifact,
    DocumentArtifact,
    DocumentLoaderSpec,
    ModelArtifact,
)
from mlrun.datastore.store_resources import get_store_resource
from mlrun.errors import MLRunInvalidArgumentError

from .artifacts.manager import ArtifactManager, dict_to_artifact, extend_artifact_path
from .datastore import store_manager
from .features import Feature
from .model import HyperParamOptions
from .secrets import SecretsStore
from .utils import (
    Logger,
    RunKeys,
    dict_to_json,
    dict_to_yaml,
    get_in,
    is_relative_path,
    logger,
    now_date,
    to_date_str,
    update_in,
)


class MLClientCtx:
    """ML Execution Client Context

    The context is generated and injected to the function using the ``function.run()``
    or manually using the :py:func:`~mlrun.run.get_or_create_ctx` call
    and provides an interface to use run params, metadata, inputs, and outputs.

    Base metadata include: uid, name, project, and iteration (for hyper params).
    Users can set labels and annotations using :py:func:`~set_label`, :py:func:`~set_annotation`.
    Access parameters and secrets using :py:func:`~get_param`, :py:func:`~get_secret`.
    Access input data objects using :py:func:`~get_input`.
    Store results, artifacts, and real-time metrics using the :py:func:`~log_result`,
    :py:func:`~log_artifact`, :py:func:`~log_dataset` and :py:func:`~log_model` methods.

    See doc for the individual params and methods
    """

    kind = "run"

    def __init__(self, autocommit=False, tmp="", log_stream=None):
        self._uid = ""
        self.name = ""
        self._iteration = 0
        self._project = ""
        self._tag = ""
        self._secrets_manager = SecretsStore()

        # Runtime db service interfaces
        self._rundb = None
        self._tmpfile = tmp
        self._logger = log_stream or logger
        self._log_level = "info"
        self._autocommit = autocommit
        self._notifications = []
        self._state_thresholds = {}

        self._labels = {}
        self._annotations = {}
        self._node_selector = {}

        self._function = ""
        self._parameters = {}
        self._hyperparams = {}
        self._hyper_param_options = HyperParamOptions()
        self._in_path = ""
        self.artifact_path = ""
        self._inputs = {}
        self._outputs = []

        self._results = {}
        # Tracks the execution state, completion of runs is not decided by the execution
        # as there may be multiple executions for a single run (e.g mpi)
        self._state = "created"
        self._error = None
        self._commit = ""
        self._host = None
        self._start_time = self._last_update = now_date()
        self._iteration_results = None
        self._children = []
        self._parent = None
        self._handler = None

        self._project_object = None
        self._allow_empty_resources = None
        self._reset_on_run = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_value:
            self.set_state(error=exc_value, commit=False)
        self.commit(completed=True)

    @property
    def uid(self):
        """Unique run id"""
        if self._iteration:
            return f"{self._uid}-{self._iteration}"
        return self._uid

    @property
    def tag(self):
        """Run tag (uid or workflow id if exists)"""
        return (
            self._labels.get(mlrun_constants.MLRunInternalLabels.workflow) or self._uid
        )

    @property
    def state(self):
        """Execution state"""
        return self._state

    @property
    def iteration(self):
        """Child iteration index, for hyperparameters"""
        return self._iteration

    @property
    def project(self):
        """Project name, runs can be categorized by projects"""
        return self._project

    @property
    def logger(self) -> Logger:
        """Built-in logger interface

        Example::

            context.logger.info("Started experiment..", param=5)

        """
        return self._logger

    @property
    def log_level(self):
        """Get the logging level, e.g. 'debug', 'info', 'error'"""
        return self._log_level

    @log_level.setter
    def log_level(self, value: str):
        """Set the logging level, e.g. 'debug', 'info', 'error'"""
        level = logging.getLevelName(value.upper())
        self._logger.set_logger_level(level)
        self._log_level = value

    @property
    def parameters(self):
        """Dictionary of run parameters (read-only)"""
        return deepcopy(self._parameters)

    @property
    def inputs(self):
        """Dictionary of input data item urls (read-only)"""
        return self._inputs

    @property
    def results(self):
        """Dictionary of results (read-only)"""
        return deepcopy(self._results)

    @property
    def artifacts(self):
        """Dictionary of artifacts (read-only)"""
        return deepcopy(self._artifacts_manager.artifact_list())

    @property
    def artifact_uris(self):
        """Dictionary of artifact URIs (read-only)"""
        return deepcopy(self._artifacts_manager.artifact_uris)

    @property
    def in_path(self):
        """Default input path for data objects"""
        return self._in_path

    @property
    def out_path(self):
        """Default output path for artifacts"""
        logger.warning("out_path will soon be deprecated, use artifact_path")
        return self.artifact_path

    @property
    def labels(self):
        """Dictionary with labels (read-only)"""
        return deepcopy(self._labels)

    @property
    def node_selector(self):
        """Dictionary with node selectors (read-only)"""
        return deepcopy(self._node_selector)

    @property
    def annotations(self):
        """Dictionary with annotations (read-only)"""
        return deepcopy(self._annotations)

    def get_child_context(self, with_parent_params=False, **params):
        """Get child context (iteration)

        Allow sub experiments (epochs, hyper-param, ..) under a parent
        will create a new iteration, log_xx will update the child only
        use commit_children() to save all the children and specify the best run.

        Example::

            def handler(context: mlrun.MLClientCtx, data: mlrun.DataItem):
                df = data.as_df()
                best_accuracy = accuracy_sum = 0
                for param in param_list:
                    with context.get_child_context(myparam=param) as child:
                        accuracy = child_handler(child, df, **child.parameters)
                        accuracy_sum += accuracy
                        child.log_result("accuracy", accuracy)
                        if accuracy > best_accuracy:
                            child.mark_as_best()
                            best_accuracy = accuracy

                context.log_result("avg_accuracy", accuracy_sum / len(param_list))

        :param params:  Extra (or override) params to parent context
        :param with_parent_params:  Child will copy the parent parameters and add to them

        :return: Child context
        """
        if self.iteration != 0:
            raise MLRunInvalidArgumentError(
                "cannot create child from a child iteration!"
            )
        ctx = deepcopy(self.to_dict())
        if not with_parent_params:
            update_in(ctx, ["spec", "parameters"], {})
        if params:
            for key, val in params.items():
                update_in(ctx, ["spec", "parameters", key], val)

        update_in(ctx, ["metadata", "iteration"], len(self._children) + 1)
        ctx["status"] = {}
        ctx = MLClientCtx.from_dict(
            ctx, self._rundb, self._autocommit, log_stream=self._logger
        )
        ctx._parent = self
        self._children.append(ctx)
        return ctx

    def update_child_iterations(
        self, best_run=0, commit_children=False, completed=True
    ):
        """Update children results in the parent, and optionally mark the best.

        :param best_run:  Marks the child iteration number (starts from 1)
        :param commit_children:  Commit all child runs to the db
        :param completed:  Mark children as completed
        """
        if not self._children:
            return
        if commit_children:
            for child in self._children:
                child.commit(completed=completed)
        results = [child.to_dict() for child in self._children]
        summary, df = mlrun.runtimes.utils.results_to_iter(results, None, self)
        task = results[best_run - 1] if best_run else None
        self.log_iteration_results(best_run, summary, task)
        mlrun.runtimes.utils.log_iter_artifacts(self, df, summary[0])

    def mark_as_best(self):
        """mark a child as the best iteration result, see .get_child_context()"""
        if not self._parent or not self._iteration:
            raise MLRunInvalidArgumentError(
                "can only mark a child run as best iteration"
            )
        self._parent.log_iteration_results(self._iteration, None, self.to_dict())

    def get_store_resource(self, url, secrets: Optional[dict] = None):
        """Get mlrun data resource (feature set/vector, artifact, item) from url.

        Example::

            feature_vector = context.get_store_resource(
                "store://feature-vectors/default/myvec"
            )
            dataset = context.get_store_resource("store://artifacts/default/mydata")

        :param url:    Store resource uri/path, store://<type>/<project>/<name>:<version>
                       Types: artifacts | feature-sets | feature-vectors
        :param secrets: Additional secrets to use when accessing the store resource
        """
        return get_store_resource(
            url,
            db=self._rundb,
            secrets=self._secrets_manager,
            data_store_secrets=secrets,
        )

    def get_dataitem(self, url, secrets: Optional[dict] = None):
        """Get mlrun dataitem from url

        Example::

            data = context.get_dataitem("s3://my-bucket/file.csv").as_df()

        :param url:    Data-item uri/path
        :param secrets: Additional secrets to use when accessing the data-item
        """
        return self._data_stores.object(url=url, secrets=secrets)

    def set_logger_stream(self, stream):
        self._logger.replace_handler_stream("default", stream)

    def get_meta(self) -> dict:
        """Reserved for internal use"""
        uri = f"{self._project}/{self.uid}" if self._project else self.uid
        resp = {
            "name": self.name,
            "kind": "run",
            "uri": uri,
            "owner": get_in(self._labels, mlrun_constants.MLRunInternalLabels.owner),
        }
        if mlrun_constants.MLRunInternalLabels.workflow in self._labels:
            resp[mlrun_constants.MLRunInternalLabels.workflow] = self._labels[
                mlrun_constants.MLRunInternalLabels.workflow
            ]
        return resp

    @classmethod
    def from_dict(
        cls,
        attrs: dict,
        rundb="",
        autocommit=False,
        tmp="",
        host=None,
        log_stream=None,
        is_api=False,
        store_run=True,
        include_status=False,
    ):
        """Create execution context from dict"""
        self = cls(autocommit=autocommit, tmp=tmp, log_stream=log_stream)

        meta = attrs.get("metadata")
        if meta:
            self._uid = meta.get("uid", self._uid or uuid.uuid4().hex)
            self._iteration = meta.get("iteration", self._iteration)
            self.name = meta.get("name", self.name)
            self._project = meta.get("project", self._project)
            self._annotations = meta.get("annotations", self._annotations)
            self._labels = meta.get("labels", self._labels)
        spec = attrs.get("spec")
        if spec:
            self._secrets_manager = SecretsStore.from_list(spec.get(RunKeys.secrets))
            self._log_level = spec.get("log_level", self._log_level)
            self._function = spec.get("function", self._function)
            self._parameters = spec.get("parameters", self._parameters)
            self._handler = spec.get("handler", self._handler)
            if not self._iteration:
                self._hyperparams = spec.get("hyperparams", self._hyperparams)
                self._hyper_param_options = spec.get(
                    "hyper_param_options", self._hyper_param_options
                )
                if isinstance(self._hyper_param_options, dict):
                    self._hyper_param_options = HyperParamOptions.from_dict(
                        self._hyper_param_options
                    )
            self._outputs = spec.get("outputs", self._outputs)
            self._allow_empty_resources = spec.get(
                "allow_empty_resources", self._allow_empty_resources
            )
            self.artifact_path = spec.get(RunKeys.output_path, self.artifact_path)
            self._in_path = spec.get(RunKeys.input_path, self._in_path)
            inputs = spec.get(RunKeys.inputs)
            self._notifications = spec.get("notifications", self._notifications)
            self._state_thresholds = spec.get(
                "state_thresholds", self._state_thresholds
            )
            self._node_selector = spec.get("node_selector", self._node_selector)
            self._reset_on_run = spec.get("reset_on_run", self._reset_on_run)

        self._init_dbs(rundb)

        if spec:
            # init data related objects (require DB & Secrets to be set first)
            self._data_stores.from_dict(spec)
            if inputs and isinstance(inputs, dict):
                for k, v in inputs.items():
                    if v:
                        self._set_input(k, v)

        if host and not is_api:
            self.set_label(mlrun_constants.MLRunInternalLabels.host, host)

        start = get_in(attrs, "status.start_time")
        if start:
            start = parser.parse(start) if isinstance(start, str) else start
            self._start_time = start
        self._state = "running"

        status = attrs.get("status")
        if include_status and status:
            self._results = status.get("results", self._results)
            for artifact in status.get("artifacts", []):
                artifact_obj = dict_to_artifact(artifact)
                self._artifacts_manager.artifact_uris[artifact_obj.key] = (
                    artifact_obj.uri
                )
            for key, uri in status.get("artifact_uris", {}).items():
                self._artifacts_manager.artifact_uris[key] = uri
            self._state = status.get("state", self._state)

        # No need to store the run for every worker
        if store_run and self.is_logging_worker():
            self.store_run()
        return self

    def artifact_subpath(self, *subpaths):
        """Subpaths under output path artifacts path

        Example::

            data_path = context.artifact_subpath("data")

        """
        return os.path.join(self.artifact_path, *subpaths)

    def set_label(self, key: str, value, replace: bool = True):
        """Set/record a specific label

        Example::

            context.set_label("framework", "sklearn")

        """
        if not self.is_logging_worker():
            logger.warning(
                "Setting labels is only supported in the logging worker, ignoring"
            )
            return

        if replace or not self._labels.get(key):
            self._labels[key] = str(value)

    def set_annotation(self, key: str, value, replace: bool = True):
        """Set/record a specific annotation

        Example::

            context.set_annotation("comment", "some text")

        """
        if replace or not self._annotations.get(key):
            self._annotations[key] = str(value)

    def get_param(self, key: str, default=None):
        """Get a run parameter, or use the provided default if not set

        Example::

            p1 = context.get_param("p1", 0)
        """
        if key not in self._parameters:
            self._parameters[key] = default
            if default:
                self._update_run()
            return default
        return self._parameters[key]

    def get_project_object(self) -> Optional["mlrun.MlrunProject"]:
        """
        Get the MLRun project object by the project name set in the context.

        :returns: The project object or None if it couldn't be retrieved.
        """
        return self._load_project_object()

    def get_project_param(self, key: str, default=None):
        """Get a parameter from the run's project's parameters"""
        if not self._load_project_object():
            return default

        return self._project_object.get_param(key, default)

    def get_secret(self, key: str):
        """Get a key based secret e.g. DB password from the context.
        Secrets can be specified when invoking a run through vault, files, env, ..

        Example::

            access_key = context.get_secret("ACCESS_KEY")
        """
        return mlrun.get_secret_or_env(key, secret_provider=self._secrets_manager)

    def get_input(self, key: str, url: str = ""):
        """
        Get an input :py:class:`~mlrun.DataItem` object,
        data objects have methods such as .get(), .download(), .url, .. to access the actual data.
        Requires access to the data store secrets if configured.

        Example::

            data = context.get_input("my_data").get()

        :param key:  The key name for the input url entry.
        :param url:  The url of the input data (file, stream, ..) - optional, saved in the inputs dictionary
                     if the key is not already present.

        :return:     :py:class:`~mlrun.datastore.base.DataItem` object
        """
        if key not in self._inputs:
            self._set_input(key, url)

        url = self._inputs[key]
        return self._data_stores.object(
            url,
            key,
            project=self._project,
            allow_empty_resources=self._allow_empty_resources,
        )

    def log_result(self, key: str, value, commit=False):
        """Log a scalar result value

        Example::

            context.log_result("accuracy", 0.85)

        :param key:    Result key
        :param value:  Result value
        :param commit: Commit (write to DB now vs wait for the end of the run)
        """
        self._results[str(key)] = _cast_result(value)
        self._update_run(commit=commit)

    def log_results(self, results: dict, commit=False):
        """Log a set of scalar result values

        Example::

            context.log_results({"accuracy": 0.85, "loss": 0.2})

        :param results:  Key/value dict or results
        :param commit:   Commit (write to DB now vs wait for the end of the run)
        """
        if not isinstance(results, dict):
            raise MLRunInvalidArgumentError("Results must be in the form of dict")

        for p in results.keys():
            self._results[str(p)] = _cast_result(results[p])
        self._update_run(commit=commit)

    def log_iteration_results(self, best, summary: list, task: dict, commit=False):
        """Reserved for internal use"""

        if best:
            # Recreate the best iteration context for the interface of getting its artifacts
            best_context = MLClientCtx.from_dict(
                task, store_run=False, include_status=True
            )
            self._results["best_iteration"] = best
            for key, result in best_context.results.items():
                self._results[key] = result
            for key, artifact_uri in best_context.artifact_uris.items():
                self._artifacts_manager.artifact_uris[key] = artifact_uri
                artifact = best_context.get_artifact(key)
                self._artifacts_manager.link_artifact(
                    self.project,
                    self.name,
                    self.tag,
                    key,
                    self.iteration,
                    artifact.target_path,
                    db_key=artifact.db_key,
                    link_iteration=best,
                )

        if summary is not None:
            self._iteration_results = summary
        if commit:
            self._update_run(commit=True)

    def log_artifact(
        self,
        item,
        body=None,
        local_path=None,
        artifact_path=None,
        tag="",
        viewer=None,
        target_path="",
        src_path=None,
        upload=None,
        labels=None,
        format=None,
        db_key=None,
        **kwargs,
    ) -> Artifact:
        """Log an output artifact and optionally upload it to datastore

        Example::

            context.log_artifact(
                "some-data",
                body=b"abc is 123",
                local_path="model.txt",
                labels={"framework": "xgboost"},
            )


        :param item:          Artifact key or artifact object (can be any type, such as dataset, model, feature store)
        :param body:          Will use the body as the artifact content
        :param local_path:    Path to the local file we upload, will also be use
                              as the destination subpath (under "artifact_path")
        :param artifact_path: Target artifact path (when not using the default)
                              To define a subpath under the default location use:
                              `artifact_path=context.artifact_subpath('data')`
        :param tag:           Version tag
        :param viewer:        Kubeflow viewer type
        :param target_path:   Absolute target path (instead of using artifact_path + local_path)
        :param src_path:      Deprecated, use local_path
        :param upload:        Whether to upload the artifact to the datastore. If not provided, and the `local_path`
                              is not a directory, upload occurs by default. Directories are uploaded only when this
                              flag is explicitly set to `True`.
        :param labels:        A set of key/value labels to tag the artifact with
        :param format:        Optional, format to use (e.g. csv, parquet, ..)
        :param db_key:        The key to use in the artifact DB table, by default its run name + '_' + key
                              db_key=False will not register it in the artifacts table

        :returns: Artifact object
        """
        local_path = src_path or local_path
        item = self._artifacts_manager.log_artifact(
            self,
            item,
            body=body,
            local_path=local_path,
            artifact_path=extend_artifact_path(artifact_path, self.artifact_path),
            target_path=target_path,
            tag=tag,
            viewer=viewer,
            upload=upload,
            labels=labels,
            db_key=db_key,
            format=format,
            **kwargs,
        )
        self._update_run()
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
        stats=None,
        db_key=None,
        target_path="",
        extra_data=None,
        label_column: Optional[str] = None,
        **kwargs,
    ) -> DatasetArtifact:
        """Log a dataset artifact and optionally upload it to datastore

        If the dataset exists with the same key and tag, it will be overwritten.

        Example::

            raw_data = {
                "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
                "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
                "age": [42, 52, 36, 24, 73],
                "testScore": [25, 94, 57, 62, 70],
            }
            df = pd.DataFrame(
                raw_data, columns=["first_name", "last_name", "age", "testScore"]
            )
            context.log_dataset("mydf", df=df, stats=True)

        :param key:           Artifact key
        :param df:            Dataframe object
        :param label_column:  Name of the label column (the one holding the target (y) values)
        :param local_path:    Path to the local dataframe file that exists locally.
                              The given file extension will be used to save the dataframe to a file
                              If the file exists, it will be uploaded to the datastore instead of the given df.
        :param artifact_path: Target artifact path (when not using the default)
                              to define a subpath under the default location use:
                              `artifact_path=context.artifact_subpath('data')`
        :param tag:           Version tag
        :param format:        Optional, format to use (e.g. csv, parquet, ..)
        :param target_path:   Absolute target path (instead of using artifact_path + local_path)
        :param preview:       Number of lines to store as preview in the artifact metadata
        :param stats:         Calculate and store dataset stats in the artifact metadata
        :param extra_data:    Key/value list of extra files/charts to link with this dataset
        :param upload:        Upload to datastore (default is True)
        :param labels:        A set of key/value labels to tag the artifact with
        :param db_key:        The key to use in the artifact DB table, by default its run name + '_' + key
                              db_key=False will not register it in the artifacts table

        :returns: Dataset artifact object
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
            self._artifacts_manager.log_artifact(
                self,
                ds,
                local_path=local_path,
                artifact_path=extend_artifact_path(artifact_path, self.artifact_path),
                target_path=target_path,
                tag=tag,
                upload=upload,
                db_key=db_key,
                labels=labels,
            ),
        )
        self._update_run()
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
        inputs: Optional[list[Feature]] = None,
        outputs: Optional[list[Feature]] = None,
        feature_vector: Optional[str] = None,
        feature_weights: Optional[list] = None,
        training_set=None,
        label_column: Optional[Union[str, list]] = None,
        extra_data=None,
        db_key=None,
        **kwargs,
    ) -> ModelArtifact:
        """Log a model artifact and optionally upload it to datastore

        Example::

            context.log_model(
                "model",
                body=dumps(model),
                model_file="model.pkl",
                metrics=context.results,
                training_set=training_df,
                label_column="label",
                feature_vector=feature_vector_uri,
                labels={"app": "fraud"},
            )

        :param key:             Artifact key or artifact class ()
        :param body:            Will use the body as the artifact content
        :param model_file:      Path to the local model file we upload (see also model_dir)
                                or to a model file data url (e.g. `http://host/path/model.pkl`)
        :param model_dir:       Path to the local dir holding the model file and extra files
        :param artifact_path:   Target artifact path (when not using the default)
                                to define a subpath under the default location use:
                                `artifact_path=context.artifact_subpath('data')`
        :param framework:       Name of the ML framework
        :param algorithm:       Training algorithm name
        :param tag:             Version tag
        :param metrics:         Key/value dict of model metrics
        :param parameters:      Key/value dict of model parameters
        :param inputs:          Ordered list of model input features (name, type, ..)
        :param outputs:         Ordered list of model output/result elements (name, type, ..)
        :param upload:          Upload to datastore (default is True)
        :param labels:          A set of key/value labels to tag the artifact with
        :param feature_vector:  Feature store feature vector uri (store://feature-vectors/<project>/<name>[:tag])
        :param feature_weights: List of feature weights, one per input column
        :param training_set:    Training set dataframe, used to infer inputs & outputs
        :param label_column:    Which columns in the training set are the label (target) columns
        :param extra_data:      Key/value list of extra files/charts to link with this dataset
                                value can be absolute path | relative path (to model dir) | bytes | artifact object
        :param db_key:          The key to use in the artifact DB table, by default its run name + '_' + key
                                db_key=False will not register it in the artifacts table

        :returns: Model artifact object
        """

        if training_set is not None and inputs:
            raise MLRunInvalidArgumentError(
                "Cannot specify inputs and training set together"
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
            self._artifacts_manager.log_artifact(
                self,
                model,
                artifact_path=extend_artifact_path(artifact_path, self.artifact_path),
                tag=tag,
                upload=upload,
                db_key=db_key,
                labels=labels,
            ),
        )
        self._update_run()
        return item

    def log_document(
        self,
        key: str = "",
        tag: str = "",
        local_path: str = "",
        artifact_path: Optional[str] = None,
        document_loader_spec: DocumentLoaderSpec = DocumentLoaderSpec(),
        upload: Optional[bool] = False,
        labels: Optional[dict[str, str]] = None,
        target_path: Optional[str] = None,
        db_key: Optional[str] = None,
        **kwargs,
    ) -> DocumentArtifact:
        """
        Log a document as an artifact.

        :param key: Optional artifact key. If not provided, will be derived from local_path
                or target_path using DocumentArtifact.key_from_source()
        :param tag: Version tag
        :param local_path: path to the local file we upload, will also be use
                        as the destination subpath (under "artifact_path")
        :param artifact_path: Target artifact path (when not using the default)
                            to define a subpath under the default location use:
                            `artifact_path=context.artifact_subpath('data')`
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
        :param target_path: Path to the local file
        :param db_key: The key to use in the artifact DB table, by default its run name + '_' + key
                       db_key=False will not register it in the artifacts table
        :param kwargs: Additional keyword arguments
        :return: DocumentArtifact object

        Example:
            >>> # Log a PDF document with custom loader
            >>> project.log_document(
            ...     local_path="path/to/doc.pdf",
            ...     document_loader_spec=DocumentLoaderSpec(
            ...         loader_class_name="langchain_community.document_loaders.PDFLoader",
            ...         src_name="file_path",
            ...         kwargs={"extract_images": True},
            ...     ),
            ... )
        """
        original_source = local_path or target_path

        if not key and not original_source:
            raise ValueError(
                "Must provide either 'key' parameter or 'local_path'/'target_path' to derive the key from"
            )
        if not key:
            key = DocumentArtifact.key_from_source(original_source)

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

        item = self._artifacts_manager.log_artifact(
            self,
            doc_artifact,
            artifact_path=extend_artifact_path(artifact_path, self.artifact_path),
            tag=tag,
            upload=upload,
            labels=labels,
            local_path=local_path,
            target_path=target_path,
            db_key=db_key,
        )
        self._update_run()
        return item

    def get_cached_artifact(self, key):
        """Return a logged artifact from cache (for potential updates)"""
        warnings.warn(
            "get_cached_artifact is deprecated in 1.8.0 and will be removed in 1.11.0. Use get_artifact instead.",
            FutureWarning,
        )
        return self.get_artifact(key)

    def get_artifact(
        self, key, tag=None, iter=None, tree=None, uid=None
    ) -> Optional[Artifact]:
        cached_artifact_uri = self._artifacts_manager.artifact_uris.get(key, None)
        if tag or iter or tree or uid or (not cached_artifact_uri):
            project = self.get_project_object()
            return project.get_artifact(key=key, tag=tag, iter=iter, tree=tree, uid=uid)
        else:
            return self.get_store_resource(cached_artifact_uri)

    def update_artifact(self, artifact_object: Artifact):
        """Update an artifact object in the DB and the cached uri"""
        self._artifacts_manager.update_artifact(self, artifact_object)

    def commit(self, message: str = "", completed=False):
        """Save run state and optionally add a commit message

        :param message:   Commit message to save in the run
        :param completed: Mark run as completed
        """
        # Changing state to completed is allowed only when the execution is in running state
        if self._state != "running":
            completed = False

        if message:
            self._annotations["message"] = message
        if completed:
            self._state = "completed"

        if self._parent:
            self._parent.update_child_iterations()
            self._parent._last_update = now_date()
            self._parent._update_run(commit=True, message=message)

        if self._children:
            self.update_child_iterations(commit_children=True, completed=completed)
        self._last_update = now_date()
        self._update_run(commit=True, message=message)
        if completed and not self.iteration:
            mlrun.runtimes.utils.global_context.set(None)

    def set_state(
        self,
        execution_state: Optional[str] = None,
        error: Optional[str] = None,
        commit=True,
    ):
        """
        Modify and store the execution state or mark an error and update the run state accordingly.
        This method allows to set the run state to 'completed' in the DB which is discouraged.
        Completion of runs should be decided externally to the execution context.

        :param execution_state:     set execution state
        :param error:               error message (if exist will set the state to error)
        :param commit:              will immediately update the state in the DB
        """
        # TODO: The execution context should not set the run state to completed.
        #  Create a separate state for the execution in the run object.
        updates = {"status.last_update": now_date().isoformat()}

        if error is not None:
            self._state = "error"
            self._error = str(error)
            updates["status.state"] = "error"
            updates["status.error"] = error
        elif (
            execution_state
            and execution_state != self._state
            and self._state != "error"
        ):
            self._state = execution_state
            updates["status.state"] = execution_state
        self._last_update = now_date()

        if self._rundb and commit:
            self._rundb.update_run(
                updates, self._uid, self.project, iter=self._iteration
            )

    def set_hostname(self, host: str):
        """Update the hostname, for internal use"""
        self._host = host
        if self._rundb:
            updates = {"status.host": host}
            self._rundb.update_run(
                updates, self._uid, self.project, iter=self._iteration
            )

    def get_notifications(self, unmask_secret_params=False):
        """
        Get the list of notifications

        :param unmask_secret_params: Used as a workaround for sending notification from workflow-runner.
                                     When used, if the notification will be saved again a new secret will be created.
        """

        # Get the full notifications from the DB since the run context does not contain the params due to bloating
        run = self._rundb.read_run(
            self.uid, format_=mlrun.common.formatters.RunFormat.notifications
        )

        notifications = []
        for notification in run["spec"]["notifications"]:
            notification: mlrun.model.Notification = mlrun.model.Notification.from_dict(
                notification
            )
            # Fill the secret params from the project secret. We cannot use the server side internal secret mechanism
            # here as it is the client side.
            # TODO: This is a workaround to allow the notification to get the secret params from project secret
            #       instead of getting them from the internal project secret that should be mounted.
            #       We should mount the internal project secret that was created to the workflow-runner
            #       and get the secret from there.
            if unmask_secret_params:
                try:
                    notification.enrich_unmasked_secret_params_from_project_secret()
                    notifications.append(notification)
                except mlrun.errors.MLRunValueError:
                    logger.warning(
                        "Failed to fill secret params from project secret for notification."
                        "Skip this notification.",
                        notification=notification.name,
                    )

        return notifications

    def to_dict(self):
        """Convert the run context to a dictionary"""

        def set_if_not_none(_struct, key, val):
            if val:
                _struct[key] = val

        struct = {
            "kind": "run",
            "metadata": {
                "name": self.name,
                "uid": self._uid,
                "iteration": self._iteration,
                "project": self._project,
                "labels": self._labels,
                "annotations": self._annotations,
            },
            "spec": {
                "function": self._function,
                "log_level": self._log_level,
                "parameters": self._parameters,
                "handler": self._handler,
                "outputs": self._outputs,
                RunKeys.output_path: self.artifact_path,
                RunKeys.inputs: self._inputs,
                "notifications": self._notifications,
                "state_thresholds": self._state_thresholds,
                "node_selector": self._node_selector,
            },
            "status": {
                "results": self._results,
                "start_time": to_date_str(self._start_time),
                "last_update": to_date_str(self._last_update),
            },
        }

        # Completion of runs is not decided by the execution as there may be
        # multiple executions for a single run (e.g. mpi)
        if self._state != "completed":
            struct["status"]["state"] = self._state

        if not self._iteration:
            struct["spec"]["hyperparams"] = self._hyperparams
            struct["spec"]["hyper_param_options"] = self._hyper_param_options.to_dict()

        set_if_not_none(struct["status"], "error", self._error)
        set_if_not_none(struct["status"], "commit", self._commit)
        set_if_not_none(struct["status"], "iterations", self._iteration_results)

        struct["status"][RunKeys.artifact_uris] = self._artifacts_manager.artifact_uris
        self._data_stores.to_dict(struct["spec"])
        return struct

    def to_yaml(self):
        """Convert the run context to a yaml buffer"""
        return dict_to_yaml(self.to_dict())

    def to_json(self):
        """Convert the run context to a json buffer"""
        return dict_to_json(self.to_dict())

    def store_run(self):
        """
        Store the run object in the DB - removes missing fields.
        Use _update_run for coherent updates.
        Should be called by the logging worker only (see is_logging_worker()).
        """
        self._write_tmpfile()
        if self._rundb:
            self._rundb.store_run(
                self.to_dict(), self._uid, self.project, iter=self._iteration
            )

    def is_logging_worker(self):
        """
        Check if the current worker is the logging worker.

        :return: True if the context belongs to the logging worker and False otherwise.
        """
        # If it's a OpenMPI job, get the global rank and compare to the logging rank (worker) set in MLRun's
        # configuration:
        labels = self.labels
        if (
            mlrun_constants.MLRunInternalLabels.host in labels
            and labels.get(mlrun_constants.MLRunInternalLabels.kind, "job") == "mpijob"
        ):
            # The host (pod name) of each worker is created by k8s, and by default it uses the rank number as the id in
            # the following template: ...-worker-<rank>
            rank = int(
                labels[mlrun_constants.MLRunInternalLabels.host].rsplit("-", 1)[1]
            )
            return rank == mlrun.mlconf.packagers.logging_worker

        # Single worker is always the logging worker:
        return True

    def _update_run(self, commit=False, message=""):
        """
        Update the required fields in the run object instead of overwriting existing values with empty ones

        :param commit:  Commit the changes to the DB if autocommit is not set or update the tmpfile alone
        :param message: Commit message
        """
        self._merge_tmpfile()
        if commit or self._autocommit:
            self._commit = message
            if self._rundb:
                self._rundb.update_run(
                    self._get_updates(), self._uid, self.project, iter=self._iteration
                )

    def _get_updates(self):
        def set_if_not_none(_struct, key, val):
            if val:
                _struct[key] = val

        struct = {
            "metadata.annotations": self._annotations,
            "spec.parameters": self._parameters,
            "spec.outputs": self._outputs,
            "spec.inputs": self._inputs,
            "status.results": self._results,
            "status.start_time": to_date_str(self._start_time),
            "status.last_update": to_date_str(self._last_update),
        }

        # Completion of runs is decided by the API runs monitoring as there may be
        # multiple executions for a single run (e.g. mpi).
        # For kinds that are not monitored by the API (local) we allow changing the state.
        run_kind = self.labels.get(mlrun_constants.MLRunInternalLabels.kind, "")
        if (
            mlrun.runtimes.RuntimeKinds.is_local_runtime(run_kind)
            or self._state != "completed"
        ):
            struct["status.state"] = self._state

        if self.is_logging_worker():
            struct["metadata.labels"] = self._labels

        set_if_not_none(struct, "status.error", self._error)
        set_if_not_none(struct, "status.commit", self._commit)
        set_if_not_none(struct, "status.iterations", self._iteration_results)

        struct[f"status.{RunKeys.artifact_uris}"] = (
            self._artifacts_manager.artifact_uris
        )
        return struct

    def _init_dbs(self, rundb):
        if rundb:
            if isinstance(rundb, str):
                self._rundb = mlrun.db.get_run_db(rundb, secrets=self._secrets_manager)
            else:
                self._rundb = rundb
        else:
            self._rundb = mlrun.get_run_db()
        self._data_stores = store_manager.set(self._secrets_manager, db=self._rundb)
        self._artifacts_manager = ArtifactManager(db=self._rundb)

    def _load_project_object(self) -> Optional["mlrun.MlrunProject"]:
        if not self._project_object:
            if not self._project:
                self.logger.warning(
                    "Project cannot be loaded without a project name set in the context"
                )
                return None
            if not self._rundb:
                self.logger.warning(
                    "Cannot retrieve project data - MLRun DB is not accessible"
                )
                return None
            self._project_object = self._rundb.get_project(self._project)
        return self._project_object

    def _set_input(self, key, url=""):
        if url is None:
            return
        if not url:
            url = key
        if self.in_path and is_relative_path(url):
            url = os.path.join(self._in_path, url)
        self._inputs[key] = url

    def _merge_tmpfile(self):
        if not self._tmpfile:
            return

        loaded_run = self._read_tmpfile()
        dict_run = self.to_dict()
        if loaded_run:
            for key, val in dict_run.items():
                update_in(loaded_run, key, val)
        else:
            loaded_run = dict_run

        self._write_tmpfile(json=dict_to_json(loaded_run))

    def _read_tmpfile(self):
        if self._tmpfile:
            with open(self._tmpfile) as fp:
                return yaml.safe_load(fp)

        return None

    def _write_tmpfile(self, json=None):
        self.last_update = now_date()
        if self._tmpfile:
            data = json or self.to_json()
            with open(self._tmpfile, "w") as fp:
                fp.write(data)
                fp.close()


def _cast_result(value):
    if isinstance(value, (int, str, float)):
        return value
    if isinstance(value, list):
        return [_cast_result(v) for v in value]
    if isinstance(value, dict):
        return {k: _cast_result(v) for k, v in value.items()}
    if isinstance(value, (np.int64, np.integer)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
