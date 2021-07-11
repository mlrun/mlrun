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

import os
import uuid
from copy import deepcopy
from datetime import datetime
from typing import List, Union

import numpy as np

import mlrun
from mlrun.artifacts import ModelArtifact
from mlrun.datastore.store_resources import get_store_resource
from mlrun.errors import MLRunInvalidArgumentError

from .artifacts import ArtifactManager, DatasetArtifact
from .datastore import store_manager
from .db import get_run_db
from .features import Feature
from .model import HyperParamOptions
from .secrets import SecretsStore
from .utils import (
    dict_to_json,
    dict_to_yaml,
    get_in,
    logger,
    now_date,
    run_keys,
    to_date_str,
    update_in,
)


class MLClientCtx(object):
    """ML Execution Client Context

    the context is generated and injected to the function using the ``function.run()``
    or manually using the :py:func:`~mlrun.run.get_or_create_ctx` call
    and provides an interface to use run params, metadata, inputs, and outputs

    base metadata include: uid, name, project, and iteration (for hyper params)
    users can set labels and annotations using :py:func:`~set_label`, :py:func:`~set_annotation`
    access parameters and secrets using :py:func:`~get_param`, :py:func:`~get_secret`
    access input data objects using :py:func:`~get_input`
    store results, artifacts, and real-time metrics using the :py:func:`~log_result`,
    :py:func:`~log_artifact`, :py:func:`~log_dataset` and :py:func:`~log_model` methods

    see doc for the individual params and methods
    """

    kind = "run"

    def __init__(self, autocommit=False, tmp="", log_stream=None):
        self._uid = ""
        self.name = ""
        self._iteration = 0
        self._project = ""
        self._tag = ""
        self._secrets_manager = SecretsStore()

        # runtime db service interfaces
        self._rundb = None
        self._tmpfile = tmp
        self._logger = log_stream or logger
        self._log_level = "info"
        self._matrics_db = None
        self._autocommit = autocommit

        self._labels = {}
        self._annotations = {}

        self._function = ""
        self._parameters = {}
        self._hyperparams = {}
        self._hyper_param_options = HyperParamOptions()
        self._in_path = ""
        self.artifact_path = ""
        self._inputs = {}
        self._outputs = []

        self._results = {}
        self._state = "created"
        self._error = None
        self._commit = ""
        self._host = None
        self._start_time = now_date()
        self._last_update = now_date()
        self._iteration_results = None
        self._children = []
        self._parent = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_value:
            self.set_state(error=exc_value, commit=False)
        self.commit(completed=exc_value is None)

    def get_child_context(self, with_parent_params=False, **params):
        """get child context (iteration)

        allow sub experiments (epochs, hyper-param, ..) under a parent
        will create a new iteration, log_xx will update the child only
        use commit_children() to save all the children and specify the best run

        example::

            def handler(context: mlrun.MLClientCtx, data: mlrun.DataItem):
                df = data.as_df()
                best_accuracy = accuracy_sum = 0
                for param in param_list:
                    with context.get_child_context(myparam=param) as child:
                        accuracy = child_handler(child, df, **child.parameters)
                        accuracy_sum += accuracy
                        child.log_result('accuracy', accuracy)
                        if accuracy > best_accuracy:
                            child.mark_as_best()
                            best_accuracy = accuracy

                context.log_result('avg_accuracy', accuracy_sum / len(param_list))

        :param params:  extra (or override) params to parent context
        :param with_parent_params:  child will copy the parent parameters and add to them

        :return: child context
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

    def update_child_iterations(self, best_run=0, commit_children=False):
        """update children results in the parent, and optionally mark the best

        :param best_run:  marks the child iteration number (starts from 1)
        :param commit_children:  commit all child runs to the db
        """
        if not self._children:
            return
        if commit_children:
            for child in self._children:
                child.commit()
        results = [child.to_dict() for child in self._children]
        summary = mlrun.runtimes.utils.results_to_iter(results, None, self)
        task = results[best_run - 1] if best_run else None
        self.log_iteration_results(best_run, summary, task)

    def mark_as_best(self):
        """mark a child as the best iteration result, see .get_child_context()"""
        if not self._parent or not self._iteration:
            raise MLRunInvalidArgumentError(
                "can only mark a child run as best iteration"
            )
        self._parent.log_iteration_results(self._iteration, None, self.to_dict())

    def get_store_resource(self, url):
        """get mlrun data resource (feature set/vector, artifact, item) from url

        example::

            feature_vector = context.get_store_resource("store://feature-vectors/default/myvec")
            dataset = context.get_store_resource("store://artifacts/default/mydata")

        :param uri:    store resource uri/path, store://<type>/<project>/<name>:<version>
                       types: artifacts | feature-sets | feature-vectors
        """
        return get_store_resource(url, db=self._rundb, secrets=self._secrets_manager)

    def get_dataitem(self, url):
        """get mlrun dataitem from url

        example::

            data = context.get_dataitem("s3://my-bucket/file.csv").as_df()

        """
        return self._data_stores.object(url=url)

    def set_logger_stream(self, stream):
        self._logger.replace_handler_stream("default", stream)

    def _init_dbs(self, rundb):
        if rundb:
            if isinstance(rundb, str):
                self._rundb = get_run_db(rundb, secrets=self._secrets_manager)
            else:
                self._rundb = rundb
        else:
            self._rundb = mlrun.get_run_db()
        self._data_stores = store_manager.set(self._secrets_manager, db=self._rundb)
        self._artifacts_manager = ArtifactManager(db=self._rundb)

    def get_meta(self):
        """Reserved for internal use"""
        uri = f"{self._project}/{self.uid}" if self._project else self.uid
        resp = {
            "name": self.name,
            "kind": "run",
            "uri": uri,
            "owner": get_in(self._labels, "owner"),
        }
        if "workflow" in self._labels:
            resp["workflow"] = self._labels["workflow"]
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
    ):
        """create execution context from dict"""

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
            self._secrets_manager = SecretsStore.from_list(spec.get(run_keys.secrets))
            self._log_level = spec.get("log_level", self._log_level)
            self._function = spec.get("function", self._function)
            self._parameters = spec.get("parameters", self._parameters)
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
            self.artifact_path = spec.get(run_keys.output_path, self.artifact_path)
            self._in_path = spec.get(run_keys.input_path, self._in_path)
            inputs = spec.get(run_keys.inputs)

        self._init_dbs(rundb)

        if spec and not is_api:
            # init data related objects (require DB & Secrets to be set first), skip when running in the api service
            self._data_stores.from_dict(spec)
            if inputs and isinstance(inputs, dict):
                for k, v in inputs.items():
                    if v:
                        self._set_input(k, v)

        if host and not is_api:
            self.set_label("host", host)

        start = get_in(attrs, "status.start_time")
        if start:
            self._start_time = start
        self._state = "running"
        self._update_db(commit=True)
        return self

    @property
    def uid(self):
        """Unique run id"""
        if self._iteration:
            return f"{self._uid}-{self._iteration}"
        return self._uid

    @property
    def tag(self):
        """run tag (uid or workflow id if exists)"""
        return self._labels.get("workflow", self._uid)

    @property
    def iteration(self):
        """child iteration index, for hyper parameters """
        return self._iteration

    @property
    def project(self):
        """project name, runs can be categorized by projects"""
        return self._project

    @property
    def logger(self):
        """built-in logger interface

        example::

            context.logger.info("started experiment..", param=5)

        """
        return self._logger

    @property
    def log_level(self):
        """get the logging level, e.g. 'debug', 'info', 'error'"""
        return self._log_level

    @log_level.setter
    def log_level(self, value: str):
        """set the logging level, e.g. 'debug', 'info', 'error'"""
        self._log_level = value
        print(f"changed log level to: {value}")

    @property
    def parameters(self):
        """dictionary of run parameters (read-only)"""
        return deepcopy(self._parameters)

    @property
    def inputs(self):
        """dictionary of input data items (read-only)"""
        return self._inputs

    @property
    def results(self):
        """dictionary of results (read-only)"""
        return deepcopy(self._results)

    @property
    def artifacts(self):
        """dictionary of artifacts (read-only)"""
        return deepcopy(self._artifacts_manager.artifact_list())

    @property
    def in_path(self):
        """default input path for data objects"""
        return self._in_path

    @property
    def out_path(self):
        """default output path for artifacts"""
        logger.info(".out_path will soon be deprecated, use .artifact_path")
        return self.artifact_path

    def artifact_subpath(self, *subpaths):
        """subpaths under output path artifacts path

        example::

            data_path=context.artifact_subpath('data')

        """
        return os.path.join(self.artifact_path, *subpaths)

    @property
    def labels(self):
        """dictionary with labels (read-only)"""
        return deepcopy(self._labels)

    def set_label(self, key: str, value, replace: bool = True):
        """set/record a specific label

        example::

            context.set_label("framework", "sklearn")

        """
        if replace or not self._labels.get(key):
            self._labels[key] = str(value)

    @property
    def annotations(self):
        """dictionary with annotations (read-only)"""
        return deepcopy(self._annotations)

    def set_annotation(self, key: str, value, replace: bool = True):
        """set/record a specific annotation

        example::

            context.set_annotation("comment", "some text")

        """
        if replace or not self._annotations.get(key):
            self._annotations[key] = str(value)

    def get_param(self, key: str, default=None):
        """get a run parameter, or use the provided default if not set

        example::

            p1 = context.get_param("p1", 0)
        """
        if key not in self._parameters:
            self._parameters[key] = default
            if default:
                self._update_db()
            return default
        return self._parameters[key]

    def get_secret(self, key: str):
        """get a key based secret e.g. DB password from the context
        secrets can be specified when invoking a run through vault, files, env, ..

        example::

            access_key = context.get_secret("ACCESS_KEY")
        """
        if self._secrets_manager:
            return self._secrets_manager.get(key)
        return None

    def _set_input(self, key, url=""):
        if url is None:
            return
        if not url:
            url = key
        if self.in_path and not (url.startswith("/") or "://" in url):
            url = os.path.join(self._in_path, url)
        obj = self._data_stores.object(url, key, project=self._project)
        self._inputs[key] = obj
        return obj

    def get_input(self, key: str, url: str = ""):
        """get an input :py:class:`~mlrun.DataItem` object, data objects have methods such as
        .get(), .download(), .url, .. to access the actual data

        example::

            data = context.get_input("my_data").get()
        """
        if key not in self._inputs:
            return self._set_input(key, url)
        else:
            return self._inputs[key]

    def log_result(self, key: str, value, commit=False):
        """log a scalar result value

        example::

            context.log_result('accuracy', 0.85)

        :param key:    result key
        :param value:  result value
        :param commit: commit (write to DB now vs wait for the end of the run)
        """
        self._results[str(key)] = _cast_result(value)
        self._update_db(commit=commit)

    def log_results(self, results: dict, commit=False):
        """log a set of scalar result values

        example::

            context.log_results({'accuracy': 0.85, 'loss': 0.2})

        :param results:  key/value dict or results
        :param commit:   commit (write to DB now vs wait for the end of the run)
        """
        if not isinstance(results, dict):
            raise MLRunInvalidArgumentError(
                "(multiple) results must be in the form of dict"
            )

        for p in results.keys():
            self._results[str(p)] = _cast_result(results[p])
        self._update_db(commit=commit)

    def log_iteration_results(self, best, summary: list, task: dict, commit=False):
        """Reserved for internal use"""

        if best:
            self._results["best_iteration"] = best
            for k, v in get_in(task, ["status", "results"], {}).items():
                self._results[k] = v
            for a in get_in(task, ["status", run_keys.artifacts], []):
                self._artifacts_manager.artifacts[a["key"]] = a
                self._artifacts_manager.link_artifact(
                    self.project,
                    self.name,
                    self.tag,
                    a["key"],
                    self.iteration,
                    a["target_path"],
                    link_iteration=best,
                )

        if summary is not None:
            self._iteration_results = summary
        if commit:
            self._update_db(commit=True)

    def log_metric(self, key: str, value, timestamp=None, labels=None):
        """TBD, log a real-time time-series metric"""
        labels = {} if labels is None else labels
        if not timestamp:
            timestamp = datetime.now()
        if self._rundb:
            self._rundb.store_metric({key: value}, timestamp, labels)

    def log_metrics(self, keyvals: dict, timestamp=None, labels=None):
        """TBD, log a set of real-time time-series metrics"""
        labels = {} if labels is None else labels
        if not timestamp:
            timestamp = datetime.now()
        if self._rundb:
            self._rundb.store_metric(keyvals, timestamp, labels)

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
    ):
        """log an output artifact and optionally upload it to datastore

        example::

            context.log_artifact(
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
        :param tag:           version tag
        :param viewer:        kubeflow viewer type
        :param target_path:   absolute target path (instead of using artifact_path + local_path)
        :param src_path:      deprecated, use local_path
        :param upload:        upload to datastore (default is True)
        :param labels:        a set of key/value labels to tag the artifact with
        :param db_key:        the key to use in the artifact DB table, by default
                              its run name + '_' + key
                              db_key=False will not register it in the artifacts table

        :returns: artifact object
        """
        local_path = src_path or local_path
        item = self._artifacts_manager.log_artifact(
            self,
            item,
            body=body,
            local_path=local_path,
            artifact_path=artifact_path or self.artifact_path,
            target_path=target_path,
            tag=tag,
            viewer=viewer,
            upload=upload,
            labels=labels,
            db_key=db_key,
            format=format,
        )
        self._update_db()
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
        db_key=None,
        target_path="",
        extra_data=None,
        **kwargs,
    ):
        """log a dataset artifact and optionally upload it to datastore

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
        :param db_key:        the key to use in the artifact DB table, by default
                              its run name + '_' + key
                              db_key=False will not register it in the artifacts table

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

        item = self._artifacts_manager.log_artifact(
            self,
            ds,
            local_path=local_path,
            artifact_path=artifact_path or self.artifact_path,
            target_path=target_path,
            tag=tag,
            upload=upload,
            db_key=db_key,
            labels=labels,
        )
        self._update_db()
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
        inputs: List[Feature] = None,
        outputs: List[Feature] = None,
        feature_vector: str = None,
        feature_weights: list = None,
        training_set=None,
        label_column: Union[str, list] = None,
        extra_data=None,
        db_key=None,
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
        :param db_key:          the key to use in the artifact DB table, by default
                                its run name + '_' + key
                                db_key=False will not register it in the artifacts table

        :returns: artifact object
        """

        if training_set is not None and inputs:
            raise MLRunInvalidArgumentError(
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

        item = self._artifacts_manager.log_artifact(
            self,
            model,
            local_path=model_dir,
            artifact_path=artifact_path or self.artifact_path,
            tag=tag,
            upload=upload,
            db_key=db_key,
            labels=labels,
        )
        self._update_db()
        return item

    def commit(self, message: str = "", completed=False):
        """save run state and optionally add a commit message

        :param message:   commit message to save in the run
        :param completed: mark run as completed
        """
        if message:
            self._annotations["message"] = message
        if completed:
            self._state = "completed"

        if self._parent:
            self._parent.update_child_iterations()
            self._parent._last_update = now_date()
            self._parent._update_db(commit=True, message=message)

        if self._children:
            self.update_child_iterations(commit_children=True)
        self._last_update = now_date()
        self._update_db(commit=True, message=message)

    def set_state(self, state: str = None, error: str = None, commit=True):
        """modify and store the run state or mark an error

        :param state:   set run state
        :param error:   error message (if exist will set the state to error)
        :param commit:  will immediately update the state in the DB
        """
        updates = {"status.last_update": now_date().isoformat()}

        if error:
            self._state = "error"
            self._error = str(error)
            updates["status.state"] = "error"
            updates["status.error"] = error
        elif state and state != self._state and self._state != "error":
            self._state = state
            updates["status.state"] = state
        self._last_update = now_date()

        if self._rundb and commit:
            self._rundb.update_run(
                updates, self._uid, self.project, iter=self._iteration
            )

    def set_hostname(self, host: str):
        """update the hostname, for internal use"""
        self._host = host
        if self._rundb:
            updates = {"status.host": host}
            self._rundb.update_run(
                updates, self._uid, self.project, iter=self._iteration
            )

    def to_dict(self):
        """convert the run context to a dictionary"""

        def set_if_valid(struct, key, val):
            if val:
                struct[key] = val

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
                "outputs": self._outputs,
                run_keys.output_path: self.artifact_path,
                run_keys.inputs: {k: v.artifact_url for k, v in self._inputs.items()},
            },
            "status": {
                "state": self._state,
                "results": self._results,
                "start_time": to_date_str(self._start_time),
                "last_update": to_date_str(self._last_update),
            },
        }

        if not self._iteration:
            struct["spec"]["hyperparams"] = self._hyperparams
            struct["spec"]["hyper_param_options"] = self._hyper_param_options.to_dict()

        set_if_valid(struct["status"], "error", self._error)
        set_if_valid(struct["status"], "commit", self._commit)

        if self._iteration_results:
            struct["status"]["iterations"] = self._iteration_results
        struct["status"][run_keys.artifacts] = self._artifacts_manager.artifact_list()
        self._data_stores.to_dict(struct["spec"])
        return struct

    def to_yaml(self):
        """convert the run context to a yaml buffer"""
        return dict_to_yaml(self.to_dict())

    def to_json(self):
        """convert the run context to a json buffer"""
        return dict_to_json(self.to_dict())

    def _update_db(self, commit=False, message=""):
        self.last_update = now_date()
        if self._tmpfile:
            data = self.to_json()
            with open(self._tmpfile, "w") as fp:
                fp.write(data)
                fp.close()

        if commit or self._autocommit:
            self._commit = message
            if self._rundb:
                self._rundb.store_run(
                    self.to_dict(), self._uid, self.project, iter=self._iteration
                )


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
