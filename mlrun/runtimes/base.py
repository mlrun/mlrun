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
import traceback
import uuid
from abc import ABC, abstractmethod
from ast import literal_eval
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from os import environ
from typing import Dict, List, Tuple, Union, Optional

from kubernetes import client
from kubernetes.client.rest import ApiException
from sqlalchemy.orm import Session

import mlrun.errors
import mlrun.utils.regex
import mlrun.utils.helpers
from mlrun.api import schemas
from mlrun.api.constants import LogSources
from mlrun.api.db.base import DBInterface
from mlrun.utils.helpers import verify_field_regex, generate_function_uri
from .constants import PodPhases, RunStates
from .generators import get_generator
from .utils import calc_hash, RunError, results_to_iter
from ..config import config
from ..datastore import store_manager
from ..db import get_run_db, get_or_set_dburl, RunDBError
from ..execution import MLClientCtx
from ..k8s_utils import get_k8s_helper
from ..kfpops import write_kfpmeta, mlrun_op
from ..lists import RunList
from ..model import RunObject, ModelObj, RunTemplate, BaseMetadata, ImageBuilder
from ..secrets import SecretsStore
from ..utils import (
    get_in,
    update_in,
    logger,
    is_ipython,
    now_date,
    enrich_image_url,
    dict_to_yaml,
    dict_to_json,
    get_parsed_docker_registry,
)


class FunctionStatus(ModelObj):
    def __init__(self, state=None, build_pod=None):
        self.state = state
        self.build_pod = build_pod


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


class FunctionSpec(ModelObj):
    def __init__(
        self,
        command=None,
        args=None,
        image=None,
        mode=None,
        build=None,
        entry_points=None,
        description=None,
        workdir=None,
        default_handler=None,
        pythonpath=None,
    ):

        self.command = command or ""
        self.image = image or ""
        self.mode = mode
        self.args = args or []
        self.rundb = None
        self.description = description or ""
        self.workdir = workdir
        self.pythonpath = pythonpath

        self._build = None
        self.build = build
        self.default_handler = default_handler
        # TODO: type verification (FunctionEntrypoint dict)
        self.entry_points = entry_points or {}

    @property
    def build(self) -> ImageBuilder:
        return self._build

    @build.setter
    def build(self, build):
        self._build = self._verify_dict(build, "build", ImageBuilder)


class BaseRuntime(ModelObj):
    kind = "base"
    _is_nested = False
    _is_remote = False
    _dict_fields = ["kind", "metadata", "spec", "status"]

    def __init__(self, metadata=None, spec=None):
        self._metadata = None
        self.metadata = metadata
        self.kfp = None
        self._spec = None
        self.spec = spec
        self._db_conn = None
        self._secrets = None
        self._k8s = None
        self._is_built = False
        self.is_child = False
        self._status = None
        self.status = None
        self._is_api_server = False
        self.verbose = False

    def set_db_connection(self, conn, is_api=False):
        if not self._db_conn:
            self._db_conn = conn
        self._is_api_server = is_api

    @property
    def metadata(self) -> BaseMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, "metadata", BaseMetadata)

    @property
    def spec(self) -> FunctionSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", FunctionSpec)

    @property
    def status(self) -> FunctionStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", FunctionStatus)

    def _get_k8s(self):
        return get_k8s_helper()

    def set_label(self, key, value):
        self.metadata.labels[key] = str(value)
        return self

    @property
    def is_deployed(self):
        return True

    def _is_remote_api(self):
        db = self._get_db()
        if db and db.kind == "http":
            return True
        return False

    def _use_remote_api(self):
        if (
            self._is_remote
            and not self._is_api_server
            and self._get_db()
            and self._get_db().kind == "http"
        ):
            return True
        return False

    def _function_uri(self, tag=None, hash_key=None):
        return generate_function_uri(
            self.metadata.project,
            self.metadata.name,
            tag=tag or self.metadata.tag,
            hash_key=hash_key,
        )

    def _ensure_run_db(self):
        self.spec.rundb = self.spec.rundb or get_or_set_dburl()

    def _get_db(self):
        self._ensure_run_db()
        if not self._db_conn:
            if self.spec.rundb:
                self._db_conn = get_run_db(self.spec.rundb).connect(self._secrets)
        return self._db_conn

    def run(
        self,
        runspec: RunObject = None,
        handler=None,
        name: str = "",
        project: str = "",
        params: dict = None,
        inputs: dict = None,
        out_path: str = "",
        workdir: str = "",
        artifact_path: str = "",
        watch: bool = True,
        schedule: Union[str, schemas.ScheduleCronTrigger] = None,
        verbose=None,
        scrape_metrics=False,
        local=False,
        local_code_path=None,
    ):
        """Run a local or remote task.

        :param runspec:        run template object or dict (see RunTemplate)
        :param handler:        pointer or name of a function handler
        :param name:           execution name
        :param project:        project name
        :param params:         input parameters (dict)
        :param inputs:         input objects (dict of key: path)
        :param out_path:       default artifact output path
        :param artifact_path:  default artifact output path (will replace out_path)
        :param workdir:        default input artifacts path
        :param watch:          watch/follow run log
        :param schedule:       ScheduleCronTrigger class instance or a standard crontab expression string (which
        will be converted to the class using its `from_crontab` constructor. see this link for help:
        https://apscheduler.readthedocs.io/en/v3.6.3/modules/triggers/cron.html#module-apscheduler.triggers.cron
        :param verbose:        add verbose prints/logs
        :param scrape_metrics: whether to add the `mlrun/scrape-metrics` label to this run's resources
        :param local:           run the function locally vs on the runtime/cluster
        :param local_code_path: path of the code for local runs & debug

        :return: run context object (dict) with run metadata, results and
            status
        """
        if local:
            # allow local run simulation with a flip of a flag
            command = self
            if local_code_path:
                project = project or self.metadata.project
                name = name or self.metadata.name
                command = local_code_path
            return mlrun.run_local(
                runspec,
                command,
                name,
                workdir=workdir,
                project=project,
                handler=handler,
                params=params,
                inputs=inputs,
                artifact_path=artifact_path,
            )

        if runspec:
            runspec = deepcopy(runspec)
            if isinstance(runspec, str):
                runspec = literal_eval(runspec)
            if not isinstance(runspec, (dict, RunTemplate, RunObject)):
                raise ValueError(
                    "task/runspec is not a valid task object,"
                    " type={}".format(type(runspec))
                )

        if isinstance(runspec, RunTemplate):
            runspec = RunObject.from_template(runspec)
        if isinstance(runspec, dict) or runspec is None:
            runspec = RunObject.from_dict(runspec)

        runspec.spec.handler = (
            handler or runspec.spec.handler or self.spec.default_handler or ""
        )
        if runspec.spec.handler and self.kind not in ["handler", "dask"]:
            runspec.spec.handler = runspec.spec.handler_name

        def_name = self.metadata.name
        if runspec.spec.handler_name:
            def_name += "-" + runspec.spec.handler_name
        runspec.metadata.name = name or runspec.metadata.name or def_name
        verify_field_regex(
            "run.metadata.name", runspec.metadata.name, mlrun.utils.regex.run_name
        )
        runspec.metadata.project = (
            project
            or runspec.metadata.project
            or self.metadata.project
            or config.default_project
        )
        runspec.spec.parameters = params or runspec.spec.parameters
        runspec.spec.inputs = inputs or runspec.spec.inputs
        runspec.spec.verbose = verbose or runspec.spec.verbose
        runspec.spec.scrape_metrics = scrape_metrics or runspec.spec.scrape_metrics
        runspec.spec.output_path = out_path or artifact_path or runspec.spec.output_path
        runspec.spec.input_path = (
            workdir or runspec.spec.input_path or self.spec.workdir
        )

        spec = runspec.spec
        if self.spec.mode and self.spec.mode == "noctx":
            params = spec.parameters or {}
            for k, v in params.items():
                self.spec.args += ["--{}".format(k), str(v)]

        if spec.secret_sources:
            self._secrets = SecretsStore.from_list(spec.secret_sources)

        # update run metadata (uid, labels) and store in DB
        meta = runspec.metadata
        meta.uid = meta.uid or uuid.uuid4().hex
        runspec.spec.output_path = runspec.spec.output_path or config.artifact_path
        if runspec.spec.output_path:
            runspec.spec.output_path = runspec.spec.output_path.replace(
                "{{run.uid}}", meta.uid
            )
            runspec.spec.output_path = runspec.spec.output_path.replace(
                "{{run.project}}", runspec.metadata.project
            )
        if is_local(runspec.spec.output_path):
            logger.warning(
                "artifact path is not defined or is local,"
                " artifacts will not be visible in the UI"
            )
            if self.kind not in ["", "local", "handler", "dask"]:
                raise ValueError(
                    "absolute artifact_path must be specified"
                    " when running remote tasks"
                )
        db = self._get_db()

        if not self.is_deployed:
            raise RunError(
                "function image is not built/ready, use .build() method first"
            )

        if self.verbose:
            logger.info("runspec:\n{}".format(runspec.to_yaml()))

        if "V3IO_USERNAME" in environ and "v3io_user" not in meta.labels:
            meta.labels["v3io_user"] = environ.get("V3IO_USERNAME")

        if not self.is_child:
            dbstr = "self" if self._is_api_server else self.spec.rundb
            logger.info(
                "starting run {} uid={} DB={}".format(meta.name, meta.uid, dbstr)
            )
            meta.labels["kind"] = self.kind
            if "owner" not in meta.labels:
                meta.labels["owner"] = environ.get("V3IO_USERNAME") or getpass.getuser()
            if runspec.spec.output_path:
                runspec.spec.output_path = runspec.spec.output_path.replace(
                    "{{run.user}}", meta.labels["owner"]
                )

            if db and self.kind != "handler":
                struct = self.to_dict()
                hash_key = db.store_function(
                    struct, self.metadata.name, self.metadata.project, versioned=True
                )
                runspec.spec.function = self._function_uri(hash_key=hash_key)

        # execute the job remotely (to a k8s cluster via the API service)
        if self._use_remote_api():
            if self._secrets:
                runspec.spec.secret_sources = self._secrets.to_serial()
            try:
                resp = db.submit_job(runspec, schedule=schedule)
                if schedule:
                    logger.info("task scheduled, {}".format(resp))
                    return

                if resp:
                    txt = get_in(resp, "status.status_text")
                    if txt:
                        logger.info(txt)
                if watch or self.kfp:
                    runspec.logs(True, self._get_db())
                    resp = self._get_db_run(runspec)
            except Exception as err:
                logger.error("got remote run err, {}".format(err))
                result = None
                # if we got a schedule no reason to do post_run stuff (it purposed to update the run status with error,
                # but there's no run in case of schedule)
                if not schedule:
                    result = self._post_run(task=runspec, err=err)
                return self._wrap_run_result(
                    result, runspec, schedule=schedule, err=err
                )
            return self._wrap_run_result(resp, runspec, schedule=schedule)

        elif self._is_remote and not self._is_api_server and not self.kfp:
            logger.warning(
                "warning!, Api url not set, " "trying to exec remote runtime locally"
            )

        execution = MLClientCtx.from_dict(runspec.to_dict(), db, autocommit=False)

        # create task generator (for child runs) from spec
        task_generator = None
        if not self._is_nested:
            task_generator = get_generator(spec, execution)

        last_err = None
        if task_generator:
            # multiple runs (based on hyper params or params file)
            generator = task_generator.generate(runspec)
            results = self._run_many(generator, execution, runspec)
            results_to_iter(results, runspec, execution)
            result = execution.to_dict()

        else:
            # single run
            try:
                resp = self._run(runspec, execution)
                if watch and self.kind not in ["", "handler", "local"]:
                    state = runspec.logs(True, self._get_db())
                    if state != "succeeded":
                        logger.warning("run ended with state {}".format(state))
                result = self._post_run(resp, task=runspec)
            except RunError as err:
                last_err = err
                result = self._post_run(task=runspec, err=err)

        return self._wrap_run_result(result, runspec, schedule=schedule, err=last_err)

    def _wrap_run_result(
        self, result: dict, runspec: RunObject, schedule=None, err=None
    ):
        # if the purpose was to schedule (and not to run) nothing to wrap
        if schedule:
            return

        if result and self.kfp and err is None:
            write_kfpmeta(result)

        # show ipython/jupyter result table widget
        results_tbl = RunList()
        if result:
            results_tbl.append(result)
        else:
            logger.info("no returned result (job may still be in progress)")
            results_tbl.append(runspec.to_dict())
        if is_ipython and config.ipython_widget:
            results_tbl.show()

            uid = runspec.metadata.uid
            proj = (
                "--project {}".format(runspec.metadata.project)
                if runspec.metadata.project
                else ""
            )
            print(
                "to track results use .show() or .logs() or in CLI: \n"
                "!mlrun get run {} {} , !mlrun logs {} {}".format(uid, proj, uid, proj)
            )

        if result:
            run = RunObject.from_dict(result)
            logger.info("run executed, status={}".format(run.status.state))
            if run.status.state == "error":
                if self._is_remote and not self.is_child:
                    print("runtime error: {}".format(run.status.error))
                raise RunError(run.status.error)
            return run

        return None

    def _get_db_run(self, task: RunObject = None):
        if self._get_db() and task:
            project = task.metadata.project
            uid = task.metadata.uid
            iter = task.metadata.iteration
            try:
                return self._get_db().read_run(uid, project, iter=iter)
            except RunDBError:
                return None
        if task:
            return task.to_dict()

    def _generate_runtime_env(self, runobj: RunObject):
        runtime_env = {"MLRUN_EXEC_CONFIG": runobj.to_json()}
        if runobj.spec.verbose:
            runtime_env["MLRUN_LOG_LEVEL"] = "debug"
        if config.httpdb.api_url:
            runtime_env["MLRUN_DBPATH"] = config.httpdb.api_url
        if self.metadata.namespace or config.namespace:
            runtime_env["MLRUN_NAMESPACE"] = self.metadata.namespace or config.namespace
        return runtime_env

    def _get_cmd_args(self, runobj: RunObject, with_mlrun: bool):
        extra_env = self._generate_runtime_env(runobj)
        if self.spec.pythonpath:
            extra_env["PYTHONPATH"] = self.spec.pythonpath
        args = []
        command = self.spec.command
        code = (
            self.spec.build.functionSourceCode if hasattr(self.spec, "build") else None
        )

        if (code or runobj.spec.handler) and self.spec.mode == "pass":
            raise ValueError('cannot use "pass" mode with code or handler')

        if code:
            extra_env["MLRUN_EXEC_CODE"] = code

        if with_mlrun:
            args = ["run", "--name", runobj.metadata.name, "--from-env"]
            if not code:
                args += [command]
            command = "mlrun"

        if runobj.spec.handler:
            args += ["--handler", runobj.spec.handler]
        if self.spec.args:
            args += self.spec.args
        return command, args, extra_env

    def _run(self, runspec: RunObject, execution) -> dict:
        pass

    def _run_many(self, tasks, execution, runobj: RunObject) -> RunList:
        results = RunList()
        for task in tasks:
            try:
                # self.store_run(task)
                resp = self._run(task, execution)
                resp = self._post_run(resp, task=task)
            except RunError as err:
                task.status.state = "error"
                task.status.error = str(err)
                resp = self._post_run(task=task, err=err)
            results.append(resp)
        return results

    def store_run(self, runobj: RunObject):
        if self._get_db() and runobj:
            project = runobj.metadata.project
            uid = runobj.metadata.uid
            iter = runobj.metadata.iteration
            self._get_db().store_run(runobj.to_dict(), uid, project, iter=iter)

    def _store_run_dict(self, rundict: dict):
        if self._get_db() and rundict:
            project = get_in(rundict, "metadata.project", "")
            uid = get_in(rundict, "metadata.uid")
            iter = get_in(rundict, "metadata.iteration", 0)
            self._get_db().store_run(rundict, uid, project, iter=iter)

    def _post_run(self, resp: dict = None, task: RunObject = None, err=None) -> dict:
        """update the task state in the DB"""
        was_none = False
        if resp is None and task:
            was_none = True
            resp = self._get_db_run(task)

            if not resp:
                self.store_run(task)
                return task.to_dict()

            if task.status.status_text:
                update_in(resp, "status.status_text", task.status.status_text)

        if resp is None:
            return None

        if not isinstance(resp, dict):
            raise ValueError("post_run called with type {}".format(type(resp)))

        updates = None
        last_state = get_in(resp, "status.state", "")
        if last_state == "error" or err:
            updates = {"status.last_update": now_date().isoformat()}
            updates["status.state"] = "error"
            update_in(resp, "status.state", "error")
            if err:
                update_in(resp, "status.error", str(err))
            err = get_in(resp, "status.error")
            if err:
                updates["status.error"] = str(err)
        elif not was_none and last_state != "completed":
            updates = {"status.last_update": now_date().isoformat()}
            updates["status.state"] = "completed"
            update_in(resp, "status.state", "completed")

        if self._get_db() and updates:
            project = get_in(resp, "metadata.project")
            uid = get_in(resp, "metadata.uid")
            iter = get_in(resp, "metadata.iteration", 0)
            self._get_db().update_run(updates, uid, project, iter=iter)

        return resp

    def _force_handler(self, handler):
        if not handler:
            raise RunError("handler must be provided for {} runtime".format(self.kind))

    def full_image_path(self, image=None):
        image = image or self.spec.image or ""

        image = enrich_image_url(image)
        if not image.startswith("."):
            return image
        registry, _ = get_parsed_docker_registry()
        if registry:
            return "{}/{}".format(registry, image[1:])
        if "IGZ_NAMESPACE_DOMAIN" in environ:
            return "docker-registry.{}:80/{}".format(
                environ.get("IGZ_NAMESPACE_DOMAIN"), image[1:]
            )
        raise RunError("local container registry is not defined")

    def as_step(
        self,
        runspec: RunObject = None,
        handler=None,
        name: str = "",
        project: str = "",
        params: dict = None,
        hyperparams=None,
        selector="",
        inputs: dict = None,
        outputs: dict = None,
        workdir: str = "",
        artifact_path: str = "",
        image: str = "",
        labels: dict = None,
        use_db=True,
        verbose=None,
        scrape_metrics=False,
    ):
        """Run a local or remote task.

        :param runspec:         run template object or dict (see RunTemplate)
        :param handler:         name of the function handler
        :param name:            execution name
        :param project:         project name
        :param params:          input parameters (dict)
        :param hyperparams:     hyper parameters
        :param selector:        selection criteria for hyper params
        :param inputs:          input objects (dict of key: path)
        :param outputs:         list of outputs which can pass in the workflow
        :param artifact_path:   default artifact output path (replace out_path)
        :param workdir:         default input artifacts path
        :param image:           container image to use
        :param labels:          labels to tag the job/run with ({key:val, ..})
        :param use_db:          save function spec in the db (vs the workflow file)
        :param verbose:         add verbose prints/logs
        :param scrape_metrics:  whether to add the `mlrun/scrape-metrics` label to this run's resources

        :return: KubeFlow containerOp
        """

        # if self.spec.image and not image:
        #     image = self.full_image_path()

        if use_db:
            hash_key = self.save(versioned=True, refresh=True)
            url = "db://" + self._function_uri(hash_key=hash_key)
        else:
            url = None

        if runspec is not None:
            verify_field_regex(
                "run.metadata.name", runspec.metadata.name, mlrun.utils.regex.run_name
            )

        return mlrun_op(
            name,
            project,
            function=self,
            func_url=url,
            runobj=runspec,
            handler=handler,
            params=params,
            hyperparams=hyperparams,
            selector=selector,
            inputs=inputs,
            outputs=outputs,
            job_image=image,
            labels=labels,
            out_path=artifact_path,
            in_path=workdir,
            verbose=verbose,
            scrape_metrics=scrape_metrics,
        )

    def export(self, target="", format=".yaml", secrets=None, strip=True):
        """save function spec to a local/remote path (default to
        ./function.yaml)"""
        if self.kind == "handler":
            raise ValueError(
                "cannot export local handler function, use "
                + "code_to_function() to serialize your function"
            )
        calc_hash(self)
        struct = self.to_dict(strip=strip)
        if format == ".yaml":
            data = dict_to_yaml(struct)
        else:
            data = dict_to_json(struct)
        stores = store_manager.set(secrets)
        target = target or "function.yaml"
        datastore, subpath = stores.get_or_create_store(target)
        datastore.put(subpath, data)
        logger.info("function spec saved to path: {}".format(target))
        return self

    def save(self, tag="", versioned=False, refresh=False):
        db = self._get_db()
        if not db:
            logger.error("database connection is not configured")
            return ""

        if refresh and self._is_remote_api():
            try:
                meta = self.metadata
                db_func = db.get_function(meta.name, meta.project, meta.tag)
                if db_func and "status" in db_func:
                    self.status = db_func["status"]
                    if (
                        self.status.state
                        and self.status.state == "ready"
                        and "nuclio_name" not in self.status
                    ):
                        self.spec.image = get_in(db_func, "spec.image", self.spec.image)
            except Exception:
                pass

        tag = tag or self.metadata.tag

        obj = self.to_dict()
        logger.debug("saving function: {}, tag: {}".format(self.metadata.name, tag))
        hash_key = db.store_function(
            obj, self.metadata.name, self.metadata.project, tag, versioned
        )
        return hash_key

    def to_dict(self, fields=None, exclude=None, strip=False):
        struct = super().to_dict(fields, exclude=exclude)
        if strip:
            if "status" in struct:
                del struct["status"]
        return struct

    def doc(self):
        print("function:", self.metadata.name)
        print(self.spec.description)
        if self.spec.default_handler:
            print("default handler:", self.spec.default_handler)
        if self.spec.entry_points:
            print("entry points:")
            for name, entry in self.spec.entry_points.items():
                print("  {}: {}".format(name, entry.get("doc", "")))
                params = entry.get("parameters")
                if params:
                    for p in params:
                        line = p["name"]
                        if "type" in p:
                            line += "({})".format(p["type"])
                        line += "  - " + p.get("doc", "")
                        if "default" in p:
                            line += ", default={}".format(p["default"])
                        print("    " + line)


def is_local(url):
    if not url:
        return True
    return "://" not in url and not url.startswith("/")


class BaseRuntimeHandler(ABC):
    @staticmethod
    @abstractmethod
    def _get_object_label_selector(object_id: str) -> str:
        """
        Should return the label selector should be used to get only resources of a specific object (with id object_id)
        """
        pass

    def list_resources(self, label_selector: str = None) -> Dict:
        k8s_helper = get_k8s_helper()
        namespace = k8s_helper.resolve_namespace()
        label_selector = self._resolve_label_selector(label_selector)
        pods = self._list_pods(namespace, label_selector)
        pod_resources = self._build_pod_resources(pods)
        crd_objects = self._list_crd_objects(namespace, label_selector)
        crd_resources = self._build_crd_resources(crd_objects)
        response = self._build_list_resources_response(pod_resources, crd_resources)
        response = self._enrich_list_resources_response(
            response, namespace, label_selector
        )
        return response

    def delete_resources(
        self,
        db: DBInterface,
        db_session: Session,
        label_selector: str = None,
        force: bool = False,
        grace_period: int = config.runtime_resources_deletion_grace_period,
    ):
        k8s_helper = get_k8s_helper()
        namespace = k8s_helper.resolve_namespace()
        label_selector = self._resolve_label_selector(label_selector)
        crd_group, crd_version, crd_plural = self._get_crd_info()
        if crd_group and crd_version and crd_plural:
            deleted_resources = self._delete_crd_resources(
                db, db_session, namespace, label_selector, force, grace_period
            )
        else:
            deleted_resources = self._delete_pod_resources(
                db, db_session, namespace, label_selector, force, grace_period
            )
        self._delete_resources(
            db,
            db_session,
            namespace,
            deleted_resources,
            label_selector,
            force,
            grace_period,
        )

    def delete_runtime_object_resources(
        self,
        db: DBInterface,
        db_session: Session,
        object_id: str,
        label_selector: str = None,
        force: bool = False,
        grace_period: int = config.runtime_resources_deletion_grace_period,
    ):
        object_label_selector = self._get_object_label_selector(object_id)
        if label_selector:
            label_selector = ",".join([object_label_selector, label_selector])
        else:
            label_selector = object_label_selector
        self.delete_resources(db, db_session, label_selector, force, grace_period)

    def monitor_runs(
        self, db: DBInterface, db_session: Session,
    ):
        k8s_helper = get_k8s_helper()
        namespace = k8s_helper.resolve_namespace()
        label_selector = self._get_default_label_selector()
        crd_group, crd_version, crd_plural = self._get_crd_info()
        runtime_resource_is_crd = False
        if crd_group and crd_version and crd_plural:
            runtime_resource_is_crd = True
            runtime_resources = self._list_crd_objects(namespace, label_selector)
        else:
            runtime_resources = self._list_pods(namespace, label_selector)
        project_run_uid_map = self._list_runs_for_monitoring(db, db_session)
        for runtime_resource in runtime_resources:
            try:
                self._monitor_runtime_resource(
                    db,
                    db_session,
                    project_run_uid_map,
                    runtime_resource,
                    runtime_resource_is_crd,
                    namespace,
                )
            except Exception as exc:
                logger.warning(
                    "Failed monitoring runtime resource. Continuing",
                    runtime_resource_name=runtime_resource["metadata"]["name"],
                    namespace=namespace,
                    exc=str(exc),
                )

    def _enrich_list_resources_response(
        self, response: Dict, namespace: str, label_selector: str = None
    ) -> Dict:
        """
        Override this to list resources other then pods or CRDs (which are handled by the base class)
        """
        return response

    def _delete_resources(
        self,
        db: DBInterface,
        db_session: Session,
        namespace: str,
        deleted_resources: List[Dict],
        label_selector: str = None,
        force: bool = False,
        grace_period: int = config.runtime_resources_deletion_grace_period,
    ):
        """
        Override this to handle deletion of resources other then pods or CRDs (which are handled by the base class)
        Note that this is happening before the deletion of the CRDs or the pods
        """
        pass

    def _resolve_crd_object_status_info(
        self, db: DBInterface, db_session: Session, crd_object
    ) -> Tuple[bool, Optional[datetime], Optional[str]]:
        """
        Override this if the runtime has CRD resources.
        :return: Tuple with:
        1. bool determining whether the crd object is in terminal state
        2. datetime of when the crd object got into terminal state (only when the crd object in terminal state)
        3. the desired run state matching the crd object state
        """
        return False, None, None

    def _resolve_pod_status_info(
        self, db: DBInterface, db_session: Session, pod: Dict
    ) -> Tuple[bool, Optional[datetime], Optional[str]]:
        """
        :return: Tuple with:
        1. bool determining whether the pod is in terminal state
        2. datetime of when the pod got into terminal state (only when the pod in terminal state)
        3. the run state matching the pod state
        """
        in_terminal_state = pod["status"]["phase"] in PodPhases.terminal_phases()
        run_state = PodPhases.pod_phase_to_run_state(pod["status"]["phase"])
        last_container_completion_time = None
        if in_terminal_state:
            for container_status in pod["status"].get("container_statuses", []):
                if container_status.get("state", {}).get("terminated"):
                    container_completion_time = container_status["state"][
                        "terminated"
                    ].get("finished_at")

                    # take latest completion time
                    if (
                        not last_container_completion_time
                        or last_container_completion_time < container_completion_time
                    ):
                        last_container_completion_time = container_completion_time

        return in_terminal_state, last_container_completion_time, run_state

    @staticmethod
    def _get_default_label_selector() -> str:
        """
        Override this to add a default label selector
        """
        return ""

    @staticmethod
    def _get_crd_info() -> Tuple[str, str, str]:
        """
        Override this if the runtime has CRD resources. this should return the CRD info:
        crd group, crd version, crd plural
        """
        return "", "", ""

    @staticmethod
    def _consider_run_on_resources_deletion() -> bool:
        """
        Some resources are tightly coupled to mlrun Run object, for example, for each Run of a Funtion of the job kind
        a kubernetes job is being generated, on the opposite a Function of the daskjob kind generates a dask cluster,
        and every Run is being excuted using this cluster, i.e. no resources are created for the Run.
        This function should return true for runtimes in which Run are coupled to the underlying resources and therefore
        aspects of the Run (like its state) should be taken into consideration on resources deletion
        """
        return False

    def _list_pods(self, namespace: str, label_selector: str = None) -> List:
        k8s_helper = get_k8s_helper()
        pods = k8s_helper.list_pods(namespace, selector=label_selector)
        # when we work with custom objects (list_namespaced_custom_object) it's always a dict, to be able to generalize
        # code working on runtime resource (either a custom object or a pod) we're transforming to dicts
        pods = [pod.to_dict() for pod in pods]
        return pods

    def _list_crd_objects(self, namespace: str, label_selector: str = None) -> List:
        k8s_helper = get_k8s_helper()
        crd_group, crd_version, crd_plural = self._get_crd_info()
        crd_objects = []
        if crd_group and crd_version and crd_plural:
            try:
                crd_objects = k8s_helper.crdapi.list_namespaced_custom_object(
                    crd_group,
                    crd_version,
                    namespace,
                    crd_plural,
                    label_selector=label_selector,
                )
            except ApiException as e:
                # ignore error if crd is not defined
                if e.status != 404:
                    raise
            else:
                crd_objects = crd_objects["items"]
        return crd_objects

    def _resolve_label_selector(self, label_selector: str = None) -> str:
        default_label_selector = self._get_default_label_selector()

        if label_selector:
            label_selector = ",".join([default_label_selector, label_selector])
        else:
            label_selector = default_label_selector

        return label_selector

    def _delete_pod_resources(
        self,
        db: DBInterface,
        db_session: Session,
        namespace: str,
        label_selector: str = None,
        force: bool = False,
        grace_period: int = config.runtime_resources_deletion_grace_period,
    ) -> List[Dict]:
        k8s_helper = get_k8s_helper()
        pods = k8s_helper.v1api.list_namespaced_pod(
            namespace, label_selector=label_selector
        )
        deleted_pods = []
        for pod in pods.items:
            pod_dict = pod.to_dict()

            # best effort - don't let one failure in pod deletion to cut the whole operation
            try:
                (
                    in_terminal_state,
                    last_update,
                    run_state,
                ) = self._resolve_pod_status_info(db, db_session, pod_dict)
                if not force:
                    if not in_terminal_state:
                        continue

                    # give some grace period if we have last update time
                    now = datetime.now(timezone.utc)
                    if (
                        last_update is not None
                        and last_update + timedelta(seconds=float(grace_period)) > now
                    ):
                        continue

                if self._consider_run_on_resources_deletion():
                    try:
                        self._pre_deletion_runtime_resource_run_actions(
                            db, db_session, pod_dict, run_state
                        )
                    except Exception as exc:
                        # Don't prevent the deletion for failure in the pre deletion run actions
                        logger.warning(
                            "Failure in pod run pre-deletion actions. Continuing",
                            exc=repr(exc),
                            pod_name=pod.metadata.name,
                        )

                self._delete_pod(namespace, pod)
                deleted_pods.append(pod_dict)
            except Exception as exc:
                logger.warning(
                    f"Cleanup failed processing pod {pod.metadata.name}: {repr(exc)}. Continuing"
                )
        return deleted_pods

    def _delete_crd_resources(
        self,
        db: DBInterface,
        db_session: Session,
        namespace: str,
        label_selector: str = None,
        force: bool = False,
        grace_period: int = config.runtime_resources_deletion_grace_period,
    ) -> List[Dict]:
        k8s_helper = get_k8s_helper()
        crd_group, crd_version, crd_plural = self._get_crd_info()
        deleted_crds = []
        try:
            crd_objects = k8s_helper.crdapi.list_namespaced_custom_object(
                crd_group,
                crd_version,
                namespace,
                crd_plural,
                label_selector=label_selector,
            )
        except ApiException as e:
            # ignore error if crd is not defined
            if e.status != 404:
                raise
        else:
            for crd_object in crd_objects["items"]:
                # best effort - don't let one failure in pod deletion to cut the whole operation
                try:
                    (
                        in_terminal_state,
                        last_update,
                        desired_run_state,
                    ) = self._resolve_crd_object_status_info(db, db_session, crd_object)
                    if not force:
                        if not in_terminal_state:
                            continue

                        # give some grace period if we have last update time
                        now = datetime.now(timezone.utc)
                        if (
                            last_update is not None
                            and last_update + timedelta(seconds=float(grace_period))
                            > now
                        ):
                            continue

                    if self._consider_run_on_resources_deletion():

                        try:
                            self._pre_deletion_runtime_resource_run_actions(
                                db, db_session, crd_object, desired_run_state
                            )
                        except Exception as exc:
                            # Don't prevent the deletion for failure in the pre deletion run actions
                            logger.warning(
                                "Failure in crd object run pre-deletion actions. Continuing",
                                exc=str(exc),
                                crd_object_name=crd_object["metadata"]["name"],
                            )

                    self._delete_crd(
                        namespace, crd_group, crd_version, crd_plural, crd_object
                    )
                    deleted_crds.append(crd_object)
                except Exception:
                    exc = traceback.format_exc()
                    crd_object_name = crd_object["metadata"]["name"]
                    logger.warning(
                        f"Cleanup failed processing CRD object {crd_object_name}: {exc}. Continuing"
                    )
        return deleted_crds

    def _pre_deletion_runtime_resource_run_actions(
        self,
        db: DBInterface,
        db_session: Session,
        runtime_resource: Dict,
        run_state: str,
    ):
        project, uid = self._resolve_runtime_resource_run(runtime_resource)

        # if cannot resolve related run nothing to do
        if not uid:
            logger.warning(
                "Could not resolve run uid from runtime resource. Skipping pre-deletion actions",
                runtime_resource=runtime_resource,
            )
            raise ValueError("Could not resolve run uid from runtime resource")

        logger.info(
            "Performing pre-deletion actions before cleaning up runtime resources",
            project=project,
            uid=uid,
        )

        self._ensure_run_state(db, db_session, project, uid, run_state)

        self._ensure_run_logs_collected(db, db_session, project, uid)

    def _is_runtime_resource_run_in_terminal_state(
        self, db: DBInterface, db_session: Session, runtime_resource: Dict,
    ) -> Tuple[bool, Optional[datetime]]:
        """
        A runtime can have different underlying resources (like pods or CRDs) - to generalize we call it runtime
        resource. This function will verify whether the Run object related to this runtime resource is in transient
        state. This is useful in order to determine whether an object can be removed. for example, a kubejob's pod
        might be in completed state, but we would like to verify that the run is completed as well to verify the logs
        were collected before we're removing the pod.

        :returns: bool determining whether the run in terminal state, and the last update time if it exists
        """
        project, uid = self._resolve_runtime_resource_run(runtime_resource)

        # if no uid, assume in terminal state
        if not uid:
            return True, None

        run = db.read_run(db_session, uid, project)
        last_update = None
        last_update_str = run.get("status", {}).get("last_update")
        if last_update_str is not None:
            last_update = datetime.fromisoformat(last_update_str)

        if run.get("status", {}).get("state") not in RunStates.terminal_states():
            return False, last_update

        return True, last_update

    def _list_runs_for_monitoring(
        self, db: DBInterface, db_session: Session,
    ):
        runs = db.list_runs(db_session, project="*")
        project_run_uid_map = {}
        run_with_missing_data = []
        duplicated_runs = []
        for run in runs:
            project = run.get("metadata", {}).get("project")
            uid = run.get("metadata", {}).get("uid")
            if not uid or not project:
                run_with_missing_data.append(run.get("metadata", {}))
                continue
            current_run = project_run_uid_map.setdefault(project, {}).get(uid)

            # sanity
            if current_run:
                duplicated_runs = {
                    "monitored_run": current_run.get(["metadata"]),
                    "duplicated_run": run.get(["metadata"]),
                }
                continue

            project_run_uid_map[project][uid] = run

        # If there are duplications or runs with missing data it probably won't be fixed
        # Monitoring is running periodically and we don't want to log on every problem we found which will spam the log
        # so we're aggregating the problems and logging only once per aggregation
        if duplicated_runs:
            logger.warning(
                "Found duplicated runs (same uid). Heuristically monitoring the first one found",
                duplicated_runs=duplicated_runs,
            )

        if run_with_missing_data:
            logger.warning(
                "Found runs with missing data. They will not be monitored",
                run_with_missing_data=run_with_missing_data,
            )

        return project_run_uid_map

    def _monitor_runtime_resource(
        self,
        db: DBInterface,
        db_session: Session,
        project_run_uid_map: Dict,
        runtime_resource: Dict,
        runtime_resource_is_crd: bool,
        namespace: str,
    ):
        project, uid = self._resolve_runtime_resource_run(runtime_resource)
        if not project or not uid:
            logger.warning(
                "Could not resolve run project or uid from runtime resource, can not monitor run. Continuing",
                project=project,
                uid=uid,
                runtime_resource_name=runtime_resource["metadata"]["name"],
                namespace=namespace,
            )
            return
        run = project_run_uid_map.get(project, {}).get(uid)
        if runtime_resource_is_crd:
            (_, _, run_state,) = self._resolve_crd_object_status_info(
                db, db_session, runtime_resource
            )
        else:
            (_, _, run_state,) = self._resolve_pod_status_info(
                db, db_session, runtime_resource
            )
        _, updated_run_state = self._ensure_run_state(
            db, db_session, project, uid, run_state, run, search_run=False,
        )
        if updated_run_state in RunStates.terminal_states():
            self._ensure_run_logs_collected(db, db_session, project, uid)

    @staticmethod
    def _get_run_label_selector(project: str, run_uid: str):
        return f"mlrun/project={project},mlrun/uid={run_uid}"

    @staticmethod
    def _ensure_run_logs_collected(
        db: DBInterface, db_session: Session, project: str, uid: str
    ):
        # import here to avoid circular imports
        import mlrun.api.crud as crud

        log_file_exists = crud.Logs.log_file_exists(project, uid)
        if not log_file_exists:
            _, logs_from_k8s = crud.Logs.get_logs(
                db_session, project, uid, source=LogSources.K8S
            )
            if logs_from_k8s:
                logger.info("Storing run logs", project=project, uid=uid)
                crud.Logs.store_log(logs_from_k8s, project, uid, append=False)

    @staticmethod
    def _ensure_run_state(
        db: DBInterface,
        db_session: Session,
        project: str,
        uid: str,
        run_state: str,
        run: Dict = None,
        search_run: bool = True,
    ) -> Tuple[bool, str]:
        if run is None:
            run = {}
        if search_run:
            try:
                run = db.read_run(db_session, uid, project)
            except mlrun.errors.MLRunNotFoundError:
                run = {}
        if not run:
            logger.warning(
                "Run not found. A new run will be created",
                project=project,
                uid=uid,
                desired_run_state=run_state,
                search_run=search_run,
            )
            run = {"metadata": {"project": project, "uid": uid}}
        db_run_state = run.get("status", {}).get("state")
        if db_run_state:
            if db_run_state == run_state:
                return False, run_state
            # if the current run state is terminal and different than the desired - log
            if db_run_state in RunStates.terminal_states():

                # This can happen when the SDK running in the user's Run updates the Run's state to terminal, but
                # before it exits, when the runtime resource is still running, the API monitoring (here) is executed
                if run_state not in RunStates.terminal_states():
                    now = datetime.now(timezone.utc)
                    last_update_str = run.get("status", {}).get("last_update")
                    if last_update_str is not None:
                        last_update = datetime.fromisoformat(last_update_str)
                        debounce_period = config.runs_monitoring_interval
                        if last_update > now - timedelta(
                            seconds=float(debounce_period)
                        ):
                            logger.warning(
                                "Monitoring found non-terminal state on runtime resource but record has recently "
                                "updated to terminal state. Debouncing",
                                project=project,
                                uid=uid,
                                db_run_state=db_run_state,
                                run_state=run_state,
                                last_update=last_update,
                                now=now,
                                debounce_period=debounce_period,
                            )
                            return False, run_state

                logger.warning(
                    "Run record has terminal state but monitoring found different state on runtime resource. Changing",
                    project=project,
                    uid=uid,
                    db_run_state=db_run_state,
                    run_state=run_state,
                )

        logger.info("Updating run state", run_state=run_state)
        run.setdefault("status", {})["state"] = run_state
        run.setdefault("status", {})["last_update"] = now_date().isoformat()
        db.store_run(db_session, run, uid, project)

        return True, run_state

    @staticmethod
    def _resolve_runtime_resource_run(runtime_resource: Dict) -> Tuple[str, str]:
        project = (
            runtime_resource.get("metadata", {}).get("labels", {}).get("mlrun/project")
        )
        if not project:
            project = config.default_project
        uid = runtime_resource.get("metadata", {}).get("labels", {}).get("mlrun/uid")
        return project, uid

    @staticmethod
    def _delete_crd(namespace, crd_group, crd_version, crd_plural, crd_object):
        k8s_helper = get_k8s_helper()
        name = crd_object["metadata"]["name"]
        try:
            k8s_helper.crdapi.delete_namespaced_custom_object(
                crd_group,
                crd_version,
                namespace,
                crd_plural,
                name,
                client.V1DeleteOptions(),
            )
            logger.info(
                "Deleted crd object",
                name=name,
                namespace=namespace,
                crd_plural=crd_plural,
            )
        except ApiException as e:
            # ignore error if crd object is already removed
            if e.status != 404:
                raise

    @staticmethod
    def _delete_pod(namespace, pod):
        k8s_helper = get_k8s_helper()
        try:
            k8s_helper.v1api.delete_namespaced_pod(pod.metadata.name, namespace)
            logger.info("Deleted pod", pod=pod.metadata.name)
        except ApiException as e:
            # ignore error if pod is already removed
            if e.status != 404:
                raise

    @staticmethod
    def _build_pod_resources(pods) -> List:
        pod_resources = []
        for pod in pods:
            pod_resources.append(
                {
                    "name": pod["metadata"]["name"],
                    "labels": pod["metadata"]["labels"],
                    "status": pod["status"],
                }
            )
        return pod_resources

    @staticmethod
    def _build_crd_resources(custom_objects) -> List:
        crd_resources = []
        for custom_object in custom_objects:
            crd_resources.append(
                {
                    "name": custom_object["metadata"]["name"],
                    "labels": custom_object["metadata"]["labels"],
                    "status": custom_object["status"],
                }
            )
        return crd_resources

    @staticmethod
    def _build_list_resources_response(
        pod_resources: List = None, crd_resources: List = None
    ) -> Dict:
        if crd_resources is None:
            crd_resources = []
        if pod_resources is None:
            pod_resources = []
        return {
            "crd_resources": crd_resources,
            "pod_resources": pod_resources,
        }
