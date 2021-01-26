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

import tempfile
import time
from datetime import datetime
from os import path, remove
from typing import List, Dict, Union

import kfp
import requests
import semver
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import mlrun
import mlrun.projects
from mlrun.api import schemas
from mlrun.errors import MLRunInvalidArgumentError
from .base import RunDBError, RunDBInterface
from ..config import config
from ..feature_store.model import FeatureSet, FeatureVector
from ..lists import RunList, ArtifactList
from ..utils import dict_to_json, logger, new_pipe_meta, datetime_to_iso

default_project = config.default_project

_artifact_keys = [
    "format",
    "inline",
    "key",
    "src_path",
    "target_path",
    "viewer",
]


def bool2str(val):
    return "yes" if val else "no"


http_adapter = HTTPAdapter(
    max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
)


class HTTPRunDB(RunDBInterface):
    kind = "http"

    def __init__(self, base_url, user="", password="", token=""):
        self.base_url = base_url
        self.user = user
        self.password = password
        self.token = token
        self.server_version = ""
        self.session = None

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}({self.base_url!r})"

    def api_call(
        self,
        method,
        path,
        error=None,
        params=None,
        body=None,
        json=None,
        headers=None,
        timeout=20,
    ):
        url = f"{self.base_url}/api/{path}"
        kw = {
            key: value
            for key, value in (
                ("params", params),
                ("data", body),
                ("json", json),
                ("headers", headers),
            )
            if value is not None
        }

        if self.user:
            kw["auth"] = (self.user, self.password)
        elif self.token:
            kw["headers"] = {"Authorization": "Bearer " + self.token}

        if not self.session:
            self.session = requests.Session()
            self.session.mount("http://", http_adapter)
            self.session.mount("https://", http_adapter)

        try:
            resp = self.session.request(
                method, url, timeout=timeout, verify=False, **kw
            )
        except requests.RequestException as err:
            error = error or "{} {}, error: {}".format(method, url, err)
            raise RunDBError(error) from err

        if not resp.ok:
            if resp.content:
                try:
                    data = resp.json()
                    reason = data.get("detail", {}).get("reason", "")
                except Exception:
                    reason = ""
            if reason:
                error = error or "{} {}, error: {}".format(method, url, reason)
                raise RunDBError(error)

            try:
                resp.raise_for_status()
            except requests.RequestException as err:
                error = error or "{} {}, error: {}".format(method, url, err)
                raise RunDBError(error) from err

        return resp

    def _path_of(self, prefix, project, uid):
        project = project or default_project
        return f"{prefix}/{project}/{uid}"

    def connect(self, secrets=None):
        resp = self.api_call("GET", "healthz", timeout=5)
        try:
            server_cfg = resp.json()
            self.server_version = server_cfg["version"]
            self._validate_version_compatibility(self.server_version, config.version)
            config.namespace = config.namespace or server_cfg.get("namespace")
            if (
                "namespace" in server_cfg
                and server_cfg["namespace"] != config.namespace
            ):
                logger.warning(
                    "warning!, server ({}) and client ({}) namespace dont match".format(
                        server_cfg["namespace"], config.namespace
                    )
                )

            # get defaults from remote server
            config.remote_host = config.remote_host or server_cfg.get("remote_host")
            config.mpijob_crd_version = config.mpijob_crd_version or server_cfg.get(
                "mpijob_crd_version"
            )
            config.ui.url = config.resolve_ui_url() or server_cfg.get("ui_url")
            # This is has a default value, therefore config.ui.projects_prefix will always have a value, prioritize the
            # API value first
            config.ui.projects_prefix = (
                server_cfg.get("ui_projects_prefix") or config.ui.projects_prefix
            )
            config.artifact_path = config.artifact_path or server_cfg.get(
                "artifact_path"
            )
            config.spark_app_image = config.spark_app_image or server_cfg.get(
                "spark_app_image"
            )
            config.spark_app_image_tag = config.spark_app_image_tag or server_cfg.get(
                "spark_app_image_tag"
            )
            config.httpdb.builder.docker_registry = (
                config.httpdb.builder.docker_registry
                or server_cfg.get("docker_registry")
            )
        except Exception:
            pass
        return self

    def store_log(self, uid, project="", body=None, append=False):
        if not body:
            return

        path = self._path_of("log", project, uid)
        params = {"append": bool2str(append)}
        error = f"store log {project}/{uid}"
        self.api_call("POST", path, error, params, body)

    def get_log(self, uid, project="", offset=0, size=-1):
        params = {"offset": offset, "size": size}
        path = self._path_of("log", project, uid)
        error = f"get log {project}/{uid}"
        resp = self.api_call("GET", path, error, params=params)
        if resp.headers:
            state = resp.headers.get("x-mlrun-run-state", "")
            return state.lower(), resp.content

        return "unknown", resp.content

    def watch_log(self, uid, project="", watch=True, offset=0):
        state, text = self.get_log(uid, project, offset=offset)
        if text:
            print(text.decode())
        if watch:
            nil_resp = 0
            while state in ["pending", "running"]:
                offset += len(text)
                if nil_resp < 3:
                    time.sleep(3)
                else:
                    time.sleep(10)
                state, text = self.get_log(uid, project, offset=offset)
                if text:
                    nil_resp = 0
                    print(text.decode(), end="")
                else:
                    nil_resp += 1

        return state

    def store_run(self, struct, uid, project="", iter=0):
        path = self._path_of("run", project, uid)
        params = {"iter": iter}
        error = f"store run {project}/{uid}"
        body = _as_json(struct)
        self.api_call("POST", path, error, params=params, body=body)

    def update_run(self, updates: dict, uid, project="", iter=0):
        path = self._path_of("run", project, uid)
        params = {"iter": iter}
        error = f"update run {project}/{uid}"
        body = _as_json(updates)
        self.api_call("PATCH", path, error, params=params, body=body)

    def read_run(self, uid, project="", iter=0):
        path = self._path_of("run", project, uid)
        params = {"iter": iter}
        error = f"get run {project}/{uid}"
        resp = self.api_call("GET", path, error, params=params)
        return resp.json()["data"]

    def del_run(self, uid, project="", iter=0):
        path = self._path_of("run", project, uid)
        params = {"iter": iter}
        error = f"del run {project}/{uid}"
        self.api_call("DELETE", path, error, params=params)

    def list_runs(
        self,
        name=None,
        uid=None,
        project=None,
        labels=None,
        state=None,
        sort=True,
        last=0,
        iter=False,
        start_time_from: datetime = None,
        start_time_to: datetime = None,
        last_update_time_from: datetime = None,
        last_update_time_to: datetime = None,
    ):

        project = project or default_project
        params = {
            "name": name,
            "uid": uid,
            "project": project,
            "label": labels or [],
            "state": state,
            "sort": bool2str(sort),
            "iter": bool2str(iter),
            "start_time_from": datetime_to_iso(start_time_from),
            "start_time_to": datetime_to_iso(start_time_to),
            "last_update_time_from": datetime_to_iso(last_update_time_from),
            "last_update_time_to": datetime_to_iso(last_update_time_to),
        }
        error = "list runs"
        resp = self.api_call("GET", "runs", error, params=params)
        return RunList(resp.json()["runs"])

    def del_runs(self, name=None, project=None, labels=None, state=None, days_ago=0):
        project = project or default_project
        params = {
            "name": name,
            "project": project,
            "label": labels or [],
            "state": state,
            "days_ago": str(days_ago),
        }
        error = "del runs"
        self.api_call("DELETE", "runs", error, params=params)

    def store_artifact(self, key, artifact, uid, iter=None, tag=None, project=""):
        path = self._path_of("artifact", project, uid) + "/" + key
        params = {
            "tag": tag,
        }
        if iter:
            params["iter"] = str(iter)

        error = f"store artifact {project}/{uid}/{key}"

        body = _as_json(artifact)
        self.api_call("POST", path, error, params=params, body=body)

    def read_artifact(self, key, tag=None, iter=None, project=""):
        project = project or default_project
        tag = tag or "latest"
        path = "projects/{}/artifact/{}?tag={}".format(project, key, tag)
        error = f"read artifact {project}/{key}"
        params = {"iter": str(iter)} if iter else {}
        resp = self.api_call("GET", path, error, params=params)
        return resp.json()["data"]

    def del_artifact(self, key, tag=None, project=""):
        path = self._path_of("artifact", project, key)  # TODO: uid?
        params = {
            "key": key,
            "tag": tag,
        }
        error = f"del artifact {project}/{key}"
        self.api_call("DELETE", path, error, params=params)

    def list_artifacts(
        self, name=None, project=None, tag=None, labels=None, since=None, until=None
    ):
        project = project or default_project
        params = {
            "name": name,
            "project": project,
            "tag": tag,
            "label": labels or [],
        }
        error = "list artifacts"
        resp = self.api_call("GET", "artifacts", error, params=params)
        values = ArtifactList(resp.json()["artifacts"])
        values.tag = tag
        return values

    def del_artifacts(self, name=None, project=None, tag=None, labels=None, days_ago=0):
        project = project or default_project
        params = {
            "name": name,
            "project": project,
            "tag": tag,
            "label": labels or [],
            "days_ago": str(days_ago),
        }
        error = "del artifacts"
        self.api_call("DELETE", "artifacts", error, params=params)

    def list_artifact_tags(self, project=None):
        project = project or default_project
        error_message = f"Failed listing artifact tags. project={project}"
        response = self.api_call(
            "GET", f"/projects/{project}/artifact-tags", error_message
        )
        return response.json()

    def store_function(self, function, name, project="", tag=None, versioned=False):
        params = {"tag": tag, "versioned": versioned}
        project = project or default_project
        path = self._path_of("func", project, name)

        error = f"store function {project}/{name}"
        resp = self.api_call(
            "POST", path, error, params=params, body=dict_to_json(function)
        )

        # hash key optional to be backwards compatible to API v<0.4.10 in which it wasn't in the response
        return resp.json().get("hash_key")

    def get_function(self, name, project="", tag=None, hash_key=""):
        params = {"tag": tag, "hash_key": hash_key}
        project = project or default_project
        path = self._path_of("func", project, name)
        error = f"get function {project}/{name}"
        resp = self.api_call("GET", path, error, params=params)
        return resp.json()["func"]

    def delete_function(self, name: str, project: str = ""):
        project = project or default_project
        path = f"projects/{project}/functions/{name}"
        error_message = f"Failed deleting function {project}/{name}"
        self.api_call("DELETE", path, error_message)

    def list_functions(self, name=None, project=None, tag=None, labels=None):
        params = {
            "project": project or default_project,
            "name": name,
            "tag": tag,
            "label": labels or [],
        }
        error = "list functions"
        resp = self.api_call("GET", "funcs", error, params=params)
        return resp.json()["funcs"]

    def list_runtimes(self, label_selector: str = None) -> List:
        params = {"label_selector": label_selector}
        error = "list runtimes"
        resp = self.api_call("GET", "runtimes", error, params=params)
        return resp.json()

    def get_runtime(self, kind: str, label_selector: str = None) -> Dict:
        params = {"label_selector": label_selector}
        path = f"runtimes/{kind}"
        error = f"get runtime {kind}"
        resp = self.api_call("GET", path, error, params=params)
        return resp.json()

    def delete_runtimes(
        self,
        label_selector: str = None,
        force: bool = False,
        grace_period: int = config.runtime_resources_deletion_grace_period,
    ):
        params = {
            "label_selector": label_selector,
            "force": force,
            "grace_period": grace_period,
        }
        error = "delete runtimes"
        self.api_call("DELETE", "runtimes", error, params=params)

    def delete_runtime(
        self,
        kind: str,
        label_selector: str = None,
        force: bool = False,
        grace_period: int = config.runtime_resources_deletion_grace_period,
    ):
        params = {
            "label_selector": label_selector,
            "force": force,
            "grace_period": grace_period,
        }
        path = f"runtimes/{kind}"
        error = f"delete runtime {kind}"
        self.api_call("DELETE", path, error, params=params)

    def delete_runtime_object(
        self,
        kind: str,
        object_id: str,
        label_selector: str = None,
        force: bool = False,
        grace_period: int = config.runtime_resources_deletion_grace_period,
    ):
        params = {
            "label_selector": label_selector,
            "force": force,
            "grace_period": grace_period,
        }
        path = f"runtimes/{kind}/{object_id}"
        error = f"delete runtime object {kind} {object_id}"
        self.api_call("DELETE", path, error, params=params)

    def create_schedule(self, project: str, schedule: schemas.ScheduleInput):
        project = project or default_project
        path = f"projects/{project}/schedules"

        error_message = f"Failed creating schedule {project}/{schedule.name}"
        self.api_call("POST", path, error_message, body=dict_to_json(schedule.dict()))

    def update_schedule(
        self, project: str, name: str, schedule: schemas.ScheduleUpdate
    ):
        project = project or default_project
        path = f"projects/{project}/schedules/{name}"

        error_message = f"Failed updating schedule {project}/{name}"
        self.api_call("PUT", path, error_message, body=dict_to_json(schedule.dict()))

    def get_schedule(
        self, project: str, name: str, include_last_run: bool = False
    ) -> schemas.ScheduleOutput:
        project = project or default_project
        path = f"projects/{project}/schedules/{name}"
        error_message = f"Failed getting schedule for {project}/{name}"
        resp = self.api_call(
            "GET", path, error_message, params={"include_last_run": include_last_run}
        )
        return schemas.ScheduleOutput(**resp.json())

    def list_schedules(
        self,
        project: str,
        name: str = None,
        kind: schemas.ScheduleKinds = None,
        include_last_run: bool = False,
    ) -> schemas.SchedulesOutput:
        project = project or default_project
        params = {"kind": kind, "name": name, "include_last_run": include_last_run}
        path = f"projects/{project}/schedules"
        error_message = f"Failed listing schedules for {project} ? {kind} {name}"
        resp = self.api_call("GET", path, error_message, params=params)
        return schemas.SchedulesOutput(**resp.json())

    def delete_schedule(self, project: str, name: str):
        project = project or default_project
        path = f"projects/{project}/schedules/{name}"
        error_message = f"Failed deleting schedule {project}/{name}"
        self.api_call("DELETE", path, error_message)

    def invoke_schedule(self, project: str, name: str):
        project = project or default_project
        path = f"projects/{project}/schedules/{name}/invoke"
        error_message = f"Failed invoking schedule {project}/{name}"
        self.api_call("POST", path, error_message)

    def remote_builder(self, func, with_mlrun, mlrun_version_specifier=None):
        try:
            req = {"function": func.to_dict(), "with_mlrun": bool2str(with_mlrun)}
            if mlrun_version_specifier:
                req["mlrun_version_specifier"] = mlrun_version_specifier
            resp = self.api_call("POST", "build/function", json=req)
        except OSError as err:
            logger.error("error submitting build task: {}".format(err))
            raise OSError("error: cannot submit build, {}".format(err))

        if not resp.ok:
            logger.error("bad resp!!\n{}".format(resp.text))
            raise ValueError("bad function run response")

        return resp.json()

    def get_builder_status(
        self, func, offset=0, logs=True, last_log_timestamp=0, verbose=False
    ):
        try:
            params = {
                "name": func.metadata.name,
                "project": func.metadata.project,
                "tag": func.metadata.tag,
                "logs": bool2str(logs),
                "offset": str(offset),
                "last_log_timestamp": str(last_log_timestamp),
                "verbose": bool2str(verbose),
            }
            resp = self.api_call("GET", "build/status", params=params)
        except OSError as err:
            logger.error("error getting build status: {}".format(err))
            raise OSError("error: cannot get build status, {}".format(err))

        if not resp.ok:
            logger.warning("failed resp, {}".format(resp.text))
            raise RunDBError("bad function build response")

        if resp.headers:
            func.status.state = resp.headers.get("x-mlrun-function-status", "")
            last_log_timestamp = float(
                resp.headers.get("x-mlrun-last-timestamp", "0.0")
            )
            if func.kind in mlrun.runtimes.RuntimeKinds.nuclio_runtimes():
                func.status.address = resp.headers.get("x-mlrun-address", "")
                func.status.nuclio_name = resp.headers.get("x-mlrun-name", "")
            else:
                func.status.build_pod = resp.headers.get("builder_pod", "")
                func.spec.image = resp.headers.get("function_image", "")

        text = ""
        if resp.content:
            text = resp.content.decode()
        return text, last_log_timestamp

    def remote_start(self, func_url) -> schemas.BackgroundTask:
        try:
            req = {"functionUrl": func_url}
            resp = self.api_call(
                "POST",
                "start/function",
                json=req,
                timeout=int(config.submit_timeout) or 60,
            )
        except OSError as err:
            logger.error("error starting function: {}".format(err))
            raise OSError("error: cannot start function, {}".format(err))

        if not resp.ok:
            logger.error("bad resp!!\n{}".format(resp.text))
            raise ValueError("bad function start response")

        return schemas.BackgroundTask(**resp.json())

    def get_background_task(self, project: str, name: str,) -> schemas.BackgroundTask:
        project = project or default_project
        path = f"projects/{project}/background-tasks/{name}"
        error_message = (
            f"Failed getting background task. project={project}, name={name}"
        )
        response = self.api_call("GET", path, error_message)
        return schemas.BackgroundTask(**response.json())

    def remote_status(self, kind, selector):
        try:
            req = {"kind": kind, "selector": selector}
            resp = self.api_call("POST", "status/function", json=req)
        except OSError as err:
            logger.error("error starting function: {}".format(err))
            raise OSError("error: cannot start function, {}".format(err))

        if not resp.ok:
            logger.error("bad resp!!\n{}".format(resp.text))
            raise ValueError("bad function status response")

        return resp.json()["data"]

    def submit_job(
        self, runspec, schedule: Union[str, schemas.ScheduleCronTrigger] = None
    ):
        try:
            req = {"task": runspec.to_dict()}
            if schedule:
                if isinstance(schedule, schemas.ScheduleCronTrigger):
                    schedule = schedule.dict()
                req["schedule"] = schedule
            timeout = (int(config.submit_timeout) or 120) + 20
            resp = self.api_call("POST", "submit_job", json=req, timeout=timeout)
        except OSError as err:
            logger.error("error submitting task: {}".format(err))
            raise OSError("error: cannot submit task, {}".format(err))

        if not resp.ok:
            logger.error("bad resp!!\n{}".format(resp.text))
            raise ValueError("bad function run response, {}".format(resp.text))

        resp = resp.json()
        return resp["data"]

    def submit_pipeline(
        self,
        pipeline,
        arguments=None,
        experiment=None,
        run=None,
        namespace=None,
        artifact_path=None,
        ops=None,
        ttl=None,
    ):

        if isinstance(pipeline, str):
            pipe_file = pipeline
        else:
            pipe_file = tempfile.mktemp(suffix=".yaml")
            conf = new_pipe_meta(artifact_path, ttl, ops)
            kfp.compiler.Compiler().compile(
                pipeline, pipe_file, type_check=False, pipeline_conf=conf
            )

        if pipe_file.endswith(".yaml"):
            headers = {"content-type": "application/yaml"}
        elif pipe_file.endswith(".zip"):
            headers = {"content-type": "application/zip"}
        else:
            raise ValueError("pipeline file must be .yaml or .zip")
        if arguments:
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be dict type")
            headers["pipeline-arguments"] = str(arguments)

        if not path.isfile(pipe_file):
            raise OSError("file {} doesnt exist".format(pipe_file))
        with open(pipe_file, "rb") as fp:
            data = fp.read()
        if not isinstance(pipeline, str):
            remove(pipe_file)

        try:
            params = {"namespace": namespace, "experiment": experiment, "run": run}
            resp = self.api_call(
                "POST",
                "submit_pipeline",
                params=params,
                timeout=20,
                body=data,
                headers=headers,
            )
        except OSError as err:
            logger.error("error cannot submit pipeline: {}".format(err))
            raise OSError("error: cannot cannot submit pipeline, {}".format(err))

        if not resp.ok:
            logger.error("bad resp!!\n{}".format(resp.text))
            raise ValueError("bad submit pipeline response, {}".format(resp.text))

        resp = resp.json()
        logger.info("submitted pipeline {} id={}".format(resp["name"], resp["id"]))
        return resp["id"]

    def list_pipelines(
        self,
        project: str,
        namespace: str = None,
        sort_by: str = "",
        page_token: str = "",
        filter_: str = "",
        format_: Union[
            str, mlrun.api.schemas.Format
        ] = mlrun.api.schemas.Format.metadata_only,
        page_size: int = None,
    ) -> mlrun.api.schemas.PipelinesOutput:
        if project != "*" and (page_token or page_size or sort_by or filter_):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Filtering by project can not be used together with pagination, sorting, or custom filter"
            )
        if isinstance(format_, mlrun.api.schemas.Format):
            format_ = format_.value
        params = {
            "namespace": namespace,
            "sort_by": sort_by,
            "page_token": page_token,
            "filter": filter_,
            "format": format_,
            "page_size": page_size,
        }

        error_message = f"Failed listing pipelines, query: {params}"
        response = self.api_call(
            "GET", f"projects/{project}/pipelines", error_message, params=params
        )
        return mlrun.api.schemas.PipelinesOutput(**response.json())

    def get_pipeline(self, run_id: str, namespace: str = None, timeout: int = 10):

        try:
            query = ""
            if namespace:
                query = "namespace={}".format(namespace)
            resp = self.api_call(
                "GET", "pipelines/{}?{}".format(run_id, query), timeout=timeout
            )
        except OSError as err:
            logger.error("error cannot get pipeline: {}".format(err))
            raise OSError("error: cannot get pipeline, {}".format(err))

        if not resp.ok:
            logger.error("bad resp!!\n{}".format(resp.text))
            raise ValueError("bad get pipeline response, {}".format(resp.text))

        return resp.json()

    @staticmethod
    def _resolve_reference(tag, uid):
        if uid and tag:
            raise MLRunInvalidArgumentError("both uid and tag were provided")
        return uid or tag or "latest"

    def create_feature_set(
        self, feature_set: Union[dict, schemas.FeatureSet], project="", versioned=True
    ) -> dict:
        if isinstance(feature_set, schemas.FeatureSet):
            feature_set = feature_set.dict()

        project = (
            project or feature_set["metadata"].get("project", None) or default_project
        )
        path = f"projects/{project}/feature-sets"
        params = {"versioned": versioned}

        name = feature_set["metadata"]["name"]
        error_message = f"Failed creating feature-set {project}/{name}"
        resp = self.api_call(
            "POST", path, error_message, params=params, body=dict_to_json(feature_set),
        )
        return resp.json()

    def get_feature_set(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ) -> FeatureSet:
        project = project or default_project
        reference = self._resolve_reference(tag, uid)
        path = f"projects/{project}/feature-sets/{name}/references/{reference}"
        error_message = f"Failed retrieving feature-set {project}/{name}"
        resp = self.api_call("GET", path, error_message)
        return FeatureSet.from_dict(resp.json())

    def list_features(
        self,
        project: str,
        name: str = None,
        tag: str = None,
        entities: List[str] = None,
        labels: List[str] = None,
    ) -> List[dict]:
        project = project or default_project
        params = {
            "name": name,
            "tag": tag,
            "entity": entities or [],
            "label": labels or [],
        }

        path = f"projects/{project}/features"

        error_message = f"Failed listing features, project: {project}, query: {params}"
        resp = self.api_call("GET", path, error_message, params=params)
        return resp.json()["features"]

    def list_entities(
        self, project: str, name: str = None, tag: str = None, labels: List[str] = None,
    ) -> List[dict]:
        project = project or default_project
        params = {
            "name": name,
            "tag": tag,
            "label": labels or [],
        }

        path = f"projects/{project}/entities"

        error_message = f"Failed listing entities, project: {project}, query: {params}"
        resp = self.api_call("GET", path, error_message, params=params)
        return resp.json()["entities"]

    def list_feature_sets(
        self,
        project: str = "",
        name: str = None,
        tag: str = None,
        state: str = None,
        entities: List[str] = None,
        features: List[str] = None,
        labels: List[str] = None,
    ) -> List[FeatureSet]:
        project = project or default_project
        params = {
            "name": name,
            "state": state,
            "tag": tag,
            "entity": entities or [],
            "feature": features or [],
            "label": labels or [],
        }

        path = f"projects/{project}/feature-sets"

        error_message = (
            f"Failed listing feature-sets, project: {project}, query: {params}"
        )
        resp = self.api_call("GET", path, error_message, params=params)
        feature_sets = resp.json()["feature_sets"]
        if feature_sets:
            return [FeatureSet.from_dict(obj) for obj in feature_sets]

    def store_feature_set(
        self,
        feature_set: Union[dict, schemas.FeatureSet],
        name=None,
        project="",
        tag=None,
        uid=None,
        versioned=True,
    ) -> dict:
        reference = self._resolve_reference(tag, uid)
        params = {"versioned": versioned}

        if isinstance(feature_set, schemas.FeatureSet):
            feature_set = feature_set.dict()

        name = name or feature_set["metadata"]["name"]
        project = project or feature_set["metadata"].get("project") or default_project
        path = f"projects/{project}/feature-sets/{name}/references/{reference}"
        error_message = f"Failed storing feature-set {project}/{name}"
        resp = self.api_call(
            "PUT", path, error_message, params=params, body=dict_to_json(feature_set)
        )
        return resp.json()

    def patch_feature_set(
        self,
        name,
        feature_set_update: dict,
        project="",
        tag=None,
        uid=None,
        patch_mode: Union[str, schemas.PatchMode] = schemas.PatchMode.replace,
    ):
        project = project or default_project
        reference = self._resolve_reference(tag, uid)
        if isinstance(patch_mode, schemas.PatchMode):
            patch_mode = patch_mode.value
        headers = {schemas.HeaderNames.patch_mode: patch_mode}
        path = f"projects/{project}/feature-sets/{name}/references/{reference}"
        error_message = f"Failed updating feature-set {project}/{name}"
        self.api_call(
            "PATCH",
            path,
            error_message,
            body=dict_to_json(feature_set_update),
            headers=headers,
        )

    def delete_feature_set(self, name, project=""):
        project = project or default_project
        path = f"projects/{project}/feature-sets/{name}"
        error_message = f"Failed deleting feature-set {name}"
        self.api_call("DELETE", path, error_message)

    def create_feature_vector(
        self,
        feature_vector: Union[dict, schemas.FeatureVector],
        project="",
        versioned=True,
    ) -> dict:
        if isinstance(feature_vector, schemas.FeatureVector):
            feature_vector = feature_vector.dict()

        project = (
            project
            or feature_vector["metadata"].get("project", None)
            or default_project
        )
        path = f"projects/{project}/feature-vectors"
        params = {"versioned": versioned}

        name = feature_vector["metadata"]["name"]
        error_message = f"Failed creating feature-vector {project}/{name}"
        resp = self.api_call(
            "POST",
            path,
            error_message,
            params=params,
            body=dict_to_json(feature_vector),
        )
        return resp.json()

    def get_feature_vector(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ) -> FeatureVector:
        project = project or default_project
        reference = self._resolve_reference(tag, uid)
        path = f"projects/{project}/feature-vectors/{name}/references/{reference}"
        error_message = f"Failed retrieving feature-vector {project}/{name}"
        resp = self.api_call("GET", path, error_message)
        return FeatureVector.from_dict(resp.json())

    def list_feature_vectors(
        self,
        project: str = "",
        name: str = None,
        tag: str = None,
        state: str = None,
        labels: List[str] = None,
    ) -> List[FeatureVector]:
        project = project or default_project
        params = {
            "name": name,
            "state": state,
            "tag": tag,
            "label": labels or [],
        }

        path = f"projects/{project}/feature-vectors"

        error_message = (
            f"Failed listing feature-vectors, project: {project}, query: {params}"
        )
        resp = self.api_call("GET", path, error_message, params=params)
        feature_vectors = resp.json()["feature_vectors"]
        if feature_vectors:
            return [FeatureVector.from_dict(obj) for obj in feature_vectors]

    def store_feature_vector(
        self,
        feature_vector: Union[dict, schemas.FeatureVector],
        name=None,
        project="",
        tag=None,
        uid=None,
        versioned=True,
    ) -> dict:
        reference = self._resolve_reference(tag, uid)
        params = {"versioned": versioned}

        if isinstance(feature_vector, schemas.FeatureVector):
            feature_vector = feature_vector.dict()

        name = name or feature_vector["metadata"]["name"]
        project = (
            project or feature_vector["metadata"].get("project") or default_project
        )
        path = f"projects/{project}/feature-vectors/{name}/references/{reference}"
        error_message = f"Failed storing feature-vector {project}/{name}"
        resp = self.api_call(
            "PUT", path, error_message, params=params, body=dict_to_json(feature_vector)
        )
        return resp.json()

    def patch_feature_vector(
        self,
        name,
        feature_vector_update: dict,
        project="",
        tag=None,
        uid=None,
        patch_mode: Union[str, schemas.PatchMode] = schemas.PatchMode.replace,
    ):
        reference = self._resolve_reference(tag, uid)
        project = project or default_project
        if isinstance(patch_mode, schemas.PatchMode):
            patch_mode = patch_mode.value
        headers = {schemas.HeaderNames.patch_mode: patch_mode}
        path = f"projects/{project}/feature-vectors/{name}/references/{reference}"
        error_message = f"Failed updating feature-vector {project}/{name}"
        self.api_call(
            "PATCH",
            path,
            error_message,
            body=dict_to_json(feature_vector_update),
            headers=headers,
        )

    def delete_feature_vector(self, name, project=""):
        project = project or default_project
        path = f"projects/{project}/feature-vectors/{name}"
        error_message = f"Failed deleting feature-vector {name}"
        self.api_call("DELETE", path, error_message)

    def list_projects(
        self,
        owner: str = None,
        format_: Union[str, mlrun.api.schemas.Format] = mlrun.api.schemas.Format.full,
        labels: List[str] = None,
        state: Union[str, mlrun.api.schemas.ProjectState] = None,
    ) -> List[Union[mlrun.projects.MlrunProject, str]]:
        if isinstance(state, mlrun.api.schemas.ProjectState):
            state = state.value
        if isinstance(format_, mlrun.api.schemas.Format):
            format_ = format_.value
        params = {
            "owner": owner,
            "state": state,
            "format": format_,
            "label": labels or [],
        }

        error_message = f"Failed listing projects, query: {params}"
        response = self.api_call("GET", "projects", error_message, params=params)
        if format_ == mlrun.api.schemas.Format.name_only:
            return response.json()["projects"]
        elif format_ == mlrun.api.schemas.Format.full:
            return [
                mlrun.projects.MlrunProject.from_dict(project_dict)
                for project_dict in response.json()["projects"]
            ]
        else:
            raise NotImplementedError(
                f"Provided format is not supported. format={format_}"
            )

    def get_project(self, name: str) -> mlrun.projects.MlrunProject:
        if not name:
            raise MLRunInvalidArgumentError("Name must be provided")

        path = f"projects/{name}"
        error_message = f"Failed retrieving project {name}"
        response = self.api_call("GET", path, error_message)
        return mlrun.projects.MlrunProject.from_dict(response.json())

    def delete_project(
        self,
        name: str,
        deletion_strategy: Union[
            str, mlrun.api.schemas.DeletionStrategy
        ] = mlrun.api.schemas.DeletionStrategy.default(),
    ):
        path = f"projects/{name}"
        if isinstance(deletion_strategy, schemas.DeletionStrategy):
            deletion_strategy = deletion_strategy.value
        headers = {schemas.HeaderNames.deletion_strategy: deletion_strategy}
        error_message = f"Failed deleting project {name}"
        self.api_call("DELETE", path, error_message, headers=headers)

    def store_project(
        self,
        name: str,
        project: Union[dict, mlrun.projects.MlrunProject, mlrun.api.schemas.Project],
    ) -> mlrun.projects.MlrunProject:
        path = f"projects/{name}"
        error_message = f"Failed storing project {name}"
        if isinstance(project, mlrun.api.schemas.Project):
            project = project.dict()
        elif isinstance(project, mlrun.projects.MlrunProject):
            project = project.to_dict()
        response = self.api_call(
            "PUT", path, error_message, body=dict_to_json(project),
        )
        return mlrun.projects.MlrunProject.from_dict(response.json())

    def patch_project(
        self,
        name: str,
        project: dict,
        patch_mode: Union[str, schemas.PatchMode] = schemas.PatchMode.replace,
    ) -> mlrun.projects.MlrunProject:
        path = f"projects/{name}"
        if isinstance(patch_mode, schemas.PatchMode):
            patch_mode = patch_mode.value
        headers = {schemas.HeaderNames.patch_mode: patch_mode}
        error_message = f"Failed patching project {name}"
        response = self.api_call(
            "PATCH", path, error_message, body=dict_to_json(project), headers=headers
        )
        return mlrun.projects.MlrunProject.from_dict(response.json())

    def create_project(
        self,
        project: Union[dict, mlrun.projects.MlrunProject, mlrun.api.schemas.Project],
    ) -> mlrun.projects.MlrunProject:
        if isinstance(project, mlrun.api.schemas.Project):
            project = project.dict()
        elif isinstance(project, mlrun.projects.MlrunProject):
            project = project.to_dict()
        error_message = f"Failed creating project {project['metadata']['name']}"
        response = self.api_call(
            "POST", "projects", error_message, body=dict_to_json(project),
        )
        return mlrun.projects.MlrunProject.from_dict(response.json())

    def create_project_secrets(
        self,
        project: str,
        provider: Union[
            str, schemas.SecretProviderName
        ] = schemas.SecretProviderName.vault,
        secrets: dict = None,
    ):
        if isinstance(provider, schemas.SecretProviderName):
            provider = provider.value
        path = f"projects/{project}/secrets"
        secrets_input = schemas.SecretsData(secrets=secrets, provider=provider)
        body = secrets_input.dict()
        error_message = f"Failed creating secret provider {project}/{provider}"
        self.api_call(
            "POST", path, error_message, body=dict_to_json(body),
        )

    def get_project_secrets(
        self,
        project: str,
        token: str,
        provider: Union[
            str, schemas.SecretProviderName
        ] = schemas.SecretProviderName.vault,
        secrets: List[str] = None,
    ) -> schemas.SecretsData:
        if isinstance(provider, schemas.SecretProviderName):
            provider = provider.value
        path = f"projects/{project}/secrets"
        params = {"provider": provider, "secret": secrets}
        headers = {schemas.HeaderNames.secret_store_token: token}
        error_message = f"Failed retrieving secrets {project}/{provider}"
        result = self.api_call(
            "GET", path, error_message, params=params, headers=headers,
        )
        return schemas.SecretsData(**result.json())

    def create_user_secrets(
        self,
        user: str,
        provider: Union[
            str, schemas.SecretProviderName
        ] = schemas.SecretProviderName.vault,
        secrets: dict = None,
    ):
        if isinstance(provider, schemas.SecretProviderName):
            provider = provider.value
        path = "user-secrets"
        secrets_creation_request = schemas.UserSecretCreationRequest(
            user=user, provider=provider, secrets=secrets,
        )
        body = secrets_creation_request.dict()
        error_message = f"Failed creating user secrets - {user}"
        self.api_call(
            "POST", path, error_message, body=dict_to_json(body),
        )

    @staticmethod
    def _validate_version_compatibility(server_version, client_version):
        try:
            parsed_server_version = semver.VersionInfo.parse(server_version)
            parsed_client_version = semver.VersionInfo.parse(client_version)
        except ValueError:
            # This will mostly happen in dev scenarios when the version is unstable and such - therefore we're ignoring
            logger.warning(
                "Unable to parse server or client version. Assuming compatible",
                server_version=server_version,
                client_version=client_version,
            )
            return
        if (
            parsed_server_version.major != parsed_client_version.major
            or parsed_server_version.minor != parsed_client_version.minor
        ):
            message = "Server and client versions are incompatible"
            logger.warning(
                message,
                parsed_server_version=parsed_server_version,
                parsed_client_version=parsed_client_version,
            )
            raise mlrun.errors.MLRunIncompatibleVersionError(message)


def _as_json(obj):
    fn = getattr(obj, "to_json", None)
    if fn:
        return fn()
    return dict_to_json(obj)
