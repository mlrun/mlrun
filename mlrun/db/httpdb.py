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

import json
import tempfile
import time
from os import path, remove, environ
from typing import List, Dict, Union

import kfp
import requests
import mlrun
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from mlrun.api import schemas
from .base import RunDBError, RunDBInterface
from ..config import config
from ..lists import RunList, ArtifactList
from ..utils import dict_to_json, logger, new_pipe_meta

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
            if self.server_version != config.version:
                logger.warning(
                    "warning!, server ({}) and client ({}) ver dont match".format(
                        self.server_version, config.version
                    )
                )
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
            config.ui_url = config.ui_url or server_cfg.get("ui_url")
            config.artifact_path = config.artifact_path or server_cfg.get(
                "artifact_path"
            )
            if (
                "docker_registry" in server_cfg
                and "DEFAULT_DOCKER_REGISTRY" not in environ
            ):
                environ["DEFAULT_DOCKER_REGISTRY"] = server_cfg["docker_registry"]

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

    def store_function(self, function, name, project="", tag=None, versioned=False):
        params = {"tag": tag, "versioned": versioned}
        project = project or default_project
        path = self._path_of("func", project, name)

        error = f"store function {project}/{name}"
        resp = self.api_call(
            "POST", path, error, params=params, body=json.dumps(function)
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
        self.api_call("POST", path, error_message, body=json.dumps(schedule.dict()))

    def get_schedule(self, project: str, name: str) -> schemas.ScheduleOutput:
        project = project or default_project
        path = f"projects/{project}/schedules/{name}"
        error_message = f"Failed getting schedule for {project}/{name}"
        resp = self.api_call("GET", path, error_message)
        return schemas.ScheduleOutput(**resp.json())

    def list_schedules(
        self, project: str, name: str = None, kind: schemas.ScheduleKinds = None
    ) -> schemas.SchedulesOutput:
        project = project or default_project
        params = {"kind": kind, "name": name}
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

    def delete_project(self, name: str):
        path = f"projects/{name}"
        error_message = f"Failed deleting project {name}"
        self.api_call("DELETE", path, error_message)

    def remote_builder(self, func, with_mlrun):
        try:
            req = {"function": func.to_dict(), "with_mlrun": bool2str(with_mlrun)}
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

    def remote_start(self, func_url):
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

        return resp.json()["data"]

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


def _as_json(obj):
    fn = getattr(obj, "to_json", None)
    if fn:
        return fn()
    return dict_to_json(obj)
