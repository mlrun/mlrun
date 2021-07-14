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

import http
import os
import tempfile
import time
from datetime import datetime
from os import path, remove
from typing import Dict, List, Optional, Union

import kfp
import requests
import semver
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import mlrun
import mlrun.projects
from mlrun.api import schemas
from mlrun.errors import MLRunInvalidArgumentError

from ..api.schemas import ModelEndpoint
from ..config import config
from ..feature_store import FeatureSet, FeatureVector
from ..lists import ArtifactList, RunList
from ..utils import datetime_to_iso, dict_to_json, logger, new_pipe_meta
from .base import RunDBError, RunDBInterface

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
    """ Interface for accessing and manipulating the :py:mod:`mlrun` persistent store, maintaining the full state
    and catalog of objects that MLRun uses. The :py:class:`HTTPRunDB` class serves as a client-side proxy to the MLRun
    API service which maintains the actual data-store, accesses the server through REST APIs.

    The class provides functions for accessing and modifying the various objects that are used by MLRun in its
    operation. The functions provided follow some standard guidelines, which are:

    - Every object in MLRun exists in the context of a project (except projects themselves). When referencing an object
      through any API, a project name must be provided. The default for most APIs is for an empty project name, which
      will be replaced by the name of the default project (usually ``default``). Therefore, if performing an API to
      list functions, for example, and not providing a project name - the result will not be functions from all
      projects but rather from the ``default`` project.
    - Many objects can be assigned labels, and listed/queried by label. The label parameter for query APIs allows for
      listing objects that:

      - Have a specific label, by asking for ``label="<label_name>"``. In this case the actual value of the label
        doesn't matter and every object with that label will be returned
      - Have a label with a specific value. This is done by specifying ``label="<label_name>=<label_value>"``. In this
        case only objects whose label matches the value will be returned

    - Most objects have a ``create`` method as well as a ``store`` method. Create can only be called when such an
      does not exist yet, while store allows for either creating a new object or overwriting an existing object.
    - Some objects have a ``versioned`` option, in which case overwriting the same object with a different version of
      it does not delete the previous version, but rather creates a new version of the object and keeps both versions.
      Versioned objects usually have a ``uid`` property which is based on their content and allows to reference a
      specific version of an object (other than tagging objects, which also allows for easy referencing).
    - Many objects have both a ``store`` function and a ``patch`` function. These are used in the same way as the
      corresponding REST verbs - a ``store`` is passed a full object and will basically perform a PUT operation,
      replacing the full object (if it exists) while ``patch`` receives just a dictionary containing the differences to
      be applied to the object, and will merge those changes to the existing object. The ``patch``
      operation also has a strategy assigned to it which determines how the merge logic should behave.
      The strategy can be either ``replace`` or ``additive``. For further details on those strategies, refer
      to https://pypi.org/project/mergedeep/
    """

    kind = "http"

    def __init__(self, base_url, user="", password="", token=""):
        self.base_url = base_url
        self.user = user
        self.password = password
        self.token = token
        self.server_version = ""
        self.session = None
        self._wait_for_project_terminal_state_retry_interval = 3
        self._wait_for_project_deletion_interval = 3

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
        timeout=45,
    ):
        """ Perform a direct REST API call on the :py:mod:`mlrun` API server.

            Caution:
                For advanced usage - prefer using the various APIs exposed through this class, rather than
                directly invoking REST calls.

            :param method: REST method (POST, GET, PUT...)
            :param path: Path to endpoint executed, for example ``"projects"``
            :param error: Error to return if API invocation fails
            :param body: Payload to be passed in the call. If using JSON objects, prefer using the ``json`` param
            :param json: JSON payload to be passed in the call
            :param headers: REST headers, passed as a dictionary: ``{"<header-name>": "<header-value>"}``
            :param timeout: API call timeout

            :return: Python HTTP response object
        """
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
            error = error or f"{method} {url}, error: {err}"
            raise RunDBError(error) from err

        if not resp.ok:
            if resp.content:
                try:
                    data = resp.json()
                    reason = data.get("detail", {}).get("reason", "")
                except Exception:
                    reason = ""
            if reason:
                error = error or f"{method} {url}, error: {reason}"
                mlrun.errors.raise_for_status(resp, error)

            mlrun.errors.raise_for_status(resp)

        return resp

    def _path_of(self, prefix, project, uid):
        project = project or config.default_project
        return f"{prefix}/{project}/{uid}"

    def connect(self, secrets=None):
        """ Connect to the MLRun API server. Must be called prior to executing any other method.
        The code utilizes the URL for the API server from the configuration - ``mlconf.dbpath``.

        For example::

            mlconf.dbpath = mlconf.dbpath or 'http://mlrun-api:8080'
            db = get_run_db().connect()
        """
        # hack to allow unit tests to instantiate HTTPRunDB without a real server behind
        if "mock-server" in self.base_url:
            return
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
                    f"warning!, server ({server_cfg['namespace']}) and client ({config.namespace})"
                    " namespace don't match"
                )

            # get defaults from remote server
            config.remote_host = config.remote_host or server_cfg.get("remote_host")
            config.mpijob_crd_version = config.mpijob_crd_version or server_cfg.get(
                "mpijob_crd_version"
            )
            config.ui.url = config.resolve_ui_url() or server_cfg.get("ui_url")
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
            config.httpdb.api_url = config.httpdb.api_url or server_cfg.get("api_url")
            # These have a default value, therefore local config will always have a value, prioritize the
            # API value first
            config.ui.projects_prefix = (
                server_cfg.get("ui_projects_prefix") or config.ui.projects_prefix
            )
            config.kfp_image = server_cfg.get("kfp_image") or config.kfp_image
            config.dask_kfp_image = (
                server_cfg.get("dask_kfp_image") or config.dask_kfp_image
            )
            config.scrape_metrics = (
                server_cfg.get("scrape_metrics")
                if server_cfg.get("scrape_metrics") is not None
                else config.scrape_metrics
            )
            config.hub_url = server_cfg.get("hub_url") or config.hub_url
        except Exception:
            pass
        return self

    def store_log(self, uid, project="", body=None, append=False):
        """ Save a log persistently.

        :param uid: Log unique ID
        :param project: Project name for which this log belongs
        :param body: The actual log to store
        :param append: Whether to append the log provided in ``body`` to an existing log with the same ``uid`` or to
            create a new log. If set to ``False``, an existing log with same ``uid`` will be overwritten
        """

        if not body:
            return

        path = self._path_of("log", project, uid)
        params = {"append": bool2str(append)}
        error = f"store log {project}/{uid}"
        self.api_call("POST", path, error, params, body)

    def get_log(self, uid, project="", offset=0, size=-1):
        """ Retrieve a log.

        :param uid: Log unique ID
        :param project: Project name for which the log belongs
        :param offset: Retrieve partial log, get up to ``size`` bytes starting at offset ``offset``
            from beginning of log
        :param size: See ``offset``. If set to ``-1`` (the default) will retrieve all data to end of log.
        :returns: The following objects:

            - state - The state of the runtime object which generates this log, if it exists. In case no known state
              exists, this will be ``unknown``.
            - content - The actual log content.
        """

        params = {"offset": offset, "size": size}
        path = self._path_of("log", project, uid)
        error = f"get log {project}/{uid}"
        resp = self.api_call("GET", path, error, params=params)
        if resp.headers:
            state = resp.headers.get("x-mlrun-run-state", "")
            return state.lower(), resp.content

        return "unknown", resp.content

    def watch_log(self, uid, project="", watch=True, offset=0):
        """ Retrieve logs of a running process, and watch the progress of the execution until it completes. This
        method will print out the logs and continue to periodically poll for, and print, new logs as long as the
        state of the runtime which generates this log is either ``pending`` or ``running``.

        :param uid: The uid of the log object to watch.
        :param project: Project that the log belongs to.
        :param watch: If set to ``True`` will continue tracking the log as described above. Otherwise this function
            is practically equivalent to the :py:func:`~get_log` function.
        :param offset: Minimal offset in the log to watch.
        :returns: The final state of the log being watched.
        """

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
        """ Store run details in the DB. This method is usually called from within other :py:mod:`mlrun` flows
        and not called directly by the user."""

        path = self._path_of("run", project, uid)
        params = {"iter": iter}
        error = f"store run {project}/{uid}"
        body = _as_json(struct)
        self.api_call("POST", path, error, params=params, body=body)

    def update_run(self, updates: dict, uid, project="", iter=0):
        """ Update the details of a stored run in the DB."""

        path = self._path_of("run", project, uid)
        params = {"iter": iter}
        error = f"update run {project}/{uid}"
        body = _as_json(updates)
        self.api_call("PATCH", path, error, params=params, body=body)

    def abort_run(self, uid, project="", iter=0):
        """
        Abort a running run - will remove the run's runtime resources and mark its state as aborted
        """
        self.update_run(
            {"status.state": mlrun.runtimes.constants.RunStates.aborted},
            uid,
            project,
            iter,
        )

    def read_run(self, uid, project="", iter=0):
        """ Read the details of a stored run from the DB.

        :param uid: The run's unique ID.
        :param project: Project name.
        :param iter: Iteration within a specific execution.
        """

        path = self._path_of("run", project, uid)
        params = {"iter": iter}
        error = f"get run {project}/{uid}"
        resp = self.api_call("GET", path, error, params=params)
        return resp.json()["data"]

    def del_run(self, uid, project="", iter=0):
        """ Delete details of a specific run from DB.

        :param uid: Unique ID for the specific run to delete.
        :param project: Project that the run belongs to.
        :param iter: Iteration within a specific task.
        """

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
    ) -> RunList:
        """ Retrieve a list of runs, filtered by various options.
        Example::

            runs = db.list_runs(name='download', project='iris', labels='owner=admin')
            # If running in Jupyter, can use the .show() function to display the results
            db.list_runs(name='', project=project_name).show()


        :param name: Name of the run to retrieve.
        :param uid: Unique ID of the run.
        :param project: Project that the runs belongs to.
        :param labels: List runs that have a specific label assigned. Currently only a single label filter can be
            applied, otherwise result will be empty.
        :param state: List only runs whose state is specified.
        :param sort: Whether to sort the result according to their start time. Otherwise results will be
            returned by their internal order in the DB (order will not be guaranteed).
        :param last: Deprecated - currently not used.
        :param iter: If ``True`` return runs from all iterations. Otherwise, return only runs whose ``iter`` is 0.
        :param start_time_from: Filter by run start time in ``[start_time_from, start_time_to]``.
        :param start_time_to: Filter by run start time in ``[start_time_from, start_time_to]``.
        :param last_update_time_from: Filter by run last update time in ``(last_update_time_from,
            last_update_time_to)``.
        :param last_update_time_to: Filter by run last update time in ``(last_update_time_from, last_update_time_to)``.
        """

        project = project or config.default_project
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
        """ Delete a group of runs identified by the parameters of the function.

        Example::

            db.del_runs(state='completed')

        :param name: Name of the task which the runs belong to.
        :param project: Project to which the runs belong.
        :param labels: Filter runs that are labeled using these specific label values.
        :param state: Filter only runs which are in this state.
        :param days_ago: Filter runs whose start time is newer than this parameter.
        """

        project = project or config.default_project
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
        """ Store an artifact in the DB.

        :param key: Identifying key of the artifact.
        :param artifact: The actual artifact to store.
        :param uid: A unique ID for this specific version of the artifact.
        :param iter: The task iteration which generated this artifact. If ``iter`` is not ``None`` the iteration will
            be added to the key provided to generate a unique key for the artifact of the specific iteration.
        :param tag: Tag of the artifact.
        :param project: Project that the artifact belongs to.
        """

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
        """ Read an artifact, identified by its key, tag and iteration."""

        project = project or config.default_project
        tag = tag or "latest"
        path = f"projects/{project}/artifact/{key}?tag={tag}"
        error = f"read artifact {project}/{key}"
        params = {"iter": str(iter)} if iter else {}
        resp = self.api_call("GET", path, error, params=params)
        return resp.json()["data"]

    def del_artifact(self, key, tag=None, project=""):
        """ Delete an artifact."""

        path = self._path_of("artifact", project, key)  # TODO: uid?
        params = {
            "key": key,
            "tag": tag,
        }
        error = f"del artifact {project}/{key}"
        self.api_call("DELETE", path, error, params=params)

    def list_artifacts(
        self,
        name=None,
        project=None,
        tag=None,
        labels=None,
        since=None,
        until=None,
        iter: int = None,
        best_iteration: bool = False,
    ) -> ArtifactList:
        """ List artifacts filtered by various parameters.

        Examples::

            # Show latest version of all artifacts in project
            latest_artifacts = db.list_artifacts('', tag='latest', project='iris')
            # check different artifact versions for a specific artifact
            result_versions = db.list_artifacts('results', tag='*', project='iris')

        :param name: Name of artifacts to retrieve. Name is used as a like query, and is not case-sensitive. This means
            that querying for ``name`` may return artifacts named ``my_Name_1`` or ``surname``.
        :param project: Project name.
        :param tag: Return artifacts assigned this tag.
        :param labels: Return artifacts that have these labels.
        :param since: Not in use in :py:class:`HTTPRunDB`.
        :param until: Not in use in :py:class:`HTTPRunDB`.
        :param iter: Return artifacts from a specific iteration (where ``iter=0`` means the root iteration). If
            ``None`` (default) return artifacts from all iterations.
        :param best_iteration: Returns the artifact which belongs to the best iteration of a given run, in the case of
            artifacts generated from a hyper-param run. If only a single iteration exists, will return the artifact
            from that iteration. If using ``best_iter``, the ``iter`` parameter must not be used.
        """

        project = project or config.default_project
        params = {
            "name": name,
            "project": project,
            "tag": tag,
            "label": labels or [],
            "iter": iter,
            "best-iteration": best_iteration,
        }
        error = "list artifacts"
        resp = self.api_call("GET", "artifacts", error, params=params)
        values = ArtifactList(resp.json()["artifacts"])
        values.tag = tag
        return values

    def del_artifacts(self, name=None, project=None, tag=None, labels=None, days_ago=0):
        """ Delete artifacts referenced by the parameters.

        :param name: Name of artifacts to delete. Note that this is a like query, and is case-insensitive. See
            :py:func:`~list_artifacts` for more details.
        :param project: Project that artifacts belong to.
        :param tag: Choose artifacts who are assigned this tag.
        :param labels: Choose artifacts which are labeled.
        :param days_ago: This parameter is deprecated and not used.
        """
        project = project or config.default_project
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
        """ Return a list of all the tags assigned to artifacts in the scope of the given project."""

        project = project or config.default_project
        error_message = f"Failed listing artifact tags. project={project}"
        response = self.api_call(
            "GET", f"/projects/{project}/artifact-tags", error_message
        )
        return response.json()

    def store_function(self, function, name, project="", tag=None, versioned=False):
        """ Store a function object. Function is identified by its name and tag, and can be versioned."""

        params = {"tag": tag, "versioned": versioned}
        project = project or config.default_project
        path = self._path_of("func", project, name)

        error = f"store function {project}/{name}"
        resp = self.api_call(
            "POST", path, error, params=params, body=dict_to_json(function)
        )

        # hash key optional to be backwards compatible to API v<0.4.10 in which it wasn't in the response
        return resp.json().get("hash_key")

    def get_function(self, name, project="", tag=None, hash_key=""):
        """ Retrieve details of a specific function, identified by its name and potentially a tag or function hash."""

        params = {"tag": tag, "hash_key": hash_key}
        project = project or config.default_project
        path = self._path_of("func", project, name)
        error = f"get function {project}/{name}"
        resp = self.api_call("GET", path, error, params=params)
        return resp.json()["func"]

    def delete_function(self, name: str, project: str = ""):
        """ Delete a function belonging to a specific project."""

        project = project or config.default_project
        path = f"projects/{project}/functions/{name}"
        error_message = f"Failed deleting function {project}/{name}"
        self.api_call("DELETE", path, error_message)

    def list_functions(self, name=None, project=None, tag=None, labels=None):
        """ Retrieve a list of functions, filtered by specific criteria.

        :param name: Return only functions with a specific name.
        :param project: Return functions belonging to this project. If not specified, the default project is used.
        :param tag: Return function versions with specific tags.
        :param labels: Return functions that have specific labels assigned to them.
        :returns: List of function objects (as dictionary).
        """

        params = {
            "project": project or config.default_project,
            "name": name,
            "tag": tag,
            "label": labels or [],
        }
        error = "list functions"
        resp = self.api_call("GET", "funcs", error, params=params)
        return resp.json()["funcs"]

    def list_runtimes(self, label_selector: str = None) -> List:
        """ List current runtime resources, which are usually (but not limited to) Kubernetes pods or CRDs.
        Function applies for runs of type ``['dask', 'job', 'spark', 'mpijob']``, and will return per runtime
        kind a list of the resources (which may have already completed their execution).

        :param label_selector: A label filter that will be passed to Kubernetes for filtering the results according
            to their labels.
        """
        params = {"label_selector": label_selector}
        error = "list runtimes"
        resp = self.api_call("GET", "runtimes", error, params=params)
        return resp.json()

    def get_runtime(self, kind: str, label_selector: str = None) -> Dict:
        """ Return a list of runtime resources of a given kind, and potentially matching a specified label.
        There may be multiple runtime resources returned from this function. This function is similar to the
        :py:func:`~list_runtimes` function, only it focuses on a specific ``kind``, rather than list all runtimes
        of all kinds which generate runtime pods.

        Example::

            project_pods = db.get_runtime('job', label_selector='mlrun/project=iris')['resources']['pod_resources']
            for pod in project_pods:
                print(pod["name"])

        :param kind: The kind of runtime to query. May be one of ``['dask', 'job', 'spark', 'mpijob']``
        :param label_selector: A label filter that will be passed to Kubernetes for filtering the results according
            to their labels.

        """

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
        """ Delete all runtimes which are matching the specific label selector provided. This will delete runtimes
        of all applicable kinds. For deleting runtimes of a specific kind, use the :py:func:`~delete_runtime` function.

        :param label_selector: Delete runtimes with this label assigned.
        :param force: Force deletion. This parameter is passed to the Kubernetes deletion API for force-delete of pods.
        :param grace_period: Grace period for the deleted resources before they are evacuated. This is passed to the
            Kubernetes deletion API.
        """

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
        """ Delete runtimes of a specific kind. See :py:func:`~delete_runtimes` for more details."""

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
        """ Delete a specific runtime object identified by its ID. The object ID can be retrieved from the
        runtime query functions, and used to target a specific runtime to delete.
        The parameters are the same as those used in :py:func:`~delete_runtimes`.
        """

        params = {
            "label_selector": label_selector,
            "force": force,
            "grace_period": grace_period,
        }
        path = f"runtimes/{kind}/{object_id}"
        error = f"delete runtime object {kind} {object_id}"
        self.api_call("DELETE", path, error, params=params)

    def create_schedule(self, project: str, schedule: schemas.ScheduleInput):
        """ Create a new schedule on the given project. The details on the actual object to schedule as well as the
        schedule itself are within the schedule object provided.
        The :py:class:`~ScheduleCronTrigger` follows the guidelines in
        https://apscheduler.readthedocs.io/en/v3.6.3/modules/triggers/cron.html.
        It also supports a :py:func:`~ScheduleCronTrigger.from_crontab` function that accepts a
        crontab-formatted string (see https://en.wikipedia.org/wiki/Cron for more information on the format).

        Example::

            from mlrun.api import schemas

            # Execute the get_data_func function every Tuesday at 15:30
            schedule = schemas.ScheduleInput(
                name="run_func_on_tuesdays",
                kind="job",
                scheduled_object=get_data_func,
                cron_trigger=schemas.ScheduleCronTrigger(day_of_week='tue', hour=15, minute=30),
            )
            db.create_schedule(project_name, schedule)
        """

        project = project or config.default_project
        path = f"projects/{project}/schedules"

        error_message = f"Failed creating schedule {project}/{schedule.name}"
        self.api_call("POST", path, error_message, body=dict_to_json(schedule.dict()))

    def update_schedule(
        self, project: str, name: str, schedule: schemas.ScheduleUpdate
    ):
        """ Update an existing schedule, replace it with the details contained in the schedule object."""

        project = project or config.default_project
        path = f"projects/{project}/schedules/{name}"

        error_message = f"Failed updating schedule {project}/{name}"
        self.api_call("PUT", path, error_message, body=dict_to_json(schedule.dict()))

    def get_schedule(
        self, project: str, name: str, include_last_run: bool = False
    ) -> schemas.ScheduleOutput:
        """ Retrieve details of the schedule in question. Besides returning the details of the schedule object itself,
        this function also returns the next scheduled run for this specific schedule, as well as potentially the
        results of the last run executed through this schedule.

        :param project: Project name.
        :param name: Name of the schedule object to query.
        :param include_last_run: Whether to include the results of the schedule's last run in the response.
        """

        project = project or config.default_project
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
        """ Retrieve list of schedules of specific name or kind.

        :param project: Project name.
        :param name: Name of schedule to retrieve. Can be omitted to list all schedules.
        :param kind: Kind of schedule objects to retrieve, can be either ``job`` or ``pipeline``.
        :param include_last_run: Whether to return for each schedule returned also the results of the last run of
            that schedule.
        """

        project = project or config.default_project
        params = {"kind": kind, "name": name, "include_last_run": include_last_run}
        path = f"projects/{project}/schedules"
        error_message = f"Failed listing schedules for {project} ? {kind} {name}"
        resp = self.api_call("GET", path, error_message, params=params)
        return schemas.SchedulesOutput(**resp.json())

    def delete_schedule(self, project: str, name: str):
        """ Delete a specific schedule by name. """

        project = project or config.default_project
        path = f"projects/{project}/schedules/{name}"
        error_message = f"Failed deleting schedule {project}/{name}"
        self.api_call("DELETE", path, error_message)

    def invoke_schedule(self, project: str, name: str):
        """ Execute the object referenced by the schedule immediately. """

        project = project or config.default_project
        path = f"projects/{project}/schedules/{name}/invoke"
        error_message = f"Failed invoking schedule {project}/{name}"
        self.api_call("POST", path, error_message)

    def remote_builder(
        self, func, with_mlrun, mlrun_version_specifier=None, skip_deployed=False
    ):
        """ Build the pod image for a function, for execution on a remote cluster. This is executed by the MLRun
        API server, and creates a Docker image out of the function provided and any specific build
        instructions provided within. This is a pre-requisite for remotely executing a function, unless using
        a pre-deployed image.

        :param func: Function to build.
        :param with_mlrun: Whether to add MLRun package to the built package. This is not required if using a base
            image that already has MLRun in it.
        :param mlrun_version_specifier: Version of MLRun to include in the built image.
        :param skip_deployed: Skip the build if we already have an image for the function.
        """

        try:
            req = {
                "function": func.to_dict(),
                "with_mlrun": bool2str(with_mlrun),
                "skip_deployed": skip_deployed,
            }
            if mlrun_version_specifier:
                req["mlrun_version_specifier"] = mlrun_version_specifier
            resp = self.api_call("POST", "build/function", json=req)
        except OSError as err:
            logger.error(f"error submitting build task: {err}")
            raise OSError(f"error: cannot submit build, {err}")

        if not resp.ok:
            logger.error(f"bad resp!!\n{resp.text}")
            raise ValueError("bad function run response")

        return resp.json()

    def get_builder_status(
        self, func, offset=0, logs=True, last_log_timestamp=0, verbose=False
    ):
        """ Retrieve the status of a build operation currently in progress.

        :param func: Function object that is being built.
        :param offset: Offset into the build logs to retrieve logs from.
        :param logs: Should build logs be retrieved.
        :param last_log_timestamp: Last timestamp of logs that were already retrieved. Function will return only logs
            later than this parameter.
        :param verbose: Add verbose logs into the output.
        :returns: The following parameters:

            - Text of builder logs.
            - Timestamp of last log retrieved, to be used in subsequent calls to this function.

            The function also updates internal members of the ``func`` object to reflect build process info.
        """

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
            logger.error(f"error getting build status: {err}")
            raise OSError(f"error: cannot get build status, {err}")

        if not resp.ok:
            logger.warning(f"failed resp, {resp.text}")
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
        """ Execute a function remotely, Used for ``dask`` functions.

        :param func_url: URL to the function to be executed.
        :returns: A BackgroundTask object, with details on execution process and its status.
        """

        try:
            req = {"functionUrl": func_url}
            resp = self.api_call(
                "POST",
                "start/function",
                json=req,
                timeout=int(config.submit_timeout) or 60,
            )
        except OSError as err:
            logger.error(f"error starting function: {err}")
            raise OSError(f"error: cannot start function, {err}")

        if not resp.ok:
            logger.error(f"bad resp!!\n{resp.text}")
            raise ValueError("bad function start response")

        return schemas.BackgroundTask(**resp.json())

    def get_background_task(self, project: str, name: str,) -> schemas.BackgroundTask:
        """ Retrieve updated information on a background task being executed."""

        project = project or config.default_project
        path = f"projects/{project}/background-tasks/{name}"
        error_message = (
            f"Failed getting background task. project={project}, name={name}"
        )
        response = self.api_call("GET", path, error_message)
        return schemas.BackgroundTask(**response.json())

    def remote_status(self, kind, selector):
        """ Retrieve status of a function being executed remotely (relevant to ``dask`` functions).

        :param kind: The kind of the function, currently ``dask`` is supported.
        :param selector: Selector clause to be applied to the Kubernetes status query to filter the results.
        """

        try:
            req = {"kind": kind, "selector": selector}
            resp = self.api_call("POST", "status/function", json=req)
        except OSError as err:
            logger.error(f"error starting function: {err}")
            raise OSError(f"error: cannot start function, {err}")

        if not resp.ok:
            logger.error(f"bad resp!!\n{resp.text}")
            raise ValueError("bad function status response")

        return resp.json()["data"]

    def submit_job(
        self, runspec, schedule: Union[str, schemas.ScheduleCronTrigger] = None
    ):
        """ Submit a job for remote execution.

        :param runspec: The runtime object spec (Task) to execute.
        :param schedule: Whether to schedule this job using a Cron trigger. If not specified, the job will be submitted
            immediately.
        """

        try:
            req = {"task": runspec.to_dict()}
            if schedule:
                if isinstance(schedule, schemas.ScheduleCronTrigger):
                    schedule = schedule.dict()
                req["schedule"] = schedule
            timeout = (int(config.submit_timeout) or 120) + 20
            resp = self.api_call("POST", "submit_job", json=req, timeout=timeout)
        except OSError as err:
            logger.error(f"error submitting task: {err}")
            raise OSError(f"error: cannot submit task, {err}")

        if not resp.ok:
            logger.error(f"bad resp!!\n{resp.text}")
            raise ValueError(f"bad function run response, {resp.text}")

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
        """ Submit a KFP pipeline for execution.

        :param pipeline: Pipeline function or path to .yaml/.zip pipeline file.
        :param arguments: A dictionary of arguments to pass to the pipeline.
        :param experiment: A name to assign for the specific experiment.
        :param run: A name for this specific run.
        :param namespace: Kubernetes namespace to execute the pipeline in.
        :param artifact_path: A path to artifacts used by this pipeline.
        :param ops: Transformers to apply on all ops in the pipeline.
        :param ttl: Set the TTL for the pipeline after its completion.
        """

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
            raise OSError(f"file {pipe_file} doesnt exist")
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
            logger.error(f"error cannot submit pipeline: {err}")
            raise OSError(f"error: cannot cannot submit pipeline, {err}")

        if not resp.ok:
            logger.error(f"bad resp!!\n{resp.text}")
            raise ValueError(f"bad submit pipeline response, {resp.text}")

        resp = resp.json()
        logger.info(f"submitted pipeline {resp['name']} id={resp['id']}")
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
        """ Retrieve a list of KFP pipelines. This function can be invoked to get all pipelines from all projects,
        by specifying ``project=*``, in which case pagination can be used and the various sorting and pagination
        properties can be applied. If a specific project is requested, then the pagination options cannot be
        used and pagination is not applied.

        :param project: Project name. Can be ``*`` for query across all projects.
        :param namespace: Kubernetes namespace in which the pipelines are executing.
        :param sort_by: Field to sort the results by.
        :param page_token: Use for pagination, to retrieve next page.
        :param filter_: Kubernetes filter to apply to the query, can be used to filter on specific object fields.
        :param format_: Result format. Can be one of:

            - ``full`` - return the full objects.
            - ``metadata_only`` (default) - return just metadata of the pipelines objects.
            - ``name_only`` - return just the names of the pipeline objects.
        :param page_size: Size of a single page when applying pagination.
        """

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
        """ Retrieve details of a specific pipeline using its run ID (as provided when the pipeline was executed)."""

        try:
            query = ""
            if namespace:
                query = f"namespace={namespace}"
            resp = self.api_call("GET", f"pipelines/{run_id}?{query}", timeout=timeout)
        except OSError as err:
            logger.error(f"error cannot get pipeline: {err}")
            raise OSError(f"error: cannot get pipeline, {err}")

        if not resp.ok:
            logger.error(f"bad resp!!\n{resp.text}")
            raise ValueError(f"bad get pipeline response, {resp.text}")

        return resp.json()

    @staticmethod
    def _resolve_reference(tag, uid):
        if uid and tag:
            raise MLRunInvalidArgumentError("both uid and tag were provided")
        return uid or tag or "latest"

    def create_feature_set(
        self, feature_set: Union[dict, schemas.FeatureSet], project="", versioned=True
    ) -> dict:
        """ Create a new :py:class:`~mlrun.feature_store.FeatureSet` and save in the :py:mod:`mlrun` DB. The
        feature-set must not previously exist in the DB.

        :param feature_set: The new :py:class:`~mlrun.feature_store.FeatureSet` to create.
        :param project: Name of project this feature-set belongs to.
        :param versioned: Whether to maintain versions for this feature-set. All versions of a versioned object
            will be kept in the DB and can be retrieved until explicitly deleted.
        :returns: The :py:class:`~mlrun.feature_store.FeatureSet` object (as dict).
        """
        if isinstance(feature_set, schemas.FeatureSet):
            feature_set = feature_set.dict()

        project = (
            project
            or feature_set["metadata"].get("project", None)
            or config.default_project
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
        """ Retrieve a ~mlrun.feature_store.FeatureSet` object. If both ``tag`` and ``uid`` are not specified, then
        the object tagged ``latest`` will be retrieved.

        :param name: Name of object to retrieve.
        :param project: Project the FeatureSet belongs to.
        :param tag: Tag of the specific object version to retrieve.
        :param uid: uid of the object to retrieve (can only be used for versioned objects).
        """

        project = project or config.default_project
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
        """ List feature-sets which contain specific features. This function may return multiple versions of the same
        feature-set if a specific tag is not requested. Note that the various filters of this function actually
        refer to the feature-set object containing the features, not to the features themselves.

        :param project: Project which contains these features.
        :param name: Name of the feature to look for. The name is used in a like query, and is not case-sensitive. For
            example, looking for ``feat`` will return features which are named ``MyFeature`` as well as ``defeat``.
        :param tag: Return feature-sets which contain the features looked for, and are tagged with the specific tag.
        :param entities: Return only feature-sets which contain an entity whose name is contained in this list.
        :param labels: Return only feature-sets which are labeled as requested.
        :returns: A list of mapping from feature to a digest of the feature-set, which contains the feature-set
            meta-data. Multiple entries may be returned for any specific feature due to multiple tags or versions
            of the feature-set.
        """

        project = project or config.default_project
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
        """ Retrieve a list of entities and their mapping to the containing feature-sets. This function is similar
        to the :py:func:`~list_features` function, and uses the same logic. However, the entities are matched
        against the name rather than the features.
        """

        project = project or config.default_project
        params = {
            "name": name,
            "tag": tag,
            "label": labels or [],
        }

        path = f"projects/{project}/entities"

        error_message = f"Failed listing entities, project: {project}, query: {params}"
        resp = self.api_call("GET", path, error_message, params=params)
        return resp.json()["entities"]

    @staticmethod
    def _generate_partition_by_params(partition_by, rows_per_partition, sort_by, order):
        if isinstance(partition_by, schemas.FeatureStorePartitionByField):
            partition_by = partition_by.value
        if isinstance(sort_by, schemas.SortField):
            sort_by = sort_by.value
        if isinstance(order, schemas.OrderType):
            order = order.value

        return {
            "partition-by": partition_by,
            "rows-per-partition": rows_per_partition,
            "partition-sort-by": sort_by,
            "partition-order": order,
        }

    def list_feature_sets(
        self,
        project: str = "",
        name: str = None,
        tag: str = None,
        state: str = None,
        entities: List[str] = None,
        features: List[str] = None,
        labels: List[str] = None,
        partition_by: Union[schemas.FeatureStorePartitionByField, str] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Union[schemas.SortField, str] = None,
        partition_order: Union[schemas.OrderType, str] = schemas.OrderType.desc,
    ) -> List[FeatureSet]:
        """ Retrieve a list of feature-sets matching the criteria provided.

        :param project: Project name.
        :param name: Name of feature-set to match. This is a like query, and is case-insensitive.
        :param tag: Match feature-sets with specific tag.
        :param state: Match feature-sets with a specific state.
        :param entities: Match feature-sets which contain entities whose name is in this list.
        :param features: Match feature-sets which contain features whose name is in this list.
        :param labels: Match feature-sets which have these labels.
        :param partition_by: Field to group results by. Only allowed value is `name`. When `partition_by` is specified,
            the `partition_sort_by` parameter must be provided as well.
        :param rows_per_partition: How many top rows (per sorting defined by `partition_sort_by` and `partition_order`)
            to return per group. Default value is 1.
        :param partition_sort_by: What field to sort the results by, within each partition defined by `partition_by`.
            Currently the only allowed value is `updated`.
        :param partition_order: Order of sorting within partitions - `asc` or `desc`. Default is `desc`.
        :returns: List of matching :py:class:`~mlrun.feature_store.FeatureSet` objects.
        """

        project = project or config.default_project

        params = {
            "name": name,
            "state": state,
            "tag": tag,
            "entity": entities or [],
            "feature": features or [],
            "label": labels or [],
        }
        if partition_by:
            params.update(
                self._generate_partition_by_params(
                    partition_by, rows_per_partition, partition_sort_by, partition_order
                )
            )

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
        """ Save a :py:class:`~mlrun.feature_store.FeatureSet` object in the :py:mod:`mlrun` DB. The
        feature-set can be either a new object or a modification to existing object referenced by the params of
        the function.

        :param feature_set: The :py:class:`~mlrun.feature_store.FeatureSet` to store.
        :param project: Name of project this feature-set belongs to.
        :param tag: The ``tag`` of the object to replace in the DB, for example ``latest``.
        :param uid: The ``uid`` of the object to replace in the DB. If using this parameter, the modified object
            must have the same ``uid`` of the previously-existing object. This cannot be used for non-versioned objects.
        :param versioned: Whether to maintain versions for this feature-set. All versions of a versioned object
            will be kept in the DB and can be retrieved until explicitly deleted.
        :returns: The :py:class:`~mlrun.feature_store.FeatureSet` object (as dict).
        """

        reference = self._resolve_reference(tag, uid)
        params = {"versioned": versioned}

        if isinstance(feature_set, schemas.FeatureSet):
            feature_set = feature_set.dict()

        name = name or feature_set["metadata"]["name"]
        project = (
            project or feature_set["metadata"].get("project") or config.default_project
        )
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
        """ Modify (patch) an existing :py:class:`~mlrun.feature_store.FeatureSet` object.
        The object is identified by its name (and project it belongs to), as well as optionally a ``tag`` or its
        ``uid`` (for versioned object). If both ``tag`` and ``uid`` are omitted then the object with tag ``latest``
        is modified.

        :param name: Name of the object to patch.
        :param feature_set_update: The modifications needed in the object. This parameter only has the changes in it,
            not a full object.
            Example::

                feature_set_update = {"status": {"processed" : True}}

            Will apply the field ``status.processed`` to the existing object.
        :param project: Project which contains the modified object.
        :param tag: The tag of the object to modify.
        :param uid: uid of the object to modify.
        :param patch_mode: The strategy for merging the changes with the existing object. Can be either ``replace``
            or ``additive``.
        """
        project = project or config.default_project
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

    def delete_feature_set(self, name, project="", tag=None, uid=None):
        """ Delete a :py:class:`~mlrun.feature_store.FeatureSet` object from the DB.
        If ``tag`` or ``uid`` are specified, then just the version referenced by them will be deleted. Using both
        is not allowed.
        If none are specified, then all instances of the object whose name is ``name`` will be deleted.
        """
        project = project or config.default_project
        path = f"projects/{project}/feature-sets/{name}"

        if tag or uid:
            reference = self._resolve_reference(tag, uid)
            path = path + f"/references/{reference}"

        error_message = f"Failed deleting feature-set {name}"
        self.api_call("DELETE", path, error_message)

    def create_feature_vector(
        self,
        feature_vector: Union[dict, schemas.FeatureVector],
        project="",
        versioned=True,
    ) -> dict:
        """ Create a new :py:class:`~mlrun.feature_store.FeatureVector` and save in the :py:mod:`mlrun` DB.

        :param feature_vector: The new :py:class:`~mlrun.feature_store.FeatureVector` to create.
        :param project: Name of project this feature-vector belongs to.
        :param versioned: Whether to maintain versions for this feature-vector. All versions of a versioned object
            will be kept in the DB and can be retrieved until explicitly deleted.
        :returns: The :py:class:`~mlrun.feature_store.FeatureVector` object (as dict).
        """
        if isinstance(feature_vector, schemas.FeatureVector):
            feature_vector = feature_vector.dict()

        project = (
            project
            or feature_vector["metadata"].get("project", None)
            or config.default_project
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
        """ Return a specific feature-vector referenced by its tag or uid. If none are provided, ``latest`` tag will
        be used. """

        project = project or config.default_project
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
        partition_by: Union[schemas.FeatureStorePartitionByField, str] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Union[schemas.SortField, str] = None,
        partition_order: Union[schemas.OrderType, str] = schemas.OrderType.desc,
    ) -> List[FeatureVector]:
        """ Retrieve a list of feature-vectors matching the criteria provided.

        :param project: Project name.
        :param name: Name of feature-vector to match. This is a like query, and is case-insensitive.
        :param tag: Match feature-vectors with specific tag.
        :param state: Match feature-vectors with a specific state.
        :param labels: Match feature-vectors which have these labels.
        :param partition_by: Field to group results by. Only allowed value is `name`. When `partition_by` is specified,
            the `partition_sort_by` parameter must be provided as well.
        :param rows_per_partition: How many top rows (per sorting defined by `partition_sort_by` and `partition_order`)
            to return per group. Default value is 1.
        :param partition_sort_by: What field to sort the results by, within each partition defined by `partition_by`.
            Currently the only allowed value is `updated`.
        :param partition_order: Order of sorting within partitions - `asc` or `desc`. Default is `desc`.
        :returns: List of matching :py:class:`~mlrun.feature_store.FeatureVector` objects.
        """

        project = project or config.default_project

        params = {
            "name": name,
            "state": state,
            "tag": tag,
            "label": labels or [],
        }
        if partition_by:
            params.update(
                self._generate_partition_by_params(
                    partition_by, rows_per_partition, partition_sort_by, partition_order
                )
            )

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
        """ Store a :py:class:`~mlrun.feature_store.FeatureVector` object in the :py:mod:`mlrun` DB. The
        feature-vector can be either a new object or a modification to existing object referenced by the params
        of the function.

        :param feature_vector: The :py:class:`~mlrun.feature_store.FeatureVector` to store.
        :param project: Name of project this feature-vector belongs to.
        :param tag: The ``tag`` of the object to replace in the DB, for example ``latest``.
        :param uid: The ``uid`` of the object to replace in the DB. If using this parameter, the modified object
            must have the same ``uid`` of the previously-existing object. This cannot be used for non-versioned objects.
        :param versioned: Whether to maintain versions for this feature-vector. All versions of a versioned object
            will be kept in the DB and can be retrieved until explicitly deleted.
        :returns: The :py:class:`~mlrun.feature_store.FeatureVector` object (as dict).
        """

        reference = self._resolve_reference(tag, uid)
        params = {"versioned": versioned}

        if isinstance(feature_vector, schemas.FeatureVector):
            feature_vector = feature_vector.dict()

        name = name or feature_vector["metadata"]["name"]
        project = (
            project
            or feature_vector["metadata"].get("project")
            or config.default_project
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
        """ Modify (patch) an existing :py:class:`~mlrun.feature_store.FeatureVector` object.
        The object is identified by its name (and project it belongs to), as well as optionally a ``tag`` or its
        ``uid`` (for versioned object). If both ``tag`` and ``uid`` are omitted then the object with tag ``latest``
        is modified.

        :param name: Name of the object to patch.
        :param feature_vector_update: The modifications needed in the object. This parameter only has the changes in it,
            not a full object.
        :param project: Project which contains the modified object.
        :param tag: The tag of the object to modify.
        :param uid: uid of the object to modify.
        :param patch_mode: The strategy for merging the changes with the existing object. Can be either ``replace``
            or ``additive``.
        """
        reference = self._resolve_reference(tag, uid)
        project = project or config.default_project
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

    def delete_feature_vector(self, name, project="", tag=None, uid=None):
        """ Delete a :py:class:`~mlrun.feature_store.FeatureVector` object from the DB.
        If ``tag`` or ``uid`` are specified, then just the version referenced by them will be deleted. Using both
        is not allowed.
        If none are specified, then all instances of the object whose name is ``name`` will be deleted.
        """
        project = project or config.default_project
        path = f"projects/{project}/feature-vectors/{name}"
        if tag or uid:
            reference = self._resolve_reference(tag, uid)
            path = path + f"/references/{reference}"

        error_message = f"Failed deleting feature-vector {name}"
        self.api_call("DELETE", path, error_message)

    def list_projects(
        self,
        owner: str = None,
        format_: Union[str, mlrun.api.schemas.Format] = mlrun.api.schemas.Format.full,
        labels: List[str] = None,
        state: Union[str, mlrun.api.schemas.ProjectState] = None,
    ) -> List[Union[mlrun.projects.MlrunProject, str]]:
        """ Return a list of the existing projects, potentially filtered by specific criteria.

        :param owner: List only projects belonging to this specific owner.
        :param format_: Format of the results. Possible values are:

            - ``full`` (default value) - Return full project objects.
            - ``name_only`` - Return just the names of the projects.

        :param labels: Filter by labels attached to the project.
        :param state: Filter by project's state. Can be either ``online`` or ``archived``.
        """

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
        """ Get details for a specific project."""

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
        """ Delete a project.

        :param name: Name of the project to delete.
        :param deletion_strategy: How to treat child objects of the project. Possible values are:

            - ``restrict`` (default) - Project must not have any child objects when deleted. If using this mode while
              child objects exist, the operation will fail.
            - ``cascade`` - Automatically delete all child objects when deleting the project.
        """

        path = f"projects/{name}"
        if isinstance(deletion_strategy, schemas.DeletionStrategy):
            deletion_strategy = deletion_strategy.value
        headers = {schemas.HeaderNames.deletion_strategy: deletion_strategy}
        error_message = f"Failed deleting project {name}"
        response = self.api_call("DELETE", path, error_message, headers=headers)
        if response.status_code == http.HTTPStatus.ACCEPTED:
            return self._wait_for_project_to_be_deleted(name)

    def store_project(
        self,
        name: str,
        project: Union[dict, mlrun.projects.MlrunProject, mlrun.api.schemas.Project],
    ) -> mlrun.projects.MlrunProject:
        """ Store a project in the DB. This operation will overwrite existing project of the same name if exists."""

        path = f"projects/{name}"
        error_message = f"Failed storing project {name}"
        if isinstance(project, mlrun.api.schemas.Project):
            project = project.dict()
        elif isinstance(project, mlrun.projects.MlrunProject):
            project = project.to_dict()
        response = self.api_call(
            "PUT", path, error_message, body=dict_to_json(project),
        )
        if response.status_code == http.HTTPStatus.ACCEPTED:
            return self._wait_for_project_to_reach_terminal_state(name)
        return mlrun.projects.MlrunProject.from_dict(response.json())

    def patch_project(
        self,
        name: str,
        project: dict,
        patch_mode: Union[str, schemas.PatchMode] = schemas.PatchMode.replace,
    ) -> mlrun.projects.MlrunProject:
        """ Patch an existing project object.

        :param name: Name of project to patch.
        :param project: The actual changes to the project object.
        :param patch_mode: The strategy for merging the changes with the existing object. Can be either ``replace``
            or ``additive``.
        """

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
        """ Create a new project. A project with the same name must not exist prior to creation."""

        if isinstance(project, mlrun.api.schemas.Project):
            project = project.dict()
        elif isinstance(project, mlrun.projects.MlrunProject):
            project = project.to_dict()
        project_name = project["metadata"]["name"]
        error_message = f"Failed creating project {project_name}"
        response = self.api_call(
            "POST", "projects", error_message, body=dict_to_json(project),
        )
        if response.status_code == http.HTTPStatus.ACCEPTED:
            return self._wait_for_project_to_reach_terminal_state(project_name)
        return mlrun.projects.MlrunProject.from_dict(response.json())

    def _wait_for_project_to_reach_terminal_state(
        self, project_name: str
    ) -> mlrun.projects.MlrunProject:
        def _verify_project_in_terminal_state():
            project = self.get_project(project_name)
            if (
                project.status.state
                not in mlrun.api.schemas.ProjectState.terminal_states()
            ):
                raise Exception(
                    f"Project not in terminal state. State: {project.status.state}"
                )
            return project

        return mlrun.utils.helpers.retry_until_successful(
            self._wait_for_project_terminal_state_retry_interval,
            120,
            logger,
            False,
            _verify_project_in_terminal_state,
        )

    def _wait_for_project_to_be_deleted(self, project_name: str):
        def _verify_project_deleted():
            projects = self.list_projects(format_=mlrun.api.schemas.Format.name_only)
            if project_name in projects:
                raise Exception("Project still exists")

        return mlrun.utils.helpers.retry_until_successful(
            self._wait_for_project_deletion_interval,
            120,
            logger,
            False,
            _verify_project_deleted,
        )

    def create_project_secrets(
        self,
        project: str,
        provider: Union[
            str, schemas.SecretProviderName
        ] = schemas.SecretProviderName.vault,
        secrets: dict = None,
    ):
        """ Create project-context secrets using either ``vault`` or ``kubernetes`` provider.
        When using with Vault, this will create needed Vault structures for storing secrets in project-context, and
        store a set of secret values. The method generates Kubernetes service-account and the Vault authentication
        structures that are required for function Pods to authenticate with Vault and be able to extract secret values
        passed as part of their context.

        Note:
                This method used with Vault is currently in technical preview, and requires a HashiCorp Vault
                infrastructure properly set up and connected to the MLRun API server.

        When used with Kubernetes, this will make sure that the project-specific k8s secret is created, and will
        populate it with the secrets provided, replacing their values if they exist.

        :param project: The project context for which to generate the infra and store secrets.
        :param provider: The name of the secrets-provider to work with. Accepts a
            :py:class:`~mlrun.api.schemas.secret.SecretProviderName` enum.
        :param secrets: A set of secret values to store.
            Example::

                secrets = {'password': 'myPassw0rd', 'aws_key': '111222333'}
                db.create_project_secrets(
                    "project1",
                    provider=mlrun.api.schemas.SecretProviderName.vault,
                    secrets=secrets
                )
        """
        if isinstance(provider, schemas.SecretProviderName):
            provider = provider.value
        path = f"projects/{project}/secrets"
        secrets_input = schemas.SecretsData(secrets=secrets, provider=provider)
        body = secrets_input.dict()
        error_message = f"Failed creating secret provider {project}/{provider}"
        self.api_call(
            "POST", path, error_message, body=dict_to_json(body),
        )

    def list_project_secrets(
        self,
        project: str,
        token: str = None,
        provider: Union[
            str, schemas.SecretProviderName
        ] = schemas.SecretProviderName.vault,
        secrets: List[str] = None,
    ) -> schemas.SecretsData:
        """ Retrieve project-context secrets from Vault.

        Note:
                This method for Vault functionality is currently in technical preview, and requires a HashiCorp Vault
                infrastructure properly set up and connected to the MLRun API server.

        :param project: The project name.
        :param token: Vault token to use for retrieving secrets.
            Must be a valid Vault token, with permissions to retrieve secrets of the project in question.
        :param provider: The name of the secrets-provider to work with. Currently only ``vault`` is accepted.
        :param secrets: A list of secret names to retrieve. An empty list ``[]`` will retrieve all secrets assigned
            to this specific project. ``kubernetes`` provider only supports an empty list.
        """

        if isinstance(provider, schemas.SecretProviderName):
            provider = provider.value

        if provider == schemas.SecretProviderName.vault.value and not token:
            raise MLRunInvalidArgumentError(
                "A vault token must be provided when accessing vault secrets"
            )

        path = f"projects/{project}/secrets"
        params = {"provider": provider, "secret": secrets}
        headers = {schemas.HeaderNames.secret_store_token: token}
        error_message = f"Failed retrieving secrets {project}/{provider}"
        result = self.api_call(
            "GET", path, error_message, params=params, headers=headers,
        )
        return schemas.SecretsData(**result.json())

    def list_project_secret_keys(
        self,
        project: str,
        provider: Union[
            str, schemas.SecretProviderName
        ] = schemas.SecretProviderName.vault,
        token: str = None,
    ) -> schemas.SecretKeysData:
        """ Retrieve project-context secret keys from Vault or Kubernetes.

        Note:
                This method for Vault functionality is currently in technical preview, and requires a HashiCorp Vault
                infrastructure properly set up and connected to the MLRun API server.

        :param project: The project name.
        :param provider: The name of the secrets-provider to work with. Accepts a
            :py:class:`~mlrun.api.schemas.secret.SecretProviderName` enum.
        :param token: Vault token to use for retrieving secrets. Only in use if ``provider`` is ``vault``.
            Must be a valid Vault token, with permissions to retrieve secrets of the project in question.
        """

        if isinstance(provider, schemas.SecretProviderName):
            provider = provider.value

        if provider == schemas.SecretProviderName.vault.value and not token:
            raise MLRunInvalidArgumentError(
                "A vault token must be provided when accessing vault secrets"
            )

        path = f"projects/{project}/secret-keys"
        params = {"provider": provider}
        headers = (
            {schemas.HeaderNames.secret_store_token: token}
            if provider == schemas.SecretProviderName.vault.value
            else None
        )
        error_message = f"Failed retrieving secret keys {project}/{provider}"
        result = self.api_call(
            "GET", path, error_message, params=params, headers=headers,
        )
        return schemas.SecretKeysData(**result.json())

    def delete_project_secrets(
        self,
        project: str,
        provider: Union[
            str, schemas.SecretProviderName
        ] = schemas.SecretProviderName.kubernetes,
        secrets: List[str] = None,
    ):
        """ Delete project-context secrets from Kubernetes.

        :param project: The project name.
        :param provider: The name of the secrets-provider to work with. Currently only ``kubernetes`` is supported.
        :param secrets: A list of secret names to delete. An empty list will delete all secrets assigned
            to this specific project.
        """
        if isinstance(provider, schemas.SecretProviderName):
            provider = provider.value

        path = f"projects/{project}/secrets"
        params = {"provider": provider, "secret": secrets}
        error_message = f"Failed deleting secrets {project}/{provider}"
        self.api_call(
            "DELETE", path, error_message, params=params,
        )

    def create_user_secrets(
        self,
        user: str,
        provider: Union[
            str, schemas.SecretProviderName
        ] = schemas.SecretProviderName.vault,
        secrets: dict = None,
    ):
        """ Create user-context secret in Vault. Please refer to :py:func:`create_project_secrets` for more details
        and status of this functionality.

        Note:
                This method is currently in technical preview, and requires a HashiCorp Vault infrastructure
                properly set up and connected to the MLRun API server.

        :param user: The user context for which to generate the infra and store secrets.
        :param provider: The name of the secrets-provider to work with. Currently only ``vault`` is supported.
        :param secrets: A set of secret values to store within the Vault.
        """
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

    def create_or_patch(
        self,
        project: str,
        endpoint_id: str,
        model_endpoint: ModelEndpoint,
        access_key: Optional[str] = None,
    ):
        """
        Creates or updates a KV record with the given model_endpoint record

        :param project: The name of the project
        :param endpoint_id: The id of the endpoint
        :param model_endpoint: An object representing the model endpoint
        :param access_key: V3IO access key, when None, will be look for in environ
        """
        access_key = access_key or os.environ.get("V3IO_ACCESS_KEY")
        if not access_key:
            raise MLRunInvalidArgumentError(
                "access_key must be initialized, either by passing it as an argument or by populating a "
                "V3IO_ACCESS_KEY environment parameter"
            )

        path = f"projects/{project}/model-endpoints/{endpoint_id}"
        self.api_call(
            method="PUT",
            path=path,
            body=model_endpoint.json(),
            headers={"X-V3io-Session-Key": access_key},
        )

    def delete_endpoint_record(
        self, project: str, endpoint_id: str, access_key: Optional[str] = None,
    ):
        """
        Deletes the KV record of a given model endpoint, project and endpoint_id are used for lookup

        :param project: The name of the project
        :param endpoint_id: The id of the endpoint
        :param access_key: V3IO access key, when None, will be look for in environ
        """
        access_key = access_key or os.environ.get("V3IO_ACCESS_KEY")
        if not access_key:
            raise MLRunInvalidArgumentError(
                "access_key must be initialized, either by passing it as an argument or by populating a "
                "V3IO_ACCESS_KEY environment parameter"
            )

        path = f"projects/{project}/model-endpoints/{endpoint_id}"
        self.api_call(
            method="DELETE", path=path, headers={"X-V3io-Session-Key": access_key},
        )

    def list_endpoints(
        self,
        project: str,
        model: Optional[str] = None,
        function: Optional[str] = None,
        labels: List[str] = None,
        start: str = "now-1h",
        end: str = "now",
        metrics: Optional[List[str]] = None,
        access_key: Optional[str] = None,
    ) -> schemas.ModelEndpointList:
        """
        Returns a list of ModelEndpointState objects. Each object represents the current state of a model endpoint.
        This functions supports filtering by the following parameters:
        1) model
        2) function
        3) labels
        By default, when no filters are applied, all available endpoints for the given project will be listed.

        In addition, this functions provides a facade for listing endpoint related metrics. This facade is time-based
        and depends on the 'start' and 'end' parameters. By default, when the metrics parameter is None, no metrics are
        added to the output of this function.

        :param project: The name of the project
        :param model: The name of the model to filter by
        :param function: The name of the function to filter by
        :param labels: A list of labels to filter by. Label filters work by either filtering a specific value of a label
        (i.e. list("key==value")) or by looking for the existence of a given key (i.e. "key")
        :param metrics: A list of metrics to return for each endpoint, read more in 'TimeMetric'
        :param start: The start time of the metrics
        :param end: The end time of the metrics
        :param access_key: V3IO access key, when None, will be look for in environ
        """
        access_key = access_key or os.environ.get("V3IO_ACCESS_KEY")
        if not access_key:
            raise MLRunInvalidArgumentError(
                "access_key must be initialized, either by passing it as an argument or by populating a "
                "V3IO_ACCESS_KEY environment parameter"
            )

        path = f"projects/{project}/model-endpoints"
        response = self.api_call(
            method="GET",
            path=path,
            params={
                "model": model,
                "function": function,
                "labels": labels,
                "start": start,
                "end": end,
                "metrics": metrics,
            },
            headers={"X-V3io-Session-Key": access_key},
        )
        return schemas.ModelEndpointList(**response.json())

    def get_endpoint(
        self,
        project: str,
        endpoint_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        feature_analysis: bool = False,
        access_key: Optional[str] = None,
    ) -> schemas.ModelEndpoint:
        """
        Returns a ModelEndpoint object with additional metrics and feature related data.

        :param project: The name of the project
        :param endpoint_id: The id of the model endpoint
        :param metrics: A list of metrics to return for each endpoint, read more in 'TimeMetric'
        :param start: The start time of the metrics
        :param end: The end time of the metrics
        :param feature_analysis: When True, the base feature statistics and current feature statistics will be added to
        the output of the resulting object
        :param access_key: V3IO access key, when None, will be look for in environ
        """
        access_key = access_key or os.environ.get("V3IO_ACCESS_KEY")
        if not access_key:
            raise MLRunInvalidArgumentError(
                "access_key must be initialized, either by passing it as an argument or by populating a "
                "V3IO_ACCESS_KEY environment parameter"
            )

        path = f"projects/{project}/model-endpoints/{endpoint_id}"
        response = self.api_call(
            method="GET",
            path=path,
            params={
                "start": start,
                "end": end,
                "metrics": metrics,
                "feature_analysis": feature_analysis,
            },
            headers={"X-V3io-Session-Key": access_key},
        )
        return schemas.ModelEndpoint(**response.json())


def _as_json(obj):
    fn = getattr(obj, "to_json", None)
    if fn:
        return fn()
    return dict_to_json(obj)
