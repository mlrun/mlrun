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
import enum
import http
import re
import tempfile
import time
import traceback
import typing
import warnings
from datetime import datetime, timedelta
from os import path, remove
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import kfp
import requests
import semver

import mlrun
import mlrun.common.schemas
import mlrun.model_monitoring.model_endpoint
import mlrun.platforms
import mlrun.projects
from mlrun.errors import MLRunInvalidArgumentError, err_to_str

from ..artifacts import Artifact
from ..config import config
from ..datastore.datastore_profile import DatastoreProfile2Json
from ..feature_store import FeatureSet, FeatureVector
from ..lists import ArtifactList, RunList
from ..runtimes import BaseRuntime
from ..utils import (
    datetime_to_iso,
    dict_to_json,
    logger,
    new_pipe_metadata,
    normalize_name,
    version,
)
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


class HTTPRunDB(RunDBInterface):
    """Interface for accessing and manipulating the :py:mod:`mlrun` persistent store, maintaining the full state
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
    # by default we don't retry on POST request as they are usually not idempotent
    # here is a list of identified POST requests that are idempotent ('store' vs 'create') and can be retried
    RETRIABLE_POST_PATHS = [
        r"\/?projects\/.+\/artifacts\/.+\/.+",
        r"\/?run\/.+\/.+",
    ]

    def __init__(self, url):
        self.server_version = ""
        self.session = None
        self._wait_for_project_terminal_state_retry_interval = 3
        self._wait_for_background_task_terminal_state_retry_interval = 3
        self._wait_for_project_deletion_interval = 3
        self.client_version = version.Version().get()["version"]
        self.python_version = str(version.Version().get_python_version())

        self._enrich_and_validate(url)

    def _enrich_and_validate(self, url):
        parsed_url = urlparse(url)
        scheme = parsed_url.scheme.lower()
        if scheme not in ("http", "https"):
            raise ValueError(
                f"Invalid URL scheme {scheme} for HTTPRunDB, only http(s) is supported"
            )

        endpoint = parsed_url.hostname
        if parsed_url.port:
            endpoint += f":{parsed_url.port}"
        base_url = f"{parsed_url.scheme}://{endpoint}{parsed_url.path}"

        username = parsed_url.username or config.httpdb.user
        password = parsed_url.password or config.httpdb.password

        username, password, token = mlrun.platforms.add_or_refresh_credentials(
            parsed_url.hostname, username, password, config.httpdb.token
        )

        self.base_url = base_url
        self.user = username
        self.password = password
        self.token = token

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}({self.base_url!r})"

    @staticmethod
    def get_api_path_prefix(version: str = None) -> str:
        """
        :param version: API version to use, None (the default) will mean to use the default value from mlconf,
         for un-versioned api set an empty string.
        """
        if version is not None:
            return f"api/{version}" if version else "api"

        api_version_path = (
            f"api/{config.api_base_version}" if config.api_base_version else "api"
        )
        return api_version_path

    def get_base_api_url(self, path: str, version: str = None) -> str:
        path_prefix = self.get_api_path_prefix(version)
        url = f"{self.base_url}/{path_prefix}/{path}"
        return url

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
        version=None,
    ):
        """Perform a direct REST API call on the :py:mod:`mlrun` API server.

        Caution:
            For advanced usage - prefer using the various APIs exposed through this class, rather than
            directly invoking REST calls.

        :param method: REST method (POST, GET, PUT...)
        :param path: Path to endpoint executed, for example ``"projects"``
        :param error: Error to return if API invocation fails
        :param params: Rest parameters, passed as a dictionary: ``{"<param-name>": <"param-value">}``
        :param body: Payload to be passed in the call. If using JSON objects, prefer using the ``json`` param
        :param json: JSON payload to be passed in the call
        :param headers: REST headers, passed as a dictionary: ``{"<header-name>": "<header-value>"}``
        :param timeout: API call timeout
        :param version: API version to use, None (the default) will mean to use the default value from config,
         for un-versioned api set an empty string.

        :return: Python HTTP response object
        """
        url = self.get_base_api_url(path, version)
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
            # Iguazio auth doesn't support passing token through bearer, so use cookie instead
            if mlrun.platforms.iguazio.is_iguazio_session(self.token):
                session_cookie = f'j:{{"sid": "{self.token}"}}'
                cookies = {
                    "session": session_cookie,
                }
                kw["cookies"] = cookies
            else:
                if "Authorization" not in kw.setdefault("headers", {}):
                    kw["headers"].update({"Authorization": "Bearer " + self.token})

        if mlrun.common.schemas.HeaderNames.client_version not in kw.setdefault(
            "headers", {}
        ):
            kw["headers"].update(
                {
                    mlrun.common.schemas.HeaderNames.client_version: self.client_version,
                    mlrun.common.schemas.HeaderNames.python_version: self.python_version,
                }
            )

        # requests no longer supports header values to be enum (https://github.com/psf/requests/pull/6154)
        # convert to strings. Do the same for params for niceness
        for dict_ in [headers, params]:
            if dict_ is not None:
                for key in dict_.keys():
                    if isinstance(dict_[key], enum.Enum):
                        dict_[key] = dict_[key].value

        # if the method is POST, we need to update the session with the appropriate retry policy
        if not self.session or method == "POST":
            retry_on_post = self._is_retry_on_post_allowed(method, path)
            self.session = self._init_session(retry_on_post)

        try:
            response = self.session.request(
                method, url, timeout=timeout, verify=False, **kw
            )
        except requests.RequestException as exc:
            error = f"{err_to_str(exc)}: {error}" if error else err_to_str(exc)
            raise mlrun.errors.MLRunRuntimeError(error) from exc

        if not response.ok:
            if response.content:
                try:
                    data = response.json()
                    error_details = data.get("detail", {})
                    if not error_details:
                        logger.warning("Failed parsing error response body", data=data)
                except Exception:
                    error_details = ""
                if error_details:
                    error_details = f"details: {error_details}"
                    error = f"{error} {error_details}" if error else error_details
                    mlrun.errors.raise_for_status(response, error)

            mlrun.errors.raise_for_status(response, error)

        return response

    def _init_session(self, retry_on_post: bool = False):
        return mlrun.utils.HTTPSessionWithRetry(
            retry_on_exception=config.httpdb.retry_api_call_on_exception
            == mlrun.common.schemas.HTTPSessionRetryMode.enabled.value,
            retry_on_post=retry_on_post,
        )

    def _path_of(self, prefix, project, uid):
        project = project or config.default_project
        return f"{prefix}/{project}/{uid}"

    def _is_retry_on_post_allowed(self, method, path: str):
        """
        Check if the given path is allowed to be retried on POST method
        :param method:  used to verify that the method is POST since if there is no session initialized there is no
                        need to initialize it with retry policy for POST when the method is not POST
        :param path:    the path to check
        :return:        True if the path is allowed to be retried on POST method and method is POST, False otherwise
        """
        return method == "POST" and any(
            re.match(regex, path) for regex in self.RETRIABLE_POST_PATHS
        )

    def connect(self, secrets=None):
        """Connect to the MLRun API server. Must be called prior to executing any other method.
        The code utilizes the URL for the API server from the configuration - ``mlconf.dbpath``.

        For example::

            mlconf.dbpath = mlconf.dbpath or 'http://mlrun-api:8080'
            db = get_run_db().connect()
        """
        # hack to allow unit tests to instantiate HTTPRunDB without a real server behind
        if "mock-server" in self.base_url:
            return
        resp = self.api_call("GET", "client-spec", timeout=5)
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
            if config.ce.mode and config.ce.mode != server_cfg.get("ce_mode", ""):
                logger.warning(
                    f"warning!, server ({server_cfg['ce_mode']}) and client ({config.ce.mode})"
                    " CE mode don't match"
                )
            config.ce = server_cfg.get("ce") or config.ce

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
            config.spark_history_server_path = (
                config.spark_history_server_path
                or server_cfg.get("spark_history_server_path")
            )
            config.httpdb.builder.docker_registry = (
                config.httpdb.builder.docker_registry
                or server_cfg.get("docker_registry")
            )
            config.httpdb.api_url = config.httpdb.api_url or server_cfg.get("api_url")
            config.nuclio_version = config.nuclio_version or server_cfg.get(
                "nuclio_version"
            )
            config.default_function_priority_class_name = (
                config.default_function_priority_class_name
                or server_cfg.get("default_function_priority_class_name")
            )
            config.valid_function_priority_class_names = (
                config.valid_function_priority_class_names
                or server_cfg.get("valid_function_priority_class_names")
            )
            config.artifacts.calculate_hash = (
                config.artifacts.calculate_hash
                if config.artifacts.calculate_hash is not None
                else server_cfg.get("calculate_artifact_hash")
            )
            config.artifacts.generate_target_path_from_artifact_hash = (
                config.artifacts.generate_target_path_from_artifact_hash
                if config.artifacts.generate_target_path_from_artifact_hash is not None
                else server_cfg.get("generate_artifact_target_path_from_artifact_hash")
            )

            config.redis.url = config.redis.url or server_cfg.get("redis_url")
            # allow client to set the default partial WA for lack of support of per-target auxiliary options
            config.redis.type = config.redis.type or server_cfg.get("redis_type")

            config.sql.url = config.sql.url or server_cfg.get("sql_url")
            # These have a default value, therefore local config will always have a value, prioritize the
            # API value first
            config.ui.projects_prefix = (
                server_cfg.get("ui_projects_prefix") or config.ui.projects_prefix
            )
            config.kfp_image = server_cfg.get("kfp_image") or config.kfp_image
            config.kfp_url = server_cfg.get("kfp_url") or config.kfp_url
            config.dask_kfp_image = (
                server_cfg.get("dask_kfp_image") or config.dask_kfp_image
            )
            config.scrape_metrics = (
                server_cfg.get("scrape_metrics")
                if server_cfg.get("scrape_metrics") is not None
                else config.scrape_metrics
            )
            config.default_function_node_selector = (
                server_cfg.get("default_function_node_selector")
                or config.default_function_node_selector
            )
            config.igz_version = server_cfg.get("igz_version") or config.igz_version
            config.storage.auto_mount_type = (
                server_cfg.get("auto_mount_type") or config.storage.auto_mount_type
            )
            config.storage.auto_mount_params = (
                server_cfg.get("auto_mount_params") or config.storage.auto_mount_params
            )
            config.spark_operator_version = (
                server_cfg.get("spark_operator_version")
                or config.spark_operator_version
            )
            config.default_tensorboard_logs_path = (
                server_cfg.get("default_tensorboard_logs_path")
                or config.default_tensorboard_logs_path
            )
            config.default_function_pod_resources = (
                server_cfg.get("default_function_pod_resources")
                or config.default_function_pod_resources
            )
            config.function_defaults.preemption_mode = (
                server_cfg.get("default_preemption_mode")
                or config.function_defaults.preemption_mode
            )
            config.preemptible_nodes.node_selector = (
                server_cfg.get("preemptible_nodes_node_selector")
                or config.preemptible_nodes.node_selector
            )
            config.preemptible_nodes.tolerations = (
                server_cfg.get("preemptible_nodes_tolerations")
                or config.preemptible_nodes.tolerations
            )
            config.force_run_local = (
                server_cfg.get("force_run_local") or config.force_run_local
            )
            config.function = server_cfg.get("function") or config.function
            config.httpdb.logs = server_cfg.get("logs") or config.httpdb.logs
            config.model_endpoint_monitoring.store_type = (
                server_cfg.get("model_endpoint_monitoring_store_type")
                or config.model_endpoint_monitoring.store_type
            )
            config.model_endpoint_monitoring.endpoint_store_connection = (
                server_cfg.get("model_endpoint_monitoring_endpoint_store_connection")
                or config.model_endpoint_monitoring.endpoint_store_connection
            )
            config.packagers = server_cfg.get("packagers") or config.packagers
            server_data_prefixes = server_cfg.get("feature_store_data_prefixes") or {}
            for prefix in ["default", "nosql", "redisnosql"]:
                server_prefix_value = server_data_prefixes.get(prefix)
                if server_prefix_value is not None:
                    setattr(
                        config.feature_store.data_prefixes, prefix, server_prefix_value
                    )

        except Exception as exc:
            logger.warning(
                "Failed syncing config from server",
                exc=err_to_str(exc),
                traceback=traceback.format_exc(),
            )
        return self

    def store_log(self, uid, project="", body=None, append=False):
        """Save a log persistently.

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
        """Retrieve a log.

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
        """Retrieve logs of a running process, and watch the progress of the execution until it completes. This
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
            print(text.decode(errors=mlrun.mlconf.httpdb.logs.decode.errors))
        if watch:
            nil_resp = 0
            while state in ["pending", "running"]:
                offset += len(text)
                # if we get 3 nil responses in a row, increase the sleep time to 10 seconds
                # TODO: refactor this to use a conditional backoff mechanism
                if nil_resp < 3:
                    time.sleep(int(mlrun.mlconf.httpdb.logs.pull_logs_default_interval))
                else:
                    time.sleep(
                        int(
                            mlrun.mlconf.httpdb.logs.pull_logs_backoff_no_logs_default_interval
                        )
                    )
                state, text = self.get_log(uid, project, offset=offset)
                if text:
                    nil_resp = 0
                    print(
                        text.decode(errors=mlrun.mlconf.httpdb.logs.decode.errors),
                        end="",
                    )
                else:
                    nil_resp += 1
        else:
            offset += len(text)

        return state, offset

    def store_run(self, struct, uid, project="", iter=0):
        """Store run details in the DB. This method is usually called from within other :py:mod:`mlrun` flows
        and not called directly by the user."""

        path = self._path_of("run", project, uid)
        params = {"iter": iter}
        error = f"store run {project}/{uid}"
        body = _as_json(struct)
        self.api_call("POST", path, error, params=params, body=body)

    def update_run(self, updates: dict, uid, project="", iter=0, timeout=45):
        """Update the details of a stored run in the DB."""

        path = self._path_of("run", project, uid)
        params = {"iter": iter}
        error = f"update run {project}/{uid}"
        body = _as_json(updates)
        self.api_call("PATCH", path, error, params=params, body=body, timeout=timeout)

    def abort_run(self, uid, project="", iter=0, timeout=45):
        """
        Abort a running run - will remove the run's runtime resources and mark its state as aborted
        """
        self.update_run(
            {"status.state": mlrun.runtimes.constants.RunStates.aborted},
            uid,
            project,
            iter,
            timeout,
        )

    def read_run(self, uid, project="", iter=0):
        """Read the details of a stored run from the DB.

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
        """Delete details of a specific run from DB.

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
        name: Optional[str] = None,
        uid: Optional[Union[str, List[str]]] = None,
        project: Optional[str] = None,
        labels: Optional[Union[str, List[str]]] = None,
        state: Optional[str] = None,
        sort: bool = True,
        last: int = 0,
        iter: bool = False,
        start_time_from: Optional[datetime] = None,
        start_time_to: Optional[datetime] = None,
        last_update_time_from: Optional[datetime] = None,
        last_update_time_to: Optional[datetime] = None,
        partition_by: Optional[
            Union[mlrun.common.schemas.RunPartitionByField, str]
        ] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Optional[Union[mlrun.common.schemas.SortField, str]] = None,
        partition_order: Union[
            mlrun.common.schemas.OrderType, str
        ] = mlrun.common.schemas.OrderType.desc,
        max_partitions: int = 0,
        with_notifications: bool = False,
    ) -> RunList:
        """
        Retrieve a list of runs, filtered by various options.
        If no filter is provided, will return runs from the last week.

        Example::

            runs = db.list_runs(name='download', project='iris', labels=['owner=admin', 'kind=job'])
            # If running in Jupyter, can use the .show() function to display the results
            db.list_runs(name='', project=project_name).show()


        :param name: Name of the run to retrieve.
        :param uid: Unique ID of the run, or a list of run UIDs.
        :param project: Project that the runs belongs to.
        :param labels: List runs that have specific labels assigned. a single or multi label filter can be
            applied.
        :param state: List only runs whose state is specified.
        :param sort: Whether to sort the result according to their start time. Otherwise, results will be
            returned by their internal order in the DB (order will not be guaranteed).
        :param last: Deprecated - currently not used.
        :param iter: If ``True`` return runs from all iterations. Otherwise, return only runs whose ``iter`` is 0.
        :param start_time_from: Filter by run start time in ``[start_time_from, start_time_to]``.
        :param start_time_to: Filter by run start time in ``[start_time_from, start_time_to]``.
        :param last_update_time_from: Filter by run last update time in ``(last_update_time_from,
            last_update_time_to)``.
        :param last_update_time_to: Filter by run last update time in ``(last_update_time_from, last_update_time_to)``.
        :param partition_by: Field to group results by. Only allowed value is `name`. When `partition_by` is specified,
            the `partition_sort_by` parameter must be provided as well.
        :param rows_per_partition: How many top rows (per sorting defined by `partition_sort_by` and `partition_order`)
            to return per group. Default value is 1.
        :param partition_sort_by: What field to sort the results by, within each partition defined by `partition_by`.
            Currently the only allowed values are `created` and `updated`.
        :param partition_order: Order of sorting within partitions - `asc` or `desc`. Default is `desc`.
        :param max_partitions: Maximal number of partitions to include in the result. Default is `0` which means no
            limit.
        :param with_notifications: Return runs with notifications, and join them to the response. Default is `False`.
        """

        project = project or config.default_project

        if (
            not name
            and not uid
            and not labels
            and not state
            and not last
            and not start_time_from
            and not start_time_to
            and not last_update_time_from
            and not last_update_time_to
            and not partition_by
            and not partition_sort_by
            and not iter
        ):
            # default to last week on no filter
            start_time_from = datetime.now() - timedelta(days=7)
            partition_by = mlrun.common.schemas.RunPartitionByField.name
            partition_sort_by = mlrun.common.schemas.SortField.updated

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
            "with-notifications": with_notifications,
        }

        if partition_by:
            params.update(
                self._generate_partition_by_params(
                    mlrun.common.schemas.RunPartitionByField,
                    partition_by,
                    rows_per_partition,
                    partition_sort_by,
                    partition_order,
                    max_partitions,
                )
            )
        error = "list runs"
        resp = self.api_call("GET", "runs", error, params=params)
        return RunList(resp.json()["runs"])

    def del_runs(self, name=None, project=None, labels=None, state=None, days_ago=0):
        """Delete a group of runs identified by the parameters of the function.

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
        """Store an artifact in the DB.

        :param key: Identifying key of the artifact.
        :param artifact: The actual artifact to store.
        :param uid: A unique ID for this specific version of the artifact.
        :param iter: The task iteration which generated this artifact. If ``iter`` is not ``None`` the iteration will
            be added to the key provided to generate a unique key for the artifact of the specific iteration.
        :param tag: Tag of the artifact.
        :param project: Project that the artifact belongs to.
        """

        endpoint_path = f"projects/{project}/artifacts/{uid}/{key}"
        params = {
            "tag": tag,
        }
        if iter:
            params["iter"] = str(iter)

        error = f"store artifact {project}/{uid}/{key}"

        body = _as_json(artifact)
        self.api_call("POST", endpoint_path, error, params=params, body=body)

    def read_artifact(self, key, tag=None, iter=None, project=""):
        """Read an artifact, identified by its key, tag and iteration."""

        project = project or config.default_project
        tag = tag or "latest"
        endpoint_path = f"projects/{project}/artifacts/{key}?tag={tag}"
        error = f"read artifact {project}/{key}"
        # explicitly set artifacts format to 'full' since old servers may default to 'legacy'
        params = {"format": mlrun.common.schemas.ArtifactsFormat.full.value}
        if iter:
            params["iter"] = str(iter)
        resp = self.api_call("GET", endpoint_path, error, params=params)
        return resp.json()["data"]

    def del_artifact(self, key, tag=None, project=""):
        """Delete an artifact."""

        endpoint_path = f"projects/{project}/artifacts/{key}"
        params = {
            "key": key,
            "tag": tag,
        }
        error = f"del artifact {project}/{key}"
        self.api_call("DELETE", endpoint_path, error, params=params)

    def list_artifacts(
        self,
        name=None,
        project=None,
        tag=None,
        labels: Optional[Union[Dict[str, str], List[str]]] = None,
        since=None,
        until=None,
        iter: int = None,
        best_iteration: bool = False,
        kind: str = None,
        category: Union[str, mlrun.common.schemas.ArtifactCategories] = None,
    ) -> ArtifactList:
        """List artifacts filtered by various parameters.

        Examples::

            # Show latest version of all artifacts in project
            latest_artifacts = db.list_artifacts('', tag='latest', project='iris')
            # check different artifact versions for a specific artifact
            result_versions = db.list_artifacts('results', tag='*', project='iris')
            # Show artifacts with label filters - both uploaded and of binary type
            result_labels = db.list_artifacts('results', tag='*', project='iris', labels=['uploaded', 'type=binary'])

        :param name: Name of artifacts to retrieve. Name with '~' prefix is used as a like query, and is not
            case-sensitive. This means that querying for ``~name`` may return artifacts named
            ``my_Name_1`` or ``surname``.
        :param project: Project name.
        :param tag: Return artifacts assigned this tag.
        :param labels: Return artifacts that have these labels. Labels can either be a dictionary {"label": "value"} or
            a list of "label=value" (match label key and value) or "label" (match just label key) strings.
        :param since: Not in use in :py:class:`HTTPRunDB`.
        :param until: Not in use in :py:class:`HTTPRunDB`.
        :param iter: Return artifacts from a specific iteration (where ``iter=0`` means the root iteration). If
            ``None`` (default) return artifacts from all iterations.
        :param best_iteration: Returns the artifact which belongs to the best iteration of a given run, in the case of
            artifacts generated from a hyper-param run. If only a single iteration exists, will return the artifact
            from that iteration. If using ``best_iter``, the ``iter`` parameter must not be used.
        :param kind: Return artifacts of the requested kind.
        :param category: Return artifacts of the requested category.
        """

        project = project or config.default_project

        labels = labels or []
        if isinstance(labels, dict):
            labels = [f"{key}={value}" for key, value in labels.items()]

        params = {
            "name": name,
            "tag": tag,
            "label": labels,
            "iter": iter,
            "best-iteration": best_iteration,
            "kind": kind,
            "category": category,
            "format": mlrun.common.schemas.ArtifactsFormat.full.value,
        }
        error = "list artifacts"
        endpoint_path = f"projects/{project}/artifacts"
        resp = self.api_call("GET", endpoint_path, error, params=params)
        values = ArtifactList(resp.json()["artifacts"])
        values.tag = tag
        return values

    def del_artifacts(self, name=None, project=None, tag=None, labels=None, days_ago=0):
        """Delete artifacts referenced by the parameters.

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
            "tag": tag,
            "label": labels or [],
            "days_ago": str(days_ago),
        }
        error = "del artifacts"
        endpoint_path = f"projects/{project}/artifacts"
        self.api_call("DELETE", endpoint_path, error, params=params)

    def list_artifact_tags(
        self,
        project=None,
        category: Union[str, mlrun.common.schemas.ArtifactCategories] = None,
    ) -> List[str]:
        """Return a list of all the tags assigned to artifacts in the scope of the given project."""

        project = project or config.default_project
        error_message = f"Failed listing artifact tags. project={project}"
        params = {"category": category} if category else {}

        response = self.api_call(
            "GET", f"projects/{project}/artifact-tags", error_message, params=params
        )
        return response.json()["tags"]

    def store_function(
        self,
        function: typing.Union[mlrun.runtimes.BaseRuntime, dict],
        name,
        project="",
        tag=None,
        versioned=False,
    ):
        """Store a function object. Function is identified by its name and tag, and can be versioned."""
        name = mlrun.utils.normalize_name(name)
        if hasattr(function, "to_dict"):
            function = function.to_dict()

        params = {"tag": tag, "versioned": versioned}
        project = project or config.default_project
        path = f"projects/{project}/functions/{name}"

        error = f"store function {project}/{name}"
        resp = self.api_call(
            "POST", path, error, params=params, body=dict_to_json(function)
        )

        # hash key optional to be backwards compatible to API v<0.4.10 in which it wasn't in the response
        return resp.json().get("hash_key")

    def get_function(self, name, project="", tag=None, hash_key=""):
        """Retrieve details of a specific function, identified by its name and potentially a tag or function hash."""

        params = {"tag": tag, "hash_key": hash_key}
        project = project or config.default_project
        path = f"projects/{project}/functions/{name}"
        error = f"get function {project}/{name}"
        resp = self.api_call("GET", path, error, params=params)
        return resp.json()["func"]

    def delete_function(self, name: str, project: str = ""):
        """Delete a function belonging to a specific project."""

        project = project or config.default_project
        path = f"projects/{project}/functions/{name}"
        error_message = f"Failed deleting function {project}/{name}"
        self.api_call("DELETE", path, error_message)

    def list_functions(self, name=None, project=None, tag=None, labels=None):
        """Retrieve a list of functions, filtered by specific criteria.

        :param name: Return only functions with a specific name.
        :param project: Return functions belonging to this project. If not specified, the default project is used.
        :param tag: Return function versions with specific tags.
        :param labels: Return functions that have specific labels assigned to them.
        :returns: List of function objects (as dictionary).
        """
        project = project or config.default_project
        params = {
            "name": name,
            "tag": tag,
            "label": labels or [],
        }
        error = "list functions"
        path = f"projects/{project}/functions"
        resp = self.api_call("GET", path, error, params=params)
        return resp.json()["funcs"]

    def list_runtime_resources(
        self,
        project: Optional[str] = None,
        label_selector: Optional[str] = None,
        kind: Optional[str] = None,
        object_id: Optional[str] = None,
        group_by: Optional[
            mlrun.common.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ) -> Union[
        mlrun.common.schemas.RuntimeResourcesOutput,
        mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
        mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
    ]:
        """List current runtime resources, which are usually (but not limited to) Kubernetes pods or CRDs.
        Function applies for runs of type `['dask', 'job', 'spark', 'remote-spark', 'mpijob']`, and will return per
        runtime kind a list of the runtime resources (which may have already completed their execution).

        :param project: Get only runtime resources of a specific project, by default None, which will return only the
            projects you're authorized to see.
        :param label_selector: A label filter that will be passed to Kubernetes for filtering the results according
            to their labels.
        :param kind: The kind of runtime to query. May be one of `['dask', 'job', 'spark', 'remote-spark', 'mpijob']`
        :param object_id: The identifier of the mlrun object to query its runtime resources. for most function runtimes,
            runtime resources are per Run, for which the identifier is the Run's UID. For dask runtime, the runtime
            resources are per Function, for which the identifier is the Function's name.
        :param group_by: Object to group results by. Allowed values are `job` and `project`.
        """
        params = {
            "label-selector": label_selector,
            "group-by": group_by,
            "kind": kind,
            "object-id": object_id,
        }
        project_path = project if project else "*"
        error = "Failed listing runtime resources"
        response = self.api_call(
            "GET", f"projects/{project_path}/runtime-resources", error, params=params
        )
        if group_by is None:
            structured_list = [
                mlrun.common.schemas.KindRuntimeResources(**kind_runtime_resources)
                for kind_runtime_resources in response.json()
            ]
            return structured_list
        elif group_by == mlrun.common.schemas.ListRuntimeResourcesGroupByField.job:
            structured_dict = {}
            for project, job_runtime_resources_map in response.json().items():
                for job_id, runtime_resources in job_runtime_resources_map.items():
                    structured_dict.setdefault(project, {})[
                        job_id
                    ] = mlrun.common.schemas.RuntimeResources(**runtime_resources)
            return structured_dict
        elif group_by == mlrun.common.schemas.ListRuntimeResourcesGroupByField.project:
            structured_dict = {}
            for project, kind_runtime_resources_map in response.json().items():
                for kind, runtime_resources in kind_runtime_resources_map.items():
                    structured_dict.setdefault(project, {})[
                        kind
                    ] = mlrun.common.schemas.RuntimeResources(**runtime_resources)
            return structured_dict
        else:
            raise NotImplementedError(
                f"Provided group by field is not supported. group_by={group_by}"
            )

    def delete_runtime_resources(
        self,
        project: Optional[str] = None,
        label_selector: Optional[str] = None,
        kind: Optional[str] = None,
        object_id: Optional[str] = None,
        force: bool = False,
        grace_period: int = None,
    ) -> mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput:
        """Delete all runtime resources which are in terminal state.

        :param project: Delete only runtime resources of a specific project, by default None, which will delete only
            from the projects you're authorized to delete from.
        :param label_selector: Delete only runtime resources matching the label selector.
        :param kind: The kind of runtime to delete. May be one of `['dask', 'job', 'spark', 'remote-spark', 'mpijob']`
        :param object_id: The identifier of the mlrun object to delete its runtime resources. for most function
            runtimes, runtime resources are per Run, for which the identifier is the Run's UID. For dask runtime, the
            runtime resources are per Function, for which the identifier is the Function's name.
        :param force: Force deletion - delete the runtime resource even if it's not in terminal state or if the grace
            period didn't pass.
        :param grace_period: Grace period given to the runtime resource before they are actually removed, counted from
            the moment they moved to terminal state (defaults to mlrun.mlconf.runtime_resources_deletion_grace_period).

        :returns: :py:class:`~mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput` listing the runtime resources
            that were removed.
        """
        if grace_period is None:
            grace_period = config.runtime_resources_deletion_grace_period
            logger.info(
                "Using default grace period for runtime resources deletion",
                grace_period=grace_period,
            )

        params = {
            "label-selector": label_selector,
            "kind": kind,
            "object-id": object_id,
            "force": force,
            "grace-period": grace_period,
        }
        error = "Failed deleting runtime resources"
        project_path = project if project else "*"
        response = self.api_call(
            "DELETE",
            f"projects/{project_path}/runtime-resources",
            error,
            params=params,
        )
        structured_dict = {}
        for project, kind_runtime_resources_map in response.json().items():
            for kind, runtime_resources in kind_runtime_resources_map.items():
                structured_dict.setdefault(project, {})[
                    kind
                ] = mlrun.common.schemas.RuntimeResources(**runtime_resources)
        return structured_dict

    def create_schedule(
        self, project: str, schedule: mlrun.common.schemas.ScheduleInput
    ):
        """Create a new schedule on the given project. The details on the actual object to schedule as well as the
        schedule itself are within the schedule object provided.
        The :py:class:`~ScheduleCronTrigger` follows the guidelines in
        https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html.
        It also supports a :py:func:`~ScheduleCronTrigger.from_crontab` function that accepts a
        crontab-formatted string (see https://en.wikipedia.org/wiki/Cron for more information on the format and
        note that the 0 weekday is always monday).


        Example::

            from mlrun.common import schemas

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
        self, project: str, name: str, schedule: mlrun.common.schemas.ScheduleUpdate
    ):
        """Update an existing schedule, replace it with the details contained in the schedule object."""

        project = project or config.default_project
        path = f"projects/{project}/schedules/{name}"

        error_message = f"Failed updating schedule {project}/{name}"
        self.api_call("PUT", path, error_message, body=dict_to_json(schedule.dict()))

    def get_schedule(
        self, project: str, name: str, include_last_run: bool = False
    ) -> mlrun.common.schemas.ScheduleOutput:
        """Retrieve details of the schedule in question. Besides returning the details of the schedule object itself,
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
        return mlrun.common.schemas.ScheduleOutput(**resp.json())

    def list_schedules(
        self,
        project: str,
        name: str = None,
        kind: mlrun.common.schemas.ScheduleKinds = None,
        include_last_run: bool = False,
    ) -> mlrun.common.schemas.SchedulesOutput:
        """Retrieve list of schedules of specific name or kind.

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
        return mlrun.common.schemas.SchedulesOutput(**resp.json())

    def delete_schedule(self, project: str, name: str):
        """Delete a specific schedule by name."""

        project = project or config.default_project
        path = f"projects/{project}/schedules/{name}"
        error_message = f"Failed deleting schedule {project}/{name}"
        self.api_call("DELETE", path, error_message)

    def invoke_schedule(self, project: str, name: str):
        """Execute the object referenced by the schedule immediately."""

        project = project or config.default_project
        path = f"projects/{project}/schedules/{name}/invoke"
        error_message = f"Failed invoking schedule {project}/{name}"
        self.api_call("POST", path, error_message)

    def remote_builder(
        self,
        func,
        with_mlrun,
        mlrun_version_specifier=None,
        skip_deployed=False,
        builder_env=None,
    ):
        """Build the pod image for a function, for execution on a remote cluster. This is executed by the MLRun
        API server, and creates a Docker image out of the function provided and any specific build
        instructions provided within. This is a pre-requisite for remotely executing a function, unless using
        a pre-deployed image.

        :param func: Function to build.
        :param with_mlrun: Whether to add MLRun package to the built package. This is not required if using a base
            image that already has MLRun in it.
        :param mlrun_version_specifier: Version of MLRun to include in the built image.
        :param skip_deployed: Skip the build if we already have an image for the function.
        :param builder_env:   Kaniko builder pod env vars dict (for config/credentials)
        """

        try:
            req = {
                "function": func.to_dict(),
                "with_mlrun": bool2str(with_mlrun),
                "skip_deployed": skip_deployed,
            }
            if mlrun_version_specifier:
                req["mlrun_version_specifier"] = mlrun_version_specifier
            if builder_env:
                req["builder_env"] = builder_env
            resp = self.api_call("POST", "build/function", json=req)
        except OSError as err:
            logger.error(f"error submitting build task: {err_to_str(err)}")
            raise OSError(f"error: cannot submit build, {err_to_str(err)}")

        if not resp.ok:
            logger.error(f"bad resp!!\n{resp.text}")
            raise ValueError("bad function run response")

        return resp.json()

    def get_builder_status(
        self,
        func: BaseRuntime,
        offset: int = 0,
        logs: bool = True,
        last_log_timestamp: float = 0.0,
        verbose: bool = False,
    ):
        """Retrieve the status of a build operation currently in progress.

        :param func:                Function object that is being built.
        :param offset:              Offset into the build logs to retrieve logs from.
        :param logs:                Should build logs be retrieved.
        :param last_log_timestamp:  Last timestamp of logs that were already retrieved. Function will return only logs
                                    later than this parameter.
        :param verbose:             Add verbose logs into the output.

        :returns: The following parameters:

            - Text of builder logs.
            - Timestamp of last log retrieved, to be used in subsequent calls to this function.

            The function also updates internal members of the ``func`` object to reflect build process info.
        """

        try:
            params = {
                "name": normalize_name(func.metadata.name),
                "project": func.metadata.project,
                "tag": func.metadata.tag,
                "logs": bool2str(logs),
                "offset": str(offset),
                "last_log_timestamp": str(last_log_timestamp),
                "verbose": bool2str(verbose),
            }
            resp = self.api_call("GET", "build/status", params=params)
        except OSError as err:
            logger.error(f"error getting build status: {err_to_str(err)}")
            raise OSError(f"error: cannot get build status, {err_to_str(err)}")

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
                func.status.internal_invocation_urls = resp.headers.get(
                    "x-mlrun-internal-invocation-urls", ""
                ).split(",")
                func.status.external_invocation_urls = resp.headers.get(
                    "x-mlrun-external-invocation-urls", ""
                ).split(",")
                func.status.container_image = resp.headers.get(
                    "x-mlrun-container-image", ""
                )
            else:
                func.status.build_pod = resp.headers.get("builder_pod", "")
                func.spec.image = resp.headers.get("function_image", "")

        text = ""
        if resp.content:
            text = resp.content.decode()
        return text, last_log_timestamp

    def start_function(
        self, func_url: str = None, function: "mlrun.runtimes.BaseRuntime" = None
    ) -> mlrun.common.schemas.BackgroundTask:
        """Execute a function remotely, Used for ``dask`` functions.

        :param func_url: URL to the function to be executed.
        :param function: The function object to start, not needed here.
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
            logger.error(f"error starting function: {err_to_str(err)}")
            raise OSError(f"error: cannot start function, {err_to_str(err)}")

        if not resp.ok:
            logger.error(f"bad resp!!\n{resp.text}")
            raise ValueError("bad function start response")

        return mlrun.common.schemas.BackgroundTask(**resp.json())

    def get_project_background_task(
        self,
        project: str,
        name: str,
    ) -> mlrun.common.schemas.BackgroundTask:
        """Retrieve updated information on a project background task being executed."""

        project = project or config.default_project
        path = f"projects/{project}/background-tasks/{name}"
        error_message = (
            f"Failed getting project background task. project={project}, name={name}"
        )
        response = self.api_call("GET", path, error_message)
        return mlrun.common.schemas.BackgroundTask(**response.json())

    def get_background_task(self, name: str) -> mlrun.common.schemas.BackgroundTask:
        """Retrieve updated information on a background task being executed."""

        path = f"background-tasks/{name}"
        error_message = f"Failed getting background task. name={name}"
        response = self.api_call("GET", path, error_message)
        return mlrun.common.schemas.BackgroundTask(**response.json())

    def function_status(self, project, name, kind, selector):
        """Retrieve status of a function being executed remotely (relevant to ``dask`` functions).

        :param project:     The project of the function
        :param name:        The name of the function
        :param kind:        The kind of the function, currently ``dask`` is supported.
        :param selector:    Selector clause to be applied to the Kubernetes status query to filter the results.
        """

        try:
            req = {"kind": kind, "selector": selector, "project": project, "name": name}
            resp = self.api_call("POST", "status/function", json=req)
        except OSError as err:
            logger.error(f"error starting function: {err_to_str(err)}")
            raise OSError(f"error: cannot start function, {err_to_str(err)}")

        if not resp.ok:
            logger.error(f"bad resp!!\n{resp.text}")
            raise ValueError("bad function status response")

        return resp.json()["data"]

    def submit_job(
        self,
        runspec,
        schedule: Union[str, mlrun.common.schemas.ScheduleCronTrigger] = None,
    ):
        """Submit a job for remote execution.

        :param runspec: The runtime object spec (Task) to execute.
        :param schedule: Whether to schedule this job using a Cron trigger. If not specified, the job will be submitted
            immediately.
        """

        try:
            req = {"task": runspec.to_dict()}
            if schedule:
                if isinstance(schedule, mlrun.common.schemas.ScheduleCronTrigger):
                    schedule = schedule.dict()
                req["schedule"] = schedule
            timeout = (int(config.submit_timeout) or 120) + 20
            resp = self.api_call("POST", "submit_job", json=req, timeout=timeout)

        except requests.HTTPError as err:
            logger.error(f"error submitting task: {err_to_str(err)}")
            # not creating a new exception here, in order to keep the response and status code in the exception
            raise

        except OSError as err:
            logger.error(f"error submitting task: {err_to_str(err)}")
            raise OSError("error: cannot submit task") from err

        if not resp.ok:
            logger.error(f"bad resp!!\n{resp.text}")
            raise ValueError(f"bad function run response, {resp.text}")

        resp = resp.json()
        return resp["data"]

    def submit_pipeline(
        self,
        project,
        pipeline,
        arguments=None,
        experiment=None,
        run=None,
        namespace=None,
        artifact_path=None,
        ops=None,
        # TODO: deprecated, remove in 1.6.0
        ttl=None,
        cleanup_ttl=None,
    ):
        """Submit a KFP pipeline for execution.

        :param project: The project of the pipeline
        :param pipeline: Pipeline function or path to .yaml/.zip pipeline file.
        :param arguments: A dictionary of arguments to pass to the pipeline.
        :param experiment: A name to assign for the specific experiment.
        :param run: A name for this specific run.
        :param namespace: Kubernetes namespace to execute the pipeline in.
        :param artifact_path: A path to artifacts used by this pipeline.
        :param ops: Transformers to apply on all ops in the pipeline.
        :param ttl: pipeline cleanup ttl in secs (time to wait after workflow completion, at which point the workflow
            and all its resources are deleted) (deprecated, use cleanup_ttl instead)
        :param cleanup_ttl: pipeline cleanup ttl in secs (time to wait after workflow completion, at which point the
                            workflow and all its resources are deleted)
        """
        if ttl:
            warnings.warn(
                "'ttl' is deprecated, use 'cleanup_ttl' instead. "
                "This will be removed in 1.6.0",
                # TODO: Remove this in 1.6.0
                FutureWarning,
            )

        if isinstance(pipeline, str):
            pipe_file = pipeline
        else:
            pipe_file = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False).name
            conf = new_pipe_metadata(
                artifact_path=artifact_path,
                cleanup_ttl=cleanup_ttl or ttl,
                op_transformers=ops,
            )
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
            headers[mlrun.common.schemas.HeaderNames.pipeline_arguments] = str(
                arguments
            )

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
                f"projects/{project}/pipelines",
                params=params,
                timeout=20,
                body=data,
                headers=headers,
            )
        except OSError as err:
            logger.error("Error: Cannot submit pipeline", err=err_to_str(err))
            raise OSError(f"Error: Cannot submit pipeline, {err_to_str(err)}")

        if not resp.ok:
            logger.error("Failed to submit pipeline", respones_text=resp.text)
            raise ValueError(f"Failed to submit pipeline, {resp.text}")

        resp = resp.json()
        logger.info(
            "Pipeline submitted successfully", pipeline_name=resp["name"], id=resp["id"]
        )
        return resp["id"]

    def list_pipelines(
        self,
        project: str,
        namespace: str = None,
        sort_by: str = "",
        page_token: str = "",
        filter_: str = "",
        format_: Union[
            str, mlrun.common.schemas.PipelinesFormat
        ] = mlrun.common.schemas.PipelinesFormat.metadata_only,
        page_size: int = None,
    ) -> mlrun.common.schemas.PipelinesOutput:
        """Retrieve a list of KFP pipelines. This function can be invoked to get all pipelines from all projects,
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

        if project != "*" and (page_token or page_size):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Filtering by project can not be used together with pagination"
            )
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
        return mlrun.common.schemas.PipelinesOutput(**response.json())

    def get_pipeline(
        self,
        run_id: str,
        namespace: str = None,
        timeout: int = 30,
        format_: Union[
            str, mlrun.common.schemas.PipelinesFormat
        ] = mlrun.common.schemas.PipelinesFormat.summary,
        project: str = None,
    ):
        """Retrieve details of a specific pipeline using its run ID (as provided when the pipeline was executed)."""

        params = {}
        if namespace:
            params["namespace"] = namespace
        params["format"] = format_
        project_path = project if project else "*"
        resp = self.api_call(
            "GET",
            f"projects/{project_path}/pipelines/{run_id}",
            params=params,
            timeout=timeout,
        )

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
        self,
        feature_set: Union[dict, mlrun.common.schemas.FeatureSet, FeatureSet],
        project="",
        versioned=True,
    ) -> dict:
        """Create a new :py:class:`~mlrun.feature_store.FeatureSet` and save in the :py:mod:`mlrun` DB. The
        feature-set must not previously exist in the DB.

        :param feature_set: The new :py:class:`~mlrun.feature_store.FeatureSet` to create.
        :param project: Name of project this feature-set belongs to.
        :param versioned: Whether to maintain versions for this feature-set. All versions of a versioned object
            will be kept in the DB and can be retrieved until explicitly deleted.
        :returns: The :py:class:`~mlrun.feature_store.FeatureSet` object (as dict).
        """
        if isinstance(feature_set, mlrun.common.schemas.FeatureSet):
            feature_set = feature_set.dict()
        elif isinstance(feature_set, FeatureSet):
            feature_set = feature_set.to_dict()

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
            "POST",
            path,
            error_message,
            params=params,
            body=dict_to_json(feature_set),
        )
        return resp.json()

    def get_feature_set(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ) -> FeatureSet:
        """Retrieve a ~mlrun.feature_store.FeatureSet` object. If both ``tag`` and ``uid`` are not specified, then
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
        """List feature-sets which contain specific features. This function may return multiple versions of the same
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
        self,
        project: str,
        name: str = None,
        tag: str = None,
        labels: List[str] = None,
    ) -> List[dict]:
        """Retrieve a list of entities and their mapping to the containing feature-sets. This function is similar
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
    def _generate_partition_by_params(
        partition_by_cls,
        partition_by,
        rows_per_partition,
        sort_by,
        order,
        max_partitions=None,
    ):
        partition_params = {
            "partition-by": partition_by,
            "rows-per-partition": rows_per_partition,
            "partition-sort-by": sort_by,
            "partition-order": order,
        }
        if max_partitions is not None:
            partition_params["max-partitions"] = max_partitions
        return partition_params

    def list_feature_sets(
        self,
        project: str = "",
        name: str = None,
        tag: str = None,
        state: str = None,
        entities: List[str] = None,
        features: List[str] = None,
        labels: List[str] = None,
        partition_by: Union[
            mlrun.common.schemas.FeatureStorePartitionByField, str
        ] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Union[mlrun.common.schemas.SortField, str] = None,
        partition_order: Union[
            mlrun.common.schemas.OrderType, str
        ] = mlrun.common.schemas.OrderType.desc,
    ) -> List[FeatureSet]:
        """Retrieve a list of feature-sets matching the criteria provided.

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
            Currently the only allowed value are `created` and `updated`.
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
                    mlrun.common.schemas.FeatureStorePartitionByField,
                    partition_by,
                    rows_per_partition,
                    partition_sort_by,
                    partition_order,
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
        feature_set: Union[dict, mlrun.common.schemas.FeatureSet, FeatureSet],
        name=None,
        project="",
        tag=None,
        uid=None,
        versioned=True,
    ) -> dict:
        """Save a :py:class:`~mlrun.feature_store.FeatureSet` object in the :py:mod:`mlrun` DB. The
        feature-set can be either a new object or a modification to existing object referenced by the params of
        the function.

        :param feature_set: The :py:class:`~mlrun.feature_store.FeatureSet` to store.
        :param name:    Name of feature set.
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

        if isinstance(feature_set, mlrun.common.schemas.FeatureSet):
            feature_set = feature_set.dict()
        elif isinstance(feature_set, FeatureSet):
            feature_set = feature_set.to_dict()

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
        patch_mode: Union[
            str, mlrun.common.schemas.PatchMode
        ] = mlrun.common.schemas.PatchMode.replace,
    ):
        """Modify (patch) an existing :py:class:`~mlrun.feature_store.FeatureSet` object.
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
        headers = {mlrun.common.schemas.HeaderNames.patch_mode: patch_mode}
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
        """Delete a :py:class:`~mlrun.feature_store.FeatureSet` object from the DB.
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
        feature_vector: Union[dict, mlrun.common.schemas.FeatureVector, FeatureVector],
        project="",
        versioned=True,
    ) -> dict:
        """Create a new :py:class:`~mlrun.feature_store.FeatureVector` and save in the :py:mod:`mlrun` DB.

        :param feature_vector: The new :py:class:`~mlrun.feature_store.FeatureVector` to create.
        :param project: Name of project this feature-vector belongs to.
        :param versioned: Whether to maintain versions for this feature-vector. All versions of a versioned object
            will be kept in the DB and can be retrieved until explicitly deleted.
        :returns: The :py:class:`~mlrun.feature_store.FeatureVector` object (as dict).
        """
        if isinstance(feature_vector, mlrun.common.schemas.FeatureVector):
            feature_vector = feature_vector.dict()
        elif isinstance(feature_vector, FeatureVector):
            feature_vector = feature_vector.to_dict()

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
        """Return a specific feature-vector referenced by its tag or uid. If none are provided, ``latest`` tag will
        be used."""

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
        partition_by: Union[
            mlrun.common.schemas.FeatureStorePartitionByField, str
        ] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Union[mlrun.common.schemas.SortField, str] = None,
        partition_order: Union[
            mlrun.common.schemas.OrderType, str
        ] = mlrun.common.schemas.OrderType.desc,
    ) -> List[FeatureVector]:
        """Retrieve a list of feature-vectors matching the criteria provided.

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
            Currently the only allowed values are `created` and `updated`.
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
                    mlrun.common.schemas.FeatureStorePartitionByField,
                    partition_by,
                    rows_per_partition,
                    partition_sort_by,
                    partition_order,
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
        feature_vector: Union[dict, mlrun.common.schemas.FeatureVector, FeatureVector],
        name=None,
        project="",
        tag=None,
        uid=None,
        versioned=True,
    ) -> dict:
        """Store a :py:class:`~mlrun.feature_store.FeatureVector` object in the :py:mod:`mlrun` DB. The
        feature-vector can be either a new object or a modification to existing object referenced by the params
        of the function.

        :param feature_vector: The :py:class:`~mlrun.feature_store.FeatureVector` to store.
        :param name:    Name of feature vector.
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

        if isinstance(feature_vector, mlrun.common.schemas.FeatureVector):
            feature_vector = feature_vector.dict()
        elif isinstance(feature_vector, FeatureVector):
            feature_vector = feature_vector.to_dict()

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
        patch_mode: Union[
            str, mlrun.common.schemas.PatchMode
        ] = mlrun.common.schemas.PatchMode.replace,
    ):
        """Modify (patch) an existing :py:class:`~mlrun.feature_store.FeatureVector` object.
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
        headers = {mlrun.common.schemas.HeaderNames.patch_mode: patch_mode}
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
        """Delete a :py:class:`~mlrun.feature_store.FeatureVector` object from the DB.
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

    def tag_objects(
        self,
        project: str,
        tag_name: str,
        objects: Union[mlrun.common.schemas.TagObjects, dict],
        replace: bool = False,
    ):
        """Tag a list of objects.

        :param project: Project which contains the objects.
        :param tag_name: The tag to set on the objects.
        :param objects: The objects to tag.
        :param replace: Whether to replace the existing tags of the objects or to add the new tag to them.
        """

        path = f"projects/{project}/tags/{tag_name}"
        error_message = f"Failed to tag {tag_name} on objects {objects}"
        method = "POST" if replace else "PUT"
        self.api_call(
            method,
            path,
            error_message,
            body=dict_to_json(
                objects.dict()
                if isinstance(objects, mlrun.common.schemas.TagObjects)
                else objects
            ),
        )

    def delete_objects_tag(
        self,
        project: str,
        tag_name: str,
        tag_objects: Union[mlrun.common.schemas.TagObjects, dict],
    ):
        """Delete a tag from a list of objects.

        :param project: Project which contains the objects.
        :param tag_name: The tag to delete from the objects.
        :param tag_objects: The objects to delete the tag from.

        """
        path = f"projects/{project}/tags/{tag_name}"
        error_message = f"Failed deleting tag from {tag_name}"
        self.api_call(
            "DELETE",
            path,
            error_message,
            body=dict_to_json(
                tag_objects.dict()
                if isinstance(tag_objects, mlrun.common.schemas.TagObjects)
                else tag_objects
            ),
        )

    def tag_artifacts(
        self,
        artifacts: Union[List[Artifact], List[dict], Artifact, dict],
        project: str,
        tag_name: str,
        replace: bool = False,
    ):
        """Tag a list of artifacts.

        :param artifacts: The artifacts to tag. Can be a list of :py:class:`~mlrun.artifacts.Artifact` objects or
            dictionaries, or a single object.
        :param project: Project which contains the artifacts.
        :param tag_name: The tag to set on the artifacts.
        :param replace: If True, replace existing tags, otherwise append to existing tags.
        """
        tag_objects = self._resolve_artifacts_to_tag_objects(artifacts)
        self.tag_objects(project, tag_name, objects=tag_objects, replace=replace)

    def delete_artifacts_tags(
        self,
        artifacts,
        project: str,
        tag_name: str,
    ):
        """Delete tag from a list of artifacts.

        :param artifacts: The artifacts to delete the tag from. Can be a list of :py:class:`~mlrun.artifacts.Artifact`
            objects or dictionaries, or a single object.
        :param project: Project which contains the artifacts.
        :param tag_name: The tag to set on the artifacts.
        """
        tag_objects = self._resolve_artifacts_to_tag_objects(artifacts)
        self.delete_objects_tag(project, tag_name, tag_objects)

    def list_projects(
        self,
        owner: str = None,
        format_: Union[
            str, mlrun.common.schemas.ProjectsFormat
        ] = mlrun.common.schemas.ProjectsFormat.name_only,
        labels: List[str] = None,
        state: Union[str, mlrun.common.schemas.ProjectState] = None,
    ) -> List[Union[mlrun.projects.MlrunProject, str]]:
        """Return a list of the existing projects, potentially filtered by specific criteria.

        :param owner: List only projects belonging to this specific owner.
        :param format_: Format of the results. Possible values are:

            - ``name_only`` (default value) - Return just the names of the projects.
            - ``minimal`` - Return minimal project objects (minimization happens in the BE).
            - ``full``  - Return full project objects.

        :param labels: Filter by labels attached to the project.
        :param state: Filter by project's state. Can be either ``online`` or ``archived``.
        """

        params = {
            "owner": owner,
            "state": state,
            "format": format_,
            "label": labels or [],
        }

        error_message = f"Failed listing projects, query: {params}"
        response = self.api_call("GET", "projects", error_message, params=params)
        if format_ == mlrun.common.schemas.ProjectsFormat.name_only:
            # projects is just a list of strings
            return response.json()["projects"]

        # forwards compatibility - we want to be able to handle new formats that might be added in the future
        # if format is not known to the api, it is up to the server to return either an error or a default format
        return [
            mlrun.projects.MlrunProject.from_dict(project_dict)
            for project_dict in response.json()["projects"]
        ]

    def get_project(self, name: str) -> mlrun.projects.MlrunProject:
        """Get details for a specific project."""

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
            str, mlrun.common.schemas.DeletionStrategy
        ] = mlrun.common.schemas.DeletionStrategy.default(),
    ):
        """Delete a project.

        :param name: Name of the project to delete.
        :param deletion_strategy: How to treat child objects of the project. Possible values are:

            - ``restrict`` (default) - Project must not have any child objects when deleted. If using this mode while
              child objects exist, the operation will fail.
            - ``cascade`` - Automatically delete all child objects when deleting the project.
        """

        path = f"projects/{name}?wait-for-completion=false"
        headers = {
            mlrun.common.schemas.HeaderNames.deletion_strategy: deletion_strategy
        }
        error_message = f"Failed deleting project {name}"
        response = self.api_call("DELETE", path, error_message, headers=headers)
        if response.status_code == http.HTTPStatus.ACCEPTED:
            logger.info("Project is being deleted", project_name=name)
            self._wait_for_project_to_be_deleted(name)
        logger.info("Project deleted", project_name=name)

    def store_project(
        self,
        name: str,
        project: Union[dict, mlrun.projects.MlrunProject, mlrun.common.schemas.Project],
    ) -> mlrun.projects.MlrunProject:
        """Store a project in the DB. This operation will overwrite existing project of the same name if exists."""

        path = f"projects/{name}"
        error_message = f"Failed storing project {name}"
        if isinstance(project, mlrun.common.schemas.Project):
            project = project.dict()
        elif isinstance(project, mlrun.projects.MlrunProject):
            project = project.to_dict()
        response = self.api_call(
            "PUT",
            path,
            error_message,
            body=dict_to_json(project),
        )
        if response.status_code == http.HTTPStatus.ACCEPTED:
            return self._wait_for_project_to_reach_terminal_state(name)
        return mlrun.projects.MlrunProject.from_dict(response.json())

    def patch_project(
        self,
        name: str,
        project: dict,
        patch_mode: Union[
            str, mlrun.common.schemas.PatchMode
        ] = mlrun.common.schemas.PatchMode.replace,
    ) -> mlrun.projects.MlrunProject:
        """Patch an existing project object.

        :param name: Name of project to patch.
        :param project: The actual changes to the project object.
        :param patch_mode: The strategy for merging the changes with the existing object. Can be either ``replace``
            or ``additive``.
        """

        path = f"projects/{name}"
        headers = {mlrun.common.schemas.HeaderNames.patch_mode: patch_mode}
        error_message = f"Failed patching project {name}"
        response = self.api_call(
            "PATCH", path, error_message, body=dict_to_json(project), headers=headers
        )
        return mlrun.projects.MlrunProject.from_dict(response.json())

    def create_project(
        self,
        project: Union[dict, mlrun.projects.MlrunProject, mlrun.common.schemas.Project],
    ) -> mlrun.projects.MlrunProject:
        """Create a new project. A project with the same name must not exist prior to creation."""

        if isinstance(project, mlrun.common.schemas.Project):
            project = project.dict()
        elif isinstance(project, mlrun.projects.MlrunProject):
            project = project.to_dict()
        project_name = project["metadata"]["name"]
        error_message = f"Failed creating project {project_name}"
        response = self.api_call(
            "POST",
            # do not wait for project to reach terminal state synchronously.
            # let it start and wait for it to reach terminal state asynchronously
            "projects?wait-for-completion=false",
            error_message,
            body=dict_to_json(project),
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
                not in mlrun.common.schemas.ProjectState.terminal_states()
            ):
                raise Exception(
                    f"Project not in terminal state. State: {project.status.state}"
                )
            return project

        return mlrun.utils.helpers.retry_until_successful(
            self._wait_for_project_terminal_state_retry_interval,
            180,
            logger,
            False,
            _verify_project_in_terminal_state,
        )

    def _wait_for_background_task_to_reach_terminal_state(
        self, name: str
    ) -> mlrun.common.schemas.BackgroundTask:
        def _verify_background_task_in_terminal_state():
            background_task = self.get_background_task(name)
            state = background_task.status.state
            if state not in mlrun.common.schemas.BackgroundTaskState.terminal_states():
                raise Exception(
                    f"Background task not in terminal state. name={name}, state={state}"
                )
            return background_task

        return mlrun.utils.helpers.retry_until_successful(
            self._wait_for_background_task_terminal_state_retry_interval,
            60 * 60,
            logger,
            False,
            _verify_background_task_in_terminal_state,
        )

    def _wait_for_project_to_be_deleted(self, project_name: str):
        def _verify_project_deleted():
            projects = self.list_projects(
                format_=mlrun.common.schemas.ProjectsFormat.name_only
            )
            if project_name in projects:
                raise Exception(f"Project {project_name} still exists")

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
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: dict = None,
    ):
        """Create project-context secrets using either ``vault`` or ``kubernetes`` provider.
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
            :py:class:`~mlrun.common.schemas.secret.SecretProviderName` enum.
        :param secrets: A set of secret values to store.
            Example::

                secrets = {'password': 'myPassw0rd', 'aws_key': '111222333'}
                db.create_project_secrets(
                    "project1",
                    provider=mlrun.common.schemas.SecretProviderName.kubernetes,
                    secrets=secrets
                )
        """
        path = f"projects/{project}/secrets"
        secrets_input = mlrun.common.schemas.SecretsData(
            secrets=secrets, provider=provider
        )
        body = secrets_input.dict()
        error_message = f"Failed creating secret provider {project}/{provider}"
        self.api_call(
            "POST",
            path,
            error_message,
            body=dict_to_json(body),
        )

    def list_project_secrets(
        self,
        project: str,
        token: str = None,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: List[str] = None,
    ) -> mlrun.common.schemas.SecretsData:
        """Retrieve project-context secrets from Vault.

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

        if (
            provider == mlrun.common.schemas.SecretProviderName.vault.value
            and not token
        ):
            raise MLRunInvalidArgumentError(
                "A vault token must be provided when accessing vault secrets"
            )

        path = f"projects/{project}/secrets"
        params = {"provider": provider, "secret": secrets}
        headers = {mlrun.common.schemas.HeaderNames.secret_store_token: token}
        error_message = f"Failed retrieving secrets {project}/{provider}"
        result = self.api_call(
            "GET",
            path,
            error_message,
            params=params,
            headers=headers,
        )
        return mlrun.common.schemas.SecretsData(**result.json())

    def list_project_secret_keys(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        token: str = None,
    ) -> mlrun.common.schemas.SecretKeysData:
        """Retrieve project-context secret keys from Vault or Kubernetes.

        Note:
                This method for Vault functionality is currently in technical preview, and requires a HashiCorp Vault
                infrastructure properly set up and connected to the MLRun API server.

        :param project: The project name.
        :param provider: The name of the secrets-provider to work with. Accepts a
            :py:class:`~mlrun.common.schemas.secret.SecretProviderName` enum.
        :param token: Vault token to use for retrieving secrets. Only in use if ``provider`` is ``vault``.
            Must be a valid Vault token, with permissions to retrieve secrets of the project in question.
        """

        if (
            provider == mlrun.common.schemas.SecretProviderName.vault.value
            and not token
        ):
            raise MLRunInvalidArgumentError(
                "A vault token must be provided when accessing vault secrets"
            )

        path = f"projects/{project}/secret-keys"
        params = {"provider": provider}
        headers = (
            {mlrun.common.schemas.HeaderNames.secret_store_token: token}
            if provider == mlrun.common.schemas.SecretProviderName.vault.value
            else None
        )
        error_message = f"Failed retrieving secret keys {project}/{provider}"
        result = self.api_call(
            "GET",
            path,
            error_message,
            params=params,
            headers=headers,
        )
        return mlrun.common.schemas.SecretKeysData(**result.json())

    def delete_project_secrets(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: List[str] = None,
    ):
        """Delete project-context secrets from Kubernetes.

        :param project: The project name.
        :param provider: The name of the secrets-provider to work with. Currently only ``kubernetes`` is supported.
        :param secrets: A list of secret names to delete. An empty list will delete all secrets assigned
            to this specific project.
        """

        path = f"projects/{project}/secrets"
        params = {"provider": provider, "secret": secrets}
        error_message = f"Failed deleting secrets {project}/{provider}"
        self.api_call(
            "DELETE",
            path,
            error_message,
            params=params,
        )

    def create_user_secrets(
        self,
        user: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.vault,
        secrets: dict = None,
    ):
        """Create user-context secret in Vault. Please refer to :py:func:`create_project_secrets` for more details
        and status of this functionality.

        Note:
                This method is currently in technical preview, and requires a HashiCorp Vault infrastructure
                properly set up and connected to the MLRun API server.

        :param user: The user context for which to generate the infra and store secrets.
        :param provider: The name of the secrets-provider to work with. Currently only ``vault`` is supported.
        :param secrets: A set of secret values to store within the Vault.
        """
        path = "user-secrets"
        secrets_creation_request = mlrun.common.schemas.UserSecretCreationRequest(
            user=user,
            provider=provider,
            secrets=secrets,
        )
        body = secrets_creation_request.dict()
        error_message = f"Failed creating user secrets - {user}"
        self.api_call(
            "POST",
            path,
            error_message,
            body=dict_to_json(body),
        )

    @staticmethod
    def _validate_version_compatibility(server_version, client_version) -> bool:
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
            return True
        if (parsed_server_version.major == 0 and parsed_server_version.minor == 0) or (
            parsed_client_version.major == 0 and parsed_client_version.minor == 0
        ):
            logger.warning(
                "Server or client version is unstable. Assuming compatible",
                server_version=server_version,
                client_version=client_version,
            )
            return True
        if parsed_server_version.major != parsed_client_version.major:
            logger.warning(
                "Server and client versions are incompatible",
                parsed_server_version=parsed_server_version,
                parsed_client_version=parsed_client_version,
            )
            return False
        if parsed_server_version.minor > parsed_client_version.minor + 2:
            logger.info(
                "Backwards compatibility might not apply between the server and client version",
                parsed_server_version=parsed_server_version,
                parsed_client_version=parsed_client_version,
            )
            return False
        if parsed_client_version.minor > parsed_server_version.minor:
            logger.warning(
                "Client version with higher version than server version isn't supported,"
                " align your client to the server version",
                parsed_server_version=parsed_server_version,
                parsed_client_version=parsed_client_version,
            )
            return False
        if parsed_server_version.minor != parsed_client_version.minor:
            logger.info(
                "Server and client versions are not the same but compatible",
                parsed_server_version=parsed_server_version,
                parsed_client_version=parsed_client_version,
            )
        return True

    def create_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        model_endpoint: Union[
            mlrun.model_monitoring.model_endpoint.ModelEndpoint, dict
        ],
    ):
        """
        Creates a DB record with the given model_endpoint record.

        :param project: The name of the project.
        :param endpoint_id: The id of the endpoint.
        :param model_endpoint: An object representing the model endpoint.
        """

        if isinstance(
            model_endpoint, mlrun.model_monitoring.model_endpoint.ModelEndpoint
        ):
            model_endpoint = model_endpoint.to_dict()

        path = f"projects/{project}/model-endpoints/{endpoint_id}"
        self.api_call(
            method="POST",
            path=path,
            body=dict_to_json(model_endpoint),
        )

    def delete_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
    ):
        """
        Deletes the DB record of a given model endpoint, project and endpoint_id are used for lookup

        :param project: The name of the project
        :param endpoint_id: The id of the endpoint
        """

        path = f"projects/{project}/model-endpoints/{endpoint_id}"
        self.api_call(
            method="DELETE",
            path=path,
        )

    def list_model_endpoints(
        self,
        project: str,
        model: Optional[str] = None,
        function: Optional[str] = None,
        labels: List[str] = None,
        start: str = "now-1h",
        end: str = "now",
        metrics: Optional[List[str]] = None,
        top_level: bool = False,
        uids: Optional[List[str]] = None,
    ) -> List[mlrun.model_monitoring.model_endpoint.ModelEndpoint]:
        """
        Returns a list of `ModelEndpoint` objects. Each `ModelEndpoint` object represents the current state of a
        model endpoint. This functions supports filtering by the following parameters:
        1) model
        2) function
        3) labels
        4) top level
        5) uids
        By default, when no filters are applied, all available endpoints for the given project will be listed.

        In addition, this functions provides a facade for listing endpoint related metrics. This facade is time-based
        and depends on the 'start' and 'end' parameters. By default, when the metrics parameter is None, no metrics are
        added to the output of this function.

        :param project: The name of the project
        :param model: The name of the model to filter by
        :param function: The name of the function to filter by
        :param labels: A list of labels to filter by. Label filters work by either filtering a specific value of a
         label (i.e. list("key=value")) or by looking for the existence of a given key (i.e. "key")
        :param metrics: A list of metrics to return for each endpoint, read more in 'TimeMetric'
        :param start: The start time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` =
                                 days), or 0 for the earliest time.
        :param end: The end time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` =
                                 days), or 0 for the earliest time.
        :param top_level: if true will return only routers and endpoint that are NOT children of any router
        :param uids: if passed will return a list `ModelEndpoint` object with uid in uids
        """

        path = f"projects/{project}/model-endpoints"

        if labels and isinstance(labels, dict):
            labels = [f"{key}={value}" for key, value in labels.items()]

        response = self.api_call(
            method="GET",
            path=path,
            params={
                "model": model,
                "function": function,
                "label": labels or [],
                "start": start,
                "end": end,
                "metric": metrics or [],
                "top-level": top_level,
                "uid": uids,
            },
        )

        # Generate a list of a model endpoint dictionaries
        model_endpoints = response.json()["endpoints"]
        if model_endpoints:
            return [
                mlrun.model_monitoring.model_endpoint.ModelEndpoint.from_dict(obj)
                for obj in model_endpoints
            ]
        return []

    def get_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        feature_analysis: bool = False,
    ) -> mlrun.model_monitoring.model_endpoint.ModelEndpoint:
        """
        Returns a single `ModelEndpoint` object with additional metrics and feature related data.

        :param project:                    The name of the project
        :param endpoint_id:                The unique id of the model endpoint.
        :param start:                      The start time of the metrics. Can be represented by a string containing an
                                           RFC 3339 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                           `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or
                                           0 for the earliest time.
        :param end:                        The end time of the metrics. Can be represented by a string containing an
                                           RFC 3339 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                           `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or
                                           0 for the earliest time.
        :param metrics:                    A list of metrics to return for the model endpoint. There are pre-defined
                                           metrics for model endpoints such as predictions_per_second and
                                           latency_avg_5m but also custom metrics defined by the user. Please note that
                                           these metrics are stored in the time series DB and the results will be
                                           appeared under model_endpoint.spec.metrics.
        :param feature_analysis:           When True, the base feature statistics and current feature statistics will
                                           be added to the output of the resulting object.

        :return: A `ModelEndpoint` object.
        """

        path = f"projects/{project}/model-endpoints/{endpoint_id}"
        response = self.api_call(
            method="GET",
            path=path,
            params={
                "start": start,
                "end": end,
                "metric": metrics or [],
                "feature_analysis": feature_analysis,
            },
        )

        return mlrun.model_monitoring.model_endpoint.ModelEndpoint.from_dict(
            response.json()
        )

    def patch_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        attributes: dict,
    ):
        """
        Updates model endpoint with provided attributes.

        :param project: The name of the project.
        :param endpoint_id: The id of the endpoint.
        :param attributes: Dictionary of attributes that will be used for update the model endpoint. The keys
                           of this dictionary should exist in the target table. Note that the values should be
                           from type string or from a valid numerical type such as int or float.
                            More details about the model endpoint available attributes can be found under
                           :py:class:`~mlrun.common.schemas.ModelEndpoint`.

                           Example::

                                # Generate current stats for two features
                                current_stats = {'tvd_sum': 2.2,
                                                 'tvd_mean': 0.5,
                                                 'hellinger_sum': 3.6,
                                                 'hellinger_mean': 0.9,
                                                 'kld_sum': 24.2,
                                                 'kld_mean': 6.0,
                                                 'f1': {'tvd': 0.5, 'hellinger': 1.0, 'kld': 6.4},
                                                 'f2': {'tvd': 0.5, 'hellinger': 1.0, 'kld': 6.5}}

                                # Create attributes dictionary according to the required format
                                attributes = {`current_stats`: json.dumps(current_stats),
                                              `drift_status`: "DRIFT_DETECTED"}

        """

        attributes = {"attributes": _as_json(attributes)}
        path = f"projects/{project}/model-endpoints/{endpoint_id}"
        self.api_call(
            method="PATCH",
            path=path,
            params=attributes,
        )

    def deploy_monitoring_batch_job(
        self,
        project: str = "",
        default_batch_image: str = "mlrun/mlrun",
        with_schedule: bool = False,
    ):
        """
        Submit model monitoring batch job. By default, submit only the batch job as ML function without scheduling.
        To submit a scheduled job as well, please set with_schedule = True.

        :param project:             Project name.
        :param default_batch_image: The default image of the model monitoring batch job. By default, the image
                                    is mlrun/mlrun.
        :param with_schedule:       If true, submit the model monitoring scheduled job as well.


        :return: model monitoring batch job as a dictionary. You can easily convert the resulted function into a
                 runtime object by calling ~mlrun.new_function.
        """

        params = {
            "default_batch_image": default_batch_image,
            "with_schedule": with_schedule,
        }
        path = f"projects/{project}/jobs/batch-monitoring"

        resp = self.api_call(method="POST", path=path, params=params)
        return resp.json()["func"]

    def create_hub_source(
        self, source: Union[dict, mlrun.common.schemas.IndexedHubSource]
    ):
        """
        Add a new hub source.

        MLRun maintains an ordered list of hub sources (“sources”) Each source has
        its details registered and its order within the list. When creating a new source, the special order ``-1``
        can be used to mark this source as last in the list. However, once the source is in the MLRun list,
        its order will always be ``>0``.

        The global hub source always exists in the list, and is always the last source
        (``order = -1``). It cannot be modified nor can it be moved to another order in the list.

        The source object may contain credentials which are needed to access the datastore where the source is stored.
        These credentials are not kept in the MLRun DB, but are stored inside a kubernetes secret object maintained by
        MLRun. They are not returned through any API from MLRun.

        Example::

            import mlrun.common.schemas

            # Add a private source as the last one (will be #1 in the list)
            private_source = mlrun.common.schemas.IndexedHubSource(
                order=-1,
                source=mlrun.common.schemas.HubSource(
                    metadata=mlrun.common.schemas.HubObjectMetadata(
                        name="priv", description="a private source"
                    ),
                    spec=mlrun.common.schemas.HubSourceSpec(path="/local/path/to/source", channel="development")
                )
            )
            db.create_hub_source(private_source)

            # Add another source as 1st in the list - will push previous one to be #2
            another_source = mlrun.common.schemas.IndexedHubSource(
                order=1,
                source=mlrun.common.schemas.HubSource(
                    metadata=mlrun.common.schemas.HubObjectMetadata(
                        name="priv-2", description="another source"
                    ),
                    spec=mlrun.common.schemas.HubSourceSpec(
                        path="/local/path/to/source/2",
                        channel="development",
                        credentials={...}
                    )
                )
            )
            db.create_hub_source(another_source)

        :param source: The source and its order, of type
            :py:class:`~mlrun.common.schemas.hub.IndexedHubSource`, or in dictionary form.
        :returns: The source object as inserted into the database, with credentials stripped.
        """
        path = "hub/sources"
        if isinstance(source, mlrun.common.schemas.IndexedHubSource):
            source = source.dict()
        response = self.api_call(method="POST", path=path, json=source)
        return mlrun.common.schemas.IndexedHubSource(**response.json())

    def store_hub_source(
        self,
        source_name: str,
        source: Union[dict, mlrun.common.schemas.IndexedHubSource],
    ):
        """
        Create or replace a hub source.
        For an example of the source format and explanation of the source order logic,
        please see :py:func:`~create_hub_source`. This method can be used to modify the source itself or its
        order in the list of sources.

        :param source_name: Name of the source object to modify/create. It must match the ``source.metadata.name``
            parameter in the source itself.
        :param source: Source object to store in the database.
        :returns: The source object as stored in the DB.
        """
        path = f"hub/sources/{source_name}"
        if isinstance(source, mlrun.common.schemas.IndexedHubSource):
            source = source.dict()

        response = self.api_call(method="PUT", path=path, json=source)
        return mlrun.common.schemas.IndexedHubSource(**response.json())

    def list_hub_sources(
        self,
        item_name: Optional[str] = None,
        tag: Optional[str] = None,
        version: Optional[str] = None,
    ) -> List[mlrun.common.schemas.hub.IndexedHubSource]:
        """
        List hub sources in the MLRun DB.

        :param item_name:   Sources contain this item will be returned, If not provided all sources will be returned.
        :param tag:         Item tag to filter by, supported only if item name is provided.
        :param version:     Item version to filter by, supported only if item name is provided and tag is not.

        :returns: List of indexed hub sources.
        """
        path = "hub/sources"
        params = {}
        if item_name:
            params["item-name"] = normalize_name(item_name)
        if tag:
            params["tag"] = tag
        if version:
            params["version"] = version
        response = self.api_call(method="GET", path=path, params=params).json()
        results = []
        for item in response:
            results.append(mlrun.common.schemas.IndexedHubSource(**item))
        return results

    def get_hub_source(self, source_name: str):
        """
        Retrieve a hub source from the DB.

        :param source_name: Name of the hub source to retrieve.
        """
        path = f"hub/sources/{source_name}"
        response = self.api_call(method="GET", path=path)
        return mlrun.common.schemas.IndexedHubSource(**response.json())

    def delete_hub_source(self, source_name: str):
        """
        Delete a hub source from the DB.
        The source will be deleted from the list, and any following sources will be promoted - for example, if the
        1st source is deleted, the 2nd source will become #1 in the list.
        The global hub source cannot be deleted.

        :param source_name: Name of the hub source to delete.
        """
        path = f"hub/sources/{source_name}"
        self.api_call(method="DELETE", path=path)

    def get_hub_catalog(
        self,
        source_name: str,
        version: str = None,
        tag: str = None,
        force_refresh: bool = False,
    ):
        """
        Retrieve the item catalog for a specified hub source.
        The list of items can be filtered according to various filters, using item's metadata to filter.

        :param source_name: Name of the source.
        :param version: Filter items according to their version.
        :param tag: Filter items based on tag.
        :param force_refresh: Make the server fetch the catalog from the actual hub source,
            rather than rely on cached information which may exist from previous get requests. For example,
            if the source was re-built,
            this will make the server get the updated information. Default is ``False``.
        :returns: :py:class:`~mlrun.common.schemas.hub.HubCatalog` object, which is essentially a list
            of :py:class:`~mlrun.common.schemas.hub.HubItem` entries.
        """
        path = f"hub/sources/{source_name}/items"
        params = {
            "version": version,
            "tag": tag,
            "force-refresh": force_refresh,
        }
        response = self.api_call(method="GET", path=path, params=params)
        return mlrun.common.schemas.HubCatalog(**response.json())

    def get_hub_item(
        self,
        source_name: str,
        item_name: str,
        version: str = None,
        tag: str = "latest",
        force_refresh: bool = False,
    ):
        """
        Retrieve a specific hub item.

        :param source_name: Name of source.
        :param item_name: Name of the item to retrieve, as it appears in the catalog.
        :param version: Get a specific version of the item. Default is ``None``.
        :param tag: Get a specific version of the item identified by tag. Default is ``latest``.
        :param force_refresh: Make the server fetch the information from the actual hub
            source, rather than
            rely on cached information. Default is ``False``.
        :returns: :py:class:`~mlrun.common.schemas.hub.HubItem`.
        """
        path = (f"hub/sources/{source_name}/items/{item_name}",)
        params = {
            "version": version,
            "tag": tag,
            "force-refresh": force_refresh,
        }
        response = self.api_call(method="GET", path=path, params=params)
        return mlrun.common.schemas.HubItem(**response.json())

    def get_hub_asset(
        self,
        source_name: str,
        item_name: str,
        asset_name: str,
        version: str = None,
        tag: str = "latest",
    ):
        """
        Get hub asset from item.

        :param source_name: Name of source.
        :param item_name:   Name of the item which holds the asset.
        :param asset_name:  Name of the asset to retrieve.
        :param version: Get a specific version of the item. Default is ``None``.
        :param tag: Get a specific version of the item identified by tag. Default is ``latest``.

        :return: http response with the asset in the content attribute
        """
        path = f"hub/sources/{source_name}/items/{item_name}/assets/{asset_name}"
        params = {
            "version": version,
            "tag": tag,
        }
        response = self.api_call(method="GET", path=path, params=params)
        return response

    def verify_authorization(
        self,
        authorization_verification_input: mlrun.common.schemas.AuthorizationVerificationInput,
    ):
        """Verifies authorization for the provided action on the provided resource.

        :param authorization_verification_input: Instance of
            :py:class:`~mlrun.common.schemas.AuthorizationVerificationInput` that includes all the needed parameters for
            the auth verification
        """
        error_message = "Authorization check failed"
        self.api_call(
            "POST",
            "authorization/verifications",
            error_message,
            body=dict_to_json(authorization_verification_input.dict()),
        )

    def trigger_migrations(self) -> Optional[mlrun.common.schemas.BackgroundTask]:
        """Trigger migrations (will do nothing if no migrations are needed) and wait for them to finish if actually
        triggered
        :returns: :py:class:`~mlrun.common.schemas.BackgroundTask`.
        """
        response = self.api_call(
            "POST",
            "operations/migrations",
            "Failed triggering migrations",
        )
        if response.status_code == http.HTTPStatus.ACCEPTED:
            background_task = mlrun.common.schemas.BackgroundTask(**response.json())
            return self._wait_for_background_task_to_reach_terminal_state(
                background_task.metadata.name
            )
        return None

    def set_run_notifications(
        self,
        project: str,
        run_uid: str,
        notifications: typing.List[mlrun.model.Notification] = None,
    ):
        """
        Set notifications on a run. This will override any existing notifications on the run.
        :param project: Project containing the run.
        :param run_uid: UID of the run.
        :param notifications: List of notifications to set on the run. Default is an empty list.
        """
        notifications = notifications or []

        self.api_call(
            "PUT",
            f"projects/{project}/runs/{run_uid}/notifications",
            f"Failed to set notifications on run. uid={run_uid}, project={project}",
            json={
                "notifications": [
                    notification.to_dict() for notification in notifications
                ],
            },
        )

    def set_schedule_notifications(
        self,
        project: str,
        schedule_name: str,
        notifications: typing.List[mlrun.model.Notification] = None,
    ):
        """
        Set notifications on a schedule. This will override any existing notifications on the schedule.
        :param project: Project containing the schedule.
        :param schedule_name: Name of the schedule.
        :param notifications: List of notifications to set on the schedule. Default is an empty list.
        """
        notifications = notifications or []

        self.api_call(
            "PUT",
            f"projects/{project}/schedules/{schedule_name}/notifications",
            f"Failed to set notifications on schedule. schedule={schedule_name}, project={project}",
            json={
                "notifications": [
                    notification.to_dict() for notification in notifications
                ],
            },
        )

    def store_run_notifications(
        self,
        notification_objects: typing.List[mlrun.model.Notification],
        run_uid: str,
        project: str = None,
        mask_params: bool = True,
    ):
        """
        For internal use.
        The notification mechanism may run "locally" for certain runtimes.
        However, the updates occur in the API so nothing to do here.
        """
        pass

    def submit_workflow(
        self,
        project: str,
        name: str,
        workflow_spec: Union[
            mlrun.projects.pipelines.WorkflowSpec,
            mlrun.common.schemas.WorkflowSpec,
            dict,
        ],
        arguments: Optional[Dict] = None,
        artifact_path: Optional[str] = None,
        source: Optional[str] = None,
        run_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ):
        """
        Submitting workflow for a remote execution.

        :param project:         project name
        :param name:            workflow name
        :param workflow_spec:   the workflow spec to execute
        :param arguments:       arguments for the workflow
        :param artifact_path:   artifact target path of the workflow
        :param source:          source url of the project
        :param run_name:        run name to override the default: 'workflow-runner-<workflow name>'
        :param namespace:       kubernetes namespace if other than default

        :returns:    :py:class:`~mlrun.common.schemas.WorkflowResponse`.
        """
        image = (
            workflow_spec.image
            if hasattr(workflow_spec, "image")
            else workflow_spec.get("image", None)
        )
        workflow_name = name or (
            workflow_spec.name
            if hasattr(workflow_spec, "name")
            else workflow_spec.get("name", None)
        )
        req = {
            "arguments": arguments,
            "artifact_path": artifact_path,
            "source": source,
            "run_name": run_name,
            "namespace": namespace,
        }
        if isinstance(
            workflow_spec,
            mlrun.common.schemas.WorkflowSpec,
        ):
            req["spec"] = workflow_spec.dict()
        elif isinstance(workflow_spec, mlrun.projects.pipelines.WorkflowSpec):
            req["spec"] = workflow_spec.to_dict()
        else:
            req["spec"] = workflow_spec
        req["spec"]["image"] = image
        req["spec"]["name"] = workflow_name
        response = self.api_call(
            "POST",
            f"projects/{project}/workflows/{workflow_name}/submit",
            json=req,
        )
        return mlrun.common.schemas.WorkflowResponse(**response.json())

    def get_workflow_id(
        self,
        project: str,
        name: str,
        run_id: str,
        engine: str = "",
    ):
        """
        Retrieve workflow id from the uid of the workflow runner.

        :param project: project name
        :param name:    workflow name
        :param run_id:  the id of the workflow runner - the job that runs the workflow
        :param engine:  pipeline runner

        :returns:   :py:class:`~mlrun.common.schemas.GetWorkflowResponse`.
        """
        params = {}
        if engine:
            params["engine"] = engine
        response = self.api_call(
            "GET",
            f"projects/{project}/workflows/{name}/runs/{run_id}",
            params=params,
        )
        return mlrun.common.schemas.GetWorkflowResponse(**response.json())

    def load_project(
        self,
        name: str,
        url: str,
        secrets: Optional[Dict] = None,
        save_secrets: bool = True,
    ) -> str:
        """
        Loading a project remotely from the given source.
        :param name:            project name
        :param url:             git or tar.gz or .zip sources archive path e.g.:
                                git://github.com/mlrun/demo-xgb-project.git
                                http://mysite/archived-project.zip
                                The git project should include the project yaml file.
        :param secrets:         Secrets to store in project in order to load it from the provided url.
                                For more information see :py:func:`mlrun.load_project` function.
        :param save_secrets:    Whether to store secrets in the loaded project.
                                Setting to False will cause waiting for the process completion.

        :returns:               The terminal state of load project process.
        """
        params = {"url": url}
        body = None
        if secrets:
            provider = mlrun.common.schemas.SecretProviderName.kubernetes
            secrets_input = mlrun.common.schemas.SecretsData(
                provider=provider, secrets=secrets
            )
            body = secrets_input.dict()
        response = self.api_call(
            "POST", f"projects/{name}/load", params=params, body=dict_to_json(body)
        )
        response = response.json()
        run = mlrun.RunObject.from_dict(response["data"])
        state, _ = run.logs()

        if secrets and not save_secrets:
            self.delete_project_secrets(project=name, secrets=list(secrets.keys()))
            if state != "completed":
                logger.error("Load project task failed, deleting project")
                self.delete_project(name, mlrun.common.schemas.DeletionStrategy.cascade)

        return state

    def get_datastore_profile(
        self, name: str, project: str
    ) -> Optional[mlrun.common.schemas.DatastoreProfile]:
        project = project or config.default_project
        path = self._path_of("projects", project, "datastore-profiles") + f"/{name}"

        res = self.api_call(method="GET", path=path)
        if res:
            public_wrapper = res.json()
            datastore = DatastoreProfile2Json.create_from_json(
                public_json=public_wrapper["object"]
            )
            return datastore
        return None

    def delete_datastore_profile(self, name: str, project: str):
        project = project or config.default_project
        path = self._path_of("projects", project, "datastore-profiles") + f"/{name}"
        self.api_call(method="DELETE", path=path)
        return None

    def list_datastore_profiles(
        self, project: str
    ) -> List[mlrun.common.schemas.DatastoreProfile]:
        project = project or config.default_project
        path = self._path_of("projects", project, "datastore-profiles")

        res = self.api_call(method="GET", path=path)
        if res:
            public_wrapper = res.json()
            datastores = [
                DatastoreProfile2Json.create_from_json(x["object"])
                for x in public_wrapper
            ]
            return datastores
        return None

    def store_datastore_profile(
        self, profile: mlrun.common.schemas.DatastoreProfile, project: str
    ):
        """
        Create or replace a datastore profile.
        :returns: None
        """
        project = project or config.default_project
        path = self._path_of("projects", project, "datastore-profiles")

        self.api_call(method="PUT", path=path, json=profile.dict())


def _as_json(obj):
    fn = getattr(obj, "to_json", None)
    if fn:
        return fn()
    return dict_to_json(obj)
