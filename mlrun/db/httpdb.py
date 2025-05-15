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
import time
import traceback
import typing
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from os import environ, path, remove
from typing import Literal, Optional, Union
from urllib.parse import urlparse

import pydantic.v1
import requests
import semver
from pydantic.v1 import parse_obj_as

import mlrun
import mlrun.common.constants
import mlrun.common.formatters
import mlrun.common.runtimes
import mlrun.common.schemas
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.common.schemas.model_monitoring.model_endpoints as mm_endpoints
import mlrun.common.types
import mlrun.platforms
import mlrun.projects
import mlrun.runtimes.nuclio.api_gateway
import mlrun.runtimes.nuclio.function
import mlrun.utils
from mlrun.alerts.alert import AlertConfig
from mlrun.db.auth_utils import OAuthClientIDTokenProvider, StaticTokenProvider
from mlrun.errors import MLRunInvalidArgumentError, err_to_str
from mlrun_pipelines.utils import compile_pipeline

from ..artifacts import Artifact
from ..common.schemas import AlertActivations
from ..config import config
from ..datastore.datastore_profile import DatastoreProfile2Json
from ..feature_store import FeatureSet, FeatureVector
from ..lists import ArtifactList, RunList
from ..runtimes import BaseRuntime
from ..utils import (
    datetime_to_iso,
    dict_to_json,
    logger,
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
        self.python_version = environ.get("MLRUN_PYTHON_VERSION") or str(
            version.Version().get_python_version()
        )

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

        self.base_url = base_url
        username = parsed_url.username or config.httpdb.user
        password = parsed_url.password or config.httpdb.password
        self.token_provider = None

        if config.auth_with_client_id.enabled:
            self.token_provider = OAuthClientIDTokenProvider(
                token_endpoint=mlrun.get_secret_or_env("MLRUN_AUTH_TOKEN_ENDPOINT"),
                client_id=mlrun.get_secret_or_env("MLRUN_AUTH_CLIENT_ID"),
                client_secret=mlrun.get_secret_or_env("MLRUN_AUTH_CLIENT_SECRET"),
                timeout=config.auth_with_client_id.request_timeout,
            )
        else:
            username, password, token = mlrun.platforms.add_or_refresh_credentials(
                parsed_url.hostname, username, password, config.httpdb.token
            )

            if token:
                self.token_provider = StaticTokenProvider(token)

        self.user = username
        self.password = password

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}({self.base_url!r})"

    @staticmethod
    def get_api_path_prefix(version: Optional[str] = None) -> str:
        """
        :param version: API version to use, None (the default) will mean to use the default value from mlrun.config,
         for un-versioned api set an empty string.
        """
        if version is not None:
            return f"api/{version}" if version else "api"

        api_version_path = (
            f"api/{config.api_base_version}" if config.api_base_version else "api"
        )
        return api_version_path

    def get_base_api_url(self, path: str, version: Optional[str] = None) -> str:
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
    ) -> requests.Response:
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

        :returns: `requests.Response` HTTP response object
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
        elif self.token_provider:
            token = self.token_provider.get_token()
            if token:
                # Iguazio auth doesn't support passing token through bearer, so use cookie instead
                if self.token_provider.is_iguazio_session():
                    session_cookie = f'j:{{"sid": "{token}"}}'
                    cookies = {
                        "session": session_cookie,
                    }
                    kw["cookies"] = cookies
                else:
                    if "Authorization" not in kw.setdefault("headers", {}):
                        kw["headers"].update({"Authorization": "Bearer " + token})

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
                method,
                url,
                timeout=timeout,
                verify=config.httpdb.http.verify,
                **kw,
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

    def paginated_api_call(
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
        return_all=False,
    ) -> typing.Generator[requests.Response, None, None]:
        """
        Calls the API with pagination and yields each page of the response.

        Depending on the `return_all` parameter:
        - If `return_all` is `True`, fetches and yields all pages of results.
        - If `return_all` is False, only a single page of results is fetched and yielded.

        :param method: The HTTP method (GET, POST, etc.).
        :param path: The API endpoint path.
        :param error: Error message used for debugging if the request fails.
        :param params: The parameters to pass for the API request, including filters.
        :param body: The body of the request.
        :param json: The JSON payload for the request.
        :param headers: Custom headers for the request.
        :param timeout: Timeout for the request.
        :param version: API version, optional.
        :param return_all: If `True`, fetches all pages and returns them in one shot. If `False`, returns only
            the requested page or the next page.
        """

        def _api_call(_params):
            return self.api_call(
                method=method,
                path=path,
                error=error,
                params=_params,
                body=body,
                json=json,
                headers=headers,
                timeout=timeout,
                version=version,
            )

        page_params = self._resolve_page_params(params)
        response = _api_call(page_params)

        # yields a single page of results
        yield response

        if return_all:
            page_token = response.json().get("pagination", {}).get("page-token", None)

            while page_token:
                try:
                    # Use the page token to get the next page.
                    # No need to supply any other parameters as the token informs the pagination cache
                    # which parameters to use.
                    response = _api_call({"page-token": page_token})
                except mlrun.errors.MLRunNotFoundError:
                    # pagination token expired, we've reached the last page
                    break

                yield response
                page_token = (
                    response.json().get("pagination", {}).get("page-token", None)
                )

    @staticmethod
    def process_paginated_responses(
        responses: typing.Generator[requests.Response, None, None], key: str = "data"
    ) -> tuple[list[typing.Any], Optional[str]]:
        """
        Processes the paginated responses and returns the combined data
        """
        data = []
        page_token = None
        for response in responses:
            page_token = response.json().get("pagination", {}).get("page-token", None)
            data.extend(response.json().get(key, []))
        return data, page_token

    def _init_session(self, retry_on_post: bool = False):
        return mlrun.utils.HTTPSessionWithRetry(
            retry_on_exception=config.httpdb.retry_api_call_on_exception
            == mlrun.common.schemas.HTTPSessionRetryMode.enabled.value,
            retry_on_post=retry_on_post,
        )

    def _path_of(self, resource, project, uid=None):
        project = project or config.default_project
        _path = f"projects/{project}/{resource}"
        if uid:
            _path += f"/{uid}"
        return _path

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
        The code utilizes the URL for the API server from the configuration - ``config.dbpath``.

        For example::

            config.dbpath = config.dbpath or "http://mlrun-api:8080"
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
            config.external_platform_tracking = (
                server_cfg.get("external_platform_tracking")
                or config.external_platform_tracking
            )
            config.packagers = server_cfg.get("packagers") or config.packagers
            server_data_prefixes = server_cfg.get("feature_store_data_prefixes") or {}
            for prefix in ["default", "nosql", "redisnosql"]:
                server_prefix_value = server_data_prefixes.get(prefix)
                if server_prefix_value is not None:
                    setattr(
                        config.feature_store.data_prefixes, prefix, server_prefix_value
                    )
            config.feature_store.default_targets = (
                server_cfg.get("feature_store_default_targets")
                or config.feature_store.default_targets
            )
            config.alerts.mode = server_cfg.get("alerts_mode") or config.alerts.mode
            config.system_id = server_cfg.get("system_id") or config.system_id
            model_monitoring_store_prefixes = (
                server_cfg.get("model_endpoint_monitoring_store_prefixes") or {}
            )
            for prefix in ["default", "user_space", "monitoring_application"]:
                store_prefix_value = model_monitoring_store_prefixes.get(prefix)
                if server_prefix_value is not None:
                    setattr(
                        config.model_endpoint_monitoring.store_prefixes,
                        prefix,
                        store_prefix_value,
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

        path = self._path_of("logs", project, uid)
        params = {"append": bool2str(append)}
        error = f"store log {project}/{uid}"
        self.api_call("POST", path, error, params, body)

    def get_log(self, uid, project="", offset=0, size=None):
        """Retrieve 1 MB data of log.

        :param uid: Log unique ID
        :param project: Project name for which the log belongs
        :param offset: Retrieve partial log, get up to ``size`` bytes starting at offset ``offset``
            from beginning of log (must be >= 0)
        :param size: If set to ``-1`` will retrieve and print all data to end of the log by chunks of 1MB each.
        :returns: The following objects:

            - state - The state of the runtime object which generates this log, if it exists. In case no known state
              exists, this will be ``unknown``.
            - content - The actual log content.

            * in case size = -1, return the state and the final offset
        """
        if offset < 0:
            raise MLRunInvalidArgumentError("Offset cannot be negative")
        if size is None:
            size = int(mlrun.mlconf.httpdb.logs.pull_logs_default_size_limit)
        elif size == -1:
            logger.warning(
                "Retrieving all logs. This may be inefficient and can result in a large log."
            )
            state, offset = self.watch_log(uid, project, watch=False, offset=offset)
            return state, offset

        params = {"offset": offset, "size": size}
        path = self._path_of("logs", project, uid)
        error = f"get log {project}/{uid}"
        resp = self.api_call("GET", path, error, params=params)
        if resp.headers:
            state = resp.headers.get("x-mlrun-run-state", "")
            return state.lower(), resp.content

        return "unknown", resp.content

    def get_log_size(self, uid, project=""):
        """Retrieve log size in bytes.

        :param uid: Run UID
        :param project: Project name for which the log belongs
        :returns: The log file size in bytes for the given run UID.
        """
        path = self._path_of("logs", project, uid)
        path += "/size"
        error = f"get log size {project}/{uid}"
        resp = self.api_call("GET", path, error)
        return resp.json()["size"]

    def watch_log(self, uid, project="", watch=True, offset=0):
        """Retrieve logs of a running process by chunks of 1MB, and watch the progress of the execution until it
        completes. This method will print out the logs and continue to periodically poll for, and print,
        new logs as long as the state of the runtime which generates this log is either ``pending`` or ``running``.

        :param uid: The uid of the log object to watch.
        :param project: Project that the log belongs to.
        :param watch: If set to ``True`` will continue tracking the log as described above. Otherwise this function
            is practically equivalent to the :py:func:`~get_log` function.
        :param offset: Minimal offset in the log to watch.
        :returns: The final state of the log being watched and the final offset.
        """

        state, text = self.get_log(uid, project, offset=offset)
        if text:
            print(text.decode(errors=mlrun.mlconf.httpdb.logs.decode.errors))
        nil_resp = 0
        while True:
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

            if watch and state in [
                mlrun.common.runtimes.constants.RunStates.pending,
                mlrun.common.runtimes.constants.RunStates.running,
                mlrun.common.runtimes.constants.RunStates.created,
                mlrun.common.runtimes.constants.RunStates.aborting,
            ]:
                continue
            else:
                # the whole log was retrieved
                if len(text) == 0:
                    break

        return state, offset

    def store_run(self, struct, uid, project="", iter=0):
        """Store run details in the DB. This method is usually called from within other :py:mod:`mlrun` flows
        and not called directly by the user."""

        path = self._path_of("runs", project, uid)
        params = {"iter": iter}
        error = f"store run {project}/{uid}"
        body = _as_json(struct)
        self.api_call("POST", path, error, params=params, body=body)

    def update_run(self, updates: dict, uid, project="", iter=0, timeout=45):
        """Update the details of a stored run in the DB."""

        path = self._path_of("runs", project, uid)
        params = {"iter": iter}
        error = f"update run {project}/{uid}"
        body = _as_json(updates)
        self.api_call("PATCH", path, error, params=params, body=body, timeout=timeout)

    def abort_run(self, uid, project="", iter=0, timeout=45, status_text=""):
        """
        Abort a running run - will remove the run's runtime resources and mark its state as aborted.
        :returns: :py:class:`~mlrun.common.schemas.BackgroundTask`.
        """
        project = project or config.default_project
        params = {"iter": iter}
        updates = {}
        if status_text:
            updates["status.status_text"] = status_text
        body = _as_json(updates)

        response = self.api_call(
            "POST",
            path=f"projects/{project}/runs/{uid}/abort",
            error="Failed run abortion",
            params=params,
            body=body,
            timeout=timeout,
        )
        if response.status_code == http.HTTPStatus.ACCEPTED:
            background_task = mlrun.common.schemas.BackgroundTask(**response.json())
            return self._wait_for_background_task_to_reach_terminal_state(
                background_task.metadata.name, project=project
            )
        return None

    def push_run_notifications(
        self,
        uid,
        project="",
        timeout=45,
    ):
        """
        Push notifications for a run.

        :param uid: Unique ID of the run.
        :param project: Project that the run belongs to.
        :returns: :py:class:`~mlrun.common.schemas.BackgroundTask`.
        """
        project = project or config.default_project
        response = self.api_call(
            "POST",
            path=f"projects/{project}/runs/{uid}/push-notifications",
            error="Failed push notifications",
            timeout=timeout,
        )
        if response.status_code == http.HTTPStatus.ACCEPTED:
            background_task = mlrun.common.schemas.BackgroundTask(**response.json())
            background_task = self._wait_for_background_task_to_reach_terminal_state(
                background_task.metadata.name, project=project
            )
            if (
                background_task.status.state
                == mlrun.common.schemas.BackgroundTaskState.succeeded
            ):
                logger.info(
                    "Notifications for the run have been pushed",
                    project=project,
                    run_id=uid,
                )
            elif (
                background_task.status.state
                == mlrun.common.schemas.BackgroundTaskState.failed
            ):
                logger.error(
                    "Failed to push run notifications",
                    project=project,
                    run_id=uid,
                    error=background_task.status.error,
                )
        return None

    def push_pipeline_notifications(
        self,
        pipeline_id,
        project="",
        notifications=None,
        timeout=45,
    ):
        """
        Push notifications for a pipeline.

        :param pipeline_id: Unique ID of the pipeline(KFP).
        :param project: Project that the run belongs to.
        :param notifications: List of notifications to push.
        :returns: :py:class:`~mlrun.common.schemas.BackgroundTask`.
        """
        if notifications is None or type(notifications) is not list:
            raise MLRunInvalidArgumentError(
                "The 'notifications' parameter must be a list."
            )

        project = project or config.default_project

        response = self.api_call(
            "POST",
            path=f"projects/{project}/pipelines/{pipeline_id}/push-notifications",
            error="Failed push notifications",
            body=_as_json([notification.to_dict() for notification in notifications]),
            timeout=timeout,
        )
        if response.status_code == http.HTTPStatus.ACCEPTED:
            background_task = mlrun.common.schemas.BackgroundTask(**response.json())
            background_task = self._wait_for_background_task_to_reach_terminal_state(
                background_task.metadata.name, project=project
            )
            if (
                background_task.status.state
                == mlrun.common.schemas.BackgroundTaskState.succeeded
            ):
                logger.info(
                    "Pipeline notifications have been pushed",
                    project=project,
                    pipeline_id=pipeline_id,
                )
            elif (
                background_task.status.state
                == mlrun.common.schemas.BackgroundTaskState.failed
            ):
                logger.error(
                    "Failed to push pipeline notifications",
                    project=project,
                    pipeline_id=pipeline_id,
                    error=background_task.status.error,
                )

        return None

    def read_run(
        self,
        uid,
        project="",
        iter=0,
        format_: mlrun.common.formatters.RunFormat = mlrun.common.formatters.RunFormat.full,
    ):
        """Read the details of a stored run from the DB.

        :param uid:         The run's unique ID.
        :param project:     Project name.
        :param iter:        Iteration within a specific execution.
        :param format_:     The format in which to return the run details.
        """

        path = self._path_of("runs", project, uid)
        params = {
            "iter": iter,
            "format": format_.value,
        }
        error = f"get run {project}/{uid}"
        resp = self.api_call("GET", path, error, params=params)
        return resp.json()["data"]

    def del_run(self, uid, project="", iter=0):
        """Delete details of a specific run from DB.

        :param uid: Unique ID for the specific run to delete.
        :param project: Project that the run belongs to.
        :param iter: Iteration within a specific task.
        """

        path = self._path_of("runs", project, uid)
        params = {"iter": iter}
        error = f"del run {project}/{uid}"
        self.api_call("DELETE", path, error, params=params)

    def list_runs(
        self,
        name: Optional[str] = None,
        uid: Optional[Union[str, list[str]]] = None,
        project: Optional[str] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
        state: Optional[
            mlrun.common.runtimes.constants.RunStates
        ] = None,  # Backward compatibility
        states: typing.Optional[list[mlrun.common.runtimes.constants.RunStates]] = None,
        sort: bool = True,
        iter: bool = False,
        start_time_from: Optional[datetime] = None,
        start_time_to: Optional[datetime] = None,
        last_update_time_from: Optional[datetime] = None,
        last_update_time_to: Optional[datetime] = None,
        end_time_from: Optional[datetime] = None,
        end_time_to: Optional[datetime] = None,
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
        Retrieve a list of runs.
        The default returns the runs from the last week, partitioned by project/name.
        To override the default, specify any filter.

        Example::

            runs = db.list_runs(
                name="download", project="iris", labels=["owner=admin", "kind=job"]
            )
            # If running in Jupyter, can use the .show() function to display the results
            db.list_runs(name="", project=project_name).show()


        :param name: Name of the run to retrieve.
        :param uid: Unique ID of the run, or a list of run UIDs.
        :param project: Project that the runs belongs to. If not specified, the default project will be used.
        :param labels: Filter runs by label key-value pairs or key existence. This can be provided as:
            - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
            or `{"label": None}` to check for key existence.
            - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
            or just `"label"` for key existence.
            - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
            the specified key-value pairs or key existence.
        :param state: Deprecated - List only runs whose state is specified (will be removed in 1.10.0)
        :param states: List only runs whose state is one of the provided states.
        :param sort: Whether to sort the result according to their start time. Otherwise, results will be
            returned by their internal order in the DB (order will not be guaranteed).
        :param iter: If ``True`` return runs from all iterations. Otherwise, return only runs whose ``iter`` is 0.
        :param start_time_from: Filter by run start time in ``[start_time_from, start_time_to]``.
        :param start_time_to: Filter by run start time in ``[start_time_from, start_time_to]``.
        :param last_update_time_from: Filter by run last update time in ``(last_update_time_from,
            last_update_time_to)``.
        :param last_update_time_to: Filter by run last update time in ``(last_update_time_from, last_update_time_to)``.
        :param end_time_from: Filter by run end time in ``[end_time_from, end_time_to]``.
        :param end_time_to: Filter by run end time in ``[end_time_from, end_time_to]``.
        :param partition_by: Field to group results by. When `partition_by` is specified, the `partition_sort_by`
            parameter must be provided as well.
        :param rows_per_partition: How many top rows (per sorting defined by `partition_sort_by` and `partition_order`)
            to return per group. Default value is 1.
        :param partition_sort_by: What field to sort the results by, within each partition defined by `partition_by`.
            Currently the only allowed values are `created` and `updated`.
        :param partition_order: Order of sorting within partitions - `asc` or `desc`. Default is `desc`.
        :param max_partitions: Maximal number of partitions to include in the result. Default is `0` which means no
            limit.
        :param with_notifications: Return runs with notifications, and join them to the response. Default is `False`.
        """
        runs, _ = self._list_runs(
            name=name,
            uid=uid,
            project=project,
            labels=labels,
            state=state,
            states=states,
            sort=sort,
            iter=iter,
            start_time_from=start_time_from,
            start_time_to=start_time_to,
            last_update_time_from=last_update_time_from,
            last_update_time_to=last_update_time_to,
            end_time_from=end_time_from,
            end_time_to=end_time_to,
            partition_by=partition_by,
            rows_per_partition=rows_per_partition,
            partition_sort_by=partition_sort_by,
            partition_order=partition_order,
            max_partitions=max_partitions,
            with_notifications=with_notifications,
            return_all=True,
        )
        return runs

    def paginated_list_runs(
        self,
        *args,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        **kwargs,
    ) -> tuple[RunList, Optional[str]]:
        """List runs with support for pagination and various filtering options.

        This method retrieves a paginated list of runs based on the specified filter parameters.
        Pagination is controlled using the `page`, `page_size`, and `page_token` parameters. The method
        will return a list of runs that match the filtering criteria provided.

        For detailed information about the parameters, refer to the list_runs method:
            See :py:func:`~list_runs` for more details.

        Examples::

            # Fetch first page of runs with page size of 5
            runs, token = db.paginated_list_runs(project="my-project", page_size=5)
            # Fetch next page using the pagination token from the previous response
            runs, token = db.paginated_list_runs(project="my-project", page_token=token)
            # Fetch runs for a specific page (e.g., page 3)
            runs, token = db.paginated_list_runs(project="my-project", page=3, page_size=5)

            # Automatically iterate over all pages without explicitly specifying the page number
            runs = []
            token = None
            while True:
                page_runs, token = db.paginated_list_runs(
                    project="my-project", page_token=token, page_size=5
                )
                runs.extend(page_runs)

                # If token is None and page_runs is empty, we've reached the end (no more runs).
                # If token is None and page_runs is not empty, we've fetched the last page of runs.
                if not token:
                    break
            print(f"Total runs retrieved: {len(runs)}")

        :param page: The page number to retrieve. If not provided, the next page will be retrieved.
        :param page_size: The number of items per page to retrieve. Up to `page_size` responses are expected.
            Defaults to `mlrun.mlconf.httpdb.pagination.default_page_size` if not provided.
        :param page_token: A pagination token used to retrieve the next page of results. Should not be provided
            for the first request.

        :returns: A tuple containing the list of runs and an optional `page_token` for pagination.
        """
        return self._list_runs(
            *args,
            page=page,
            page_size=page_size,
            page_token=page_token,
            return_all=False,
            **kwargs,
        )

    def del_runs(
        self,
        name: Optional[str] = None,
        project: Optional[str] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
        state: Optional[mlrun.common.runtimes.constants.RunStates] = None,
        days_ago: int = 0,
    ):
        """Delete a group of runs identified by the parameters of the function.

        Example::

            db.del_runs(state="completed")

        :param name: Name of the task which the runs belong to.
        :param project: Project to which the runs belong.
        :param labels: Filter runs by label key-value pairs or key existence. This can be provided as:
            - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
            or `{"label": None}` to check for key existence.
            - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
            or just `"label"` for key existence.
            - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
            the specified key-value pairs or key existence.
        :param state: Filter only runs which are in this state.
        :param days_ago: Filter runs whose start time is newer than this parameter.
        """

        project = project or config.default_project
        labels = self._parse_labels(labels)
        params = {
            "name": name,
            "project": project,
            "label": labels,
            "state": state,
            "days_ago": str(days_ago),
        }
        error = "del runs"
        _path = self._path_of("runs", project)
        self.api_call("DELETE", _path, error, params=params)

    def store_artifact(
        self,
        key,
        artifact,
        iter=None,
        tag=None,
        project="",
        tree=None,
    ) -> dict[str, str]:
        """Store an artifact in the DB.

        :param key: Identifying key of the artifact.
        :param artifact: The :py:class:`~mlrun.artifacts.Artifact` to store.
        :param iter: The task iteration which generated this artifact. If ``iter`` is not ``None`` the iteration will
            be added to the key provided to generate a unique key for the artifact of the specific iteration.
        :param tag: Tag of the artifact.
        :param project: Project that the artifact belongs to.
        :param tree: The tree (producer id) which generated this artifact.
        :returns: The stored artifact dictionary.
        """
        project = project or mlrun.mlconf.default_project
        endpoint_path = f"projects/{project}/artifacts/{key}"

        error = f"store artifact {project}/{key}"

        params = {}
        if iter:
            params["iter"] = str(iter)
        if tag:
            params["tag"] = tag
        if tree:
            params["tree"] = tree

        body = _as_json(artifact)
        response = self.api_call(
            "PUT", endpoint_path, error, body=body, params=params, version="v2"
        )
        return response.json()

    def read_artifact(
        self,
        key,
        tag=None,
        iter=None,
        project="",
        tree=None,
        uid=None,
        format_: mlrun.common.formatters.ArtifactFormat = mlrun.common.formatters.ArtifactFormat.full,
    ):
        """Read an artifact, identified by its key, tag, tree and iteration.

        :param key: Identifying key of the artifact.
        :param tag: Tag of the artifact.
        :param iter: The iteration which generated this artifact (where ``iter=0`` means the root iteration).
        :param project: Project that the artifact belongs to.
        :param tree: The tree which generated this artifact.
        :param uid: A unique ID for this specific version of the artifact (the uid that was generated in the backend)
        :param format_: The format in which to return the artifact. Default is 'full'.
        """

        project = project or mlrun.mlconf.default_project
        tag = tag or "latest"
        endpoint_path = f"projects/{project}/artifacts/{key}"
        error = f"read artifact {project}/{key}"
        params = {
            "format": format_,
            "tag": tag,
            "tree": tree,
            "object-uid": uid,
        }
        if iter is not None:
            params["iter"] = str(iter)
        resp = self.api_call("GET", endpoint_path, error, params=params, version="v2")
        return resp.json()

    def del_artifact(
        self,
        key,
        tag=None,
        project="",
        tree=None,
        uid=None,
        deletion_strategy: mlrun.common.schemas.artifact.ArtifactsDeletionStrategies = (
            mlrun.common.schemas.artifact.ArtifactsDeletionStrategies.metadata_only
        ),
        secrets: Optional[dict] = None,
        iter=None,
    ):
        """Delete an artifact.

        :param key: Identifying key of the artifact.
        :param tag: Tag of the artifact.
        :param project: Project that the artifact belongs to.
        :param tree: The tree which generated this artifact.
        :param uid: A unique ID for this specific version of the artifact (the uid that was generated in the backend)
        :param deletion_strategy: The artifact deletion strategy types.
        :param secrets: Credentials needed to access the artifact data.
        """
        project = project or mlrun.mlconf.default_project
        endpoint_path = f"projects/{project}/artifacts/{key}"
        params = {
            "key": key,
            "tag": tag,
            "tree": tree,
            "object-uid": uid,
            "iter": iter,
            "deletion_strategy": deletion_strategy,
        }
        error = f"del artifact {project}/{key}"
        self.api_call(
            "DELETE",
            endpoint_path,
            error,
            params=params,
            version="v2",
            body=dict_to_json(secrets),
        )

    def list_artifacts(
        self,
        name: Optional[str] = None,
        project: Optional[str] = None,
        tag: Optional[str] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        iter: Optional[int] = None,
        best_iteration: bool = False,
        kind: Optional[str] = None,
        category: Union[str, mlrun.common.schemas.ArtifactCategories] = None,
        tree: Optional[str] = None,
        producer_uri: Optional[str] = None,
        format_: Optional[
            mlrun.common.formatters.ArtifactFormat
        ] = mlrun.common.formatters.ArtifactFormat.full,
        limit: Optional[int] = None,
        partition_by: Optional[
            Union[mlrun.common.schemas.ArtifactPartitionByField, str]
        ] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Optional[
            Union[mlrun.common.schemas.SortField, str]
        ] = mlrun.common.schemas.SortField.updated,
        partition_order: Union[
            mlrun.common.schemas.OrderType, str
        ] = mlrun.common.schemas.OrderType.desc,
    ) -> ArtifactList:
        """List artifacts filtered by various parameters.

        Examples::

            # Show latest version of all artifacts in project
            latest_artifacts = db.list_artifacts(tag="latest", project="iris")
            # check different artifact versions for a specific artifact
            result_versions = db.list_artifacts("results", tag="*", project="iris")
            # Show artifacts with label filters - both uploaded and of binary type
            result_labels = db.list_artifacts(
                "results", tag="*", project="iris", labels=["uploaded", "type=binary"]
            )

        :param name: Name of artifacts to retrieve. Name with '~' prefix is used as a like query, and is not
            case-sensitive. This means that querying for ``~name`` may return artifacts named
            ``my_Name_1`` or ``surname``.
        :param project: Project name.
        :param tag: Return artifacts assigned this tag.
        :param labels: Filter artifacts by label key-value pairs or key existence. This can be provided as:
            - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
            or `{"label": None}` to check for key existence.
            - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
            or just `"label"` for key existence.
            - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
            the specified key-value pairs or key existence.
        :param since: Return artifacts updated after this date (as datetime object).
        :param until: Return artifacts updated before this date (as datetime object).
        :param iter: Return artifacts from a specific iteration (where ``iter=0`` means the root iteration). If
            ``None`` (default) return artifacts from all iterations.
        :param best_iteration: Returns the artifact which belongs to the best iteration of a given run, in the case of
            artifacts generated from a hyper-param run. If only a single iteration exists, will return the artifact
            from that iteration. If using ``best_iter``, the ``iter`` parameter must not be used.
        :param kind:            Return artifacts of the requested kind.
        :param category:        Return artifacts of the requested category.
        :param tree:            Return artifacts of the requested tree.
        :param producer_uri:    Return artifacts produced by the requested producer URI. Producer URI usually
            points to a run and is used to filter artifacts by the run that produced them when the artifact producer id
            is a workflow id (artifact was created as part of a workflow).
        :param format_: The format in which to return the artifacts. Default is 'full'.
        :param limit: Deprecated - Maximum number of artifacts to return (will be removed in 1.11.0).
        :param partition_by: Field to group results by. When `partition_by` is specified, the `partition_sort_by`
            parameter must be provided as well.
        :param rows_per_partition: How many top rows (per sorting defined by `partition_sort_by` and `partition_order`)
            to return per group. Default value is 1.
        :param partition_sort_by: What field to sort the results by, within each partition defined by `partition_by`.
            Currently, the only allowed values are `created` and `updated`.
        :param partition_order: Order of sorting within partitions - `asc` or `desc`. Default is `desc`.
        """

        artifacts, _ = self._list_artifacts(
            name=name,
            project=project,
            tag=tag,
            labels=labels,
            since=since,
            until=until,
            iter=iter,
            best_iteration=best_iteration,
            kind=kind,
            category=category,
            tree=tree,
            producer_uri=producer_uri,
            format_=format_,
            limit=limit,
            partition_by=partition_by,
            rows_per_partition=rows_per_partition,
            partition_sort_by=partition_sort_by,
            partition_order=partition_order,
            return_all=not limit,
        )
        return artifacts

    def paginated_list_artifacts(
        self,
        *args,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        **kwargs,
    ) -> tuple[ArtifactList, Optional[str]]:
        """List artifacts with support for pagination and various filtering options.

        This method retrieves a paginated list of artifacts based on the specified filter parameters.
        Pagination is controlled using the `page`, `page_size`, and `page_token` parameters. The method
        will return a list of artifacts that match the filtering criteria provided.

        For detailed information about the parameters, refer to the list_artifacts method:
            See :py:func:`~list_artifacts` for more details.

        Examples::

            # Fetch first page of artifacts with page size of 5
            artifacts, token = db.paginated_list_artifacts(
                project="my-project", page_size=5
            )
            # Fetch next page using the pagination token from the previous response
            artifacts, token = db.paginated_list_artifacts(
                project="my-project", page_token=token
            )
            # Fetch artifacts for a specific page (e.g., page 3)
            artifacts, token = db.paginated_list_artifacts(
                project="my-project", page=3, page_size=5
            )

            # Automatically iterate over all pages without explicitly specifying the page number
            artifacts = []
            token = None
            while True:
                page_artifacts, token = db.paginated_list_artifacts(
                    project="my-project", page_token=token, page_size=5
                )
                artifacts.extend(page_artifacts)

                # If token is None and page_artifacts is empty, we've reached the end (no more artifacts).
                # If token is None and page_artifacts is not empty, we've fetched the last page of artifacts.
                if not token:
                    break
            print(f"Total artifacts retrieved: {len(artifacts)}")

        :param page: The page number to retrieve. If not provided, the next page will be retrieved.
        :param page_size: The number of items per page to retrieve. Up to `page_size` responses are expected.
            Defaults to `mlrun.mlconf.httpdb.pagination.default_page_size` if not provided.
        :param page_token: A pagination token used to retrieve the next page of results. Should not be provided
            for the first request.

        :returns: A tuple containing the list of artifacts and an optional `page_token` for pagination.
        """

        return self._list_artifacts(
            *args,
            page=page,
            page_size=page_size,
            page_token=page_token,
            return_all=False,
            **kwargs,
        )

    def del_artifacts(
        self,
        name: Optional[str] = None,
        project: Optional[str] = None,
        tag: Optional[str] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
        days_ago=0,
        tree: Optional[str] = None,
    ):
        """Delete artifacts referenced by the parameters.

        :param name: Name of artifacts to delete. Note that this is a like query, and is case-insensitive. See
            :py:func:`~list_artifacts` for more details.
        :param project: Project that artifacts belong to.
        :param tag: Choose artifacts who are assigned this tag.
        :param labels: Filter artifacts by label key-value pairs or key existence. This can be provided as:
            - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
            or `{"label": None}` to check for key existence.
            - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
            or just `"label"` for key existence.
            - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
            the specified key-value pairs or key existence.
        :param days_ago: This parameter is deprecated and not used.
        :param tree: Delete artifacts filtered by tree.
        """
        project = project or config.default_project
        labels = self._parse_labels(labels)

        params = {
            "name": name,
            "tag": tag,
            "tree": tree,
            "label": labels,
            "days_ago": str(days_ago),
        }
        error = "del artifacts"
        endpoint_path = f"projects/{project}/artifacts"
        self.api_call("DELETE", endpoint_path, error, params=params, version="v2")

    def list_artifact_tags(
        self,
        project=None,
        category: Union[str, mlrun.common.schemas.ArtifactCategories] = None,
    ) -> list[str]:
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
        response = self.api_call("DELETE", path, error_message, version="v2")
        if response.status_code == http.HTTPStatus.ACCEPTED:
            logger.info(
                "Function is being deleted", project_name=project, function_name=name
            )
            background_task = mlrun.common.schemas.BackgroundTask(**response.json())
            background_task = self._wait_for_background_task_to_reach_terminal_state(
                background_task.metadata.name, project=project
            )
            if (
                background_task.status.state
                == mlrun.common.schemas.BackgroundTaskState.succeeded
            ):
                logger.info(
                    "Function deleted", project_name=project, function_name=name
                )
            elif (
                background_task.status.state
                == mlrun.common.schemas.BackgroundTaskState.failed
            ):
                logger.info(
                    "Function deletion failed",
                    reason=background_task.status.error,
                    project_name=project,
                    function_name=name,
                )

    def list_functions(
        self,
        name: Optional[str] = None,
        project: Optional[str] = None,
        tag: Optional[str] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        kind: Optional[str] = None,
        format_: mlrun.common.formatters.FunctionFormat = mlrun.common.formatters.FunctionFormat.full,
        states: typing.Optional[list[mlrun.common.schemas.FunctionState]] = None,
    ):
        """Retrieve a list of functions, filtered by specific criteria.

        :param name: Return only functions with a specific name.
        :param project: Return functions belonging to this project. If not specified, the default project is used.
        :param tag: Return function versions with specific tags. To return only tagged functions, set tag to ``"*"``.
        :param labels: Filter functions by label key-value pairs or key existence. This can be provided as:
            - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
            or `{"label": None}` to check for key existence.
            - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
            or just `"label"` for key existence.
            - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
            the specified key-value pairs or key existence.
        :param since: Return functions updated after this date (as datetime object).
        :param until: Return functions updated before this date (as datetime object).
        :param kind: Return only functions of a specific kind.
        :param format_: The format in which to return the functions. Default is 'full'.
        :param states: Return only functions whose state is one of the provided states.
        :returns: List of function objects (as dictionary).
        """
        functions, _ = self._list_functions(
            name=name,
            project=project,
            tag=tag,
            kind=kind,
            labels=labels,
            format_=format_,
            since=since,
            until=until,
            states=states,
            return_all=True,
        )
        return functions

    def paginated_list_functions(
        self,
        *args,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        **kwargs,
    ) -> tuple[list[dict], Optional[str]]:
        """List functions with support for pagination and various filtering options.

        This method retrieves a paginated list of functions based on the specified filter parameters.
        Pagination is controlled using the `page`, `page_size`, and `page_token` parameters. The method
        will return a list of functions that match the filtering criteria provided.

        For detailed information about the parameters, refer to the list_functions method:
            See :py:func:`~list_functions` for more details.

        Examples::

            # Fetch first page of functions with page size of 5
            functions, token = db.paginated_list_functions(
                project="my-project", page_size=5
            )
            # Fetch next page using the pagination token from the previous response
            functions, token = db.paginated_list_functions(
                project="my-project", page_token=token
            )
            # Fetch functions for a specific page (e.g., page 3)
            functions, token = db.paginated_list_functions(
                project="my-project", page=3, page_size=5
            )

            # Automatically iterate over all pages without explicitly specifying the page number
            functions = []
            token = None
            while True:
                page_functions, token = db.paginated_list_functions(
                    project="my-project", page_token=token, page_size=5
                )
                functions.extend(page_functions)

                # If token is None and page_functions is empty, we've reached the end (no more functions).
                # If token is None and page_functions is not empty, we've fetched the last page of functions.
                if not token:
                    break
            print(f"Total functions retrieved: {len(functions)}")

        :param page: The page number to retrieve. If not provided, the next page will be retrieved.
        :param page_size: The number of items per page to retrieve. Up to `page_size` responses are expected.
            Defaults to `mlrun.mlconf.httpdb.pagination.default_page_size` if not provided.
        :param page_token: A pagination token used to retrieve the next page of results. Should not be provided
            for the first request.

        :returns: A tuple containing the list of functions objects (as dictionary) and an optional
            `page_token` for pagination.
        """
        return self._list_functions(
            *args,
            page=page,
            page_size=page_size,
            page_token=page_token,
            return_all=False,
            **kwargs,
        )

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
                    structured_dict.setdefault(project, {})[job_id] = (
                        mlrun.common.schemas.RuntimeResources(**runtime_resources)
                    )
            return structured_dict
        elif group_by == mlrun.common.schemas.ListRuntimeResourcesGroupByField.project:
            structured_dict = {}
            for project, kind_runtime_resources_map in response.json().items():
                for kind, runtime_resources in kind_runtime_resources_map.items():
                    structured_dict.setdefault(project, {})[kind] = (
                        mlrun.common.schemas.RuntimeResources(**runtime_resources)
                    )
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
        grace_period: Optional[int] = None,
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
            the moment they moved to terminal state
            (defaults to mlrun.mlconf.runtime_resources_deletion_grace_period).

        :returns: :py:class:`~mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput` listing the runtime resources
            that were removed.
        """
        params = {
            "label-selector": label_selector,
            "kind": kind,
            "object-id": object_id,
            "force": force,
        }
        if grace_period is not None:
            params["grace-period"] = grace_period
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
                structured_dict.setdefault(project, {})[kind] = (
                    mlrun.common.schemas.RuntimeResources(**runtime_resources)
                )
        return structured_dict

    def create_schedule(
        self, project: str, schedule: mlrun.common.schemas.ScheduleInput
    ):
        """The create_schedule functionality has been deprecated."""
        raise mlrun.errors.MLRunBadRequestError(
            "The create_schedule functionality has been deprecated."
        )

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
        name: Optional[str] = None,
        kind: mlrun.common.schemas.ScheduleKinds = None,
        include_last_run: bool = False,
        next_run_time_since: Optional[datetime] = None,
        next_run_time_until: Optional[datetime] = None,
    ) -> mlrun.common.schemas.SchedulesOutput:
        """Retrieve list of schedules of specific name or kind.

        :param project: Project name.
        :param name: Name of schedule to retrieve. Can be omitted to list all schedules.
        :param kind: Kind of schedule objects to retrieve, can be either ``job`` or ``pipeline``.
        :param include_last_run: Whether to return for each schedule returned also the results of the last run of
            that schedule.
        :param next_run_time_since: Return only schedules with next run time after this date.
        :param next_run_time_until: Return only schedules with next run time before this date.
        """

        project = project or config.default_project
        params = {
            "kind": kind,
            "name": name,
            "include_last_run": include_last_run,
            "next_run_time_since": datetime_to_iso(next_run_time_since),
            "next_run_time_until": datetime_to_iso(next_run_time_until),
        }
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
        func: BaseRuntime,
        with_mlrun: bool,
        mlrun_version_specifier: Optional[str] = None,
        skip_deployed: bool = False,
        builder_env: Optional[dict] = None,
        force_build: bool = False,
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
        :param force_build:   Force building the image, even when no changes were made
        """
        self.warn_on_s3_and_ecr_permissions_conflict(func)
        try:
            req = {
                "function": func.to_dict(),
                "with_mlrun": bool2str(with_mlrun),
                "skip_deployed": skip_deployed,
                "force_build": force_build,
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
            raise ValueError("bad submit build response")

        return resp.json()

    def deploy_nuclio_function(
        self,
        func: mlrun.runtimes.RemoteRuntime,
        builder_env: Optional[dict] = None,
    ):
        """
        Deploy a Nuclio function.

        :param func:            Function to build.
        :param builder_env:     Kaniko builder pod env vars dict (for config/credentials)
        """
        func.metadata.project = func.metadata.project or config.default_project
        self.warn_on_s3_and_ecr_permissions_conflict(func)
        try:
            req = {
                "function": func.to_dict(),
            }
            if builder_env:
                req["builder_env"] = builder_env
            _path = (
                f"projects/{func.metadata.project}/nuclio/{func.metadata.name}/deploy"
            )
            resp = self.api_call("POST", _path, json=req)
        except OSError as err:
            logger.error(f"error submitting nuclio deploy task: {err_to_str(err)}")
            raise OSError(f"error: cannot submit deploy, {err_to_str(err)}")

        if not resp.ok:
            logger.error(f"deploy nuclio - bad response:\n{resp.text}")
            raise ValueError("bad nuclio deploy response")

        return resp.json()

    def get_nuclio_deploy_status(
        self,
        func: mlrun.runtimes.RemoteRuntime,
        last_log_timestamp: float = 0.0,
        verbose: bool = False,
    ):
        """Retrieve the status of a deploy operation currently in progress.

        :param func:                Function object that is being built.
        :param last_log_timestamp:  Last timestamp of logs that were already retrieved. Function will return only logs
                                    later than this parameter.
        :param verbose:             Add verbose logs into the output.

        :returns: The following parameters:

            - Text of builder logs.
            - Timestamp of last log retrieved, to be used in subsequent calls to this function.
        """

        try:
            normalized_name = normalize_name(func.metadata.name)
            params = {
                "name": normalized_name,
                "project": func.metadata.project,
                "tag": func.metadata.tag,
                "last_log_timestamp": str(last_log_timestamp),
                "verbose": bool2str(verbose),
            }
            _path = f"projects/{func.metadata.project}/nuclio/{normalized_name}/deploy"
            resp = self.api_call("GET", _path, params=params)
        except OSError as err:
            logger.error(f"error getting deploy status: {err_to_str(err)}")
            raise OSError(f"error: cannot get deploy status, {err_to_str(err)}")

        if not resp.ok:
            logger.warning(f"failed resp, {resp.text}")
            raise RunDBError("bad function build response")

        if resp.headers:
            last_log_timestamp = float(
                resp.headers.get("x-mlrun-last-timestamp", "0.0")
            )
            mlrun.runtimes.nuclio.function.enrich_nuclio_function_from_headers(
                func, resp.headers
            )

        text = ""
        if resp.content:
            text = resp.content.decode()
        return text, last_log_timestamp

    def get_builder_status(
        self,
        func: BaseRuntime,
        offset: int = 0,
        logs: bool = True,
        last_log_timestamp: float = 0.0,
        verbose: bool = False,
        events_offset: int = 0,
    ):
        """Retrieve the status of a build operation currently in progress.

        :param func:                Function object that is being built.
        :param offset:              Offset into the build logs to retrieve logs from.
        :param logs:                Should build logs be retrieved.
        :param last_log_timestamp:  Last timestamp of logs that were already retrieved. Function will return only logs
                                    later than this parameter.
        :param verbose:             Add verbose logs into the output.
        :param events_offset:       Offset into the build events to retrieve events from.

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
                "events_offset": str(events_offset),
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

        deploy_status_text_kind = mlrun.common.constants.DeployStatusTextKind.logs
        if resp.headers:
            func.status.state = resp.headers.get("x-mlrun-function-status", "")
            last_log_timestamp = float(
                resp.headers.get("x-mlrun-last-timestamp", "0.0")
            )
            if func.kind in mlrun.runtimes.RuntimeKinds.pure_nuclio_deployed_runtimes():
                mlrun.runtimes.nuclio.function.enrich_nuclio_function_from_headers(
                    func, resp.headers
                )

            builder_pod = resp.headers.get("builder_pod", "")
            if builder_pod:
                func.status.build_pod = builder_pod

            function_image = resp.headers.get("function_image", "")
            if function_image:
                func.spec.image = function_image

            deploy_status_text_kind = resp.headers.get(
                "deploy_status_text_kind",
                mlrun.common.constants.DeployStatusTextKind.logs,
            )

        text = ""
        if resp.content:
            text = resp.content.decode()
        return text, last_log_timestamp, deploy_status_text_kind

    def start_function(
        self,
        func_url: Optional[str] = None,
        function: "mlrun.runtimes.BaseRuntime" = None,
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

    def list_project_background_tasks(
        self,
        project: Optional[str] = None,
        state: Optional[str] = None,
        created_from: Optional[datetime] = None,
        created_to: Optional[datetime] = None,
        last_update_time_from: Optional[datetime] = None,
        last_update_time_to: Optional[datetime] = None,
    ) -> list[mlrun.common.schemas.BackgroundTask]:
        """
        Retrieve updated information on project background tasks being executed.
        If no filter is provided, will return background tasks from the last week.

        :param project: Project name (defaults to mlrun.mlconf.default_project).
        :param state:   List only background tasks whose state is specified.
        :param created_from: Filter by background task created time in ``[created_from, created_to]``.
        :param created_to:  Filter by background task created time in ``[created_from, created_to]``.
        :param last_update_time_from: Filter by background task last update time in
            ``(last_update_time_from, last_update_time_to)``.
        :param last_update_time_to: Filter by background task last update time in
            ``(last_update_time_from, last_update_time_to)``.
        """
        project = project or config.default_project
        if (
            not state
            and not created_from
            and not created_to
            and not last_update_time_from
            and not last_update_time_to
        ):
            # default to last week on no filter
            created_from = datetime.now() - timedelta(days=7)

        params = {
            "state": state,
            "created_from": datetime_to_iso(created_from),
            "created_to": datetime_to_iso(created_to),
            "last_update_time_from": datetime_to_iso(last_update_time_from),
            "last_update_time_to": datetime_to_iso(last_update_time_to),
        }

        path = f"projects/{project}/background-tasks"
        error_message = f"Failed listing project background task. project={project}"
        response = self.api_call("GET", path, error_message, params=params)
        return mlrun.common.schemas.BackgroundTaskList(
            **response.json()
        ).background_tasks

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
        cleanup_ttl=None,
        timeout=60,
    ):
        """Submit a KFP pipeline for execution.

        :param project:         The project of the pipeline
        :param pipeline:        Pipeline function or path to .yaml/.zip pipeline file.
        :param arguments:       A dictionary of arguments to pass to the pipeline.
        :param experiment:      A name to assign for the specific experiment.
        :param run:             A name for this specific run.
        :param namespace:       Kubernetes namespace to execute the pipeline in.
        :param artifact_path:   A path to artifacts used by this pipeline.
        :param ops:             Transformers to apply on all ops in the pipeline.
        :param cleanup_ttl:     Pipeline cleanup ttl in secs (time to wait after workflow completion, at which point the
                                workflow and all its resources are deleted)
        :param timeout:         Timeout for the API call.
        """

        if isinstance(pipeline, str):
            pipe_file = pipeline
        else:
            pipe_file = compile_pipeline(
                artifact_path=artifact_path,
                cleanup_ttl=cleanup_ttl,
                ops=ops,
                pipeline=pipeline,
            )

        if pipe_file.endswith(".yaml"):
            headers = {"content-type": "application/yaml"}
        elif pipe_file.endswith(".zip"):
            headers = {"content-type": "application/zip"}
        else:
            raise ValueError("'pipeline' file must be .yaml or .zip")
        if arguments:
            if not isinstance(arguments, dict):
                raise ValueError("'arguments' must be dict type")
            headers[mlrun.common.schemas.HeaderNames.pipeline_arguments] = str(
                arguments
            )

        if not path.isfile(pipe_file):
            raise OSError(f"File {pipe_file} doesnt exist")
        with open(pipe_file, "rb") as fp:
            data = fp.read()
            if not data:
                raise ValueError("The compiled pipe file is empty")
        if not isinstance(pipeline, str):
            remove(pipe_file)

        try:
            params = {"namespace": namespace, "experiment": experiment, "run": run}
            resp = self.api_call(
                "POST",
                f"projects/{project}/pipelines",
                params=params,
                timeout=timeout,
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
        namespace: Optional[str] = None,
        sort_by: str = "",
        page_token: str = "",
        filter_: str = "",
        format_: Union[
            str, mlrun.common.formatters.PipelineFormat
        ] = mlrun.common.formatters.PipelineFormat.metadata_only,
        page_size: Optional[int] = None,
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
        namespace: Optional[str] = None,
        timeout: int = 30,
        format_: Union[
            str, mlrun.common.formatters.PipelineFormat
        ] = mlrun.common.formatters.PipelineFormat.summary,
        project: Optional[str] = None,
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

    def retry_pipeline(
        self,
        run_id: str,
        project: str,
        namespace: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Retry a specific pipeline run using its run ID. This function sends an API request
        to retry a pipeline run. If a project is specified, the run must belong to that
        project; otherwise, all projects are queried.

        :param run_id: The unique ID of the pipeline run to retry.
        :param namespace: Kubernetes namespace where the pipeline is running. Optional.
        :param timeout: Timeout (in seconds) for the API call. Defaults to 30 seconds.
        :param project: Name of the MLRun project associated with the pipeline.

        :raises ValueError: Raised if the API response is not successful or contains an
            error.

        :return: JSON response containing details of the retried pipeline run.
        """

        params = {}
        if namespace:
            params["namespace"] = namespace

        resp_text = ""
        resp_code = None
        try:
            resp = self.api_call(
                "POST",
                f"projects/{project}/pipelines/{run_id}/retry",
                params=params,
                timeout=timeout,
            )
            resp_code = resp.status_code
            resp_text = resp.text
            if not resp.ok:
                raise mlrun.errors.MLRunHTTPError(
                    f"Failed to retry pipeline run '{run_id}'. "
                    f"HTTP {resp_code}: {resp_text}"
                )
        except Exception as exc:
            logger.error(
                "Retry pipeline API call encountered an error.",
                run_id=run_id,
                project=project,
                namespace=namespace,
                response_code=resp_code,
                response_text=resp_text,
                error=str(exc),
            )
            if isinstance(exc, mlrun.errors.MLRunHTTPError):
                raise exc  # Re-raise known HTTP errors
            raise mlrun.errors.MLRunRuntimeError(
                f"Unexpected error while retrying pipeline run '{run_id}'."
            ) from exc

        logger.info(
            "Successfully retried pipeline run",
            run_id=run_id,
            project=project,
            namespace=namespace,
        )
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
        self,
        name: str,
        project: str = "",
        tag: Optional[str] = None,
        uid: Optional[str] = None,
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
        project: Optional[str] = None,
        name: Optional[str] = None,
        tag: Optional[str] = None,
        entities: Optional[list[str]] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
    ) -> list[dict]:
        """List feature-sets which contain specific features. This function may return multiple versions of the same
        feature-set if a specific tag is not requested. Note that the various filters of this function actually
        refer to the feature-set object containing the features, not to the features themselves.

        :param project: Project which contains these features.
        :param name: Name of the feature to look for. The name is used in a like query, and is not case-sensitive. For
            example, looking for ``feat`` will return features which are named ``MyFeature`` as well as ``defeat``.
        :param tag: Return feature-sets which contain the features looked for, and are tagged with the specific tag.
        :param entities: Return only feature-sets which contain an entity whose name is contained in this list.
        :param labels: Filter feature-sets by label key-value pairs or key existence. This can be provided as:
            - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
            or `{"label": None}` to check for key existence.
            - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
            or just `"label"` for key existence.
            - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
            the specified key-value pairs or key existence.
        :returns: A list of mapping from feature to a digest of the feature-set, which contains the feature-set
            meta-data. Multiple entries may be returned for any specific feature due to multiple tags or versions
            of the feature-set.
        """

        project = project or config.default_project
        labels = self._parse_labels(labels)
        params = {
            "name": name,
            "tag": tag,
            "entity": entities or [],
            "label": labels,
        }

        path = f"projects/{project}/features"

        error_message = f"Failed listing features, project: {project}, query: {params}"
        resp = self.api_call("GET", path, error_message, params=params)
        return resp.json()["features"]

    def list_features_v2(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        tag: Optional[str] = None,
        entities: Optional[list[str]] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
    ) -> dict[str, list[dict]]:
        """List feature-sets which contain specific features. This function may return multiple versions of the same
        feature-set if a specific tag is not requested. Note that the various filters of this function actually
        refer to the feature-set object containing the features, not to the features themselves.

        :param project: Project which contains these features.
        :param name: Name of the feature to look for. The name is used in a like query, and is not case-sensitive. For
            example, looking for ``feat`` will return features which are named ``MyFeature`` as well as ``defeat``.
        :param tag: Return feature-sets which contain the features looked for, and are tagged with the specific tag.
        :param entities: Return only feature-sets which contain an entity whose name is contained in this list.
        :param labels: Filter feature-sets by label key-value pairs or key existence. This can be provided as:
            - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
            or `{"label": None}` to check for key existence.
            - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
            or just `"label"` for key existence.
            - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
            the specified key-value pairs or key existence.
        :returns: A list of features, and a list of their corresponding feature sets.
        """

        project = project or config.default_project
        labels = self._parse_labels(labels)
        params = {
            "name": name,
            "tag": tag,
            "entity": entities or [],
            "label": labels,
        }

        path = f"projects/{project}/features"

        error_message = f"Failed listing features, project: {project}, query: {params}"
        resp = self.api_call("GET", path, error_message, params=params, version="v2")
        return resp.json()

    def list_entities(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        tag: Optional[str] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
    ) -> list[dict]:
        """Retrieve a list of entities and their mapping to the containing feature-sets. This function is similar
        to the :py:func:`~list_features` function, and uses the same logic. However, the entities are matched
        against the name rather than the features.

        :param project: The project containing the entities.
        :param name: The name of the entities to retrieve.
        :param tag: The tag of the specific entity version to retrieve.
        :param labels: Filter entities by label key-value pairs or key existence. This can be provided as:
            - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
            or `{"label": None}` to check for key existence.
            - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
            or just `"label"` for key existence.
            - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
            the specified key-value pairs or key existence.
        :returns: A list of entities.
        """

        project = project or config.default_project
        labels = self._parse_labels(labels)
        params = {
            "name": name,
            "tag": tag,
            "label": labels,
        }

        path = f"projects/{project}/entities"

        error_message = f"Failed listing entities, project: {project}, query: {params}"
        resp = self.api_call("GET", path, error_message, params=params)
        return resp.json()["entities"]

    def list_entities_v2(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        tag: Optional[str] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
    ) -> dict[str, list[dict]]:
        """Retrieve a list of entities and their mapping to the containing feature-sets. This function is similar
        to the :py:func:`~list_features_v2` function, and uses the same logic. However, the entities are matched
        against the name rather than the features.

        :param project: The project containing the entities.
        :param name: The name of the entities to retrieve.
        :param tag: The tag of the specific entity version to retrieve.
        :param labels: Filter entities by label key-value pairs or key existence. This can be provided as:
            - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
            or `{"label": None}` to check for key existence.
            - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
            or just `"label"` for key existence.
            - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
            the specified key-value pairs or key existence.
        :returns: A list of entities.
        """

        project = project or config.default_project
        labels = self._parse_labels(labels)
        params = {
            "name": name,
            "tag": tag,
            "label": labels,
        }

        path = f"projects/{project}/entities"

        error_message = f"Failed listing entities, project: {project}, query: {params}"
        resp = self.api_call("GET", path, error_message, params=params, version="v2")
        return resp.json()

    @staticmethod
    def _generate_partition_by_params(
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
        project: Optional[str] = None,
        name: Optional[str] = None,
        tag: Optional[str] = None,
        state: Optional[str] = None,
        entities: Optional[list[str]] = None,
        features: Optional[list[str]] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
        partition_by: Union[
            mlrun.common.schemas.FeatureStorePartitionByField, str
        ] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Union[mlrun.common.schemas.SortField, str] = None,
        partition_order: Union[
            mlrun.common.schemas.OrderType, str
        ] = mlrun.common.schemas.OrderType.desc,
        format_: Union[
            str, mlrun.common.formatters.FeatureSetFormat
        ] = mlrun.common.formatters.FeatureSetFormat.full,
    ) -> list[FeatureSet]:
        """Retrieve a list of feature-sets matching the criteria provided.

        :param project: Project name.
        :param name: Name of feature-set to match. This is a like query, and is case-insensitive.
        :param tag: Match feature-sets with specific tag.
        :param state: Match feature-sets with a specific state.
        :param entities: Match feature-sets which contain entities whose name is in this list.
        :param features: Match feature-sets which contain features whose name is in this list.
        :param labels: Filter feature-sets by label key-value pairs or key existence. This can be provided as:
            - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
            or `{"label": None}` to check for key existence.
            - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
            or just `"label"` for key existence.
            - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
            the specified key-value pairs or key existence.
        :param partition_by: Field to group results by. Only allowed value is `name`. When `partition_by` is specified,
            the `partition_sort_by` parameter must be provided as well.
        :param rows_per_partition: How many top rows (per sorting defined by `partition_sort_by` and `partition_order`)
            to return per group. Default value is 1.
        :param partition_sort_by: What field to sort the results by, within each partition defined by `partition_by`.
            Currently the only allowed value are `created` and `updated`.
        :param partition_order: Order of sorting within partitions - `asc` or `desc`. Default is `desc`.
        :param format_: Format of the results. Possible values are:
            - ``minimal`` - Return minimal feature set objects, not including stats and preview for each feature set.
            - ``full`` - Return full feature set objects.
        :returns: List of matching :py:class:`~mlrun.feature_store.FeatureSet` objects.
        """

        project = project or config.default_project
        labels = self._parse_labels(labels)
        params = {
            "name": name,
            "state": state,
            "tag": tag,
            "entity": entities or [],
            "feature": features or [],
            "label": labels,
            "format": format_,
        }
        if partition_by:
            params.update(
                self._generate_partition_by_params(
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

                feature_set_update = {"status": {"processed": True}}

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
        self,
        name: str,
        project: str = "",
        tag: Optional[str] = None,
        uid: Optional[str] = None,
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
        project: Optional[str] = None,
        name: Optional[str] = None,
        tag: Optional[str] = None,
        state: Optional[str] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
        partition_by: Union[
            mlrun.common.schemas.FeatureStorePartitionByField, str
        ] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Union[mlrun.common.schemas.SortField, str] = None,
        partition_order: Union[
            mlrun.common.schemas.OrderType, str
        ] = mlrun.common.schemas.OrderType.desc,
    ) -> list[FeatureVector]:
        """Retrieve a list of feature-vectors matching the criteria provided.

        :param project: Project name.
        :param name: Name of feature-vector to match. This is a like query, and is case-insensitive.
        :param tag: Match feature-vectors with specific tag.
        :param state: Match feature-vectors with a specific state.
        :param labels: Filter feature-vectors by label key-value pairs or key existence. This can be provided as:
            - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
            or `{"label": None}` to check for key existence.
            - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
            or just `"label"` for key existence.
            - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
            the specified key-value pairs or key existence.
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
        labels = self._parse_labels(labels)
        params = {
            "name": name,
            "state": state,
            "tag": tag,
            "label": labels,
        }
        if partition_by:
            params.update(
                self._generate_partition_by_params(
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
        artifacts: Union[list[Artifact], list[dict], Artifact, dict],
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
        owner: Optional[str] = None,
        format_: Union[
            str, mlrun.common.formatters.ProjectFormat
        ] = mlrun.common.formatters.ProjectFormat.name_only,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
        state: Union[str, mlrun.common.schemas.ProjectState] = None,
    ) -> list[Union[mlrun.projects.MlrunProject, str]]:
        """Return a list of the existing projects, potentially filtered by specific criteria.

        :param owner: List only projects belonging to this specific owner.
        :param format_: Format of the results. Possible values are:

            - ``name_only`` (default value) - Return just the names of the projects.
            - ``minimal`` - Return minimal project objects (minimization happens in the BE).
            - ``full``  - Return full project objects.

        :param labels: Filter projects by label key-value pairs or key existence. This can be provided as:
            - A dictionary in the format `{"label": "value"}` to match specific label key-value pairs,
            or `{"label": None}` to check for key existence.
            - A list of strings formatted as `"label=value"` to match specific label key-value pairs,
            or just `"label"` for key existence.
            - A comma-separated string formatted as `"label1=value1,label2"` to match entities with
            the specified key-value pairs or key existence.
        :param state: Filter by project's state. Can be either ``online`` or ``archived``.
        """
        labels = self._parse_labels(labels)

        params = {
            "owner": owner,
            "state": state,
            "format": format_,
            "label": labels,
        }

        error_message = f"Failed listing projects, query: {params}"
        response = self.api_call("GET", "projects", error_message, params=params)
        if format_ == mlrun.common.formatters.ProjectFormat.name_only:
            # projects is just a list of strings
            return response.json()["projects"]

        # forwards compatibility - we want to be able to handle new formats that might be added in the future
        # if format is not known to the api, it is up to the server to return either an error or a default format
        return [
            mlrun.projects.MlrunProject.from_dict(project_dict)
            for project_dict in response.json()["projects"]
        ]

    def get_project(self, name: str) -> "mlrun.MlrunProject":
        """Get details for a specific project."""

        if not name:
            raise MLRunInvalidArgumentError("Name must be provided")

        path = f"projects/{name}"
        error_message = f"Failed retrieving project {name}"
        response = self.api_call("GET", path, error_message)
        return mlrun.MlrunProject.from_dict(response.json())

    def delete_project(
        self,
        name: str,
        deletion_strategy: Union[
            str, mlrun.common.schemas.DeletionStrategy
        ] = mlrun.common.schemas.DeletionStrategy.default(),
    ) -> None:
        """Delete a project.

        :param name: Name of the project to delete.
        :param deletion_strategy: How to treat resources related to the project. Possible values are:

            - ``restrict`` (default) - Project must not have any related resources when deleted. If using
              this mode while related resources exist, the operation will fail.
            - ``cascade`` - Automatically delete all related resources when deleting the project.
        """

        headers = {
            mlrun.common.schemas.HeaderNames.deletion_strategy: deletion_strategy
        }
        error_message = f"Failed deleting project {name}"
        response = self.api_call(
            "DELETE", f"projects/{name}", error_message, headers=headers, version="v2"
        )
        if response.status_code == http.HTTPStatus.ACCEPTED:
            logger.info("Waiting for project to be deleted", project_name=name)
            background_task = mlrun.common.schemas.BackgroundTask(**response.json())
            background_task = self._wait_for_background_task_to_reach_terminal_state(
                background_task.metadata.name
            )
            if (
                background_task.status.state
                == mlrun.common.schemas.BackgroundTaskState.succeeded
            ):
                logger.info("Project deleted", project_name=name)
            elif (
                background_task.status.state
                == mlrun.common.schemas.BackgroundTaskState.failed
            ):
                logger.error(
                    "Project deletion failed",
                    project_name=name,
                    error=background_task.status.error,
                )
        elif response.status_code == http.HTTPStatus.NO_CONTENT:
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
        self, name: str, project: str = ""
    ) -> mlrun.common.schemas.BackgroundTask:
        def _verify_background_task_in_terminal_state():
            if project:
                background_task = self.get_project_background_task(project, name)
            else:
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

    def create_project_secrets(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: Optional[dict] = None,
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

                secrets = {"password": "myPassw0rd", "aws_key": "111222333"}
                db.create_project_secrets(
                    "project1",
                    provider=mlrun.common.schemas.SecretProviderName.kubernetes,
                    secrets=secrets,
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
        token: Optional[str] = None,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: Optional[list[str]] = None,
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
        token: Optional[str] = None,
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
        secrets: Optional[list[str]] = None,
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

    def get_model_endpoint_monitoring_metrics(
        self,
        project: str,
        endpoint_id: str,
        type: Literal["results", "metrics", "all"] = "all",
    ) -> list[mm_endpoints.ModelEndpointMonitoringMetric]:
        """Get application metrics/results by endpoint id and project.

        :param project: The name of the project.
        :param endpoint_id: The unique id of the model endpoint.
        :param type: The type of the metrics to return. "all" means "results" and "metrics".

        :return: A list of the application metrics or/and results for this model endpoint.
        """
        path = f"projects/{project}/model-endpoints/{endpoint_id}/metrics"
        params = {"type": type}
        error_message = (
            f"Failed to get model endpoint monitoring metrics,"
            f" endpoint_id: {endpoint_id}, project: {project}"
        )
        response = self.api_call(
            mlrun.common.types.HTTPMethod.GET,
            path,
            error_message,
            params=params,
        )
        monitoring_metrics = response.json()
        return parse_obj_as(
            list[mm_endpoints.ModelEndpointMonitoringMetric], monitoring_metrics
        )

    def get_metrics_by_multiple_endpoints(
        self,
        project: str,
        endpoint_ids: Union[str, list[str]],
        type: Literal["results", "metrics", "all"] = "all",
        events_format: mm_constants.GetEventsFormat = mm_constants.GetEventsFormat.SEPARATION,
    ) -> dict[str, list[mm_endpoints.ModelEndpointMonitoringMetric]]:
        """Get application metrics/results by endpoint id and project.

        :param project:         The name of the project.
        :param endpoint_ids:    The unique id of the model endpoint. Can be a single id or a list of ids.
        :param type:            The type of the metrics to return. "all" means "results" and "metrics".
        :param events_format:   response format:

                                separation: {"mep_id1":[...], "mep_id2":[...]}
                                intersection {"intersect_metrics":[], "intersect_results":[]}
        :return: A dictionary of application metrics and/or results for the model endpoints formatted by events_format.
        """
        path = f"projects/{project}/model-endpoints/metrics"
        params = {
            "type": type,
            "endpoint-id": endpoint_ids,
            "events-format": events_format,
        }
        error_message = (
            f"Failed to get model monitoring metrics,"
            f" endpoint_ids: {endpoint_ids}, project: {project}"
        )
        response = self.api_call(
            mlrun.common.types.HTTPMethod.GET,
            path,
            error_message,
            params=params,
        )
        monitoring_metrics_by_endpoint = response.json()
        parsed_metrics_by_endpoint = {}
        for endpoint, metrics in monitoring_metrics_by_endpoint.items():
            parsed_metrics_by_endpoint[endpoint] = parse_obj_as(
                list[mm_endpoints.ModelEndpointMonitoringMetric], metrics
            )
        return parsed_metrics_by_endpoint

    def create_user_secrets(
        self,
        user: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.vault,
        secrets: Optional[dict] = None,
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
        model_endpoint: mlrun.common.schemas.ModelEndpoint,
        creation_strategy: Optional[
            mm_constants.ModelEndpointCreationStrategy
        ] = mm_constants.ModelEndpointCreationStrategy.INPLACE,
    ) -> mlrun.common.schemas.ModelEndpoint:
        """
        Creates a DB record with the given model_endpoint record.

        :param model_endpoint: An object representing the model endpoint.
        :param creation_strategy: Strategy for creating or updating the model endpoint:
            * **overwrite**:
            1. If model endpoints with the same name exist, delete the `latest` one.
            2. Create a new model endpoint entry and set it as `latest`.
            * **inplace** (default):
            1. If model endpoints with the same name exist, update the `latest` entry.
            2. Otherwise, create a new entry.
            * **archive**:
            1. If model endpoints with the same name exist, preserve them.
            2. Create a new model endpoint with the same name and set it to `latest`.
        :return: The created model endpoint object.
        """

        path = f"projects/{model_endpoint.metadata.project}/model-endpoints"
        response = self.api_call(
            method=mlrun.common.types.HTTPMethod.POST,
            path=path,
            body=model_endpoint.json(),
            params={
                "creation-strategy": creation_strategy,
            },
        )
        return mlrun.common.schemas.ModelEndpoint(**response.json())

    def delete_model_endpoint(
        self,
        name: str,
        project: str,
        function_name: Optional[str] = None,
        function_tag: Optional[str] = None,
        endpoint_id: Optional[str] = None,
    ):
        """
        Deletes the DB record of a given model endpoint, project and endpoint_id are used for lookup

        :param name: The name of the model endpoint
        :param project: The name of the project
        :param function_name: The name of the function
        :param function_tag: The tag of the function
        :param endpoint_id: The id of the endpoint
        """
        self._check_model_endpoint_representation(
            function_name, function_tag, endpoint_id
        )
        path = f"projects/{project}/model-endpoints/{name}"
        self.api_call(
            method=mlrun.common.types.HTTPMethod.DELETE,
            path=path,
            params={
                "function-name": function_name,
                "function-tag": function_tag,
                "endpoint-id": endpoint_id,
            },
        )

    def list_model_endpoints(
        self,
        project: str,
        names: Optional[Union[str, list[str]]] = None,
        function_name: Optional[str] = None,
        function_tag: Optional[str] = None,
        model_name: Optional[str] = None,
        model_tag: Optional[str] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        tsdb_metrics: bool = False,
        metric_list: Optional[list[str]] = None,
        top_level: bool = False,
        uids: Optional[list[str]] = None,
        latest_only: bool = False,
    ) -> mlrun.common.schemas.ModelEndpointList:
        """
        List model endpoints with optional filtering by name, function name, model name, labels, and time range.

        :param project:         The name of the project
        :param names:            The name of the model endpoint, or list of names of the model endpoints
        :param function_name:   The name of the function
        :param function_tag:    The tag of the function
        :param model_name:      The name of the model
        :param model_tag:       The tag of the model
        :param labels:          A list of labels to filter by. (see mlrun.common.schemas.LabelsModel)
        :param start:           The start time to filter by.Corresponding to the `created` field.
        :param end:             The end time to filter by. Corresponding to the `created` field.
        :param tsdb_metrics:    Whether to include metrics from the time series DB.
        :param metric_list:     List of metrics to include from the time series DB. Defaults to all metrics.
                                If tsdb_metrics=False, this parameter will be ignored and no tsdb metrics
                                will be included.
        :param top_level:       Whether to return only top level model endpoints.
        :param uids:            A list of unique ids to filter by.
        :param latest_only:     Whether to return only the latest model endpoint version.
        :return:                A list of model endpoints.
        """
        path = f"projects/{project}/model-endpoints"
        labels = self._parse_labels(labels)
        if names and isinstance(names, str):
            names = [names]
        response = self.api_call(
            method=mlrun.common.types.HTTPMethod.GET,
            path=path,
            params={
                "name": names,
                "model-name": model_name,
                "model-tag": model_tag,
                "function-name": function_name,
                "function-tag": function_tag,
                "label": labels,
                "start": datetime_to_iso(start),
                "end": datetime_to_iso(end),
                "tsdb-metrics": tsdb_metrics,
                "metric": metric_list,
                "top-level": top_level,
                "uid": uids,
                "latest-only": latest_only,
            },
        )

        return mlrun.common.schemas.ModelEndpointList(**response.json())

    def get_model_endpoint(
        self,
        name: str,
        project: str,
        function_name: Optional[str] = None,
        function_tag: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        tsdb_metrics: bool = True,
        metric_list: Optional[list[str]] = None,
        feature_analysis: bool = False,
    ) -> mlrun.common.schemas.ModelEndpoint:
        """
        Returns a single `ModelEndpoint` object with additional metrics and feature related data.

        :param name:                       The name of the model endpoint
        :param project:                    The name of the project
        :param function_name:              The name of the function
        :param function_tag:               The tag of the function
        :param endpoint_id:                The id of the endpoint
        :param tsdb_metrics:               Whether to include metrics from the time series DB.
        :param metric_list:                List of metrics to include from the time series DB. Defaults to all metrics.
                                           If tsdb_metrics=False, this parameter will be ignored and no tsdb metrics
                                           will be included.
        :param feature_analysis:           Whether to include feature analysis data (feature_stats,
                                            current_stats & drift_measures).

        :return:                          A `ModelEndpoint` object.
        """
        self._check_model_endpoint_representation(
            function_name, function_tag, endpoint_id
        )
        path = f"projects/{project}/model-endpoints/{name}"
        response = self.api_call(
            method=mlrun.common.types.HTTPMethod.GET,
            path=path,
            params={
                "function-name": function_name,
                "function-tag": function_tag,
                "endpoint-id": endpoint_id,
                "tsdb-metrics": tsdb_metrics,
                "metric": metric_list,
                "feature-analysis": feature_analysis,
            },
        )

        return mlrun.common.schemas.ModelEndpoint(**response.json())

    def patch_model_endpoint(
        self,
        name: str,
        project: str,
        attributes: dict,
        function_name: Optional[str] = None,
        function_tag: Optional[str] = None,
        endpoint_id: Optional[str] = None,
    ) -> None:
        """
        Updates a model endpoint with the given attributes.

        :param name:                       The name of the model endpoint
        :param project:                    The name of the project
        :param attributes:                 The attributes to update
        :param function_name:              The name of the function
        :param function_tag:               The tag of the function
        :param endpoint_id:                The id of the endpoint
        """
        attributes_keys = list(attributes.keys())
        attributes["name"] = name
        attributes["project"] = project
        attributes["function_name"] = function_name or None
        attributes["function_tag"] = function_tag or None
        attributes["uid"] = endpoint_id or None
        model_endpoint = mlrun.common.schemas.ModelEndpoint.from_flat_dict(attributes)
        path = f"projects/{project}/model-endpoints"
        logger.info(
            "Patching model endpoint",
            attributes_keys=attributes_keys,
            model_endpoint=model_endpoint,
        )
        response = self.api_call(
            method=mlrun.common.types.HTTPMethod.PATCH,
            path=path,
            params={
                "attribute-key": attributes_keys,
            },
            body=model_endpoint.json(),
        )
        logger.info(
            "Updating model endpoint done",
            model_endpoint_uid=response.json(),
            status_code=response.status_code,
        )

    @staticmethod
    def _check_model_endpoint_representation(
        function_name: str, function_tag: str, uid: str
    ):
        if not uid and not (function_name and function_tag):
            raise MLRunInvalidArgumentError(
                "Either endpoint_uid or function_name and function_tag must be provided"
            )

    def update_model_monitoring_controller(
        self,
        project: str,
        base_period: int = 10,
        image: str = "mlrun/mlrun",
    ) -> None:
        """
        Redeploy model monitoring application controller function.

        :param project:                  Project name.
        :param base_period:              The time period in minutes in which the model monitoring controller function
                                         triggers. By default, the base period is 10 minutes.
        :param image: The image of the model monitoring controller function.
                                         By default, the image is mlrun/mlrun.
        """
        self.api_call(
            method=mlrun.common.types.HTTPMethod.PATCH,
            path=f"projects/{project}/model-monitoring/controller",
            params={
                "base_period": base_period,
                "image": image,
            },
        )

    def enable_model_monitoring(
        self,
        project: str,
        base_period: int = 10,
        image: str = "mlrun/mlrun",
        deploy_histogram_data_drift_app: bool = True,
        fetch_credentials_from_sys_config: bool = False,
    ) -> None:
        """
        Deploy model monitoring application controller, writer and stream functions.
        While the main goal of the controller function is to handle the monitoring processing and triggering
        applications, the goal of the model monitoring writer function is to write all the monitoring
        application results to the databases.
        The stream function goal is to monitor the log of the data stream. It is triggered when a new log entry
        is detected. It processes the new events into statistics that are then written to statistics databases.

        :param project:                          Project name.
        :param base_period:                      The time period in minutes in which the model monitoring controller
                                                  function triggers. By default, the base period is 10 minutes.
        :param image:                             The image of the model monitoring controller, writer & monitoring
                                                  stream functions, which are real time nuclio functions.
                                                  By default, the image is mlrun/mlrun.
        :param deploy_histogram_data_drift_app:   If true, deploy the default histogram-based data drift application.
        :param fetch_credentials_from_sys_config: If true, fetch the credentials from the system configuration.

        """
        self.api_call(
            method=mlrun.common.types.HTTPMethod.PUT,
            path=f"projects/{project}/model-monitoring/",
            params={
                "base_period": base_period,
                "image": image,
                "deploy_histogram_data_drift_app": deploy_histogram_data_drift_app,
                "fetch_credentials_from_sys_config": fetch_credentials_from_sys_config,
            },
        )

    def disable_model_monitoring(
        self,
        project: str,
        delete_resources: bool = True,
        delete_stream_function: bool = False,
        delete_histogram_data_drift_app: bool = True,
        delete_user_applications: bool = False,
        user_application_list: Optional[list[str]] = None,
    ) -> bool:
        """
        Disable model monitoring application controller, writer, stream, histogram data drift application
        and the user's applications functions, according to the given params.

        :param project:                             Project name.
        :param delete_resources:                    If True, it would delete the model monitoring controller & writer
                                                    functions. Default True
        :param delete_stream_function:              If True, it would delete model monitoring stream function,
                                                    need to use wisely because if you're deleting this function
                                                    this can cause data loss in case you will want to
                                                    enable the model monitoring capability to the project.
                                                    Default False.
        :param delete_histogram_data_drift_app:     If True, it would delete the default histogram-based data drift
                                                    application. Default False.
        :param delete_user_applications:            If True, it would delete the user's model monitoring
                                                    application according to user_application_list, Default False.
        :param user_application_list:               List of the user's model monitoring application to disable.
                                                    Default all the applications.
                                                    Note: you have to set delete_user_applications to True
                                                    in order to delete the desired application.

        :returns:                                   True if the deletion was successful, False otherwise.
        """
        response = self.api_call(
            method=mlrun.common.types.HTTPMethod.DELETE,
            path=f"projects/{project}/model-monitoring/",
            params={
                "delete_resources": delete_resources,
                "delete_stream_function": delete_stream_function,
                "delete_histogram_data_drift_app": delete_histogram_data_drift_app,
                "delete_user_applications": delete_user_applications,
                "user_application_list": user_application_list,
            },
        )
        deletion_failed = False
        if response.status_code == http.HTTPStatus.ACCEPTED:
            if delete_resources:
                logger.info(
                    "Model Monitoring is being disabled",
                    project_name=project,
                )
            if delete_user_applications:
                logger.info("User applications are being deleted", project_name=project)
            background_tasks = mlrun.common.schemas.BackgroundTaskList(
                **response.json()
            ).background_tasks
            for task in background_tasks:
                task = self._wait_for_background_task_to_reach_terminal_state(
                    task.metadata.name, project=project
                )
                if (
                    task.status.state
                    == mlrun.common.schemas.BackgroundTaskState.succeeded
                ):
                    continue
                elif (
                    task.status.state == mlrun.common.schemas.BackgroundTaskState.failed
                ):
                    deletion_failed = True
        return not deletion_failed

    def delete_model_monitoring_function(
        self, project: str, functions: list[str]
    ) -> bool:
        """
        Delete a model monitoring application.

        :param functions:            List of the model monitoring function to delete.
        :param project:              Project name.

        :returns:                    True if the deletion was successful, False otherwise.
        """
        response = self.api_call(
            method=mlrun.common.types.HTTPMethod.DELETE,
            path=f"projects/{project}/model-monitoring/functions",
            params={"functions": functions},
        )
        deletion_failed = False
        if response.status_code == http.HTTPStatus.ACCEPTED:
            logger.info("User applications are being deleted", project_name=project)
            background_tasks = mlrun.common.schemas.BackgroundTaskList(
                **response.json()
            ).background_tasks
            for task in background_tasks:
                task = self._wait_for_background_task_to_reach_terminal_state(
                    task.metadata.name, project=project
                )
                if (
                    task.status.state
                    == mlrun.common.schemas.BackgroundTaskState.succeeded
                ):
                    continue
                elif (
                    task.status.state == mlrun.common.schemas.BackgroundTaskState.failed
                ):
                    deletion_failed = True
        return not deletion_failed

    def set_model_monitoring_credentials(
        self,
        project: str,
        credentials: dict[str, Optional[str]],
        replace_creds: bool,
    ) -> None:
        """
        Set the credentials for the model monitoring application.

        :param project:     Project name.
        :param credentials: Credentials to set.
        :param replace_creds:       If True, will override the existing credentials.
        """
        self.api_call(
            method=mlrun.common.types.HTTPMethod.PUT,
            path=f"projects/{project}/model-monitoring/credentials",
            params={**credentials, "replace_creds": replace_creds},
        )

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
                    spec=mlrun.common.schemas.HubSourceSpec(
                        path="/local/path/to/source", channel="development"
                    ),
                ),
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
                        credentials={...},
                    ),
                ),
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
    ) -> list[mlrun.common.schemas.hub.IndexedHubSource]:
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
        version: Optional[str] = None,
        tag: Optional[str] = None,
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
        version: Optional[str] = None,
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
        version: Optional[str] = None,
        tag: str = "latest",
    ):
        """
        Get hub asset from item.

        :param source_name: Name of source.
        :param item_name:   Name of the item which holds the asset.
        :param asset_name:  Name of the asset to retrieve.
        :param version: Get a specific version of the item. Default is ``None``.
        :param tag: Get a specific version of the item identified by tag. Default is ``latest``.

        :returns: http response with the asset in the content attribute
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

    def list_api_gateways(self, project=None) -> mlrun.common.schemas.APIGatewaysOutput:
        """
        Returns a list of Nuclio api gateways

        :param project: optional str parameter to filter by project, if not passed, default project value is taken

        :returns: :py:class:`~mlrun.common.schemas.APIGateways`.
        """
        project = project or config.default_project
        error = "list api gateways"
        endpoint_path = f"projects/{project}/api-gateways"
        response = self.api_call("GET", endpoint_path, error)
        return mlrun.common.schemas.APIGatewaysOutput(**response.json())

    def get_api_gateway(self, name, project=None) -> mlrun.common.schemas.APIGateway:
        """
        Returns an API gateway

        :param name: API gateway name
        :param project: optional str parameter to filter by project, if not passed, default project value is taken

        :returns:  :py:class:`~mlrun.common.schemas.APIGateway`.
        """
        project = project or config.default_project
        error = "get api gateway"
        endpoint_path = f"projects/{project}/api-gateways/{name}"
        response = self.api_call("GET", endpoint_path, error)
        return mlrun.common.schemas.APIGateway(**response.json())

    def delete_api_gateway(self, name, project=None):
        """
        Deletes an API gateway

        :param name: API gateway name
        :param project: Project name
        """
        project = project or config.default_project
        error = "delete api gateway"
        endpoint_path = f"projects/{project}/api-gateways/{name}"
        self.api_call("DELETE", endpoint_path, error)

    def store_api_gateway(
        self,
        api_gateway: Union[
            mlrun.common.schemas.APIGateway,
            mlrun.runtimes.nuclio.api_gateway.APIGateway,
        ],
        project: Optional[str] = None,
    ) -> mlrun.common.schemas.APIGateway:
        """
        Stores an API Gateway.

        :param api_gateway: :py:class:`~mlrun.runtimes.nuclio.APIGateway`
            or :py:class:`~mlrun.common.schemas.APIGateway`: API Gateway entity.
        :param project: project name. Mandatory if api_gateway is mlrun.common.schemas.APIGateway.

        :returns:  :py:class:`~mlrun.common.schemas.APIGateway`.
        """

        if isinstance(api_gateway, mlrun.runtimes.nuclio.api_gateway.APIGateway):
            api_gateway = api_gateway.to_scheme()
        endpoint_path = f"projects/{project}/api-gateways/{api_gateway.metadata.name}"
        error = "store api gateways"
        response = self.api_call(
            "PUT",
            endpoint_path,
            error,
            json=api_gateway.dict(exclude_none=True),
        )
        return mlrun.common.schemas.APIGateway(**response.json())

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
        return self._wait_for_background_task_from_response(response)

    def refresh_smtp_configuration(
        self,
    ) -> Optional[mlrun.common.schemas.BackgroundTask]:
        """Refresh smtp configuration and wait for the task to finish

        :returns: :py:class:`~mlrun.common.schemas.BackgroundTask`.
        """
        response = self.api_call(
            "POST",
            "operations/refresh-smtp-configuration",
            "Failed refreshing smtp configuration",
        )
        return self._wait_for_background_task_from_response(response)

    def set_run_notifications(
        self,
        project: str,
        run_uid: str,
        notifications: Optional[list[mlrun.model.Notification]] = None,
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
        notifications: Optional[list[mlrun.model.Notification]] = None,
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
        notification_objects: list[mlrun.model.Notification],
        run_uid: str,
        project: Optional[str] = None,
        mask_params: bool = True,
    ):
        """
        For internal use.
        The notification mechanism may run "locally" for certain runtimes.
        However, the updates occur in the API so nothing to do here.
        """
        pass

    def store_alert_notifications(
        self,
        session,
        notification_objects: list[mlrun.model.Notification],
        alert_id: str,
        project: str,
        mask_params: bool = True,
    ):
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
        arguments: Optional[dict] = None,
        artifact_path: Optional[str] = None,
        source: Optional[str] = None,
        run_name: Optional[str] = None,
        namespace: Optional[str] = None,
        notifications: Optional[list[mlrun.model.Notification]] = None,
    ) -> mlrun.common.schemas.WorkflowResponse:
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
        :param notifications:   list of notifications to send when workflow execution is completed

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
        if notifications:
            req["notifications"] = [
                notification.to_dict() for notification in notifications
            ]

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
        secrets: Optional[dict] = None,
        save_secrets: bool = True,
    ) -> str:
        """
        Loading a project remotely from the given source.

        :param name:    project name
        :param url:     git or tar.gz or .zip sources archive path e.g.:
            git://github.com/mlrun/demo-xgb-project.git
            http://mysite/archived-project.zip
            The git project should include the project yaml file.
        :param secrets:         Secrets to store in project in order to load it from the provided url. For more
            information see :py:func:`mlrun.load_project` function.
        :param save_secrets:    Whether to store secrets in the loaded project. Setting to False will cause waiting
            for the process completion.

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
        _path = self._path_of("datastore-profiles", project, name)

        res = self.api_call(method="GET", path=_path)
        if res:
            public_wrapper = res.json()
            datastore = DatastoreProfile2Json.create_from_json(
                public_json=public_wrapper["object"]
            )
            return datastore
        return None

    def delete_datastore_profile(self, name: str, project: str):
        project = project or config.default_project
        _path = self._path_of("datastore-profiles", project, name)
        self.api_call(method="DELETE", path=_path)
        return None

    def list_datastore_profiles(
        self, project: str
    ) -> list[mlrun.common.schemas.DatastoreProfile]:
        project = project or config.default_project
        _path = self._path_of("datastore-profiles", project)

        res = self.api_call(method="GET", path=_path)
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
        _path = self._path_of("datastore-profiles", project)

        self.api_call(method="PUT", path=_path, json=profile.dict())

    @staticmethod
    def warn_on_s3_and_ecr_permissions_conflict(func):
        is_s3_source = func.spec.build.source and func.spec.build.source.startswith(
            "s3://"
        )
        is_ecr_image = mlrun.utils.is_ecr_url(config.httpdb.builder.docker_registry)
        if not func.spec.build.load_source_on_run and is_s3_source and is_ecr_image:
            logger.warning(
                "Building a function image to ECR and loading an S3 source to the image may require conflicting access "
                "keys. Only the permissions granted to the platform's configured secret will take affect "
                "(see mlrun.mlconf.httpdb.builder.docker_registry_secret). "
                "In case the permissions are limited to ECR scope, you may use pull_at_runtime=True instead",
                source=func.spec.build.source,
                load_source_on_run=func.spec.build.load_source_on_run,
                default_docker_registry=config.httpdb.builder.docker_registry,
            )

    def generate_event(
        self, name: str, event_data: Union[dict, mlrun.common.schemas.Event], project=""
    ):
        """
        Generate an event.

        :param name:       The name of the event.
        :param event_data: The data of the event.
        :param project:    The project that the event belongs to.
        """
        if mlrun.mlconf.alerts.mode == mlrun.common.schemas.alert.AlertsModes.disabled:
            logger.warning("Alerts are disabled, event will not be generated")

        project = project or config.default_project
        endpoint_path = f"projects/{project}/events/{name}"
        error_message = f"post event {project}/events/{name}"
        if isinstance(event_data, mlrun.common.schemas.Event):
            event_data = event_data.dict()
        self.api_call(
            "POST", endpoint_path, error_message, body=dict_to_json(event_data)
        )

    def store_alert_config(
        self,
        alert_name: str,
        alert_data: Union[dict, AlertConfig],
        project="",
        force_reset: bool = False,
    ) -> AlertConfig:
        """
        Create/modify an alert.

        :param alert_name: The name of the alert.
        :param alert_data: The data of the alert.
        :param project:    The project that the alert belongs to.
        :param force_reset: If True and the alert already exists, the alert would be reset.
        :returns:          The created/modified alert.
        """
        if not alert_data:
            raise mlrun.errors.MLRunInvalidArgumentError("Alert data must be provided")

        if mlrun.mlconf.alerts.mode == mlrun.common.schemas.alert.AlertsModes.disabled:
            logger.warning(
                "Alerts are disabled, alert will still be stored but will not be triggered"
            )

        project = project or config.default_project
        endpoint_path = f"projects/{project}/alerts/{alert_name}"
        error_message = f"put alert {project}/alerts/{alert_name}"
        alert_instance = (
            alert_data
            if isinstance(alert_data, AlertConfig)
            else AlertConfig.from_dict(alert_data)
        )
        # Validation is necessary here because users can directly invoke this function
        # through `mlrun.get_run_db().store_alert_config()`.
        alert_instance.validate_required_fields()

        alert_data = alert_instance.to_dict()
        body = _as_json(alert_data)
        params = {"force_reset": bool2str(force_reset)} if force_reset else {}
        response = self.api_call(
            "PUT", endpoint_path, error_message, params=params, body=body
        )
        return AlertConfig.from_dict(response.json())

    def get_alert_config(self, alert_name: str, project="") -> AlertConfig:
        """
        Retrieve an alert.

        :param alert_name: The name of the alert to retrieve.
        :param project:    The project that the alert belongs to.

        :returns:           The alert object.
        """
        project = project or config.default_project
        endpoint_path = f"projects/{project}/alerts/{alert_name}"
        error_message = f"get alert {project}/alerts/{alert_name}"
        response = self.api_call("GET", endpoint_path, error_message)
        return AlertConfig.from_dict(response.json())

    def list_alerts_configs(
        self, project="", limit: Optional[int] = None, offset: Optional[int] = None
    ) -> list[AlertConfig]:
        """
        Retrieve list of alerts of a project.

        :param project: The project name.
        :param limit: The maximum number of alerts to return.
            Defaults to `mlconf.alerts.default_list_alert_configs_limit` if not provided.
        :param offset: The number of alerts to skip.

        :returns: All the alerts objects of the project.
        """
        project = project or config.default_project
        endpoint_path = f"projects/{project}/alerts"
        error_message = f"get alerts {project}/alerts"
        params = {}
        # TODO: Deprecate limit and offset when pagination is implemented
        if limit:
            params["page-size"] = limit
        if offset:
            params["offset"] = offset
        response = self.api_call(
            "GET", endpoint_path, error_message, params=params
        ).json()
        results = []
        for item in response.get("alerts", []):
            results.append(AlertConfig(**item))
        return results

    def delete_alert_config(self, alert_name: str, project=""):
        """
        Delete an alert.
        :param alert_name: The name of the alert to delete.
        :param project: The project that the alert belongs to.
        """
        project = project or config.default_project
        endpoint_path = f"projects/{project}/alerts/{alert_name}"
        error_message = f"delete alert {project}/alerts/{alert_name}"
        self.api_call("DELETE", endpoint_path, error_message)

    def reset_alert_config(self, alert_name: str, project=""):
        """
        Reset an alert.

        :param alert_name: The name of the alert to reset.
        :param project: The project that the alert belongs to.
        """
        project = project or config.default_project
        endpoint_path = f"projects/{project}/alerts/{alert_name}/reset"
        error_message = f"post alert {project}/alerts/{alert_name}/reset"
        self.api_call("POST", endpoint_path, error_message)

    def get_alert_template(
        self, template_name: str
    ) -> mlrun.common.schemas.AlertTemplate:
        """
        Retrieve a specific alert template.

        :param template_name: The name of the template to retrieve.

        :returns: The template object.
        """
        endpoint_path = f"alert-templates/{template_name}"
        error_message = f"get template alert-templates/{template_name}"
        response = self.api_call("GET", endpoint_path, error_message)
        return mlrun.common.schemas.AlertTemplate(**response.json())

    def list_alert_templates(self) -> list[mlrun.common.schemas.AlertTemplate]:
        """
        Retrieve list of all alert templates.

        :returns: All the alert template objects in the database.
        """
        endpoint_path = "alert-templates"
        error_message = "get templates /alert-templates"
        response = self.api_call("GET", endpoint_path, error_message).json()
        results = []
        for item in response:
            results.append(mlrun.common.schemas.AlertTemplate(**item))
        return results

    def list_alert_activations(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        entity: Optional[str] = None,
        severity: Optional[
            list[Union[mlrun.common.schemas.alert.AlertSeverity, str]]
        ] = None,
        entity_kind: Optional[
            Union[mlrun.common.schemas.alert.EventEntityKind, str]
        ] = None,
        event_kind: Optional[Union[mlrun.common.schemas.alert.EventKind, str]] = None,
    ) -> mlrun.common.schemas.AlertActivations:
        """
        Retrieve a list of all alert activations.

        :param project: The project name to filter by. If None, results are not filtered by project.
        :param name: The alert name to filter by. Supports exact matching or partial matching if prefixed with `~`.
        :param since: Filters for alert activations occurring after this timestamp.
        :param until: Filters for alert activations occurring before this timestamp.
        :param entity: The entity ID to filter by. Supports wildcard matching if prefixed with `~`.
        :param severity: A list of severity levels to filter by (e.g., ["high", "low"]).
        :param entity_kind: The kind of entity (e.g., "job", "endpoint") to filter by.
        :param event_kind: The kind of event (e.g., ""data-drift-detected"", "failed") to filter by.

        :returns: A list of alert activations matching the provided filters.
        """

        alert_activations, _ = self._list_alert_activations(
            project=project,
            name=name,
            since=since,
            until=until,
            entity=entity,
            severity=severity,
            entity_kind=entity_kind,
            event_kind=event_kind,
            return_all=True,
        )
        return alert_activations

    def paginated_list_alert_activations(
        self,
        *args,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        **kwargs,
    ) -> tuple[AlertActivations, Optional[str]]:
        """List alerts activations with support for pagination and various filtering options.

        This method retrieves a paginated list of alert activations based on the specified filter parameters.
        Pagination is controlled using the `page`, `page_size`, and `page_token` parameters. The method
        will return a list of alert activations that match the filtering criteria provided.

        For detailed information about the parameters, refer to the list_alert_activations method:
            See :py:func:`~list_alert_activations` for more details.

        Examples::

            # Fetch first page of alert activations with page size of 5
            alert_activations, token = db.paginated_list_alert_activations(
                project="my-project", page_size=5
            )
            # Fetch next page using the pagination token from the previous response
            alert_activations, token = db.paginated_list_alert_activations(
                project="my-project", page_token=token
            )
            # Fetch alert activations for a specific page (e.g., page 3)
            alert_activations, token = db.paginated_list_alert_activations(
                project="my-project", page=3, page_size=5
            )

            # Automatically iterate over all pages without explicitly specifying the page number
            alert_activations = []
            token = None
            while True:
                page_alert_activations, token = db.paginated_list_alert_activations(
                    project="my-project", page_token=token, page_size=5
                )
                alert_activations.extend(page_alert_activations)

                # If token is None and page_alert_activations is empty, we've reached the end (no more activations).
                # If token is None and page_alert_activations is not empty, we've fetched the last page of activations.
                if not token:
                    break
            print(f"Total alert activations retrieved: {len(alert_activations)}")

        :param page: The page number to retrieve. If not provided, the next page will be retrieved.
        :param page_size: The number of items per page to retrieve. Up to `page_size` responses are expected.
            Defaults to `mlrun.mlconf.httpdb.pagination.default_page_size` if not provided.
        :param page_token: A pagination token used to retrieve the next page of results. Should not be provided
            for the first request.

        :returns: A tuple containing the list of alert activations and an optional `page_token` for pagination.
        """
        return self._list_alert_activations(
            *args,
            page=page,
            page_size=page_size,
            page_token=page_token,
            return_all=False,
            **kwargs,
        )

    def get_alert_activation(
        self,
        project,
        activation_id,
    ) -> mlrun.common.schemas.AlertActivation:
        """
        Retrieve the alert activation by id

        :param project: Project name for which the summary belongs.
        :param activation_id: alert activation id.
        :returns: alert activation object.
        """
        project = project or config.default_project

        error = "get alert activation"
        path = f"projects/{project}/alert-activations/{activation_id}"

        response = self.api_call("GET", path, error)

        return mlrun.common.schemas.AlertActivation(**response.json())

    def get_project_summary(
        self, project: Optional[str] = None
    ) -> mlrun.common.schemas.ProjectSummary:
        """
        Retrieve the summary of a project.

        :param project: Project name for which the summary belongs.
        :returns: A summary of the project.
        """
        project = project or config.default_project

        endpoint_path = f"project-summaries/{project}"
        error_message = f"Failed retrieving project summary for {project}"
        response = self.api_call("GET", endpoint_path, error_message)
        return mlrun.common.schemas.ProjectSummary(**response.json())

    @staticmethod
    def _parse_labels(
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]],
    ):
        """
        Parse labels to support providing a dictionary from the SDK,
        which may not be directly supported in the endpoints.

        :param labels: The labels to parse, which can be a dictionary, a list of strings,
                       or a comma-separated string. This function converts them into a list
                       of labels in the format 'key=value' or 'key'.
        :return: A list of parsed labels in the format 'key=value' or 'key'.
        :raises MLRunValueError: If the labels format is invalid.
        """
        try:
            return mlrun.common.schemas.common.LabelsModel(labels=labels).labels
        except pydantic.v1.error_wrappers.ValidationError as exc:
            raise mlrun.errors.MLRunValueError(
                "Invalid labels format. Must be a dictionary of strings, a list of strings, "
                "or a comma-separated string."
            ) from exc

    def _list_artifacts(
        self,
        name: Optional[str] = None,
        project: Optional[str] = None,
        tag: Optional[str] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        iter: Optional[int] = None,
        best_iteration: bool = False,
        kind: Optional[str] = None,
        category: Union[str, mlrun.common.schemas.ArtifactCategories] = None,
        tree: Optional[str] = None,
        producer_uri: Optional[str] = None,
        format_: Optional[
            mlrun.common.formatters.ArtifactFormat
        ] = mlrun.common.formatters.ArtifactFormat.full,
        limit: Optional[int] = None,
        partition_by: Optional[
            Union[mlrun.common.schemas.ArtifactPartitionByField, str]
        ] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Optional[
            Union[mlrun.common.schemas.SortField, str]
        ] = mlrun.common.schemas.SortField.updated,
        partition_order: Union[
            mlrun.common.schemas.OrderType, str
        ] = mlrun.common.schemas.OrderType.desc,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        return_all: bool = False,
    ) -> tuple[ArtifactList, Optional[str]]:
        """Handles list artifacts, both paginated and not."""

        project = project or config.default_project
        labels = self._parse_labels(labels)

        if limit:
            # TODO: Remove this in 1.11.0
            warnings.warn(
                "'limit' is deprecated and will be removed in 1.11.0. Use 'page' and 'page_size' instead.",
                FutureWarning,
            )

        params = {
            "name": name,
            "tag": tag,
            "label": labels,
            "iter": iter,
            "best-iteration": best_iteration,
            "kind": kind,
            "category": category,
            "tree": tree,
            "format": format_,
            "producer_uri": producer_uri,
            "since": datetime_to_iso(since),
            "until": datetime_to_iso(until),
            "limit": limit,
            "page": page,
            "page-size": page_size,
            "page-token": page_token,
        }

        if partition_by:
            params.update(
                self._generate_partition_by_params(
                    partition_by,
                    rows_per_partition,
                    partition_sort_by,
                    partition_order,
                )
            )
        error = "list artifacts"
        endpoint_path = f"projects/{project}/artifacts"

        # Fetch the responses, either one page or all based on `return_all`
        responses = self.paginated_api_call(
            "GET",
            endpoint_path,
            error,
            params=params,
            version="v2",
            return_all=return_all,
        )
        paginated_responses, token = self.process_paginated_responses(
            responses, "artifacts"
        )

        values = ArtifactList(paginated_responses)
        values.tag = tag
        return values, token

    def _list_functions(
        self,
        name: Optional[str] = None,
        project: Optional[str] = None,
        tag: Optional[str] = None,
        kind: Optional[str] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
        format_: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        states: typing.Optional[list[mlrun.common.schemas.FunctionState]] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        return_all: bool = False,
    ) -> tuple[list, Optional[str]]:
        """Handles list functions, both paginated and not."""

        project = project or config.default_project
        labels = self._parse_labels(labels)
        params = {
            "name": name,
            "tag": tag,
            "kind": kind,
            "label": labels,
            "since": datetime_to_iso(since),
            "until": datetime_to_iso(until),
            "format": format_,
            "state": states or None,
            "page": page,
            "page-size": page_size,
            "page-token": page_token,
        }
        error = "list functions"
        path = f"projects/{project}/functions"

        # Fetch the responses, either one page or all based on `return_all`
        responses = self.paginated_api_call(
            "GET", path, error, params=params, return_all=return_all
        )
        paginated_responses, token = self.process_paginated_responses(
            responses, "funcs"
        )
        return paginated_responses, token

    def _list_runs(
        self,
        name: Optional[str] = None,
        uid: Optional[Union[str, list[str]]] = None,
        project: Optional[str] = None,
        labels: Optional[Union[str, dict[str, Optional[str]], list[str]]] = None,
        state: Optional[
            mlrun.common.runtimes.constants.RunStates
        ] = None,  # Backward compatibility
        states: typing.Optional[list[mlrun.common.runtimes.constants.RunStates]] = None,
        sort: bool = True,
        iter: bool = False,
        start_time_from: Optional[datetime] = None,
        start_time_to: Optional[datetime] = None,
        last_update_time_from: Optional[datetime] = None,
        last_update_time_to: Optional[datetime] = None,
        end_time_from: Optional[datetime] = None,
        end_time_to: Optional[datetime] = None,
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
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        return_all: bool = False,
    ) -> tuple[RunList, Optional[str]]:
        """Handles list runs, both paginated and not."""

        project = project or config.default_project
        if with_notifications:
            logger.warning(
                "Local run notifications are not persisted in the DB, therefore local runs will not be returned when "
                "using the `with_notifications` flag."
            )

        if state:
            # TODO: Remove this in 1.10.0
            warnings.warn(
                "'state' is deprecated in 1.7.0 and will be removed in 1.10.0. Use 'states' instead.",
                FutureWarning,
            )

        labels = self._parse_labels(labels)

        if (
            not name
            and not uid
            and not labels
            and not state
            and not states
            and not start_time_from
            and not start_time_to
            and not last_update_time_from
            and not last_update_time_to
            and not end_time_from
            and not end_time_to
            and not partition_by
            and not partition_sort_by
            and not iter
        ):
            # default to last week on no filter
            start_time_from = datetime.now() - timedelta(days=7)
            partition_by = mlrun.common.schemas.RunPartitionByField.project_and_name
            partition_sort_by = mlrun.common.schemas.SortField.updated

        params = {
            "name": name,
            "uid": uid,
            "label": labels,
            "state": (
                mlrun.utils.helpers.as_list(state)
                if state is not None
                else states or None
            ),
            "sort": bool2str(sort),
            "iter": bool2str(iter),
            "start_time_from": datetime_to_iso(start_time_from),
            "start_time_to": datetime_to_iso(start_time_to),
            "last_update_time_from": datetime_to_iso(last_update_time_from),
            "last_update_time_to": datetime_to_iso(last_update_time_to),
            "end_time_from": datetime_to_iso(end_time_from),
            "end_time_to": datetime_to_iso(end_time_to),
            "with-notifications": with_notifications,
            "page": page,
            "page-size": page_size,
            "page-token": page_token,
        }

        if partition_by:
            params.update(
                self._generate_partition_by_params(
                    partition_by,
                    rows_per_partition,
                    partition_sort_by,
                    partition_order,
                    max_partitions,
                )
            )
        error = "list runs"
        _path = self._path_of("runs", project)

        # Fetch the responses, either one page or all based on `return_all`
        responses = self.paginated_api_call(
            "GET", _path, error, params=params, return_all=return_all
        )
        paginated_responses, token = self.process_paginated_responses(responses, "runs")
        return RunList(paginated_responses), token

    def _list_alert_activations(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        entity: Optional[str] = None,
        severity: Optional[
            Union[
                mlrun.common.schemas.alert.AlertSeverity,
                str,
                list[Union[mlrun.common.schemas.alert.AlertSeverity, str]],
            ]
        ] = None,
        entity_kind: Optional[
            Union[mlrun.common.schemas.alert.EventEntityKind, str]
        ] = None,
        event_kind: Optional[Union[mlrun.common.schemas.alert.EventKind, str]] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        return_all: bool = False,
    ) -> tuple[mlrun.common.schemas.AlertActivations, Optional[str]]:
        project = project or config.default_project
        params = {
            "name": name,
            "since": datetime_to_iso(since),
            "until": datetime_to_iso(until),
            "entity": entity,
            "severity": mlrun.utils.helpers.as_list(severity) if severity else None,
            "entity-kind": entity_kind,
            "event-kind": event_kind,
            "page": page,
            "page-size": page_size,
            "page-token": page_token,
        }
        error = "list alert activations"
        path = f"projects/{project}/alert-activations"

        # Fetch the responses, either one page or all based on `return_all`
        responses = self.paginated_api_call(
            "GET", path, error, params=params, return_all=return_all
        )
        paginated_responses, token = self.process_paginated_responses(
            responses, "activations"
        )
        paginated_results = mlrun.common.schemas.AlertActivations(
            activations=[
                mlrun.common.schemas.AlertActivation(**item)
                for item in paginated_responses
            ]
        )

        return paginated_results, token

    def _wait_for_background_task_from_response(self, response):
        if response.status_code == http.HTTPStatus.ACCEPTED:
            background_task = mlrun.common.schemas.BackgroundTask(**response.json())
            return self._wait_for_background_task_to_reach_terminal_state(
                background_task.metadata.name
            )
        return None

    def _resolve_page_params(self, params: typing.Optional[dict]) -> dict:
        """
        Resolve the page parameters, setting defaults where necessary.
        """
        page_params = deepcopy(params) or {}
        if page_params.get("page-token") is None and page_params.get("page") is None:
            page_params["page"] = 1
        if page_params.get("page-size") is None:
            page_size = config.httpdb.pagination.default_page_size

            if page_params.get("limit") is not None:
                page_size = page_params["limit"]

                # limit and page/page size are conflicting
                page_params.pop("limit")
            page_params["page-size"] = page_size

        # this may happen only when page-size was explicitly set along with limit
        # this is to ensure we will not get stopped by API on similar below validation
        # but rather simply fallback to use page-size.
        if page_params.get("page-size") and page_params.get("limit"):
            logger.warning(
                "Both 'limit' and 'page-size' are provided, using 'page-size'."
            )
            page_params.pop("limit")
        return page_params


def _as_json(obj):
    fn = getattr(obj, "to_json", None)
    if fn:
        return fn()
    return dict_to_json(obj)
