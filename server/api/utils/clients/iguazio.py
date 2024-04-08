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
#
import asyncio
import contextlib
import copy
import datetime
import enum
import http
import json
import threading
import typing
import urllib.parse

import aiohttp
import fastapi
import humanfriendly
import igz_mgmt.schemas.manual_events
import requests.adapters
from fastapi.concurrency import run_in_threadpool

import mlrun.common.schemas
import mlrun.config
import mlrun.errors
import mlrun.utils.helpers
import mlrun.utils.singleton
import server.api.utils.projects.remotes.leader as project_leader
from mlrun.utils import get_in, logger


class JobStates:
    completed = "completed"
    failed = "failed"
    canceled = "canceled"
    in_progress = "in_progress"

    @staticmethod
    def all():
        return [
            JobStates.completed,
            JobStates.failed,
            JobStates.canceled,
            JobStates.in_progress,
        ]

    @staticmethod
    def terminal_states():
        return [
            JobStates.completed,
            JobStates.failed,
            JobStates.canceled,
        ]


class SessionPlanes:
    data = "data"
    control = "control"

    @staticmethod
    def all():
        return [
            SessionPlanes.data,
            SessionPlanes.control,
        ]


class JobCache:
    """
    Cache for delete project jobs.
    This cache is used to avoid consecutive create/delete jobs for the same project,
    while the jobs are still in progress.
    """

    def __init__(self, ttl: str):
        self._delete_jobs = {}
        self._delete_locks = {}

        # this lock is used only for getting the project specific locks
        self._lock = threading.Lock()

        self._ttl = humanfriendly.parse_timespan(ttl)

    def get_delete_lock(self, project: str):
        with self._lock:
            if project not in self._delete_locks:
                self._delete_locks[project] = threading.Lock()
            return self._delete_locks[project]

    def get_delete_job_id(self, project: str) -> typing.Optional[str]:
        return self._delete_jobs.get(project, {}).get("job_id")

    def set_delete_job(self, project: str, job_id: str):
        self._delete_jobs[project] = {
            "job_id": job_id,
            "timestamp": datetime.datetime.now(),
        }

        # schedule cache invalidation for delete job
        self._schedule_cache_invalidation(project, job_id)

    def remove_delete_job(self, project: str):
        self._delete_jobs.pop(project, None)

    def invalidate_cache(self, project: str, job_id: str):
        # if current project job is the same as the scheduled job id, remove it from the cache
        # otherwise, it means that a new job was created for the project, and we don't want to remove it
        with self.get_delete_lock(project):
            if self.get_delete_job_id(project) == job_id:
                self.remove_delete_job(project)

    def _schedule_cache_invalidation(
        self,
        project: str,
        job_id: str,
    ):
        try:
            event_loop = asyncio.get_event_loop()
        except RuntimeError:
            event_loop = asyncio.new_event_loop()

        event_loop.call_later(self._ttl, self.invalidate_cache, project, job_id)


class Client(
    project_leader.Member,
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._session = mlrun.utils.HTTPSessionWithRetry(
            retry_on_exception=mlrun.mlconf.httpdb.projects.retry_leader_request_on_exception
            == mlrun.common.schemas.HTTPSessionRetryMode.enabled.value,
            verbose=True,
        )
        self._api_url = mlrun.mlconf.iguazio_api_url
        # The job is expected to be completed in less than 5 seconds. If 10 seconds have passed and the job
        # has not been completed, increase the interval to retry every 5 seconds
        self._wait_for_job_completion_retry_interval = mlrun.utils.create_step_backoff(
            [[1, 10], [5, None]]
        )
        self._wait_for_project_terminal_state_retry_interval = 5
        self._logger = logger.get_child("iguazio-client")
        self._igz_clients = {}

        self._job_cache = JobCache(
            ttl=mlrun.mlconf.httpdb.projects.iguazio_client_job_cache_ttl,
        )

    def verify_request_session(
        self, request: fastapi.Request
    ) -> mlrun.common.schemas.AuthInfo:
        """
        Proxy the request to one of the session verification endpoints (which will verify the session of the request)
        """
        response = self._send_request_to_api(
            "POST",
            mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint,
            "Failed verifying iguazio session",
            headers={
                "authorization": request.headers.get("authorization"),
                "cookie": request.headers.get("cookie"),
            },
        )
        return self._generate_auth_info_from_session_verification_response(
            response.headers, response.json()
        )

    def get_user_unix_id(self, session: str) -> str:
        response = self._send_request_to_api(
            "GET",
            "self",
            "Failed get iguazio user",
            session,
        )
        response_json = response.json()
        return response_json["data"]["attributes"]["uid"]

    def get_or_create_access_key(self, session: str, planes: list[str] = None) -> str:
        if planes is None:
            planes = [
                SessionPlanes.data,
                SessionPlanes.control,
            ]
        body = {
            "data": {
                "type": "access_key",
                "attributes": {"label": "MLRun", "planes": planes},
            }
        }
        response = self._send_request_to_api(
            "POST",
            "self/get_or_create_access_key",
            "Failed getting or creating iguazio access key",
            session,
            json=body,
        )
        if response.status_code == http.HTTPStatus.CREATED.value:
            self._logger.debug("Created access key in Iguazio", planes=planes)
        return response.json()["data"]["id"]

    def create_project(
        self,
        session: str,
        project: mlrun.common.schemas.Project,
        wait_for_completion: bool = True,
    ) -> bool:
        self._logger.debug("Creating project in Iguazio", project=project.metadata.name)
        body = self._transform_mlrun_project_to_iguazio_project(project)
        return self._create_project_in_iguazio(
            session,
            project.metadata.name,
            body,
            wait_for_completion,
            timeout=60,
        )

    def update_project(
        self,
        session: str,
        name: str,
        project: mlrun.common.schemas.Project,
    ):
        self._logger.debug("Updating project in Iguazio", name=name)
        body = self._transform_mlrun_project_to_iguazio_project(project)
        self._put_project_to_iguazio(session, name, body)

    def delete_project(
        self,
        session: str,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
        wait_for_completion: bool = True,
    ) -> bool:
        with self._job_cache.get_delete_lock(name):
            # check if project is already being deleted
            job_id = self._job_cache.get_delete_job_id(name)

            # the existing job might have already been completed, but the cache was not yet invalidated
            # in that case, we want to create a new deletion job
            if job_id:
                try:
                    if self._is_job_terminated(session, job_id):
                        self._job_cache.remove_delete_job(name)
                        # set job_id to None so that a new job will be created
                        job_id = None
                except mlrun.errors.MLRunNotFoundError:
                    # job not found in iguazio. consider deletion as successful
                    self._job_cache.remove_delete_job(name)
                    job_id = None

            if not job_id:
                job_id = self._delete_project_in_iguazio(
                    session, name, deletion_strategy
                )
                if not job_id:
                    # project not found in iguazio. consider deletion as successful
                    return False

                self._job_cache.set_delete_job(name, job_id)

        if wait_for_completion:
            self._logger.debug(
                "Waiting for project deletion job in Iguazio",
                name=name,
                job_id=job_id,
            )
            self._wait_for_job_completion(
                session, job_id, "Project deletion job failed"
            )
            self._logger.debug(
                "Successfully deleted project in Iguazio",
                name=name,
                job_id=job_id,
            )
            self._job_cache.invalidate_cache(name, job_id)
            return False

        return True

    def list_projects(
        self,
        session: str,
        updated_after: typing.Optional[datetime.datetime] = None,
        page_size: typing.Optional[int] = None,
    ) -> tuple[list[mlrun.common.schemas.Project], typing.Optional[datetime.datetime]]:
        project_names, latest_updated_at = self._list_project_names(
            session, updated_after, page_size
        )
        return self._list_projects_data(session, project_names), latest_updated_at

    def get_project(
        self,
        session: str,
        name: str,
    ) -> mlrun.common.schemas.Project:
        return self._get_project_from_iguazio(session, name)

    def get_project_owner(
        self,
        session: str,
        name: str,
    ) -> mlrun.common.schemas.ProjectOwner:
        response = self._get_project_from_iguazio_without_parsing(
            session, name, enrich_owner_access_key=True
        )
        iguazio_project = response.json()["data"]
        owner_username = iguazio_project.get("attributes", {}).get(
            "owner_username", None
        )
        owner_access_key = iguazio_project.get("attributes", {}).get(
            "owner_access_key", None
        )
        if not owner_username:
            raise mlrun.errors.MLRunInternalServerError(
                f"Unable to enrich project owner for project {name},"
                f" because project has no owner configured"
            )
        return mlrun.common.schemas.ProjectOwner(
            username=owner_username,
            access_key=owner_access_key,
        )

    def format_as_leader_project(
        self, project: mlrun.common.schemas.Project
    ) -> mlrun.common.schemas.IguazioProject:
        return mlrun.common.schemas.IguazioProject(
            data=self._transform_mlrun_project_to_iguazio_project(project)["data"]
        )

    @property
    def is_sync(self):
        """
        False because client is synchronous
        """
        return True

    def try_get_grafana_service_url(self, session: str) -> typing.Optional[str]:
        """
        Try to find a ready grafana app service, and return its URL
        If nothing found, returns None
        """
        self._logger.debug("Getting grafana service url from Iguazio")
        response = self._send_request_to_api(
            "GET",
            "app_services_manifests",
            "Failed getting app services manifests from Iguazio",
            session,
        )
        response_body = response.json()
        for app_services_manifest in response_body.get("data", []):
            for app_service in app_services_manifest.get("attributes", {}).get(
                "app_services", []
            ):
                if (
                    app_service.get("spec", {}).get("kind") == "grafana"
                    and app_service.get("status", {}).get("state") == "ready"
                    and len(app_service.get("status", {}).get("urls", [])) > 0
                ):
                    url_kind_to_url = {}
                    for url in app_service["status"]["urls"]:
                        url_kind_to_url[url["kind"]] = url["url"]
                    # precedence for https
                    for kind in ["https", "http"]:
                        if kind in url_kind_to_url:
                            return url_kind_to_url[kind]
        return None

    def emit_manual_event(self, access_key: str, event: igz_mgmt.Event):
        """
        Emit a manual event to Iguazio
        """
        client = self._get_igz_client(access_key)
        igz_mgmt.ManualEvents.emit(
            http_client=client, event=event, audit_tenant_id=client.tenant_id
        )

    def _get_igz_client(self, access_key: str) -> igz_mgmt.Client:
        if not self._igz_clients.get(access_key):
            self._igz_clients[access_key] = igz_mgmt.Client(
                endpoint=self._api_url,
                access_key=access_key,
            )
        return self._igz_clients[access_key]

    def _list_project_names(
        self,
        session: str,
        updated_after: typing.Optional[datetime.datetime] = None,
        page_size: typing.Optional[int] = None,
    ) -> tuple[list[str], typing.Optional[datetime.datetime]]:
        params = {}
        if updated_after is not None:
            time_string = updated_after.isoformat().split("+")[0]
            params["filter[updated_at]"] = f"[$gt]{time_string}Z"
        if page_size is None:
            page_size = (
                mlrun.mlconf.httpdb.projects.iguazio_list_projects_default_page_size
            )
        if page_size is not None:
            params["page[size]"] = int(page_size)

        # avoid getting projects that are in the process of being deleted
        # this is done to avoid race condition between deleting the project flow and sync mechanism
        params["filter[operational_status]"] = "[$ne]deleting"

        response = self._send_request_to_api(
            "GET",
            "projects",
            "Failed listing projects from Iguazio",
            session,
            params=params,
        )
        response_body = response.json()
        project_names = [
            iguazio_project["attributes"]["name"]
            for iguazio_project in response_body["data"]
        ]
        latest_updated_at = self._find_latest_updated_at(response_body)
        return project_names, latest_updated_at

    def _list_projects_data(
        self, session: str, project_names: list[str]
    ) -> list[mlrun.common.schemas.Project]:
        return [
            self._get_project_from_iguazio(session, project_name)
            for project_name in project_names
        ]

    def _find_latest_updated_at(
        self, response_body: dict
    ) -> typing.Optional[datetime.datetime]:
        latest_updated_at = None
        for iguazio_project in response_body["data"]:
            updated_at = datetime.datetime.fromisoformat(
                iguazio_project["attributes"]["updated_at"]
            )
            if latest_updated_at is None or latest_updated_at < updated_at:
                latest_updated_at = updated_at
        return latest_updated_at

    def _create_project_in_iguazio(
        self,
        session: str,
        name: str,
        body: dict,
        wait_for_completion: bool,
        **kwargs,
    ) -> bool:
        _, job_id = self._post_project_to_iguazio(session, body, **kwargs)

        if wait_for_completion:
            self._logger.debug(
                "Waiting for project creation job in Iguazio",
                name=name,
                job_id=job_id,
            )
            self._wait_for_job_completion(
                session, job_id, "Project creation job failed"
            )
            self._logger.debug(
                "Successfully created project in Iguazio",
                name=name,
                job_id=job_id,
            )
            return False
        return True

    def _delete_project_in_iguazio(
        self,
        session: str,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
    ) -> typing.Optional[str]:
        self._logger.debug(
            "Deleting project in Iguazio",
            name=name,
            deletion_strategy=deletion_strategy,
        )
        body = self._transform_mlrun_project_to_iguazio_project(
            mlrun.common.schemas.Project(
                metadata=mlrun.common.schemas.ProjectMetadata(name=name)
            )
        )
        headers = {
            "igz-project-deletion-strategy": deletion_strategy.to_iguazio_deletion_strategy(),
        }
        try:
            response = self._send_request_to_api(
                "DELETE",
                "projects",
                "Failed deleting project in Iguazio",
                session,
                headers=headers,
                json=body,
            )
            job_id = response.json()["data"]["id"]
            return job_id
        except requests.HTTPError as exc:
            if exc.response.status_code != http.HTTPStatus.NOT_FOUND.value:
                raise
            self._logger.debug(
                "Project not found in Iguazio",
                name=name,
                deletion_strategy=deletion_strategy,
            )
            raise mlrun.errors.MLRunNotFoundError(
                "Project not found in Iguazio"
            ) from exc

    def _post_project_to_iguazio(
        self,
        session: str,
        body: dict,
        **kwargs,
    ) -> tuple[mlrun.common.schemas.Project, str]:
        response = self._send_request_to_api(
            "POST",
            "projects",
            "Failed creating project in Iguazio",
            session,
            json=body,
            **kwargs,
        )
        response_body = response.json()
        return (
            self._transform_iguazio_project_to_mlrun_project(response_body["data"]),
            response_body["data"]["relationships"]["last_job"]["data"]["id"],
        )

    def _put_project_to_iguazio(
        self,
        session: str,
        name: str,
        body: dict,
        **kwargs,
    ) -> mlrun.common.schemas.Project:
        response = self._send_request_to_api(
            "PUT",
            f"projects/__name__/{name}",
            "Failed updating project in Iguazio",
            session,
            json=body,
            **kwargs,
        )
        return self._transform_iguazio_project_to_mlrun_project(response.json()["data"])

    def _get_project_from_iguazio_without_parsing(
        self, session: str, name: str, enrich_owner_access_key: bool = False
    ):
        params = {"include": "owner"}
        if enrich_owner_access_key:
            params["enrich_owner_access_key"] = "true"
        try:
            return self._send_request_to_api(
                "GET",
                f"projects/__name__/{name}",
                "Failed getting project from Iguazio",
                session,
                params=params,
            )
        except requests.HTTPError as exc:
            if exc.response.status_code != http.HTTPStatus.NOT_FOUND.value:
                raise
            self._logger.debug(
                "Project not found in Iguazio",
                name=name,
            )
            raise mlrun.errors.MLRunNotFoundError(
                "Project not found in Iguazio"
            ) from exc

    def _get_project_from_iguazio(
        self, session: str, name: str, include_owner_session: bool = False
    ) -> mlrun.common.schemas.Project:
        response = self._get_project_from_iguazio_without_parsing(session, name)
        return self._transform_iguazio_project_to_mlrun_project(response.json()["data"])

    def _wait_for_job_completion(self, session: str, job_id: str, error_message: str):
        def _verify_job_in_terminal_state():
            job_response_body = self._get_job_from_iguazio(session, job_id)
            _job_state = job_response_body["data"]["attributes"]["state"]
            if _job_state not in JobStates.terminal_states():
                raise Exception(f"Job in progress. State: {_job_state}")
            return _job_state, job_response_body["data"]["attributes"]["result"]

        job_state, job_result = mlrun.utils.helpers.retry_until_successful(
            self._wait_for_job_completion_retry_interval,
            360,
            self._logger,
            False,
            _verify_job_in_terminal_state,
        )
        if job_state != JobStates.completed:
            status_code = None
            try:
                parsed_result = json.loads(job_result)
                error_message = f"{error_message} {parsed_result['message']}"
                # status is optional
                if "status" in parsed_result:
                    status_code = int(parsed_result["status"])
            except Exception:
                pass
            if not status_code:
                raise mlrun.errors.MLRunRuntimeError(error_message)
            raise mlrun.errors.err_for_status_code(status_code, error_message)
        self._logger.debug("Job completed successfully", job_id=job_id)

    def _get_job_from_iguazio(self, session: str, job_id: str) -> dict:
        response = self._send_request_to_api(
            "GET", f"jobs/{job_id}", "Failed getting job from Iguazio", session
        )
        return response.json()

    def _send_request_to_api(
        self, method, path, error_message: str, session=None, **kwargs
    ):
        url = f"{self._api_url}/api/{path}"
        self._prepare_request_kwargs(session, path, kwargs=kwargs)
        response = self._session.request(
            method, url, verify=mlrun.config.config.httpdb.http.verify, **kwargs
        )
        if not response.ok:
            try:
                response_body = response.json()
            except Exception:
                response_body = {}
            self._handle_error_response(
                method, path, response, response_body, error_message, kwargs
            )
        return response

    def _generate_auth_info_from_session_verification_response(
        self,
        response_headers: typing.Mapping[str, typing.Any],
        response_body: typing.Mapping[typing.Any, typing.Any],
    ) -> mlrun.common.schemas.AuthInfo:
        (
            username,
            session,
            planes,
            user_unix_id,
            user_id,
            group_ids,
        ) = self._resolve_params_from_response_headers(response_headers)

        (
            user_id_from_body,
            group_ids_from_body,
        ) = self._resolve_params_from_response_body(response_body)

        # from iguazio version >= 3.5.2, user and group ids are included in the response body
        # if not, get them from the headers
        user_id = user_id_from_body or user_id
        group_ids = group_ids_from_body or group_ids

        auth_info = mlrun.common.schemas.AuthInfo(
            username=username,
            session=session,
            user_id=user_id,
            user_group_ids=group_ids,
            user_unix_id=user_unix_id,
            planes=planes,
        )
        if SessionPlanes.data in planes:
            auth_info.data_session = auth_info.session
        return auth_info

    @staticmethod
    def _resolve_params_from_response_headers(
        response_headers: typing.Mapping[str, typing.Any],
    ):
        username = response_headers.get("x-remote-user")
        session = response_headers.get("x-v3io-session-key")
        user_id = response_headers.get("x-user-id")

        gids = response_headers.get("x-user-group-ids", [])
        # "x-user-group-ids" header is a comma separated list of group ids
        if gids and not isinstance(gids, list):
            gids = gids.split(",")

        planes = response_headers.get("x-v3io-session-planes")
        if planes:
            planes = planes.split(",")
        planes = planes or []
        user_unix_id = None
        x_unix_uid = response_headers.get("x-unix-uid")
        # x-unix-uid may be 'Unknown' in case it is missing or in case of enrichment failures
        if x_unix_uid and x_unix_uid.lower() != "unknown":
            user_unix_id = int(x_unix_uid)

        return username, session, planes, user_unix_id, user_id, gids

    @staticmethod
    def _resolve_params_from_response_body(
        response_body: typing.Mapping[typing.Any, typing.Any],
    ) -> tuple[typing.Optional[str], typing.Optional[list[str]]]:
        context_auth = get_in(
            response_body, "data.attributes.context.authentication", {}
        )
        user_id = context_auth.get("user_id", None)

        gids = context_auth.get("group_ids", [])

        # some gids can be a comma separated list of group ids
        # (if taken from the headers, the next split will have no effect)
        group_ids = []
        for gid in gids:
            group_ids.extend(gid.split(","))

        return user_id, group_ids

    @staticmethod
    def _transform_mlrun_project_to_iguazio_project(
        project: mlrun.common.schemas.Project,
    ) -> dict:
        body = {
            "data": {
                "type": "project",
                "attributes": {
                    "name": project.metadata.name,
                    "description": project.spec.description,
                    "admin_status": project.spec.desired_state,
                    "mlrun_project": Client._transform_mlrun_project_to_iguazio_mlrun_project_attribute(
                        project
                    ),
                },
            }
        }
        if project.metadata.created:
            body["data"]["attributes"]["created_at"] = (
                project.metadata.created.isoformat()
            )
        if project.metadata.labels is not None:
            body["data"]["attributes"]["labels"] = (
                Client._transform_mlrun_labels_to_iguazio_labels(
                    project.metadata.labels
                )
            )
        if project.metadata.annotations is not None:
            body["data"]["attributes"]["annotations"] = (
                Client._transform_mlrun_labels_to_iguazio_labels(
                    project.metadata.annotations
                )
            )
        if project.spec.owner:
            body["data"]["attributes"]["owner_username"] = project.spec.owner
        return body

    @staticmethod
    def _transform_mlrun_project_to_iguazio_mlrun_project_attribute(
        project: mlrun.common.schemas.Project,
    ):
        project_dict = project.dict(
            exclude_unset=True,
            exclude={
                "metadata": {"name", "created", "labels", "annotations"},
                "spec": {"description", "desired_state", "owner"},
                "status": {"state"},
            },
        )
        # ensure basic fields exist (schema should take care of that but we exclude, so status might be missing for
        # example)
        for field in ["metadata", "spec", "status"]:
            project_dict.setdefault(field, {})
        return json.dumps(project_dict)

    @staticmethod
    def _transform_mlrun_labels_to_iguazio_labels(
        mlrun_labels: dict,
    ) -> list[dict]:
        iguazio_labels = []
        for label_key, label_value in mlrun_labels.items():
            iguazio_labels.append({"name": label_key, "value": label_value})
        return iguazio_labels

    @staticmethod
    def _transform_iguazio_labels_to_mlrun_labels(
        iguazio_labels: list[dict],
    ) -> dict:
        return {label["name"]: label["value"] for label in iguazio_labels}

    @staticmethod
    def _transform_iguazio_project_to_mlrun_project(
        iguazio_project,
    ) -> mlrun.common.schemas.Project:
        mlrun_project_without_common_fields = json.loads(
            iguazio_project["attributes"].get("mlrun_project", "{}")
        )
        # name is mandatory in the mlrun schema, without adding it the schema initialization will fail
        mlrun_project_without_common_fields.setdefault("metadata", {})["name"] = (
            iguazio_project["attributes"]["name"]
        )
        mlrun_project = mlrun.common.schemas.Project(
            **mlrun_project_without_common_fields
        )
        mlrun_project.metadata.created = datetime.datetime.fromisoformat(
            iguazio_project["attributes"]["created_at"]
        )
        mlrun_project.spec.desired_state = mlrun.common.schemas.ProjectDesiredState(
            iguazio_project["attributes"]["admin_status"]
        )
        mlrun_project.status.state = mlrun.common.schemas.ProjectState(
            iguazio_project["attributes"]["operational_status"]
        )
        if iguazio_project["attributes"].get("description"):
            mlrun_project.spec.description = iguazio_project["attributes"][
                "description"
            ]
        if iguazio_project["attributes"].get("labels"):
            mlrun_project.metadata.labels = (
                Client._transform_iguazio_labels_to_mlrun_labels(
                    iguazio_project["attributes"]["labels"]
                )
            )
        if iguazio_project["attributes"].get("annotations"):
            mlrun_project.metadata.annotations = (
                Client._transform_iguazio_labels_to_mlrun_labels(
                    iguazio_project["attributes"]["annotations"]
                )
            )
        if iguazio_project["attributes"].get("owner_username"):
            mlrun_project.spec.owner = iguazio_project["attributes"]["owner_username"]

        if iguazio_project["attributes"].get("default_function_node_selector"):
            mlrun_project.spec.default_function_node_selector = (
                Client._transform_iguazio_labels_to_mlrun_labels(
                    iguazio_project["attributes"]["default_function_node_selector"]
                )
            )
        return mlrun_project

    def _prepare_request_kwargs(self, session, path, *, kwargs):
        # support session being already a cookie
        session_cookie = session
        if (
            session_cookie
            and not session_cookie.startswith('j:{"sid"')
            and not session_cookie.startswith(urllib.parse.quote_plus('j:{"sid"'))
        ):
            session_cookie = f'j:{{"sid": "{session_cookie}"}}'
        if session_cookie:
            cookies = kwargs.get("cookies", {})
            # in case some dev using this function for some reason setting cookies manually through kwargs + have a
            # cookie with "session" key there + filling the session cookie - explode
            if "session" in cookies and cookies["session"] != session_cookie:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Session cookie already set"
                )
            cookies["session"] = session_cookie
            kwargs["cookies"] = cookies
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = 20
        if "projects" in path:
            if mlrun.common.schemas.HeaderNames.projects_role not in kwargs.get(
                "headers", {}
            ):
                kwargs.setdefault("headers", {})[
                    mlrun.common.schemas.HeaderNames.projects_role
                ] = "mlrun"

        # requests no longer supports header values to be enum (https://github.com/psf/requests/pull/6154)
        # convert to strings. Do the same for params for niceness
        for kwarg in ["headers", "params"]:
            dict_ = kwargs.get(kwarg, {})
            for key in dict_.keys():
                if isinstance(dict_[key], enum.Enum):
                    dict_[key] = dict_[key].value

    def _handle_error_response(
        self, method, path, response, response_body, error_message, kwargs
    ):
        log_kwargs = copy.deepcopy(kwargs)

        # this can be big and spammy
        log_kwargs.pop("json", None)
        log_kwargs.update({"method": method, "path": path})
        try:
            ctx = response_body.get("meta", {}).get("ctx")
            errors = response_body.get("errors", [])
        except Exception:
            pass
        else:
            if errors:
                error_message = f"{error_message}: {str(errors)}"
            if errors or ctx:
                log_kwargs.update({"ctx": ctx, "errors": errors})

        self._logger.warning("Request to iguazio failed", **log_kwargs)
        mlrun.errors.raise_for_status(response, error_message)

    def _is_job_terminated(self, session: str, job_id: str) -> bool:
        """
        Check if the iguazio job is terminated

        :param session: iguazio session
        :param job_id: iguazio job id
        :return: True if the job is terminated, False otherwise
        """
        try:
            response = self._get_job_from_iguazio(session, job_id)
            return (
                response["data"]["attributes"]["state"] in JobStates.terminal_states()
            )
        except requests.HTTPError as exc:
            if exc.response.status_code == http.HTTPStatus.NOT_FOUND.value:
                raise mlrun.errors.MLRunNotFoundError(
                    f"Job {job_id} not found in Iguazio"
                )
            raise


class AsyncClient(Client):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._run_in_threadpool_callback = run_in_threadpool
        self._async_session: typing.Optional[mlrun.utils.AsyncClientWithRetry] = None

    @property
    def is_sync(self):
        """
        False because client is asynchronous
        """
        return False

    def __getattribute__(self, name):
        """
        This method is called when trying to access an attribute of the class.
        We override it to make sure that all *public* methods that are not async will be run in a thread pool.
          by convention/norm - public methods are methods that don't start with an underscore.
          If the method name starts with an underscore - it's a private method that was called from a public method,
          which means that it's already running in a thread pool or runs asynchronously.
        If the method is async, we don't do anything and let the async machinery handle it.

        """
        attr = super().__getattribute__(name)
        if name.startswith("_") or not callable(attr):
            return attr

        # already a coroutine
        if asyncio.iscoroutinefunction(attr):
            return attr

        # not a coroutine, run in threadpool
        def wrapper(*args, **kwargs):
            return self._run_in_threadpool_callback(attr, *args, **kwargs)

        return wrapper

    async def verify_request_session(
        self, request: fastapi.Request
    ) -> mlrun.common.schemas.AuthInfo:
        """
        Proxy the request to one of the session verification endpoints (which will verify the session of the request)
        """
        headers = {
            "authorization": request.headers.get("authorization"),
            "cookie": request.headers.get("cookie"),
            "x-request-id": request.state.request_id,
        }
        async with self._send_request_to_api_async(
            "POST",
            mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint,
            "Failed verifying iguazio session",
            headers=headers,
        ) as response:
            return self._generate_auth_info_from_session_verification_response(
                response.headers, await response.json()
            )

    @contextlib.asynccontextmanager
    async def _send_request_to_api_async(
        self, method, path: str, error_message: str, session=None, **kwargs
    ) -> aiohttp.ClientResponse:
        url = f"{self._api_url}/api/{path}"
        self._prepare_request_kwargs(session, path, kwargs=kwargs)
        await self._ensure_async_session()
        response = None
        try:
            response = await self._async_session.request(
                method, url, verify_ssl=False, **kwargs
            )
            if not response.ok:
                try:
                    response_body = await response.json()
                except Exception:
                    response_body = {}
                self._handle_error_response(
                    method, path, response, response_body, error_message, kwargs
                )
            yield response
        finally:
            if response:
                response.release()

    async def _ensure_async_session(self):
        if not self._async_session:
            self._async_session = mlrun.utils.AsyncClientWithRetry(
                retry_on_exception=mlrun.mlconf.httpdb.projects.retry_leader_request_on_exception
                == mlrun.common.schemas.HTTPSessionRetryMode.enabled.value,
                logger=logger,
            )
