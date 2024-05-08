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
import os
import pathlib
import shutil
import typing
from http import HTTPStatus

from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.common.schemas
import mlrun.utils.singleton
import server.api.api.utils
import server.api.utils.clients.log_collector as log_collector
import server.api.utils.singletons.k8s
from mlrun.common.runtimes.constants import PodPhases
from mlrun.utils import logger
from server.api.constants import LogSources
from server.api.utils.singletons.db import get_db


class Logs(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def store_log(
        self,
        body: bytes,
        project: str,
        uid: str,
        append: bool = True,
    ):
        project = project or mlrun.mlconf.default_project
        log_file = server.api.api.utils.log_path(project, uid)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        mode = "ab" if append else "wb"
        with log_file.open(mode) as fp:
            fp.write(body)

    async def stop_logs_for_project(
        self,
        project_name: str,
    ) -> None:
        logger.debug("Stopping logs for project", project=project_name)
        await self._stop_logs(project_name)

    async def stop_logs_for_run(
        self,
        project_name: str,
        run_uid: str,
    ) -> None:
        logger.debug("Stopping logs for run", project=project_name, run_uid=run_uid)
        await self._stop_logs(project_name, [run_uid])

    async def delete_project_logs(self, project: str):
        logger.debug("Deleting logs for project", project=project)
        await self._delete_logs(project)

    async def delete_run_logs(self, project: str, run_uid: str):
        logger.debug("Deleting logs for run", project=project, run_uid=run_uid)
        await self._delete_logs(project, [run_uid])

    @staticmethod
    def delete_project_logs_legacy(
        project: str,
    ):
        project = project or mlrun.mlconf.default_project
        logs_path = server.api.api.utils.project_logs_path(project)
        if logs_path.exists():
            shutil.rmtree(str(logs_path))

    @staticmethod
    def delete_run_logs_legacy(
        project: str,
        run_uid: str,
    ):
        project = project or mlrun.mlconf.default_project
        logs_path = server.api.api.utils.log_path(project, run_uid)
        if logs_path.exists():
            shutil.rmtree(str(logs_path))

    async def get_log_size(
        self,
        project: str,
        run_uid: str,
    ):
        logger.debug("Getting log size for run", project=project, run_uid=run_uid)
        if (
            mlrun.mlconf.log_collector.mode
            == mlrun.common.schemas.LogsCollectorMode.sidecar
        ):
            return await self._get_log_size_from_log_collector(project, run_uid)

        elif (
            mlrun.mlconf.log_collector.mode
            == mlrun.common.schemas.LogsCollectorMode.best_effort
        ):
            try:
                return await self._get_log_size_from_log_collector(project, run_uid)
            except Exception as exc:
                if mlrun.mlconf.log_collector.verbose:
                    logger.warning(
                        "Failed to get logs from logs collector, falling back to legacy method",
                        exc=mlrun.errors.err_to_str(exc),
                    )
                return self._get_log_size_legacy(project, run_uid)

        elif (
            mlrun.mlconf.log_collector.mode
            == mlrun.common.schemas.LogsCollectorMode.legacy
        ):
            return self._get_log_size_legacy(project, run_uid)

        else:
            raise ValueError(
                f"Invalid log collector mode {mlrun.mlconf.log_collector.mode}"
            )

    async def get_logs(
        self,
        db_session: Session,
        project: str,
        uid: str,
        size: int = -1,
        offset: int = 0,
        source: LogSources = LogSources.AUTO,
    ) -> tuple[str, typing.AsyncIterable[bytes]]:
        """
        Get logs
        :param db_session: db session
        :param project: project name
        :param uid: run uid
        :param size: number of bytes to return (default -1, return all)
        :param offset: number of bytes to skip (default 0)
        :param source: log source (default auto) Relevant only for legacy log_collector mode
          if auto, it will use the mode configured in `mlrun.mlconf.log_collector.mode`
          if other than auto, it will fall back to legacy log_collector mode
        :return: run state and logs
        """
        project = project or mlrun.mlconf.default_project
        run = await self._get_run_for_log(db_session, project, uid)
        run_state = run.get("status", {}).get("state", "")
        log_stream = None
        if (
            mlrun.mlconf.log_collector.mode
            == mlrun.common.schemas.LogsCollectorMode.best_effort
            and source == LogSources.AUTO
        ):
            try:
                log_stream = self._get_logs_from_logs_collector(
                    project,
                    uid,
                    size,
                    offset,
                )
            except Exception as exc:
                if mlrun.mlconf.log_collector.verbose:
                    logger.warning(
                        "Failed to get logs from logs collector, falling back to legacy method",
                        exc=mlrun.errors.err_to_str(exc),
                    )
                log_stream = self._get_logs_legacy_method_generator_wrapper(
                    db_session,
                    project,
                    uid,
                    size,
                    offset,
                    source,
                    run,
                )
        elif (
            mlrun.mlconf.log_collector.mode
            == mlrun.common.schemas.LogsCollectorMode.sidecar
            and source == LogSources.AUTO
        ):
            log_stream = self._get_logs_from_logs_collector(
                project,
                uid,
                size,
                offset,
            )
        elif (
            mlrun.mlconf.log_collector.mode
            == mlrun.common.schemas.LogsCollectorMode.legacy
            or source != LogSources.AUTO
        ):
            log_stream = self._get_logs_legacy_method_generator_wrapper(
                db_session,
                project,
                uid,
                size,
                offset,
                source,
                run,
            )
        return run_state, log_stream

    @staticmethod
    async def _get_logs_from_logs_collector(
        project: str,
        run_uid: str,
        size: int = -1,
        offset: int = 0,
    ) -> typing.AsyncIterable[bytes]:
        log_collector_client = log_collector.LogCollectorClient()
        async for log in log_collector_client.get_logs(
            run_uid=run_uid,
            project=project,
            size=size,
            offset=offset,
        ):
            yield log

    def _get_logs_legacy_method(
        self,
        db_session: Session,
        project: str,
        uid: str,
        size: int = -1,
        offset: int = 0,
        source: LogSources = LogSources.AUTO,
        run: dict = None,
    ) -> bytes:
        """
        :return: bytes of the logs themselves
        """
        project = project or mlrun.mlconf.default_project
        log_contents = b""
        log_file_exists, log_file = self.log_file_exists_for_run_uid(project, uid)
        if not run:
            run = get_db().read_run(db_session, uid, project)
        if not run:
            server.api.api.utils.log_and_raise(
                HTTPStatus.NOT_FOUND.value, project=project, uid=uid
            )
        if log_file_exists and source in [LogSources.AUTO, LogSources.PERSISTENCY]:
            with log_file.open("rb") as fp:
                fp.seek(offset)
                log_contents = fp.read(size)
        elif source in [LogSources.AUTO, LogSources.K8S]:
            k8s = server.api.utils.singletons.k8s.get_k8s_helper()
            if k8s and k8s.is_running_inside_kubernetes_cluster():
                run_kind = run.get("metadata", {}).get("labels", {}).get("kind")
                pods = server.api.utils.singletons.k8s.get_k8s_helper().get_logger_pods(
                    project, uid, run_kind
                )
                if pods:
                    if len(pods) > 1:
                        # This shouldn't happen, but if it does, we log it here. No need to fail.
                        logger.debug(
                            "Got more than one pod in logger pods result",
                            run_uid=uid,
                            run_kind=run_kind,
                            project=project,
                            pods=pods,
                        )
                    pod, pod_phase = list(pods.items())[0]
                    if pod_phase != PodPhases.pending:
                        resp = server.api.utils.singletons.k8s.get_k8s_helper().logs(
                            pod
                        )
                        if resp:
                            if size == -1:
                                log_contents = resp.encode()[offset:]
                            else:
                                log_contents = resp.encode()[offset : offset + size]
        return log_contents

    async def _get_logs_legacy_method_generator_wrapper(
        self,
        db_session: Session,
        project: str,
        uid: str,
        size: int = -1,
        offset: int = 0,
        source: LogSources = LogSources.AUTO,
        run: dict = None,
    ):
        log_contents = await run_in_threadpool(
            self._get_logs_legacy_method,
            db_session,
            project,
            uid,
            size,
            offset,
            source,
            run,
        )
        yield log_contents

    @staticmethod
    async def _get_run_for_log(db_session: Session, project: str, uid: str) -> dict:
        run = await run_in_threadpool(get_db().read_run, db_session, uid, project)
        if not run:
            server.api.api.utils.log_and_raise(
                HTTPStatus.NOT_FOUND.value, project=project, uid=uid
            )
        return run

    @staticmethod
    async def _get_log_size_from_log_collector(project: str, run_uid: str) -> int:
        log_collector_client = (
            server.api.utils.clients.log_collector.LogCollectorClient()
        )
        log_file_size = await log_collector_client.get_log_size(
            project=project,
            run_uid=run_uid,
        )
        if log_file_size < 0:
            # If the log file size is negative, it means the log file doesn't exist
            raise mlrun.errors.MLRunNotFoundError(
                f"Log file for {project}/{run_uid} not found",
            )
        return log_file_size

    @staticmethod
    def _get_log_size_legacy(project: str, uid: str) -> int:
        log_file = server.api.api.utils.log_path(project, uid)
        if not log_file.exists():
            raise mlrun.errors.MLRunNotFoundError(
                f"Log file for {project}/{uid} not found",
            )
        return log_file.stat().st_size

    @staticmethod
    def log_file_exists_for_run_uid(project: str, uid: str) -> (bool, pathlib.Path):
        """
        Checks if the log file exists for the given project and uid
        A Run's log file path is: /mlrun/logs/{project}/{uid}
        :param project: project name
        :param uid: run uid
        :return: True if the log file exists, False otherwise, and the log file path
        """
        project_logs_dir = server.api.api.utils.project_logs_path(project)
        if not project_logs_dir.exists():
            return False, None

        log_file = server.api.api.utils.log_path(project, uid)
        if log_file.exists():
            return True, log_file

        return False, None

    def _list_project_logs_uids(self, project: str) -> list[str]:
        logs_path = server.api.api.utils.project_logs_path(project)
        return [
            file
            for file in os.listdir(str(logs_path))
            if os.path.isfile(os.path.join(str(logs_path), file))
        ]

    @staticmethod
    async def _stop_logs(
        project_name: str,
        run_uids: list[str] = None,
    ) -> None:
        resource = "project" if not run_uids else "run"
        try:
            log_collector_client = (
                server.api.utils.clients.log_collector.LogCollectorClient()
            )
            await log_collector_client.stop_logs(
                project=project_name,
                run_uids=run_uids,
            )
        except Exception as exc:
            logger.warning(
                f"Failed stopping logs for {resource}, Ignoring",
                exc=mlrun.errors.err_to_str(exc),
                project=project_name,
                run_uids=run_uids,
            )
        else:
            logger.debug(
                f"Successfully stopped logs for {resource}",
                project=project_name,
                run_uids=run_uids,
            )

    async def _delete_logs(self, project: str, run_uids: list[str] = None):
        resource = "project" if not run_uids else "run"
        try:
            log_collector_client = (
                server.api.utils.clients.log_collector.LogCollectorClient()
            )
            await log_collector_client.delete_logs(
                project=project,
                run_uids=run_uids,
            )
        except Exception as exc:
            logger.warning(
                f"Failed deleting {resource} logs via the log collector. Falling back to deleting logs explicitly",
                exc=mlrun.errors.err_to_str(exc),
                project=project,
                runs=run_uids,
            )

            # fallback to deleting logs explicitly if the log collector failed
            if run_uids:
                for run_uid in run_uids:
                    await run_in_threadpool(
                        self.delete_run_logs_legacy,
                        project,
                        run_uid,
                    )
            else:
                await run_in_threadpool(
                    self.delete_project_logs_legacy,
                    project,
                )

        logger.debug(
            f"Successfully deleted {resource} logs", project=project, runs=run_uids
        )
