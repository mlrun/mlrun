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
#
import os
import shutil
import typing
from http import HTTPStatus

from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.schemas
import mlrun.api.utils.clients.log_collector as log_collector
import mlrun.utils.singleton
from mlrun.api.api.utils import log_and_raise, log_path, project_logs_path
from mlrun.api.constants import LogSources
from mlrun.api.utils.singletons.db import get_db
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.runtimes.constants import PodPhases
from mlrun.utils import logger


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
        log_file = log_path(project, uid)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        mode = "ab" if append else "wb"
        with log_file.open(mode) as fp:
            fp.write(body)

    def delete_logs(
        self,
        project: str,
    ):
        project = project or mlrun.mlconf.default_project
        logs_path = project_logs_path(project)
        if logs_path.exists():
            shutil.rmtree(str(logs_path))

    async def get_logs(
        self,
        db_session: Session,
        project: str,
        uid: str,
        size: int = -1,
        offset: int = 0,
        source: LogSources = LogSources.AUTO,
    ) -> typing.Tuple[str, bytes]:
        """
        Get logs
        :param db_session: db session
        :param project: project name
        :param uid: run uid
        :param size: number of bytes to return (default -1, return all)
        :param offset: number of bytes to skip (default 0)
        :param source: log source (default auto) Relevant only for legacy log_collector mode
        :return:
        """
        project = project or mlrun.mlconf.default_project
        run = await run_in_threadpool(get_db().read_run, db_session, uid, project)
        logs = b""
        if not run:
            log_and_raise(HTTPStatus.NOT_FOUND.value, project=project, uid=uid)
        run_state = run.get("status", {}).get("state", "")
        if (
            mlrun.mlconf.logs_collector.mode
            == mlrun.api.schemas.LogsCollectorMode.best_effort
        ):
            try:
                logs = await self._get_logs_from_logs_collector(
                    project,
                    uid,
                    size,
                    offset,
                )
            except Exception as exc:
                if mlrun.mlconf.logs_collector.verbose:
                    logger.warning(
                        "Failed to get logs from logs collector, falling back to legacy method",
                        exc=exc,
                    )
                logs = await run_in_threadpool(
                    self._get_logs_legacy_method,
                    db_session,
                    project,
                    uid,
                    size,
                    offset,
                    source,
                    run,
                )
        elif (
            mlrun.mlconf.logs_collector.mode
            == mlrun.api.schemas.LogsCollectorMode.sidecar
        ):
            logs = await self._get_logs_from_logs_collector(
                project,
                uid,
                size,
                offset,
            )
        elif (
            mlrun.mlconf.logs_collector.mode
            == mlrun.api.schemas.LogsCollectorMode.legacy
        ):
            logs = await run_in_threadpool(
                self._get_logs_legacy_method,
                db_session,
                project,
                uid,
                size,
                offset,
                source,
                run,
            )
        return run_state, logs

    @staticmethod
    async def _get_logs_from_logs_collector(
        project: str,
        run_uid: str,
        size: int = -1,
        offset: int = 0,
    ) -> bytes:
        logs = await log_collector.LogCollectorClient().get_logs(
            run_uid=run_uid,
            project=project,
            size=size,
            offset=offset,
        )
        return logs

    @staticmethod
    def _get_logs_legacy_method(
        db_session: Session,
        project: str,
        uid: str,
        size: int = -1,
        offset: int = 0,
        source: LogSources = LogSources.AUTO,
        run: dict = None,
    ) -> bytes:
        """
        :return: Tuple with:
            1. str of the run state (so watchers will know whether to continue polling for logs)
            2. bytes of the logs themselves
        """
        project = project or mlrun.mlconf.default_project
        log_contents = b""
        log_file = log_path(project, uid)
        if not run:
            run = get_db().read_run(db_session, uid, project)
        if not run:
            log_and_raise(HTTPStatus.NOT_FOUND.value, project=project, uid=uid)
        if log_file.exists() and source in [LogSources.AUTO, LogSources.PERSISTENCY]:
            with log_file.open("rb") as fp:
                fp.seek(offset)
                log_contents = fp.read(size)
        elif source in [LogSources.AUTO, LogSources.K8S]:
            k8s = get_k8s()
            if k8s and k8s.is_running_inside_kubernetes_cluster():
                run_kind = run.get("metadata", {}).get("labels", {}).get("kind")
                pods = get_k8s().get_logger_pods(project, uid, run_kind)
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
                        resp = get_k8s().logs(pod)
                        if resp:
                            log_contents = resp.encode()[offset:]
        return log_contents

    def get_log_mtime(self, project: str, uid: str) -> int:
        log_file = log_path(project, uid)
        if not log_file.exists():
            raise FileNotFoundError(f"Log file does not exist: {log_file}")
        return log_file.stat().st_mtime

    def log_file_exists(self, project: str, uid: str) -> bool:
        log_file = log_path(project, uid)
        return log_file.exists()

    def _list_project_logs_uids(self, project: str) -> typing.List[str]:
        logs_path = project_logs_path(project)
        return [
            file
            for file in os.listdir(str(logs_path))
            if os.path.isfile(os.path.join(str(logs_path), file))
        ]
