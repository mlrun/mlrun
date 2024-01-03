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
import datetime
import traceback
import typing
import uuid

import fastapi
import fastapi.concurrency

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.singleton
import server.api.utils.helpers
import server.api.utils.singletons.project_member
from mlrun.utils import logger


class InternalBackgroundTasksHandler(metaclass=mlrun.utils.singleton.Singleton):
    def __init__(self):
        self._internal_background_tasks: typing.Dict[
            str, mlrun.common.schemas.BackgroundTask
        ] = {}

        # contains a lock for each background task kind, with the following format:
        # {kind: [active_name, previous_name]}
        self._background_tasks_kind_locks: typing.Dict[
            str, typing.Tuple[typing.Optional[str], typing.Optional[str]]
        ] = {}

    @server.api.utils.helpers.ensure_running_on_chief
    def create_background_task(
        self,
        background_tasks: fastapi.BackgroundTasks,
        kind: str,
        function,
        *args,
        **kwargs,
    ) -> mlrun.common.schemas.BackgroundTask:
        name = str(uuid.uuid4())
        # sanity
        if name in self._internal_background_tasks:
            raise RuntimeError("Background task name already exists")

        if self._get_active_task_name_by_kind(kind):
            raise mlrun.errors.MLRunConflictError(
                f"Background task of kind {kind} already running"
            )

        background_task = self._generate_background_task(name, kind)
        self._internal_background_tasks[name] = background_task
        self._set_active_task_name_by_kind(kind, name)
        background_tasks.add_task(
            self.background_task_wrapper,
            name=name,
            function=function,
            *args,
            **kwargs,
        )

        return self.get_background_task(name)

    @server.api.utils.helpers.ensure_running_on_chief
    def get_background_task(
        self,
        name: str,
    ) -> mlrun.common.schemas.BackgroundTask:
        """
        :return: returns the background task object and bool whether exists
        """
        if name in self._internal_background_tasks:
            return self._internal_background_tasks[name]
        else:
            return self._generate_background_task_not_found_response(name)

    @server.api.utils.helpers.ensure_running_on_chief
    def get_background_task_by_kind(
        self,
        kind: str,
    ) -> typing.Optional[mlrun.common.schemas.BackgroundTask]:
        name = self._get_active_task_name_by_kind(kind)
        if name:
            return self.get_background_task(name)
        else:
            return None

    @server.api.utils.helpers.ensure_running_on_chief
    async def background_task_wrapper(self, name: str, function, *args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(function):
                await function(*args, **kwargs)
            else:
                await fastapi.concurrency.run_in_threadpool(function, *args, **kwargs)

        except Exception as exc:
            err_str = mlrun.errors.err_to_str(exc)
            logger.warning(
                "Background task execution failed",
                function_name=function.__name__,
                exc=err_str,
                tb=traceback.format_exc(),
            )
            self._update_background_task(
                name, mlrun.common.schemas.BackgroundTaskState.failed, error=err_str
            )
        else:
            self._update_background_task(
                name, mlrun.common.schemas.BackgroundTaskState.succeeded
            )
        finally:
            self._finish_active_task_by_kind(
                self._internal_background_tasks[name].metadata.kind
            )

    def _update_background_task(
        self,
        name: str,
        state: mlrun.common.schemas.BackgroundTaskState,
        error: typing.Optional[str] = None,
    ):
        background_task = self._internal_background_tasks[name]
        background_task.status.state = state
        background_task.status.error = error
        background_task.metadata.updated = datetime.datetime.utcnow()

    @staticmethod
    def _generate_background_task_not_found_response(
        name: str, project: typing.Optional[str] = None
    ):
        # in order to keep things simple we don't persist the internal background tasks to the DB
        # If for some reason get is called and the background task doesn't exist, it means that probably we got
        # restarted, therefore we want to return a failed background task so the client will retry (if needed)
        return mlrun.common.schemas.BackgroundTask(
            metadata=mlrun.common.schemas.BackgroundTaskMetadata(
                name=name, project=project
            ),
            spec=mlrun.common.schemas.BackgroundTaskSpec(),
            status=mlrun.common.schemas.BackgroundTaskStatus(
                state=mlrun.common.schemas.BackgroundTaskState.failed,
                error="Background task not found",
            ),
        )

    @staticmethod
    def _generate_background_task(
        name: str, kind: str
    ) -> mlrun.common.schemas.BackgroundTask:
        now = datetime.datetime.utcnow()
        metadata = mlrun.common.schemas.BackgroundTaskMetadata(
            name=name,
            kind=kind,
            created=now,
            updated=now,
        )
        spec = mlrun.common.schemas.BackgroundTaskSpec()
        status = mlrun.common.schemas.BackgroundTaskStatus(
            state=mlrun.common.schemas.BackgroundTaskState.running
        )
        return mlrun.common.schemas.BackgroundTask(
            metadata=metadata, spec=spec, status=status
        )

    def _get_active_task_name_by_kind(self, kind: str):
        return self._background_tasks_kind_locks.get(kind, (None, None))[0]

    def _get_previous_task_name_by_kind(self, kind: str):
        return self._background_tasks_kind_locks.get(kind, (None, None))[1]

    def _set_active_task_name_by_kind(self, kind: str, name: str):
        self._background_tasks_kind_locks.setdefault(kind, (None, None))
        self._background_tasks_kind_locks[kind] = (
            name,
            self._background_tasks_kind_locks[kind][1],
        )

    def _finish_active_task_by_kind(self, kind: str):
        self._background_tasks_kind_locks.setdefault(kind, (None, None))

        # if we have a previous task, delete it from the internal background tasks.
        # this is done so not to have a memory leak of background tasks that are not needed anymore.
        # we'll keep history of 1 previous task per kind.
        if self._background_tasks_kind_locks[kind][1]:
            del self._internal_background_tasks[
                self._background_tasks_kind_locks[kind][1]
            ]

        self._background_tasks_kind_locks[kind] = (
            None,
            self._background_tasks_kind_locks[kind][0],
        )
