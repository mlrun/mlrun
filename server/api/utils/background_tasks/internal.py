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
import functools
import traceback
import typing
import uuid

import fastapi
import fastapi.concurrency

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.singleton
import server.api.utils.background_tasks.common
import server.api.utils.helpers
from mlrun.utils import logger


class InternalBackgroundTasksHandler(metaclass=mlrun.utils.singleton.Singleton):
    def __init__(self):
        self._internal_background_tasks: dict[
            str, mlrun.common.schemas.BackgroundTask
        ] = {}

        # contains a lock for each background task kind, with the following format:
        # {kind: [active_name, previous_name]}
        self._background_tasks_kind_locks: dict[
            str, tuple[typing.Optional[str], typing.Optional[str]]
        ] = {}

    @server.api.utils.helpers.ensure_running_on_chief
    def create_background_task(
        self,
        kind: str,
        timeout: typing.Optional[int],  # in seconds
        function,
        name: typing.Optional[str] = None,
        *args,
        **kwargs,
    ) -> tuple[typing.Callable, str]:
        name = name or str(uuid.uuid4())
        # sanity
        if name in self._internal_background_tasks:
            raise RuntimeError("Background task name already exists")

        if self._get_active_task_name_by_kind(kind):
            raise mlrun.errors.MLRunConflictError(
                f"Background task of kind '{kind}' already running"
            )

        project_name = None
        if project := kwargs.get("project", None):
            project_name = project.metadata.name if project else None

        background_task = self._generate_background_task(
            name, kind, timeout, project_name=project_name
        )
        self._internal_background_tasks[name] = background_task
        self._set_active_task_name_by_kind(kind, name)
        task = functools.partial(
            self.background_task_wrapper, background_task, function, *args, **kwargs
        )
        return task, name

    @server.api.utils.helpers.ensure_running_on_chief
    def list_background_tasks(
        self,
        name: typing.Optional[str] = None,
        kind: typing.Optional[str] = None,
    ) -> list[mlrun.common.schemas.BackgroundTask]:
        if name:
            background_task = self.get_background_task(name)
            return (
                [background_task]
                if background_task
                # filter out kind if specified
                and (not kind or background_task.metadata.kind == kind)
                else []
            )

        if kind:
            tasks = []
            if kind in self._background_tasks_kind_locks:
                tasks.extend(
                    # don't add None values from active and previous tasks
                    filter(
                        None,
                        [
                            self.get_active_background_task_by_kind(kind),
                            self.get_previous_background_task_by_kind(kind),
                        ],
                    )
                )
            return tasks

        return list(self._internal_background_tasks.values())

    @server.api.utils.helpers.ensure_running_on_chief
    def get_background_task(
        self,
        name: str,
        raise_on_not_found: bool = False,
    ) -> mlrun.common.schemas.BackgroundTask:
        """
        :return: returns the background task object and bool whether exists
        """
        if name in self._internal_background_tasks:
            background_task = self._internal_background_tasks[name]
            if server.api.utils.background_tasks.common.background_task_exceeded_timeout(
                background_task.metadata.created,
                background_task.metadata.timeout,
                background_task.status.state,
            ):
                self._update_background_task(
                    name,
                    mlrun.common.schemas.BackgroundTaskState.failed,
                    error="Timeout exceeded",
                )
                self._finish_active_task(name)
            return self._internal_background_tasks[name]
        elif raise_on_not_found:
            raise mlrun.errors.MLRunNotFoundError(f"Background task {name} not found")
        else:
            return self._generate_background_task_not_found_response(name)

    @server.api.utils.helpers.ensure_running_on_chief
    def get_active_background_task_by_kind(
        self,
        kind: str,
        raise_on_not_found: bool = False,
    ) -> typing.Optional[mlrun.common.schemas.BackgroundTask]:
        name = self._get_active_task_name_by_kind(kind)
        if name:
            return self.get_background_task(name, raise_on_not_found=raise_on_not_found)
        elif raise_on_not_found:
            raise mlrun.errors.MLRunNotFoundError(
                f"Active background task of kind '{kind}' not found"
            )
        else:
            return None

    @server.api.utils.helpers.ensure_running_on_chief
    def get_previous_background_task_by_kind(
        self,
        kind: str,
        raise_on_not_found: bool = False,
    ) -> typing.Optional[mlrun.common.schemas.BackgroundTask]:
        name = self._get_previous_task_name_by_kind(kind)
        if name:
            return self.get_background_task(name, raise_on_not_found=raise_on_not_found)
        elif raise_on_not_found:
            raise mlrun.errors.MLRunNotFoundError(
                f"Previous background task of kind '{kind}' not found"
            )
        else:
            return None

    @server.api.utils.helpers.ensure_running_on_chief
    async def background_task_wrapper(
        self,
        background_task: mlrun.common.schemas.BackgroundTask,
        function,
        *args,
        **kwargs,
    ):
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
                name=background_task.metadata.name,
                project=background_task.metadata.project,
            )
            self._update_background_task(
                background_task.metadata.name,
                mlrun.common.schemas.BackgroundTaskState.failed,
                error=err_str,
            )
        else:
            self._update_background_task(
                background_task.metadata.name,
                mlrun.common.schemas.BackgroundTaskState.succeeded,
            )
        finally:
            self._finish_active_task(background_task.metadata.name)

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
        name: str,
        kind: str,
        timeout: typing.Optional[int] = None,
        project_name: typing.Optional[str] = None,
    ) -> mlrun.common.schemas.BackgroundTask:
        now = datetime.datetime.utcnow()
        metadata = mlrun.common.schemas.BackgroundTaskMetadata(
            name=name, kind=kind, created=now, updated=now, project=project_name
        )
        if timeout and mlrun.mlconf.background_tasks.timeout_mode == "enabled":
            metadata.timeout = int(timeout)

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

    def _finish_active_task(self, name: str):
        kind = self._internal_background_tasks[name].metadata.kind
        self._background_tasks_kind_locks.setdefault(kind, (None, None))

        if self._background_tasks_kind_locks[kind][0] != name:
            logger.debug("Background task already marked as finished, skipping...")
            return

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
