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
import asyncio
import datetime
import traceback
import typing
import uuid

import fastapi
import fastapi.concurrency
import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.helpers
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.project_member
import mlrun.errors
import mlrun.utils.singleton
from mlrun.utils import logger


class ProjectBackgroundTasksHandler(metaclass=mlrun.utils.singleton.Singleton):
    def create_background_task(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        background_tasks: fastapi.BackgroundTasks,
        function,
        timeout: int = None,  # in seconds
        *args,
        **kwargs,
    ) -> mlrun.api.schemas.BackgroundTask:
        name = str(uuid.uuid4())
        mlrun.api.utils.singletons.db.get_db().store_background_task(
            db_session,
            name,
            project,
            mlrun.api.schemas.BackgroundTaskState.running,
            timeout,
        )
        background_tasks.add_task(
            self.background_task_wrapper,
            db_session,
            project,
            name,
            function,
            *args,
            **kwargs,
        )
        return self.get_background_task(db_session, name, project)

    def get_background_task(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: str,
    ) -> mlrun.api.schemas.BackgroundTask:
        return mlrun.api.utils.singletons.db.get_db().get_background_task(
            db_session, name, project
        )

    async def background_task_wrapper(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        function,
        *args,
        **kwargs,
    ):
        try:
            if asyncio.iscoroutinefunction(function):
                await function(*args, **kwargs)
            else:
                await fastapi.concurrency.run_in_threadpool(function, *args, **kwargs)
        except Exception:
            logger.warning(
                f"Failed during background task execution: {function.__name__}, exc: {traceback.format_exc()}"
            )
            mlrun.api.utils.singletons.db.get_db().store_background_task(
                db_session,
                name,
                project=project,
                state=mlrun.api.schemas.BackgroundTaskState.failed,
            )
        else:
            mlrun.api.utils.singletons.db.get_db().store_background_task(
                db_session,
                name,
                project=project,
                state=mlrun.api.schemas.BackgroundTaskState.succeeded,
            )


class InternalBackgroundTasksHandler(metaclass=mlrun.utils.singleton.Singleton):
    def __init__(self):
        self._internal_background_tasks: typing.Dict[
            str, mlrun.api.schemas.BackgroundTask
        ] = {}

    @mlrun.api.utils.helpers.ensure_running_on_chief
    def create_background_task(
        self,
        background_tasks: fastapi.BackgroundTasks,
        function,
        *args,
        **kwargs,
    ) -> mlrun.api.schemas.BackgroundTask:
        name = str(uuid.uuid4())
        # sanity
        if name in self._internal_background_tasks:
            raise RuntimeError("Background task name already exists")
        background_task = self._generate_background_task(name)
        self._internal_background_tasks[name] = background_task
        background_tasks.add_task(
            self.background_task_wrapper,
            name=name,
            function=function,
            *args,
            **kwargs,
        )

        return self.get_background_task(name)

    @mlrun.api.utils.helpers.ensure_running_on_chief
    def get_background_task(
        self,
        name: str,
    ) -> mlrun.api.schemas.BackgroundTask:
        """
        :return: returns the background task object and bool whether exists
        """
        if name in self._internal_background_tasks:
            return self._internal_background_tasks[name]
        else:
            return self._generate_background_task_not_found_response(name)

    @mlrun.api.utils.helpers.ensure_running_on_chief
    async def background_task_wrapper(self, name: str, function, *args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(function):
                await function(*args, **kwargs)
            else:
                await fastapi.concurrency.run_in_threadpool(function, *args, **kwargs)

        except Exception:
            logger.warning(
                f"Failed during background task execution: {function.__name__}, exc: {traceback.format_exc()}"
            )
            self._update_background_task(
                name, mlrun.api.schemas.BackgroundTaskState.failed
            )
        else:
            self._update_background_task(
                name, mlrun.api.schemas.BackgroundTaskState.succeeded
            )

    def _update_background_task(
        self,
        name: str,
        state: mlrun.api.schemas.BackgroundTaskState,
    ):
        background_task = self._internal_background_tasks[name]
        background_task.status.state = state
        background_task.metadata.updated = datetime.datetime.utcnow()

    @staticmethod
    def _generate_background_task_not_found_response(
        name: str, project: typing.Optional[str] = None
    ):
        # in order to keep things simple we don't persist the internal background tasks to the DB
        # If for some reason get is called and the background task doesn't exist, it means that probably we got
        # restarted, therefore we want to return a failed background task so the client will retry (if needed)
        return mlrun.api.schemas.BackgroundTask(
            metadata=mlrun.api.schemas.BackgroundTaskMetadata(
                name=name, project=project
            ),
            spec=mlrun.api.schemas.BackgroundTaskSpec(),
            status=mlrun.api.schemas.BackgroundTaskStatus(
                state=mlrun.api.schemas.BackgroundTaskState.failed
            ),
        )

    @staticmethod
    def _generate_background_task(
        name: str, project: typing.Optional[str] = None
    ) -> mlrun.api.schemas.BackgroundTask:
        now = datetime.datetime.utcnow()
        metadata = mlrun.api.schemas.BackgroundTaskMetadata(
            name=name,
            project=project,
            created=now,
            updated=now,
        )
        spec = mlrun.api.schemas.BackgroundTaskSpec()
        status = mlrun.api.schemas.BackgroundTaskStatus(
            state=mlrun.api.schemas.BackgroundTaskState.running
        )
        return mlrun.api.schemas.BackgroundTask(
            metadata=metadata, spec=spec, status=status
        )
