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
import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.singleton
import server.api.utils.background_tasks.common
import server.api.utils.helpers
import server.api.utils.singletons.db
import server.api.utils.singletons.project_member
from mlrun.utils import logger


class ProjectBackgroundTasksHandler(metaclass=mlrun.utils.singleton.Singleton):
    def create_background_task(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        background_tasks: fastapi.BackgroundTasks,
        function,
        timeout: int = None,  # in seconds
        name: str = None,
        *args,
        **kwargs,
    ) -> mlrun.common.schemas.BackgroundTask:
        name = name or str(uuid.uuid4())
        logger.debug(
            "Creating background task",
            name=name,
            project=project,
            function=function.__name__,
        )
        server.api.utils.singletons.db.get_db().store_background_task(
            db_session,
            name,
            project,
            mlrun.common.schemas.BackgroundTaskState.running,
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
    ) -> mlrun.common.schemas.BackgroundTask:
        return server.api.utils.singletons.db.get_db().get_background_task(
            db_session,
            name,
            project,
            server.api.utils.background_tasks.common.background_task_exceeded_timeout,
        )

    def list_background_tasks(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        states: typing.Optional[list[str]] = None,
        created_from: datetime.datetime = None,
        created_to: datetime.datetime = None,
        last_update_time_from: datetime.datetime = None,
        last_update_time_to: datetime.datetime = None,
    ) -> list[mlrun.common.schemas.BackgroundTask]:
        return server.api.utils.singletons.db.get_db().list_background_tasks(
            db_session,
            project,
            server.api.utils.background_tasks.common.background_task_exceeded_timeout,
            states=states,
            created_from=created_from,
            created_to=created_to,
            last_update_time_from=last_update_time_from,
            last_update_time_to=last_update_time_to,
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
        error = None
        state = mlrun.common.schemas.BackgroundTaskState.succeeded
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
                name=name,
            )
            state = mlrun.common.schemas.BackgroundTaskState.failed
            error = err_str
        finally:
            server.api.utils.singletons.db.get_db().store_background_task(
                db_session,
                name,
                project=project,
                state=state,
                error=error,
            )
