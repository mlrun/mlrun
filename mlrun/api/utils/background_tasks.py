import asyncio
import datetime
import traceback
import typing
import uuid

import fastapi
import fastapi.concurrency
import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.singletons.project_member
import mlrun.errors
import mlrun.utils.singleton
from mlrun.utils import logger


class Handler(metaclass=mlrun.utils.singleton.Singleton):
    def __init__(self):
        self._background_tasks: typing.Dict[
            str, typing.Dict[str, mlrun.api.schemas.BackgroundTask]
        ] = {}

    def create_background_task(
        self,
        db_session: sqlalchemy.orm.Session,
        leader_session: typing.Optional[str],
        project: str,
        background_tasks: fastapi.BackgroundTasks,
        function,
        *args,
        **kwargs,
    ) -> mlrun.api.schemas.BackgroundTask:
        name = str(uuid.uuid4())
        # sanity
        if name in self._background_tasks:
            raise RuntimeError("Background task name already exists")
        self._save_background_task(db_session, project, name, leader_session)
        background_tasks.add_task(
            self.background_task_wrapper, project, name, function, *args, **kwargs
        )
        return self.get_background_task(project, name)

    def _save_background_task(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        leader_session: typing.Optional[str] = None,
    ):
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project(
            db_session, project, leader_session=leader_session,
        )
        metadata = mlrun.api.schemas.BackgroundTaskMetadata(
            name=name, project=project, created=datetime.datetime.utcnow()
        )
        spec = mlrun.api.schemas.BackgroundTaskSpec()
        status = mlrun.api.schemas.BackgroundTaskStatus(
            state=mlrun.api.schemas.BackgroundTaskState.running
        )
        self._background_tasks.setdefault(project, {})[
            name
        ] = mlrun.api.schemas.BackgroundTask(
            metadata=metadata, spec=spec, status=status
        )

    def get_background_task(
        self, project: str, name: str
    ) -> mlrun.api.schemas.BackgroundTask:
        if (
            project in self._background_tasks
            and name in self._background_tasks[project]
        ):
            return self._background_tasks[project][name]
        else:
            # in order to keep things simple we don't persist the background tasks to the DB
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

    async def background_task_wrapper(
        self, project: str, name: str, function, *args, **kwargs
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
            self._update_background_task(
                project, name, mlrun.api.schemas.BackgroundTaskState.failed
            )
        else:
            self._update_background_task(
                project, name, mlrun.api.schemas.BackgroundTaskState.succeeded
            )

    def _update_background_task(
        self, project: str, name: str, state: mlrun.api.schemas.BackgroundTaskState
    ):
        background_task = self._background_tasks[project][name]
        background_task.status.state = state
        background_task.metadata.updated = datetime.datetime.utcnow()
