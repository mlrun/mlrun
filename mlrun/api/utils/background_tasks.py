import asyncio
import datetime
import traceback
import typing
import uuid

import fastapi
import fastapi.concurrency

import mlrun.api.schemas
import mlrun.api.utils.singletons.project_member
import mlrun.errors
import mlrun.utils.singleton
from mlrun.utils import logger


class Handler(metaclass=mlrun.utils.singleton.Singleton):
    def __init__(self):
        self._project_background_tasks: typing.Dict[
            str, typing.Dict[str, mlrun.api.schemas.BackgroundTask]
        ] = {}
        self._background_tasks: typing.Dict[str, mlrun.api.schemas.BackgroundTask] = {}

    def create_project_background_task(
        self,
        project: str,
        background_tasks: fastapi.BackgroundTasks,
        function,
        *args,
        **kwargs,
    ) -> mlrun.api.schemas.BackgroundTask:
        name = str(uuid.uuid4())
        # sanity
        if name in self._project_background_tasks:
            raise RuntimeError("Background task name already exists")
        background_task = self._generate_background_task(name, project)
        self._project_background_tasks.setdefault(project, {})[name] = background_task
        background_tasks.add_task(
            self.background_task_wrapper, project, name, function, *args, **kwargs
        )
        return self.get_project_background_task(project, name)

    def create_background_task(
        self, background_tasks: fastapi.BackgroundTasks, function, *args, **kwargs,
    ) -> mlrun.api.schemas.BackgroundTask:
        name = str(uuid.uuid4())
        # sanity
        if name in self._background_tasks:
            raise RuntimeError("Background task name already exists")
        background_task = self._generate_background_task(name)
        self._background_tasks[name] = background_task
        background_tasks.add_task(
            self.background_task_wrapper, None, name, function, *args, **kwargs
        )
        return self.get_background_task(name)

    def _generate_background_task(
        self, name: str, project: typing.Optional[str] = None
    ) -> mlrun.api.schemas.BackgroundTask:
        metadata = mlrun.api.schemas.BackgroundTaskMetadata(
            name=name, project=project, created=datetime.datetime.utcnow()
        )
        spec = mlrun.api.schemas.BackgroundTaskSpec()
        status = mlrun.api.schemas.BackgroundTaskStatus(
            state=mlrun.api.schemas.BackgroundTaskState.running
        )
        return mlrun.api.schemas.BackgroundTask(
            metadata=metadata, spec=spec, status=status
        )

    def get_project_background_task(
        self, project: str, name: str,
    ) -> mlrun.api.schemas.BackgroundTask:
        if (
            project in self._project_background_tasks
            and name in self._project_background_tasks[project]
        ):
            return self._project_background_tasks[project][name]
        else:
            return self._generate_background_task_not_found_response(name, project)

    def get_background_task(self, name: str,) -> mlrun.api.schemas.BackgroundTask:
        if name in self._background_tasks:
            return self._background_tasks[name]
        else:
            return self._generate_background_task_not_found_response(name)

    async def background_task_wrapper(
        self, project: typing.Optional[str], name: str, function, *args, **kwargs
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
                name, mlrun.api.schemas.BackgroundTaskState.failed, project
            )
        else:
            self._update_background_task(
                name, mlrun.api.schemas.BackgroundTaskState.succeeded, project
            )

    def _update_background_task(
        self,
        name: str,
        state: mlrun.api.schemas.BackgroundTaskState,
        project: typing.Optional[str] = None,
    ):
        if project is not None:
            background_task = self._project_background_tasks[project][name]
        else:
            background_task = self._background_tasks[name]
        background_task.status.state = state
        background_task.metadata.updated = datetime.datetime.utcnow()

    def _generate_background_task_not_found_response(
        self, name: str, project: typing.Optional[str] = None
    ):
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
