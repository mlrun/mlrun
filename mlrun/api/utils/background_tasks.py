import asyncio
import datetime
import traceback
import typing
import uuid

import fastapi
import fastapi.concurrency

import mlrun.api.schemas
import mlrun.errors
import mlrun.utils.singleton
from mlrun.utils import logger


class Handler(metaclass=mlrun.utils.singleton.Singleton):
    def __init__(self):
        self._background_tasks: typing.Dict[str, mlrun.api.schemas.BackgroundTask] = {}

    def create_background_task(
        self, background_tasks: fastapi.BackgroundTasks, function, *args, **kwargs
    ) -> mlrun.api.schemas.BackgroundTask:
        name = str(uuid.uuid4())
        # sanity
        if name in self._background_tasks:
            raise RuntimeError("Background task name already exists")
        self._save_background_task(name)
        background_tasks.add_task(
            self.background_task_wrapper, name, function, *args, **kwargs
        )
        return self.get_background_task(name)

    def _save_background_task(self, name: str):
        metadata = mlrun.api.schemas.BackgroundTaskMetadata(
            name=name, created=datetime.datetime.utcnow()
        )
        status = mlrun.api.schemas.BackgroundTaskStatus(
            state=mlrun.api.schemas.BackgroundTaskState.running
        )
        self._background_tasks[name] = mlrun.api.schemas.BackgroundTask(
            metadata=metadata, status=status
        )

    def get_background_task(self, name: str) -> mlrun.api.schemas.BackgroundTask:
        if name in self._background_tasks:
            return self._background_tasks[name]
        else:
            # in order to keep things simple we don't persist the background tasks to the DB
            # If for some reason get is called and the background task doesn't exist, it means that probably we got
            # restarted, therefore we want to return a failed background task so the client will retry (if needed)
            return mlrun.api.schemas.BackgroundTask(
                metadata=mlrun.api.schemas.BackgroundTaskMetadata(name=name),
                status=mlrun.api.schemas.BackgroundTaskStatus(
                    state=mlrun.api.schemas.BackgroundTaskState.failed
                ),
            )

    async def background_task_wrapper(
        self, background_task_name: str, function, *args, **kwargs
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
                background_task_name, mlrun.api.schemas.BackgroundTaskState.failed
            )
        finally:
            self._update_background_task(
                background_task_name, mlrun.api.schemas.BackgroundTaskState.succeeded
            )

    def _update_background_task(
        self, name: str, state: mlrun.api.schemas.BackgroundTaskState
    ):
        background_task = self._background_tasks[name]
        background_task.status.state = state
        background_task.metadata.updated = datetime.datetime.utcnow()
