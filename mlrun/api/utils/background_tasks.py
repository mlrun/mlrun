import asyncio
import datetime
import traceback
import typing
import uuid

import fastapi
import fastapi.concurrency
import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.project_member
import mlrun.errors
import mlrun.utils.singleton
from mlrun.utils import logger


class Handler(metaclass=mlrun.utils.singleton.Singleton):
    def create_background_task(
        self,
        db_session: sqlalchemy.orm.Session,
        background_tasks: fastapi.BackgroundTasks,
        function,
        project: str = None,
        timeout: int = None,
        *args,
        **kwargs,
    ) -> mlrun.api.schemas.BackgroundTask:
        name = str(uuid.uuid4())
        mlrun.api.utils.singletons.db.get_db().store_background_task(
            db_session,
            name,
            mlrun.api.schemas.BackgroundTaskState.running,
            project,
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

    @staticmethod
    def _generate_background_task(
        name: str, project: typing.Optional[str] = None
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

    def get_background_task(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: str = None,
    ) -> mlrun.api.schemas.BackgroundTask:
        return mlrun.api.utils.singletons.db.get_db().get_background_task(
            db_session, name, project
        )

    async def background_task_wrapper(
        self,
        db_session: sqlalchemy.orm.Session,
        project: typing.Optional[str],
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
                state=mlrun.api.schemas.BackgroundTaskState.failed,
                project=project,
            )
        else:
            mlrun.api.utils.singletons.db.get_db().store_background_task(
                db_session,
                name,
                state=mlrun.api.schemas.BackgroundTaskState.succeeded,
                project=project,
            )
