import asyncio
import traceback
import typing
from http import HTTPStatus
from os import environ
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.errors
from mlrun.api import schemas
from mlrun.api.db.session import create_session, close_session
from mlrun.api.db.sqldb.db import SQLDB
from mlrun.api.utils.singletons.db import get_db
from mlrun.api.utils.singletons.logs_dir import get_logs_dir
from mlrun.api.utils.singletons.scheduler import get_scheduler
from mlrun.config import config
from mlrun.db.sqldb import SQLDB as SQLRunDB
from mlrun.run import import_function, new_function
from mlrun.runtimes import get_runtime_handler
from mlrun.utils import get_in, logger, parse_function_uri


def log_and_raise(status=HTTPStatus.BAD_REQUEST.value, **kw):
    logger.error(str(kw))
    raise HTTPException(status_code=status, detail=kw)


def log_path(project, uid) -> Path:
    return get_logs_dir() / project / uid


def get_obj_path(schema, path, user=""):
    if schema:
        return schema + "://" + path
    elif path.startswith("/User/"):
        user = user or environ.get("V3IO_USERNAME", "admin")
        return "v3io:///users/" + user + path[5:]
    elif config.httpdb.data_volume and path.startswith(config.httpdb.data_volume):
        if config.httpdb.real_path:
            path = config.httpdb.real_path + path[len(config.httpdb.data_volume) - 1 :]
        return path
    return None


def get_secrets(_request: Request):
    access_key = _request.headers.get("X-V3io-Session-Key")
    return {
        "V3IO_ACCESS_KEY": access_key,
    }


def get_run_db_instance(db_session: Session):
    db = get_db()
    if isinstance(db, SQLDB):
        run_db = SQLRunDB(db.dsn, db_session, db.get_projects_cache())
    else:
        run_db = db.db
    run_db.connect()
    return run_db


def _parse_submit_run_body(db_session: Session, data):
    task = data.get("task")
    function_dict = data.get("function")
    function_url = data.get("functionUrl")
    if not function_url and task:
        function_url = get_in(task, "spec.function")
    if not (function_dict or function_url) or not task:
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason="bad JSON, need to include function/url and task objects",
        )

    # TODO: block exec for function["kind"] in ["", "local]  (must be a
    # remote/container runtime)

    if function_dict and not function_url:
        function = new_function(runtime=function_dict)
    else:
        if "://" in function_url:
            function = import_function(url=function_url)
        else:
            project, name, tag, hash_key = parse_function_uri(function_url)
            function_record = get_db().get_function(
                db_session, name, project, tag, hash_key
            )
            if not function_record:
                log_and_raise(
                    HTTPStatus.NOT_FOUND.value,
                    reason="runtime error: function {} not found".format(function_url),
                )
            function = new_function(runtime=function_record)

        if function_dict:
            # The purpose of the function dict is to enable the user to override configurations of the existing function
            # without modifying it - to do that we're creating a function object from the request function dict and
            # assign values from it to the main function object
            override_function = new_function(runtime=function_dict, kind=function.kind)
            for attribute in [
                "volumes",
                "volume_mounts",
                "env",
                "resources",
                "image_pull_policy",
                "replicas",
            ]:
                override_value = getattr(override_function.spec, attribute, None)
                if override_value:
                    if attribute == "env":
                        for env_dict in override_value:
                            function.set_env(env_dict["name"], env_dict["value"])
                    elif attribute == "volumes":
                        function.spec.update_vols_and_mounts(override_value, [])
                    elif attribute == "volume_mounts":
                        function.spec.update_vols_and_mounts([], override_value)
                    elif attribute == "resources":
                        # don't override it there are limits and requests but both are empty
                        if override_value.get("limits", {}) or override_value.get(
                            "requests", {}
                        ):
                            setattr(function.spec, attribute, override_value)
                    else:
                        setattr(function.spec, attribute, override_value)

    return function, task


async def submit_task(db_session: Session, data):
    project, function_kind, run_uid, response = await run_in_threadpool(
        _submit_task, db_session, data
    )
    if run_uid:
        # monitor in the background
        asyncio.create_task(
            run_in_threadpool(monitor_run, project, function_kind, run_uid)
        )
    return response


def monitor_run(project: str, function_kind: str, run_uid: str):
    """
    This function is running in background, i.e. outside of the context of a request
    therefore it should create its own db session
    """
    logger.info(
        "Starting to monitor run",
        project=project,
        function_kind=function_kind,
        run_uid=run_uid,
    )
    try:
        runtime_handler = get_runtime_handler(function_kind)
        db = get_db()
        db_session = create_session()
        runtime_handler.monitor_run(db, db_session, project, run_uid)
        close_session(db_session)
    except Exception as exc:
        logger.warning("Run monitoring failed", exc=str(exc))
    else:
        logger.info(
            "Run monitoring finished successfully",
            project=project,
            function_kind=function_kind,
            run_uid=run_uid,
        )


def _submit_task(db_session: Session, data) -> typing.Tuple[str, str, str, typing.Dict]:
    """
    :return: Tuple with:
        1. str of the project of the run
        2. str of the kind of the function of the run
        3. str of the uid of the run that started execution (None when it was scheduled)
        4. dict of the response info
    """
    run_uid = None
    project = None
    try:
        fn, task = _parse_submit_run_body(db_session, data)
        run_db = get_run_db_instance(db_session)
        fn.set_db_connection(run_db, True)
        logger.info("Submitting task", function=fn.to_dict(), task=task)
        # fn.spec.rundb = "http://mlrun-api:8080"
        schedule = data.get("schedule")
        if schedule:
            cron_trigger = schedule
            if isinstance(cron_trigger, dict):
                cron_trigger = schemas.ScheduleCronTrigger(**cron_trigger)
            get_scheduler().create_schedule(
                db_session,
                task["metadata"]["project"],
                task["metadata"]["name"],
                schemas.ScheduleKinds.job,
                data,
                cron_trigger,
            )
            project = task["metadata"]["project"]

            response = {
                "schedule": schedule,
                "project": task["metadata"]["project"],
                "name": task["metadata"]["name"],
            }
        else:
            run = fn.run(task, watch=False)
            run_uid = run.metadata.uid
            project = run.metadata.project
            if run:
                response = run.to_dict()

    except HTTPException:
        logger.error(traceback.format_exc())
        raise
    except mlrun.errors.MLRunHTTPStatusError:
        raise
    except Exception as err:
        logger.error(traceback.format_exc())
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value, reason="runtime error: {}".format(err)
        )

    logger.info("Submit task succeeded, returning response", response=response)
    return project, fn.kind, run_uid, {"data": response}
