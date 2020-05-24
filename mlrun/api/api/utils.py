import traceback
from http import HTTPStatus
from os import environ
from pathlib import Path

from fastapi import HTTPException
from fastapi import Request
from sqlalchemy.orm import Session

from mlrun.api.db.sqldb.db import SQLDB
from mlrun.api.singletons import get_db, get_logs_dir, get_scheduler
from mlrun.config import config
from mlrun.db.sqldb import SQLDB as SQLRunDB
from mlrun.run import import_function, new_function
from mlrun.utils import get_in, logger, parse_function_uri


def log_and_raise(status=HTTPStatus.BAD_REQUEST, **kw):
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
    elif config.httpdb.data_volume and \
            path.startswith(config.httpdb.data_volume):
        if config.httpdb.real_path:
            path = config.httpdb.real_path + \
                   path[len(config.httpdb.data_volume) - 1:]
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


def submit(db_session: Session, data):
    task = data.get("task")
    function = data.get("function")
    url = data.get("functionUrl")
    if not url and task:
        url = get_in(task, "spec.function")
    if not (function or url) or not task:
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="bad JSON, need to include function/url and task objects")

    # TODO: block exec for function["kind"] in ["", "local]  (must be a
    # remote/container runtime)

    resp = None
    try:
        if function and not url:
            fn = new_function(runtime=function)
        else:
            if "://" in url:
                fn = import_function(url=url)
            else:
                project, name, tag, hash_key = parse_function_uri(url)
                runtime = get_db().get_function(db_session, name, project, tag, hash_key)
                if not runtime:
                    log_and_raise(HTTPStatus.BAD_REQUEST, reason="runtime error: function {} not found".format(url))
                fn = new_function(runtime=runtime)

            if function:
                fn2 = new_function(runtime=function)
                for attr in ["volumes", "volume_mounts", "env", "resources",
                             "image_pull_policy", "replicas"]:
                    val = getattr(fn2.spec, attr, None)
                    if val:
                        setattr(fn.spec, attr, val)

        run_db = get_run_db_instance(db_session)
        fn.set_db_connection(run_db, True)
        logger.info("func:\n{}".format(fn.to_yaml()))
        # fn.spec.rundb = "http://mlrun-api:8080"
        schedule = data.get("schedule")
        if schedule:
            args = (task,)
            job_id = get_scheduler().add(schedule, fn, args)
            get_db().store_schedule(db_session, data)
            resp = {"schedule": schedule, "id": job_id}
        else:
            resp = fn.run(task, watch=False)

        logger.info("resp: %s", resp.to_yaml())
    except Exception as err:
        logger.error(traceback.format_exc())
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="runtime error: {}".format(err))

    if not isinstance(resp, dict):
        resp = resp.to_dict()
    return {
        "data": resp,
    }
