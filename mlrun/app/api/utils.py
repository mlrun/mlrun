import traceback
from http import HTTPStatus
from os import environ
from pathlib import Path

from fastapi import HTTPException

from mlrun.app.main import db
from mlrun.app.main import logs_dir
from mlrun.app.main import scheduler
from mlrun.config import config
from mlrun.run import import_function, new_function
from mlrun.utils import get_in, logger, parse_function_uri


def json_error(status=HTTPStatus.BAD_REQUEST, **kw):
    logger.error(str(kw))
    raise HTTPException(status_code=status, detail=kw)


def log_path(project, uid) -> Path:
    return logs_dir / project / uid


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


def get_secrets(_request):
    access_key = _request.headers.get("X-V3io-Session-Key")
    return {
        "V3IO_ACCESS_KEY": access_key,
    }


def submit(db_session, data):
    task = data.get("task")
    function = data.get("function")
    url = data.get("functionUrl")
    if not url and task:
        url = get_in(task, "spec.function")
    if not (function or url) or not task:
        return json_error(
            HTTPStatus.BAD_REQUEST,
            reason="bad JSON, need to include function/url and task objects",
        )

    # TODO: block exec for function["kind"] in ["", "local]  (must be a
    # remote/container runtime)

    try:
        if function and not url:
            fn = new_function(runtime=function)
        else:
            if "://" in url:
                fn = import_function(url=url)
            else:
                project, name, tag = parse_function_uri(url)
                runtime = db.get_function(db_session, name, project, tag)
                if not runtime:
                    return json_error(
                        HTTPStatus.BAD_REQUEST,
                        reason="runtime error: function {} not found".format(
                            url),
                    )
                fn = new_function(runtime=runtime)

            if function:
                fn2 = new_function(runtime=function)
                for attr in ["volumes", "volume_mounts", "env", "resources",
                             "image_pull_policy", "replicas"]:
                    val = getattr(fn2.spec, attr, None)
                    if val:
                        setattr(fn.spec, attr, val)

        # FIXME: discuss with yaronh
        fn.set_db_connection(_db, True)
        logger.info("func:\n{}".format(fn.to_yaml()))
        # fn.spec.rundb = "http://mlrun-api:8080"
        schedule = data.get("schedule")
        if schedule:
            args = (task,)
            job_id = scheduler.add(schedule, fn, args)
            db.store_schedule(db_session, data)
            resp = {"schedule": schedule, "id": job_id}
        else:
            resp = fn.run(task, watch=False)

        logger.info("resp: %s", resp.to_yaml())
    except Exception as err:
        logger.error(traceback.format_exc())
        return json_error(
            HTTPStatus.BAD_REQUEST,
            reason="runtime error: {}".format(err),
        )

    if not isinstance(resp, dict):
        resp = resp.to_dict()
    return {
        "data": resp,
    }
