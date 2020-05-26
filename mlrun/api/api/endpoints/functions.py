import traceback
from distutils.util import strtobool
from http import HTTPStatus
from typing import List

from fastapi import APIRouter, Depends, Request, Query, Response
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from mlrun.api.api import deps
from mlrun.api.api.utils import log_and_raise, get_run_db_instance
from mlrun.api.singletons import get_db, get_k8s
from mlrun.builder import build_runtime
from mlrun.config import config
from mlrun.run import new_function
from mlrun.runtimes import runtime_resources_map
from mlrun.utils import get_in, logger, parse_function_uri, update_in

router = APIRouter()


# curl -d@/path/to/func.json http://localhost:8080/func/prj/7?tag=0.3.2
@router.post("/func/{project}/{name}")
async def store_function(
        request: Request,
        project: str,
        name: str,
        tag: str = "",
        versioned: bool = False,
        db_session: Session = Depends(deps.get_db_session)):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="bad JSON body")

    logger.debug(data)
    logger.info(
        "store function: project=%s, name=%s, tag=%s", project, name, tag)
    await run_in_threadpool(get_db().store_function, db_session, data, name, project, tag=tag, versioned=versioned)
    return {}


# curl http://localhost:8080/log/prj/7?tag=0.2.3
@router.get("/func/{project}/{name}")
def get_function(
        project: str,
        name: str,
        tag: str = "",
        hash_key="",
        db_session: Session = Depends(deps.get_db_session)):
    func = get_db().get_function(db_session, name, project, tag, hash_key)
    return {
        "func": func,
    }


# curl http://localhost:8080/funcs?project=p1&name=x&label=l1&label=l2
@router.get("/funcs")
def list_functions(
        project: str = config.default_project,
        name: str = None,
        tag: str = None,
        labels: List[str] = Query([], alias='label'),
        db_session: Session = Depends(deps.get_db_session)):
    funcs = get_db().list_functions(db_session, name, project, tag, labels)
    return {
        "funcs": list(funcs),
    }


# curl -d@/path/to/job.json http://localhost:8080/build/function
@router.post("/build/function")
@router.post("/build/function/")
async def build_function(
        request: Request,
        db_session: Session = Depends(deps.get_db_session)):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="bad JSON body")

    logger.info("build_function:\n{}".format(data))
    function = data.get("function")
    with_mlrun = strtobool(data.get("with_mlrun", "on"))
    fn, ready = await run_in_threadpool(_build_function, db_session, function, with_mlrun)
    return {
        "data": fn.to_dict(),
        "ready": ready,
    }


# curl -d@/path/to/job.json http://localhost:8080/start/function
@router.post("/start/function")
@router.post("/start/function/")
async def start_function(
        request: Request,
        db_session: Session = Depends(deps.get_db_session)):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="bad JSON body")

    fn = await run_in_threadpool(_start_function, db_session, data)

    return {
        "data": fn.to_dict(),
    }


# curl -d@/path/to/job.json http://localhost:8080/status/function
@router.post("/status/function")
@router.post("/status/function/")
async def function_status(
        request: Request):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="bad JSON body")

    resp = await run_in_threadpool(_get_function_status, data)
    return {
        "data": resp,
    }


# curl -d@/path/to/job.json http://localhost:8080/build/status
@router.get("/build/status")
@router.get("/build/status/")
def build_status(
        name: str = "",
        project: str = "",
        tag: str = "",
        offset: int = 0,
        logs: str = "on",
        db_session: Session = Depends(deps.get_db_session)):
    logs = strtobool(logs)
    fn = get_db().get_function(db_session, name, project, tag)
    if not fn:
        log_and_raise(HTTPStatus.NOT_FOUND, name=name, project=project, tag=tag)

    state = get_in(fn, "status.state", "")
    pod = get_in(fn, "status.build_pod", "")
    image = get_in(fn, "spec.build.image", "")
    out = b""
    if not pod:
        if state == "ready":
            image = image or get_in(fn, "spec.image")
        return Response(content=out,
                        media_type="text/plain",
                        headers={"function_status": state,
                                 "function_image": image,
                                 "builder_pod": pod})

    logger.info("get pod {} status".format(pod))
    state = get_k8s().get_pod_status(pod)
    logger.info("pod state={}".format(state))

    if state == "succeeded":
        logger.info("build completed successfully")
        state = "ready"
    if state in ["failed", "error"]:
        logger.error("build {}, watch the build pod logs: {}".format(
            state, pod))

    if logs and state != "pending":
        resp = get_k8s().logs(pod)
        if resp:
            out = resp.encode()[offset:]

    update_in(fn, "status.state", state)
    if state == "ready":
        update_in(fn, "spec.image", image)

    get_db().store_function(db_session, fn, name, project, tag)

    return Response(content=out,
                    media_type="text/plain",
                    headers={"function_status": state,
                             "function_image": image,
                             "builder_pod": pod})


def _build_function(db_session, function, with_mlrun):
    fn = None
    ready = None
    try:
        fn = new_function(runtime=function)

        run_db = get_run_db_instance(db_session)
        fn.set_db_connection(run_db)
        fn.save(versioned=False)

        ready = build_runtime(fn, with_mlrun)
        fn.save(versioned=False)
        logger.info("Fn:\n %s", fn.to_yaml())
    except Exception as err:
        logger.error(traceback.format_exc())
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="runtime error: {}".format(err))
    return fn, ready


def _start_function(db_session, data):
    logger.info("start_function:\n{}".format(data))
    url = data.get("functionUrl")
    if not url:
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="runtime error: functionUrl not specified")

    project, name, tag, hash_key = parse_function_uri(url)
    runtime = get_db().get_function(db_session, name, project, tag, hash_key)
    if not runtime:
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="runtime error: function {} not found".format(url))

    fn = new_function(runtime=runtime)
    resource = runtime_resources_map.get(fn.kind)
    if "start" not in resource:
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="runtime error: 'start' not supported by this runtime")

    try:

        run_db = get_run_db_instance(db_session)
        fn.set_db_connection(run_db)
        #  resp = resource["start"](fn)  # TODO: handle resp?
        resource["start"](fn)
        fn.save(versioned=False)
        logger.info("Fn:\n %s", fn.to_yaml())
    except Exception as err:
        logger.error(traceback.format_exc())
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="runtime error: {}".format(err))

    return fn


def _get_function_status(data):
    logger.info("function_status:\n{}".format(data))
    selector = data.get("selector")
    kind = data.get("kind")
    if not selector or not kind:
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="runtime error: selector or runtime kind not specified")

    resource = runtime_resources_map.get(kind)
    if "status" not in resource:
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="runtime error: 'status' not supported by this runtime")

    resp = None
    try:
        resp = resource["status"](selector)
        logger.info("status: %s", resp)
    except Exception as err:
        logger.error(traceback.format_exc())
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="runtime error: {}".format(err))
