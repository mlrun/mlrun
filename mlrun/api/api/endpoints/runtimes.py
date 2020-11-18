from http import HTTPStatus

from fastapi import APIRouter, Depends
from fastapi import Response
from sqlalchemy.orm import Session

from mlrun.api.api import deps
from mlrun.api.api.utils import log_and_raise
from mlrun.api.utils.singletons.db import get_db
from mlrun.config import config
from mlrun.runtimes import RuntimeKinds
from mlrun.runtimes import get_runtime_handler

router = APIRouter()


@router.get("/runtimes")
def list_runtimes(label_selector: str = None):
    runtimes = []
    for kind in RuntimeKinds.runtime_with_handlers():
        runtime_handler = get_runtime_handler(kind)
        resources = runtime_handler.list_resources(label_selector)
        runtimes.append({"kind": kind, "resources": resources})
    return runtimes


@router.get("/runtimes/{kind}")
def get_runtime(kind: str, label_selector: str = None):
    if kind not in RuntimeKinds.runtime_with_handlers():
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value, kind=kind, err="Invalid runtime kind"
        )
    runtime_handler = get_runtime_handler(kind)
    resources = runtime_handler.list_resources(label_selector)
    return {
        "kind": kind,
        "resources": resources,
    }


@router.delete("/runtimes", status_code=HTTPStatus.NO_CONTENT.value)
def delete_runtimes(
    label_selector: str = None,
    force: bool = False,
    grace_period: int = config.runtime_resources_deletion_grace_period,
    db_session: Session = Depends(deps.get_db_session),
):
    for kind in RuntimeKinds.runtime_with_handlers():
        runtime_handler = get_runtime_handler(kind)
        runtime_handler.delete_resources(
            get_db(), db_session, label_selector, force, grace_period
        )
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.delete("/runtimes/{kind}", status_code=HTTPStatus.NO_CONTENT.value)
def delete_runtime(
    kind: str,
    label_selector: str = None,
    force: bool = False,
    grace_period: int = config.runtime_resources_deletion_grace_period,
    db_session: Session = Depends(deps.get_db_session),
):
    if kind not in RuntimeKinds.runtime_with_handlers():
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value, kind=kind, err="Invalid runtime kind"
        )
    runtime_handler = get_runtime_handler(kind)
    runtime_handler.delete_resources(
        get_db(), db_session, label_selector, force, grace_period
    )
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


# FIXME: find a more REST-y path
@router.delete("/runtimes/{kind}/{object_id}", status_code=HTTPStatus.NO_CONTENT.value)
def delete_runtime_object(
    kind: str,
    object_id: str,
    label_selector: str = None,
    force: bool = False,
    grace_period: int = config.runtime_resources_deletion_grace_period,
    db_session: Session = Depends(deps.get_db_session),
):
    if kind not in RuntimeKinds.runtime_with_handlers():
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value, kind=kind, err="Invalid runtime kind"
        )
    runtime_handler = get_runtime_handler(kind)
    runtime_handler.delete_runtime_object_resources(
        get_db(), db_session, object_id, label_selector, force, grace_period
    )
    return Response(status_code=HTTPStatus.NO_CONTENT.value)
