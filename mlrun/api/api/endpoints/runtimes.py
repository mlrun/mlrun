from fastapi import APIRouter, status, Response

from mlrun.api.api.utils import log_and_raise
from mlrun.runtimes import RuntimeKinds
from mlrun.runtimes import get_runtime_handler

router = APIRouter()


@router.get("/runtimes")
def list_runtimes(
        label_selector: str = None):
    runtimes = []
    for kind in RuntimeKinds.runtime_with_handlers():
        runtime_handler = get_runtime_handler(kind)
        resources = runtime_handler.list_resources(label_selector)
        runtimes.append({
            'kind': kind,
            'resources': resources,
        })
    return runtimes


@router.get("/runtimes/{kind}")
def get_runtime(
        kind: str,
        label_selector: str = None):
    if kind not in RuntimeKinds.runtime_with_handlers():
        log_and_raise(status.HTTP_400_BAD_REQUEST, kind=kind, err='Invalid runtime kind')
    runtime_handler = get_runtime_handler(kind)
    resources = runtime_handler.list_resources(label_selector)
    return {
        'kind': kind,
        'resources': resources,
    }


@router.delete("/runtimes")
def delete_runtimes(
        label_selector: str = None,
        force: bool = False):
    for kind in RuntimeKinds.runtime_with_handlers():
        runtime_handler = get_runtime_handler(kind)
        runtime_handler.delete_resources(label_selector, force)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/runtimes/{kind}")
def delete_runtime(
        kind: str,
        label_selector: str = None,
        force: bool = False):
    if kind not in RuntimeKinds.runtime_with_handlers():
        log_and_raise(status.HTTP_400_BAD_REQUEST, kind=kind, err='Invalid runtime kind')
    runtime_handler = get_runtime_handler(kind)
    runtime_handler.delete_resources(label_selector, force)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# FIXME: find a more REST-y path
@router.delete("/runtimes/{kind}/{object_id}")
def delete_runtime_object(
        kind: str,
        object_id: str,
        label_selector: str = None,
        force: bool = False):
    if kind not in RuntimeKinds.runtime_with_handlers():
        log_and_raise(status.HTTP_400_BAD_REQUEST, kind=kind, err='Invalid runtime kind')
    runtime_handler = get_runtime_handler(kind)
    runtime_handler.delete_runtime_object_resources(object_id, label_selector, force)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
