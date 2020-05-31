from fastapi import APIRouter, status, Response

from mlrun.runtimes import get_runtime_handler

router = APIRouter()


@router.get("/runtimes/{kind}")
def get_runtime(
        kind: str,
        namespace: str = None,
        label_selector: str = None):
    runtime_handler = get_runtime_handler(kind)
    resources = runtime_handler.list_resources(namespace, label_selector)
    return {
        'resources': resources
    }


@router.delete("/runtimes/{kind}")
def delete_runtime(
        kind: str,
        namespace: str = None,
        label_selector: str = None,
        running: bool = False):
    runtime_handler = get_runtime_handler(kind)
    runtime_handler.delete_resources(namespace, label_selector, running)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
