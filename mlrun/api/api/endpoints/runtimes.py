from fastapi import APIRouter, status

from mlrun.runtimes import get_runtime_handler_class

router = APIRouter()


@router.get("/runtimes/{kind}")
def get_runtime(
        kind: str,
        namespace: str = None,
        label_selector: str = None):
    runtime_handler = get_runtime_handler_class(kind)
    resources = runtime_handler.list_resources(namespace, label_selector)
    return {
        'resources': resources
    }


@router.delete("/runtimes/{kind}", status_code=status.HTTP_204_NO_CONTENT)
def delete_runtime(
        kind: str,
        namespace: str = None,
        label_selector: str = None,
        running: bool = False):
    runtime_handler = get_runtime_handler_class(kind)
    runtime_handler.delete_resources(namespace, label_selector, running)
