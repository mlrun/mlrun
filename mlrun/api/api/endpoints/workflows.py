from fastapi import APIRouter

from mlrun.k8s_utils import get_k8s_helper
from mlrun.run import list_pipelines

router = APIRouter()


# curl http://localhost:8080/workflows?full=no
@router.get("/workflows")
def list_workflows(
    experiment_id: str = None,
    namespace: str = None,
    sort_by: str = "",
    page_token: str = "",
    full: bool = False,
    page_size: int = 10,
):
    total_size, next_page_token, runs = None, None, None
    if get_k8s_helper(silent=True).is_running_inside_kubernetes_cluster():
        total_size, next_page_token, runs = list_pipelines(
            full=full,
            page_token=page_token,
            page_size=page_size,
            sort_by=sort_by,
            experiment_id=experiment_id,
            namespace=namespace,
        )
    return {
        "runs": runs or [],
        "total_size": total_size or 0,
        "next_page_token": next_page_token or None,
    }
