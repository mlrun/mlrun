from distutils.util import strtobool

from fastapi import APIRouter

from mlrun.run import list_piplines

router = APIRouter()


# curl http://localhost:8080/workflows?full=no
@router.get("/workflows")
def list_workflows(
        experiment_id: str = None,
        namespace: str = None,
        sort_by: str = "",
        page_token: str = "",
        full: str = "0",
        page_size: int = 10):
    full = strtobool(full)
    total_size, next_page_token, runs = list_piplines(
        full=full, page_token=page_token, page_size=page_size,
        sort_by=sort_by, experiment_id=experiment_id, namespace=namespace
    )
    return {
        "runs": runs,
        "total_size": total_size,
        "next_page_token": next_page_token,
    }
