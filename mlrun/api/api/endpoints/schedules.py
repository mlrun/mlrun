from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from mlrun.api.api import deps
from mlrun.api.singletons import get_db

router = APIRouter()


# curl http://localhost:8080/schedules
@router.get("/schedules")
def list_schedules(
        db_session: Session = Depends(deps.get_db_session)):
    schedules = get_db().list_schedules(db_session)
    return {
        "schedules": list(schedules),
    }
