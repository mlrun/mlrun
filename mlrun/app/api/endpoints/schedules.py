from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from mlrun.app.api import deps
from mlrun.app.db.session import get_db_instance

router = APIRouter()


# curl http://localhost:8080/schedules
@router.get("/schedules")
def list_schedules(
        db_session: Session = Depends(deps.get_db_session)):
    schedules = get_db_instance().list_schedules(db_session)
    return {
        "schedules": list(schedules),
    }
