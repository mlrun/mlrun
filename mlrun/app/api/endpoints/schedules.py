from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from mlrun.app.api import deps
from mlrun.app.main import db

router = APIRouter()


# curl http://localhost:8080/schedules
@router.get("/schedules")
def list_schedules(
        db_session: Session = Depends(deps.get_db_session)):
    schedules = db.list_schedules(db_session)
    return {
        "schedules": list(schedules),
    }
