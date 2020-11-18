from mlrun.api.db.sqldb.session import create_session
from mlrun.api.utils.scheduler import Scheduler

# TODO: something nicer
scheduler: Scheduler = None


async def initialize_scheduler():
    global scheduler
    scheduler = Scheduler()
    db_session = None
    try:
        db_session = create_session()
        await scheduler.start(db_session)
    finally:
        db_session.close()


def get_scheduler() -> Scheduler:
    global scheduler
    return scheduler
