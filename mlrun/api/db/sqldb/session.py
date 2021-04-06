from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker as SessionMaker

from mlrun.config import config

engine: Engine = None
_session_maker: SessionMaker = None


# doing lazy load to allow tests to initialize the engine
def get_engine() -> Engine:
    global engine
    if engine is None:
        _init_engine()
    return engine


def create_session() -> Session:
    session_maker = _get_session_maker()
    return session_maker()


# doing lazy load to allow tests to initialize the engine
def _get_session_maker() -> SessionMaker:
    global _session_maker
    if _session_maker is None:
        _init_session_maker()
    return _session_maker


# TODO: we accept the dsn here to enable tests to override it, the "right" thing will be that config will be easily
#  overridable by tests (today when you import the config it is already being initialized.. should be lazy load)
def _init_engine(dsn=None):
    global engine
    dsn = dsn or config.httpdb.dsn
    engine = create_engine(dsn)
    _init_session_maker()


def _init_session_maker():
    global _session_maker
    _session_maker = SessionMaker(bind=get_engine())
