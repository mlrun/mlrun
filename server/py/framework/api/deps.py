from dependency_injector import containers, providers
from services.api.api import deps


class DepsContainer(containers.DeclarativeContainer):
    db_session = providers.Callable(deps.get_db_session)
    authenticate_request = providers.Callable(deps.authenticate_request)
