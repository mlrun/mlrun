from abc import ABC, abstractmethod

import fastapi
from dependency_injector import containers, providers


class Service(ABC):
    def __init__(self):
        self.SERVICE_PREFIX = "alerts"
        self.BASE_VERSIONED_SERVICE_PREFIX = "alerts/v1"
        self.app: fastapi.FastAPI = None
        self._logger = ...  # Hierarchical logger with context?
        self._route_mappers = {}

    def initialize(self):
        self._register_routes()
        self._initialize_app()
        # self._add_middlewares()
        # self._add_exception_handlers()
        # self._start_periodic_functions()  # chief clusterization and co.

    async def handle_request(
        self,
        path,
        method,
        request,
        *args,
        **kwargs,
    ):
        callback = self._route_mappers[method].match(path)
        return await callback(
            request,
            *args,
            **kwargs,
        )

    @abstractmethod
    def _register_routes(self):
        # Routes are in the form of /<service>/?
        pass

    @abstractmethod
    def _initialize_app(self):
        # Initializes fastAPI app
        pass

    # AsyncClientWithRetry for services-communication
    def clients(self):
        return [...]


class ServiceContainer(containers.DeclarativeContainer):
    service = providers.Dependency(instance_of=Service)
