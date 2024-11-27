# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import concurrent.futures
import contextlib
import http
import traceback
import typing
from abc import ABC, abstractmethod

import fastapi
import fastapi.concurrency
import fastapi.exception_handlers
from dependency_injector import containers, providers

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils
import mlrun.utils.version
from mlrun import mlconf

import framework.api.utils
import framework.middlewares
import framework.utils.clients.chief
import framework.utils.clients.discovery
import framework.utils.periodic
from framework.utils.singletons.db import initialize_db


class Service(ABC):
    def __init__(self):
        self.service_name = mlconf.services.service_name
        self.service_prefix = f"/{self.service_name}"
        self.base_versioned_service_prefix = f"{self.service_prefix}/v1"
        self.v2_service_prefix = f"{self.service_prefix}/v2"
        self.app: fastapi.FastAPI = None
        self._logger = mlrun.utils.logger.get_child(self.service_name)
        self._mounted_services: list[Service] = []

    def initialize(self, mounts: typing.Optional[list] = None):
        self._logger.info("Initializing service", service_name=self.service_name)
        self._initialize_app()
        self._register_routes()
        self._mount_services(mounts)
        self._add_middlewares()
        self._add_exception_handlers()

    async def move_service_to_online(self):
        self._logger.info("Moving service to online", service_name=self.service_name)
        await self._move_service_to_online()

    # https://fastapi.tiangolo.com/advanced/events/
    @contextlib.asynccontextmanager
    async def lifespan(self, app_: fastapi.FastAPI):
        setup_tasks = [self._setup_service()] + [
            service._setup_service(mounted=True) for service in self._mounted_services
        ]
        await asyncio.gather(*setup_tasks)

        # Let the service run
        yield

        teardown_tasks = [self._teardown_service()] + [
            service._teardown_service(mounted=True)
            for service in self._mounted_services
        ]
        await asyncio.gather(*teardown_tasks)

    async def handle_request(
        self,
        path,
        request: fastapi.Request,
        *args,
        **kwargs,
    ):
        callback = getattr(self, path, None)
        if callback is None:
            return await self._base_handler(request, *args, **kwargs)
        return await callback(
            request,
            *args,
            **kwargs,
        )

    @abstractmethod
    async def _move_service_to_online(self):
        pass

    def _mount_services(self, mounts: typing.Optional[list] = None):
        if not mounts:
            return

        self._mounted_services = mounts
        for service in self._mounted_services:
            service.initialize()
            self.app.mount("/", service.app)

    async def _move_mounted_services_to_online(self):
        if not self._mounted_services:
            return

        tasks = [service.move_service_to_online() for service in self._mounted_services]
        await asyncio.gather(*tasks)

    @abstractmethod
    def _register_routes(self):
        pass

    def _initialize_app(self):
        # Initializes fastAPI app - each service register the routers they implement
        # API gateway registers all routers, alerts service registers alert router
        self.app = fastapi.FastAPI(
            title="MLRun",  # TODO: configure
            description="Machine Learning automation and tracking",  # TODO: configure
            version=mlconf.version,
            debug=mlconf.httpdb.debug,
            openapi_url=f"{self.service_prefix}/openapi.json",
            docs_url=f"{self.service_prefix}/docs",
            redoc_url=f"{self.service_prefix}/redoc",
            default_response_class=fastapi.responses.ORJSONResponse,
            lifespan=self.lifespan,
        )

    async def _setup_service(self, mounted: bool = False):
        """
        This method is called when the service is starting up.
        :param mounted: True if the service is mounted as a sub-service of another service, False otherwise
        """
        if not mounted:
            self._logger.info(
                "On startup event handler called",
                config=mlconf.dump_yaml(),
                version=mlrun.utils.version.Version().get(),
                service_name=self.service_name,
            )
            loop = asyncio.get_running_loop()
            loop.set_default_executor(
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=int(mlconf.httpdb.max_workers)
                )
            )

            initialize_db()
        await self._custom_setup_service()

        self._initialize_data()
        if (
            mlconf.httpdb.clusterization.worker.sync_with_chief.mode
            == mlrun.common.schemas.WaitForChiefToReachOnlineStateFeatureFlag.enabled
            and mlconf.httpdb.clusterization.role
            == mlrun.common.schemas.ClusterizationRole.worker
            and not mounted
        ):
            # in the background, wait for chief to reach online state
            self._start_chief_clusterization_spec_sync_loop()

        if mlconf.httpdb.state == mlrun.common.schemas.APIStates.online and not mounted:
            await self.move_service_to_online()

    async def _custom_setup_service(self):
        pass

    async def _teardown_service(self, mounted: bool = False):
        await self._custom_teardown_service()
        if not mounted:
            framework.utils.periodic.cancel_all_periodic_functions()

    async def _custom_teardown_service(self):
        pass

    def _add_middlewares(self):
        # middlewares, order matter
        self.app.add_middleware(
            framework.middlewares.EnsureBackendVersionMiddleware,
            backend_version=mlconf.version,
        )
        self.app.add_middleware(
            framework.middlewares.UiClearCacheMiddleware, backend_version=mlconf.version
        )
        self.app.add_middleware(
            framework.middlewares.RequestLoggerMiddleware, logger=self._logger
        )

    def _add_exception_handlers(self):
        self.app.add_exception_handler(Exception, self._generic_error_handler)
        self.app.add_exception_handler(
            mlrun.errors.MLRunHTTPStatusError, self._http_status_error_handler
        )

    async def _generic_error_handler(self, request: fastapi.Request, exc: Exception):
        error_message = repr(exc)
        return await fastapi.exception_handlers.http_exception_handler(
            # we have no specific knowledge on what was the exception and what status code fits so we simply use 500
            # This handler is mainly to put the error message in the right place in the body so the client will be able
            # to show it
            request,
            fastapi.HTTPException(status_code=500, detail=error_message),
        )

    async def _http_status_error_handler(
        self, request: fastapi.Request, exc: mlrun.errors.MLRunHTTPStatusError
    ):
        request_id = None

        # request might not have request id when the error is raised before the request id is set on middleware
        if hasattr(request.state, "request_id"):
            request_id = request.state.request_id
        status_code = exc.response.status_code
        error_message = repr(exc)
        log_message = "Request handling returned error status"

        if isinstance(exc, mlrun.errors.EXPECTED_ERRORS):
            self._logger.debug(
                log_message,
                error_message=error_message,
                status_code=status_code,
                request_id=request_id,
            )
        else:
            self._logger.warning(
                log_message,
                error_message=error_message,
                status_code=status_code,
                traceback=traceback.format_exc(),
                request_id=request_id,
            )

        return await fastapi.exception_handlers.http_exception_handler(
            request,
            fastapi.HTTPException(status_code=status_code, detail=error_message),
        )

    async def _base_handler(
        self,
        request: fastapi.Request,
        *args,
        **kwargs,
    ):
        framework.api.utils.log_and_raise(
            http.HTTPStatus.NOT_IMPLEMENTED.value,
            reason="Handler not implemented for request",
            request_url=request.url,
        )

    def _initialize_data(self):
        pass

    async def _start_periodic_functions(self):
        pass

    def _start_chief_clusterization_spec_sync_loop(self):
        # put it here first, because we need to set it before the periodic function starts
        # so the worker will be aligned with the chief state
        mlconf.httpdb.state = mlrun.common.schemas.APIStates.waiting_for_chief

        interval = int(mlconf.httpdb.clusterization.worker.sync_with_chief.interval)
        if interval > 0:
            self._logger.info(
                "Starting chief clusterization spec sync loop", interval=interval
            )
            framework.utils.periodic.run_function_periodically(
                interval,
                self._synchronize_with_chief_clusterization_spec.__name__,
                False,
                self._synchronize_with_chief_clusterization_spec,
            )

    async def _synchronize_with_chief_clusterization_spec(
        self,
    ):
        # sanity
        # if we are still in the periodic function and the worker has reached the terminal state, then cancel it
        if mlconf.httpdb.state in mlrun.common.schemas.APIStates.terminal_states():
            self._logger.debug(
                "Worker reached terminal state, canceling periodic function",
                state=mlconf.httpdb.state,
            )
            framework.utils.periodic.cancel_periodic_function(
                self._synchronize_with_chief_clusterization_spec.__name__
            )

        try:
            chief_client = framework.utils.clients.chief.Client()
            clusterization_spec = await chief_client.get_clusterization_spec(
                return_fastapi_response=False, raise_on_failure=True
            )
        except Exception as exc:
            self._logger.debug(
                "Failed receiving clusterization spec",
                exc=mlrun.errors.err_to_str(exc),
                traceback=traceback.format_exc(),
            )
        else:
            await self._align_worker_state_with_chief_state(clusterization_spec)

    async def _align_worker_state_with_chief_state(
        self,
        clusterization_spec: mlrun.common.schemas.ClusterizationSpec,
    ):
        chief_state = clusterization_spec.chief_api_state
        if not chief_state:
            self._logger.warning("Chief did not return any state")
            return

        if chief_state not in mlrun.common.schemas.APIStates.terminal_states():
            self._logger.debug(
                "Chief did not reach online state yet, will retry after sync interval",
                interval=mlconf.httpdb.clusterization.worker.sync_with_chief.interval,
                chief_state=chief_state,
            )
            # we want the worker to be aligned with chief state
            mlconf.httpdb.state = chief_state
            return

        if chief_state == mlrun.common.schemas.APIStates.online:
            self._logger.info(
                "Chief reached online state",
                service_name=self.service_name,
            )
            await self.move_service_to_online()
            self._logger.info(
                "Service state reached online", service_name=self.service_name
            )

        else:
            self._logger.info(
                "Chief state is terminal, canceling periodic chief clusterization spec pulling",
                state=mlconf.httpdb.state,
                service_name=self.service_name,
            )

        mlconf.httpdb.state = chief_state
        self._logger.info("Worker state aligned with chief state", state=chief_state)
        # if reached terminal state we cancel the periodic function
        # assumption: we can't get out of a terminal api state, so no need to continue pulling when reached one
        framework.utils.periodic.cancel_periodic_function(
            self._synchronize_with_chief_clusterization_spec.__name__
        )


class Daemon(ABC):
    def __init__(self, service_cls: Service.__class__):
        self._service: Service = service_cls()

    def initialize(self):
        self._service.initialize(self.mounts)

    @staticmethod
    def wire():
        # Wire the service container to inject the providers to the routers
        container = framework.service.ServiceContainer()
        container.wire()

    @property
    def mounts(self) -> list[Service]:
        return []

    @property
    def app(self) -> fastapi.FastAPI:
        return self._service.app

    @property
    def service(self) -> Service:
        return self._service


class ServiceContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=["framework.routers"])
    service = providers.Object(None)
