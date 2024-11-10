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
import traceback
from abc import ABC, abstractmethod

import fastapi
import fastapi.concurrency
import fastapi.exception_handlers

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils
import mlrun.utils.version
from mlrun import mlconf

import framework.middlewares
import framework.utils.periodic


class Service(ABC):
    def __init__(self):
        # TODO: make the prefixes and service name configurable
        service_name = "api"
        self.SERVICE_PREFIX = f"/{service_name}"
        self.BASE_VERSIONED_SERVICE_PREFIX = f"{self.SERVICE_PREFIX}/v1"
        self.V2_SERVICE_PREFIX = f"{self.SERVICE_PREFIX}/v2"
        self.app: fastapi.FastAPI = None
        self._logger = mlrun.utils.logger.get_child(service_name)

    def initialize(self):
        self._initialize_app()
        self._register_routes()
        self._add_middlewares()
        self._add_exception_handlers()

    @abstractmethod
    async def move_service_to_online(self):
        pass

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
            openapi_url=f"{self.SERVICE_PREFIX}/openapi.json",
            docs_url=f"{self.SERVICE_PREFIX}/docs",
            redoc_url=f"{self.SERVICE_PREFIX}/redoc",
            default_response_class=fastapi.responses.ORJSONResponse,
            lifespan=self.lifespan,
        )

    # https://fastapi.tiangolo.com/advanced/events/

    @contextlib.asynccontextmanager
    async def lifespan(self, app_: fastapi.FastAPI):
        await self._setup_service()

        # Let the service run
        yield

        await self._teardown_service()

    async def _setup_service(self):
        self._logger.info(
            "On startup event handler called",
            config=mlconf.dump_yaml(),
            version=mlrun.utils.version.Version().get(),
        )
        loop = asyncio.get_running_loop()
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(
                max_workers=int(mlconf.httpdb.max_workers)
            )
        )

        await self._custom_setup_service()

        if mlconf.httpdb.state == mlrun.common.schemas.APIStates.online:
            await self.move_service_to_online()

    async def _custom_setup_service(self):
        pass

    async def _teardown_service(self):
        await self._custom_teardown_service()
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

    def _initialize_data(self):
        pass

    async def _start_periodic_functions(self):
        pass


class Daemon(ABC):
    def __init__(self, service_cls: Service.__class__):
        self._service = service_cls()

    def initialize(self):
        self._service.initialize()

    @property
    def app(self):
        return self._service.app

    @property
    def service(self) -> Service:
        return self._service
