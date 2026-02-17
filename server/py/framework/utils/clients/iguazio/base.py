# Copyright 2023 Iguazio
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
import contextlib
import copy
import typing
from abc import ABC, abstractmethod

import aiohttp
import fastapi
from fastapi.concurrency import run_in_threadpool

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.singleton
import mlrun.utils.thread
from mlrun.utils import logger


class BaseClient(ABC, metaclass=mlrun.utils.singleton.AbstractSingleton):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._logger = logger.get_child("iguazio-client")

    @property
    def is_sync(self) -> bool:
        """Indicates whether the client is synchronous."""
        return True

    @property
    def _api_url(self) -> str:
        return mlrun.mlconf.iguazio_api_url

    @property
    def _session_verification_endpoint(self) -> str:
        return mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint

    @property
    @abstractmethod
    def _verify_session_http_method(self) -> str:
        pass

    @abstractmethod
    def _generate_auth_info_from_session_verification_response(
        self,
        response_headers: typing.Mapping[str, typing.Any],
        response_body: typing.Mapping[typing.Any, typing.Any],
    ) -> mlrun.common.schemas.AuthInfo:
        """
        Extract and return AuthInfo from a valid session verification response.
        """
        pass

    @abstractmethod
    def _prepare_request_kwargs(
        self, session: typing.Optional[str], path: str, *, kwargs: dict
    ):
        pass

    def _send_request_to_api(
        self,
        method: str,
        path: str,
        error_message: str,
        session=None,
        retry_on_post=False,
        **kwargs,
    ):
        url = f"{self._api_url}/api/{path}"
        self._prepare_request_kwargs(session, path, kwargs=kwargs)
        http_session = self._session
        if retry_on_post and self._retry_on_post_session:
            http_session = self._retry_on_post_session
        response = http_session.request(
            method, url, verify=mlrun.mlconf.iguazio_api_ssl_verify, **kwargs
        )
        if not response.ok:
            try:
                response_body = response.json()
            except Exception:
                response_body = {}
            self._handle_error_response(
                method, path, response, response_body, error_message, kwargs
            )
        return response

    def _handle_error_response(
        self,
        method: str,
        path: str,
        response: typing.Any,
        response_body: dict,
        error_message: str,
        kwargs: dict,
    ) -> None:
        log_kwargs = copy.deepcopy(kwargs)

        # this can be big and spammy
        log_kwargs.pop("json", None)

        log_kwargs.update({"method": method, "path": path})

        ctx = self._extract_ctx(response_body)
        extracted_error = self._extract_error_message(response_body)

        if extracted_error:
            error_message = f"{error_message}: {extracted_error}"
        if extracted_error or ctx:
            log_kwargs.update({"ctx": ctx, "error": extracted_error})

        self._logger.warning("Request to Iguazio failed", **log_kwargs)
        mlrun.errors.raise_for_status(response, error_message)

    @abstractmethod
    def _extract_ctx(self, response_body: dict) -> typing.Optional[str]:
        pass

    @abstractmethod
    def _extract_error_message(self, response_body: dict) -> typing.Optional[str]:
        pass


class BaseAsyncClient(BaseClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._run_in_threadpool_callback = run_in_threadpool
        self._async_sessions = mlrun.utils.thread.ThreadLocalClient(
            factory=self._get_new_async_session,
            close_callback=lambda async_session: async_session.close(),
        )

    @property
    def is_sync(self) -> bool:
        """
        False because client is asynchronous
        """
        return False

    def __getattribute__(self, name):
        """
        This method is called when trying to access an attribute of the class.
        We override it to make sure that all *public* methods that are not async will be run in a thread pool.
          by convention/norm - public methods are methods that don't start with an underscore.
          If the method name starts with an underscore - it's a private method that was called from a public method,
          which means that it's already running in a thread pool or runs asynchronously.
        If the method is async, we don't do anything and let the async machinery handle it.

        """
        attr = super().__getattribute__(name)
        if name.startswith("_") or not callable(attr):
            return attr

        # already a coroutine
        if asyncio.iscoroutinefunction(attr):
            return attr

        # not a coroutine, run in threadpool
        def wrapper(*args, **kwargs):
            return self._run_in_threadpool_callback(attr, *args, **kwargs)

        return wrapper

    async def verify_request_session(
        self, request: fastapi.Request
    ) -> mlrun.common.schemas.AuthInfo:
        """
        Proxy the request to one of the session verification endpoints (which will verify the session of the request)
        """
        # TODO: Instead of calling the session verification endpoint for IG4, we can use the Iguazio SDK's "self"
        #  method to retrieve user info and generate the AuthInfo from it. When calling it, we need to wrap it in an
        #  async client and call the method as async. As a result, this method will not be part of the base client
        #  and should be implemented separately in the Iguaziov4 client.
        headers = {
            mlrun.common.schemas.HeaderNames.authorization: request.headers.get(
                mlrun.common.schemas.HeaderNames.authorization
            ),
            mlrun.common.schemas.HeaderNames.cookie: request.headers.get(
                mlrun.common.schemas.HeaderNames.cookie, ""
            ),
            mlrun.common.schemas.HeaderNames.x_request_id: getattr(
                request.state, "request_id", ""
            ),
            mlrun.common.schemas.HeaderNames.igz_authenticator_kind: request.headers.get(
                mlrun.common.schemas.HeaderNames.igz_authenticator_kind, ""
            ),
        }
        async with (
            self._send_request_to_api_async(
                self._verify_session_http_method,
                self._session_verification_endpoint,
                "Failed verifying iguazio session",
                retry_options_override=mlrun.utils.async_http.ExponentialRetryOverride(
                    blacklisted_methods=[],  # iguazio session verification endpoint is idempotent
                    # 1, 2, 4, 8, ...
                    start_timeout=1,
                    max_timeout=30.0,
                    factor=2.0,
                ),
                headers=headers,
            ) as response
        ):
            return self._generate_auth_info_from_session_verification_response(
                response.headers, await response.json()
            )

    @contextlib.asynccontextmanager
    async def _send_request_to_api_async(
        self,
        method: str,
        path: str,
        error_message: str,
        session: typing.Optional[str] = None,
        retry_options_override: typing.Optional[
            mlrun.utils.async_http.ExponentialRetryOverride
        ] = None,
        **kwargs,
    ) -> typing.AsyncGenerator[aiohttp.ClientResponse, None]:
        url = f"{self._api_url}/api/{path}"
        self._prepare_request_kwargs(session, path, kwargs=kwargs)
        async_session = self._async_sessions.get()
        # take the session default
        retry_options = copy.deepcopy(async_session.retry_options)

        # override with cherry-picked options
        if retry_options_override:
            if retry_options_override.blacklisted_methods is not None:
                retry_options.blacklisted_methods = (
                    retry_options_override.blacklisted_methods
                )
            retry_options._start_timeout = retry_options_override._start_timeout
            retry_options._max_timeout = retry_options_override._max_timeout
            retry_options._factor = retry_options_override._factor

        response = None
        try:
            response = await async_session.request(
                method,
                url,
                verify_ssl=mlrun.mlconf.iguazio_api_ssl_verify,
                retry_options=retry_options,
                **kwargs,
            )
            if not response.ok:
                try:
                    response_body = await response.json()
                except Exception:
                    response_body = {}
                self._handle_error_response(
                    method, path, response, response_body, error_message, kwargs
                )
            yield response

        except mlrun.errors.MLRunUnauthorizedError as exc:
            self._logger.error(
                f"{error_message}: Unauthorized request to {url}",
                exc_info=mlrun.errors.err_to_str(exc),
            )
            raise

        except Exception as exc:
            self._logger.error(
                f"{error_message}: Failed to send request to API",
                exc_info=mlrun.errors.err_to_str(exc),
            )
            raise

        finally:
            if response:
                response.release()

    @staticmethod
    def _get_new_async_session():
        return mlrun.utils.AsyncClientWithRetry(
            retry_on_exception=mlrun.mlconf.httpdb.projects.retry_leader_request_on_exception
            == mlrun.common.schemas.HTTPSessionRetryMode.enabled.value,
            logger=logger,
        )
