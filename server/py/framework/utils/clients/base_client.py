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
import enum
import typing
import urllib.parse
from abc import ABC, abstractmethod

import aiohttp
import fastapi
from fastapi.concurrency import run_in_threadpool

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.singleton
from mlrun.utils import logger


class BaseClient(ABC, metaclass=mlrun.utils.singleton.AbstractSingleton):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._api_url = mlrun.mlconf.iguazio_api_url

    @property
    def is_sync(self) -> bool:
        """Indicates whether the client is synchronous."""
        return True

    @abstractmethod
    def _generate_auth_info_from_session_verification_response(
        self,
        response_headers: typing.Mapping[str, typing.Any],
        response_body: typing.Mapping[typing.Any, typing.Any],
    ) -> mlrun.common.schemas.AuthInfo:
        pass

    def _prepare_request_kwargs(self, session, path, *, kwargs):
        # support session being already a cookie
        session_cookie = session
        if (
            session_cookie
            and not session_cookie.startswith('j:{"sid"')
            and not session_cookie.startswith(urllib.parse.quote_plus('j:{"sid"'))
        ):
            session_cookie = f'j:{{"sid": "{session_cookie}"}}'
        if session_cookie:
            cookies = kwargs.get("cookies", {})
            # in case some dev using this function for some reason setting cookies manually through kwargs + have a
            # cookie with "session" key there + filling the session cookie - explode
            if "session" in cookies and cookies["session"] != session_cookie:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Session cookie already set"
                )
            cookies["session"] = session_cookie
            kwargs["cookies"] = cookies
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = 20
        if "projects" in path:
            if mlrun.common.schemas.HeaderNames.projects_role not in kwargs.get(
                "headers", {}
            ):
                kwargs.setdefault("headers", {})[
                    mlrun.common.schemas.HeaderNames.projects_role
                ] = "mlrun"

        # requests no longer supports header values to be enum (https://github.com/psf/requests/pull/6154)
        # convert to strings. Do the same for params for niceness
        for kwarg in ["headers", "params"]:
            dict_ = kwargs.get(kwarg, {})
            for key in dict_.keys():
                if isinstance(dict_[key], enum.Enum):
                    dict_[key] = dict_[key].value

    def _handle_error_response(
        self, method, path, response, response_body, error_message, kwargs
    ):
        log_kwargs = copy.deepcopy(kwargs)

        # this can be big and spammy
        log_kwargs.pop("json", None)
        log_kwargs.update({"method": method, "path": path})
        try:
            ctx = response_body.get("meta", {}).get("ctx")
            errors = response_body.get("errors", [])
        except Exception:
            pass
        else:
            if errors:
                error_message = f"{error_message}: {str(errors)}"
            if errors or ctx:
                log_kwargs.update({"ctx": ctx, "errors": errors})

        self._logger.warning("Request to iguazio failed", **log_kwargs)
        mlrun.errors.raise_for_status(response, error_message)


class BaseAsyncClient(BaseClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._run_in_threadpool_callback = run_in_threadpool
        self._async_session: typing.Optional[mlrun.utils.AsyncClientWithRetry] = None

    @property
    def is_sync(self):
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

    @property
    @abstractmethod
    def _verify_session_http_method(self) -> str:
        pass

    @property
    def _session_verification_endpoint(self) -> str:
        return mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint

    async def verify_request_session(
        self, request: fastapi.Request
    ) -> mlrun.common.schemas.AuthInfo:
        """
        Proxy the request to one of the session verification endpoints (which will verify the session of the request)
        """
        headers = {
            "authorization": request.headers.get("authorization"),
            "cookie": request.headers.get("cookie"),
            "x-request-id": request.state.request_id,
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
        method,
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
        await self._ensure_async_session()

        # take the session default
        retry_options = copy.deepcopy(self._async_session.retry_options)

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
            response = await self._async_session.request(
                method, url, verify_ssl=False, retry_options=retry_options, **kwargs
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
        finally:
            if response:
                response.release()

    async def _ensure_async_session(self):
        if not self._async_session:
            self._async_session = mlrun.utils.AsyncClientWithRetry(
                retry_on_exception=mlrun.mlconf.httpdb.projects.retry_leader_request_on_exception
                == mlrun.common.schemas.HTTPSessionRetryMode.enabled.value,
                logger=logger,
            )
