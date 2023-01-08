# Copyright 2018 Iguazio
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
#

import asyncio
import http
import typing
from typing import List, Optional

import aiohttp
import aiohttp.http_exceptions
from aiohttp_retry import ExponentialRetry, RequestParams, RetryClient, RetryOptionsBase
from aiohttp_retry.client import _RequestContext

from ..config import config
from ..errors import err_to_str
from .helpers import logger as mlrun_logger


class AsyncClientWithRetry(RetryClient):
    """
    Extends by adding retry logic on both error statuses and certain exceptions.
    """

    def __init__(
        self,
        max_retries=config.http_retry_defaults.max_retries,
        retry_backoff_factor=config.http_retry_defaults.backoff_factor,
        retry_on_exception=True,
        retry_on_status=True,
        blacklisted_methods=None,
        logger=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            retry_options=ExponentialRetryOverride(
                retry_on_exception=retry_on_exception,
                # do not retry on PUT / PATCH as they might have side effects (not truly idempotent)
                blacklisted_methods=blacklisted_methods or ["POST", "PUT", "PATCH"],
                attempts=max_retries,
                statuses={
                    http.HTTPStatus.BAD_GATEWAY.value,
                    http.HTTPStatus.SERVICE_UNAVAILABLE.value,
                    http.HTTPStatus.GATEWAY_TIMEOUT.value,
                    http.HTTPStatus.TOO_MANY_REQUESTS,
                },
                # a callback that will run on response to decide if retry is needed
                factor=retry_backoff_factor,
            ),
            logger=logger or mlrun_logger,
            raise_for_status=retry_on_status,
            **kwargs,
        )

    def _make_requests(
        self,
        params_list: List[RequestParams],
        retry_options: Optional[RetryOptionsBase] = None,
        raise_for_status: Optional[bool] = None,
    ) -> "_CustomRequestContext":
        if retry_options is None:
            retry_options = self._retry_options
        if raise_for_status is None:
            raise_for_status = self._raise_for_status
        return _CustomRequestContext(
            request_func=self._client.request,
            params_list=params_list,
            logger=self._logger,
            retry_options=retry_options,
            raise_for_status=raise_for_status,
        )


class ExponentialRetryOverride(ExponentialRetry):
    # make sure to only add exceptions that are raised early in the request. For example, ConnectionError can be raised
    # during the handling of a request, and therefore should not be retried, as the request might not be idempotent.
    HTTP_RETRYABLE_EXCEPTION = [
        # "Connection reset by peer" is raised when the server closes the connection prematurely during TCP handshake.
        ConnectionResetError,
        # "Connection aborted" and "Connection refused" happen when the server doesn't respond at all.
        ConnectionRefusedError,
        ConnectionAbortedError,
        # aiohttp exceptions that can be raised during connection establishment
        aiohttp.ClientConnectionError,
        aiohttp.ServerDisconnectedError,
    ]

    def __init__(self, retry_on_exception, blacklisted_methods, *args, **kwargs):

        # whether to retry on exceptions
        self.retry_on_exception = retry_on_exception

        # methods that should not be retried
        self.blacklisted_methods = blacklisted_methods

        # default exceptions that should be retried on (when retry_on_exception is True)
        if "exceptions" not in kwargs:
            kwargs["exceptions"] = self.HTTP_RETRYABLE_EXCEPTION
        super().__init__(*args, **kwargs)


class _CustomRequestContext(_RequestContext):
    """
    Extends by adding retry logic on both error statuses and certain exceptions.
    """

    async def _do_request(self) -> aiohttp.ClientResponse:
        current_attempt = 0
        while True:
            current_attempt += 1
            response = None
            params = None
            try:
                try:
                    params = self._params_list[current_attempt - 1]
                except IndexError:
                    params = self._params_list[-1]

                response: typing.Optional[
                    aiohttp.ClientResponse
                ] = await self._request_func(
                    params.method,
                    params.url,
                    headers=params.headers,
                    trace_request_ctx={
                        "current_attempt": current_attempt,
                        **(params.trace_request_ctx or {}),
                    },
                    **(params.kwargs or {}),
                )
                retry_wait = self._retry_options.get_timeout(
                    attempt=current_attempt, response=response
                )

                # if the response is not retryable, return it.
                # this is done to prevent the retry logic from running on non-idempotent methods such as POST.
                if not self._is_method_retryable(params.method):
                    self._response = response
                    return response

                # check if status code is retryable (given by the retry options) or last attempt
                if (
                    self._is_status_code_ok(response.status)
                    or current_attempt == self._retry_options.attempts
                ):
                    if self._raise_for_status:
                        response.raise_for_status()

                    # allow user to provide a callback to decide if retry is needed
                    is_response_correct = await self._check_response_callback(
                        params, current_attempt, retry_wait, response
                    )

                    # response is correct or last attempt, return it
                    if (
                        is_response_correct
                        or current_attempt == self._retry_options.attempts
                    ):
                        self._response = response
                        return response
                else:
                    self._logger.debug(
                        f"Request failed on retryable http code, retrying",
                        retry_wait_secs=retry_wait,
                        method=params.method,
                        url=params.url,
                        current_attempt=current_attempt,
                        max_attemps=self._retry_options.attempts,
                        status_code=response.status,
                    )

            except Exception as exc:
                if not self._retry_options.retry_on_exception:
                    self._logger.warning(
                        f"Request failed, retry on exception is disabled",
                        method=params.method,
                        url=params.url,
                        exc=err_to_str(exc),
                    )
                    raise exc

                # exhaust all attempts, stop here, return the last response
                if current_attempt >= self._retry_options.attempts:
                    if response:
                        self._response = response
                        return response
                    raise exc

                # by type
                self.verify_exception_type(exc)

                # if the response is not retryable, return now.
                # this is done to prevent the retry logic from running on non-idempotent methods such as POST.
                if not self._is_method_retryable(params.method):
                    if response:
                        self._response = response
                        return response
                    raise exc

                retry_wait = self._retry_options.get_timeout(
                    attempt=current_attempt, response=None
                )
                self._logger.debug(
                    f"Request failed on retryable exception, retrying",
                    retry_wait_secs=retry_wait,
                    method=params.method,
                    url=params.url,
                    current_attempt=current_attempt,
                    max_attemps=self._retry_options.attempts,
                    exc=err_to_str(exc),
                )

            await asyncio.sleep(retry_wait)

    def _is_method_retryable(self, method: str):
        return method not in self._retry_options.blacklisted_methods

    async def _check_response_callback(
        self, params, retry_count, retry_wait_secs, response
    ):
        if self._retry_options.evaluate_response_callback is not None:
            try:
                result = self._retry_options.evaluate_response_callback(response)
                if asyncio.iscoroutinefunction(
                    self._retry_options.evaluate_response_callback
                ):
                    return await result
                else:
                    return result
            except Exception as exc:
                self._logger.warning(
                    f"Request failed on evaluating response",
                    retry_wait_secs=retry_wait_secs,
                    method=params.method,
                    url=params.url,
                    retry_count=retry_count,
                    max_attemps=self._retry_options.attempts,
                    status_code=response.status,
                    exc=err_to_str(exc),
                )
                return False
        return True

    def verify_exception_type(self, exc):
        for exc_type in self._retry_options.exceptions:
            if isinstance(exc, exc_type):
                return
            if isinstance(exc, aiohttp.ClientConnectorError):
                if isinstance(exc.os_error, exc_type):
                    return
        raise exc
