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

import time

import requests
import requests.adapters
import requests.utils
import urllib3.exceptions
import urllib3.util.retry

from ..config import config
from ..errors import err_to_str
from . import logger


class DummyCookieJar(requests.cookies.RequestsCookieJar):
    """
    Cookie jar that doesn't store any cookies.

    This prevents identity leakage by ensuring cookies from authentication services
    are not stored or sent in subsequent requests. Note that this does NOT affect
    reading incoming cookies from request headers - you can still access request.cookies
    or request.headers.get('Cookie') to read cookies sent TO your service.
    """

    def set_cookie(self, cookie, *args, **kwargs):
        """Override to prevent storing cookies"""
        pass

    def __setitem__(self, name, value):
        """Override to prevent storing cookies"""
        pass


class HTTPSessionWithRetry(requests.Session):
    """
    Extend requests.Session to add retry logic on both error statuses and certain exceptions.
    """

    # make sure to only add exceptions that are raised early in the request. For example, ConnectionError can be raised
    # during the handling of a request, and therefore should not be retried, as the request might not be idempotent.

    # use strings because some requests exceptions are encapsulated in other exceptions, and we want to catch them all.
    HTTP_RETRYABLE_EXCEPTION_STRINGS = [
        # "Connection reset by peer" is raised when the server closes the connection prematurely during TCP handshake.
        "Connection reset by peer",
        # "Connection aborted" and "Connection refused" happen when the server doesn't respond at all.
        "Connection aborted",
        "Connection refused",
    ]

    # most of the exceptions would not be encapsulated, we want to catch them all directly.
    # this allows us more flexibility when deciding which *exactly* exception we want to retry on.
    HTTP_RETRYABLE_EXCEPTIONS = [
        # "Connection reset by peer" is raised when the server closes the connection prematurely during TCP handshake.
        ConnectionResetError,
        # "Connection aborted" and "Connection refused" happen when the server doesn't respond at all.
        ConnectionRefusedError,
        ConnectionAbortedError,
        ConnectionError,
        # often happens when the server is overloaded and can't handle the load.
        requests.exceptions.ConnectionError,
        requests.exceptions.ConnectTimeout,
        requests.exceptions.ReadTimeout,
        urllib3.exceptions.ReadTimeoutError,
        # may occur when connection breaks during the request.
        requests.exceptions.ChunkedEncodingError,
        urllib3.exceptions.InvalidChunkLength,
    ]

    def __init__(
        self,
        max_retries=config.http_retry_defaults.max_retries,
        retry_backoff_factor=config.http_retry_defaults.backoff_factor,
        retry_on_exception=True,
        retry_on_status=True,
        retry_on_post=False,
        retry_on_put=True,
        verbose=False,
    ):
        """
        Initialize a new HTTP session with retry logic.
        :param max_retries:             Maximum number of retries to attempt.
        :param retry_backoff_factor:    Wait interval retries in seconds.
        :param retry_on_exception:      Retry on the HTTP_RETRYABLE_EXCEPTIONS. defaults to True.
        :param retry_on_status:         Retry on error status codes. defaults to True.
        :param retry_on_post:           Retry on POST requests. defaults to False.
        :param retry_on_put:            Whether to allow retries on PUT requests. Actual behavior may exclude specific
                                        paths from retrying. defaults to True.
        :param verbose:                 Print debug messages.
        """
        super().__init__()

        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_on_exception = retry_on_exception
        self.verbose = verbose
        self._logger = logger.get_child("http-client")
        self._retry_methods = self._resolve_retry_methods(retry_on_post, retry_on_put)

        # Disable cookie storage to prevent identity leakage
        self.cookies = DummyCookieJar()

        if retry_on_status:
            self._http_adapter = requests.adapters.HTTPAdapter(
                max_retries=urllib3.util.retry.Retry(
                    total=self.max_retries,
                    backoff_factor=self.retry_backoff_factor,
                    status_forcelist=config.http_retry_defaults.status_codes,
                    allowed_methods=self._retry_methods,
                    # we want to retry but not to raise since we do want that last response (to parse details on the
                    # error from response body) we'll handle raising ourselves
                    raise_on_status=False,
                ),
                pool_maxsize=int(config.httpdb.max_workers),
            )

            self.mount("http://", self._http_adapter)
            self.mount("https://", self._http_adapter)

    # Attributes set in __init__ that requests.Session.__attrs__ does not cover and
    # that must survive copy/pickle. Extends the base allowlist rather than replacing
    # it. _logger is intentionally absent: it is a named child logger, rebuilt in
    # __setstate__ instead of serialized.
    _EXTRA_PICKLE_ATTRS = (
        "max_retries",
        "retry_backoff_factor",
        "retry_on_exception",
        "verbose",
        "_retry_methods",
        "_http_adapter",
    )

    def __getstate__(self) -> dict:
        """Return picklable state, extending ``requests.Session``'s allowlist.

        ``requests.Session.__getstate__`` only serializes ``__attrs__`` (headers,
        cookies, adapters, ...), which drops the retry-related attributes this
        subclass adds in ``__init__``. A copied or pickled session would then raise
        ``AttributeError`` the moment :meth:`update_retry_methods` reads
        ``_retry_methods``. We add :attr:`_EXTRA_PICKLE_ATTRS` on top of the base
        state, copying each only if present - ``_http_adapter`` is set only when
        ``retry_on_status`` is enabled, and copying by presence (rather than the
        base ``getattr(..., None)``) keeps it absent otherwise so the
        ``hasattr`` guard in :meth:`update_retry_methods` stays correct.
        ``_http_adapter`` is the same object as the mounted ``adapters`` entries;
        ``requests.adapters.HTTPAdapter`` rebuilds a fresh connection pool on
        restore, so the copy gets its own pool instead of sharing sockets.

        :return: The instance state to pickle/copy.
        """
        state = super().__getstate__()
        state.update(
            (attr, self.__dict__[attr])
            for attr in self._EXTRA_PICKLE_ATTRS
            if attr in self.__dict__
        )
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore instance state and rebuild the non-serialized logger.

        :param state: The state produced by :meth:`__getstate__`.
        """
        super().__setstate__(state)
        self._logger = logger.get_child("http-client")

    def request(self, method, url, **kwargs):
        retry_count = 0
        kwargs.setdefault("headers", {})
        kwargs["headers"]["User-Agent"] = (
            f"{requests.utils.default_user_agent()} mlrun/{config.version}"
        )
        while True:
            try:
                response = super().request(method, url, **kwargs)
                return response
            except Exception as exc:
                if not self._error_is_retryable(url, method, exc, retry_count):
                    raise exc

                self._logger.warning(
                    "Error during request handling, retrying",
                    exc=err_to_str(exc),
                    retry_count=retry_count,
                    url=url,
                    method=method,
                )
                if self.verbose:
                    self._log_exception(
                        "debug",
                        exc,
                        f"{method} {url} request failed on retryable exception, "
                        f"retrying in {self.retry_backoff_factor} seconds",
                        retry_count,
                    )
                retry_count += 1
                time.sleep(self.retry_backoff_factor)

    def _error_is_retryable(self, url, method, exc, retry_count):
        if not self.retry_on_exception:
            self._log_exception(
                "warning",
                exc,
                f"{method} {url} request failed, http retries disabled,"
                f" raising exception: {err_to_str(exc)}",
                retry_count,
            )
            return False

        # if the response is not retryable, stop retrying
        # this is done to prevent the retry logic from running on non-idempotent methods (such as POST).
        if not self._method_retryable(method):
            self._log_exception(
                "warning",
                exc,
                f"{method} {url} request failed, http retries disabled for {method} method.",
                retry_count,
            )
            return False

        if retry_count >= self.max_retries:
            self._log_exception(
                "warning",
                exc,
                f"{method} {url} request failed, max retries reached,"
                f" raising exception: {err_to_str(exc)}",
                retry_count,
            )
            return False

        def exception_is_retryable(exc, retryable_exceptions):
            def err_chain(err):
                while err:
                    yield err
                    err = err.__cause__

            return any(
                isinstance(err_in_chain, retryable_exc)
                for retryable_exc in retryable_exceptions
                for err_in_chain in err_chain(exc)
            )

        # only retryable exceptions
        exception_is_retryable = exception_is_retryable(
            exc, self.HTTP_RETRYABLE_EXCEPTIONS
        )

        if not exception_is_retryable:
            self._log_exception(
                "warning",
                exc,
                f"{method} {url} request failed on non-retryable exception,"
                f" raising exception: {err_to_str(exc)}",
                retry_count,
            )
            return False
        return True

    def update_retry_methods(self, retry_on_post: bool, retry_on_put: bool) -> None:
        """Update the retry method set on the session and its mounted adapter.

        This allows reusing a single session across requests with different
        retry policies (e.g., POST paths that are retriable vs. non-retriable),
        avoiding the overhead of creating a new session per request.

        :param retry_on_post: Whether POST requests should be retried.
        :param retry_on_put:  Whether PUT requests should be retried.
        """
        new_methods = self._resolve_retry_methods(retry_on_post, retry_on_put)

        # Skip Retry object re-allocation when the allowed methods haven't changed
        # consecutive calls with the same policy (common case) don't need a new Retry object
        if new_methods == self._retry_methods:
            return

        self._retry_methods = new_methods
        if hasattr(self, "_http_adapter"):
            self._http_adapter.max_retries = urllib3.util.retry.Retry(
                total=self.max_retries,
                backoff_factor=self.retry_backoff_factor,
                status_forcelist=config.http_retry_defaults.status_codes,
                allowed_methods=self._retry_methods,
                raise_on_status=False,
            )

    def _method_retryable(self, method: str):
        return method in self._retry_methods

    def _resolve_retry_methods(
        self, retry_on_post: bool = False, retry_on_put: bool = True
    ) -> frozenset[str]:
        methods = urllib3.util.retry.Retry.DEFAULT_ALLOWED_METHODS
        methods = methods.union({"PATCH"})
        if not retry_on_put:
            methods = methods.difference({"PUT"})
        if retry_on_post:
            methods = methods.union({"POST"})
        return frozenset(methods)

    def _log_exception(self, level, exc, message, retry_count):
        getattr(self._logger, level)(
            message,
            exception_type=type(exc),
            exception_message=err_to_str(exc),
            retry_interval=self.retry_backoff_factor,
            retry_count=retry_count,
            max_retries=self.max_retries,
        )
