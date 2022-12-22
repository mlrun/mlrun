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
import time

import requests
import requests.adapters
import urllib3.util.retry

from ..config import config
from ..errors import err_to_str
from . import logger


class HTTPSessionWithRetry(requests.Session):
    """
    Extend requests.Session to add retry logic on both error statuses and certain exceptions.
    """

    # make sure to only add exceptions that are raised early in the request. For example, ConnectionError can be raised
    # during the handling of a request, and therefore should not be retried, as the request might not be idempotent.

    HTTP_RETRYABLE_EXCEPTION_STRINGS = [
        # "Connection reset by peer" is raised when the server closes the connection prematurely during TCP handshake.
        "Connection reset by peer",
        # "Connection aborted" and "Connection refused" happen when the server doesn't respond at all.
        "Connection aborted",
        "Connection refused",
    ]

    def __init__(
        self,
        max_retries=config.http_retry_defaults.max_retries,
        retry_backoff_factor=config.http_retry_defaults.backoff_factor,
        retry_on_exception=True,
        retry_on_status=True,
        retry_on_post=False,
        verbose=False,
    ):
        """
        Initialize a new HTTP session with retry logic.
        :param max_retries: Maximum number of retries to attempt.
        :param retry_backoff_factor: Wait interval retries in seconds.
        :param retry_on_exception: Retry on the HTTP_RETRYABLE_EXCEPTIONS. defaults to True.
        :param retry_on_status: Retry on error status codes. defaults to True.
        :param retry_on_post: Retry on POST requests. defaults to False.
        :param verbose: Print debug messages.
        """
        super().__init__()

        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_on_exception = retry_on_exception
        self.verbose = verbose

        if retry_on_status:
            http_adapter = requests.adapters.HTTPAdapter(
                max_retries=urllib3.util.retry.Retry(
                    total=self.max_retries,
                    backoff_factor=self.retry_backoff_factor,
                    status_forcelist=config.http_retry_defaults.status_codes,
                    method_whitelist=self._get_retry_methods(retry_on_post),
                    # we want to retry but not to raise since we do want that last response (to parse details on the
                    # error from response body) we'll handle raising ourselves
                    raise_on_status=False,
                ),
                pool_maxsize=int(config.httpdb.max_workers),
            )

            self.mount("http://", http_adapter)
            self.mount("https://", http_adapter)

    def request(self, method, url, **kwargs):
        retry_count = 0
        while True:
            try:
                response = super().request(method, url, **kwargs)
                return response
            except Exception as exc:
                if not self.retry_on_exception:
                    self._log_exception(
                        "warning",
                        exc,
                        f"{method} {url} request failed, http retries disabled,"
                        f" raising exception: {err_to_str(exc)}",
                        retry_count,
                    )
                    raise exc

                if retry_count >= self.max_retries:
                    self._log_exception(
                        "warning",
                        exc,
                        f"{method} {url} request failed, max retries reached,"
                        f" raising exception: {err_to_str(exc)}",
                        retry_count,
                    )
                    raise exc

                # only retry on exceptions with the right message
                exception_is_retryable = any(
                    msg in str(exc) for msg in self.HTTP_RETRYABLE_EXCEPTION_STRINGS
                )

                if not exception_is_retryable:
                    self._log_exception(
                        "warning",
                        exc,
                        f"{method} {url} request failed on non-retryable exception,"
                        f" raising exception: {err_to_str(exc)}",
                        retry_count,
                    )
                    raise exc

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

    @staticmethod
    def _get_retry_methods(retry_on_post=False):
        return (
            # setting to False in order to retry on all methods, otherwise every method except POST.
            False
            if retry_on_post
            else urllib3.util.retry.Retry.DEFAULT_METHOD_WHITELIST
        )

    def _log_exception(self, level, exc, message, retry_count):
        getattr(logger, level)(
            message,
            exception_type=type(exc),
            exception_message=err_to_str(exc),
            retry_interval=self.retry_backoff_factor,
            retry_count=retry_count,
            max_retries=self.max_retries,
        )
