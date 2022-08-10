import time

import requests
import requests.adapters
import urllib3.util.retry

from ..config import config
from . import logger

DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_BACKOFF = 1


class HTTPSessionWithRetry(requests.Session):

    # make sure to only add exceptions that are raised early in the request. For example, ConnectionError can be raised
    # during the handling of a request, and therefore should not be retried, as the request might not be idempotent.
    HTTP_RETRYABLE_EXCEPTIONS = {
        # ConnectionResetError is raised when the server closes the connection prematurely during TCP handshake.
        ConnectionResetError: ["Connection reset by peer", "Connection aborted"],
        # "Connection aborted" and "Connection refused" happen when the server doesn't respond at all.
        ConnectionError: ["Connection aborted", "Connection refused"],
        ConnectionRefusedError: ["Connection refused"],
    }

    def __init__(
        self,
        max_retries=DEFAULT_RETRY_COUNT,
        retry_backoff_factor=DEFAULT_RETRY_BACKOFF,
        retry_on_exception=True,
        retry_on_status=True,
    ):
        super().__init__()

        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_on_exception = retry_on_exception

        if retry_on_status:
            http_adapter = requests.adapters.HTTPAdapter(
                max_retries=urllib3.util.retry.Retry(
                    total=self.max_retries,
                    backoff_factor=self.retry_backoff_factor,
                    status_forcelist=[500, 502, 503, 504],
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
            except tuple(self.HTTP_RETRYABLE_EXCEPTIONS.keys()) as exc:
                if not self.retry_on_exception:
                    logger.warning(
                        f"{method} {url} request failed, http retries disabled, raising exception: {exc}",
                        exception_type=type(exc),
                        exception_message=str(exc),
                        retry_interval=self.retry_backoff_factor,
                        retry_count=retry_count,
                        max_retries=self.max_retries,
                    )
                    raise exc

                if retry_count >= self.max_retries:
                    logger.warning(
                        f"Maximum retries exhausted for {method} {url} request",
                        exception_type=type(exc),
                        exception_message=str(exc),
                        retry_interval=self.retry_backoff_factor,
                        retry_count=retry_count,
                        max_retries=self.max_retries,
                    )
                    raise exc

                # only retry on exceptions with the right message
                exception_is_retryable = any(
                    [
                        msg in str(exc)
                        for msg in self.HTTP_RETRYABLE_EXCEPTIONS[type(exc)]
                    ]
                )

                if not exception_is_retryable:
                    logger.warning(
                        f"{method} {url} request failed on non-retryable exception",
                        exception_type=type(exc),
                        exception_message=str(exc),
                    )
                    raise exc

                logger.debug(
                    f"{method} {url} request failed on retryable exception, "
                    f"retrying in {self.retry_backoff_factor} seconds",
                    exception_type=type(exc),
                    exception_message=str(exc),
                    retry_interval=self.retry_backoff_factor,
                    retry_count=retry_count,
                    max_retries=self.max_retries,
                )
                retry_count += 1
                time.sleep(self.retry_backoff_factor)
