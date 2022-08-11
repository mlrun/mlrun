import time

import requests
import requests.adapters
import urllib3.util.retry

from ..config import config
from . import logger

DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_BACKOFF = 1


class HTTPSessionWithRetry(requests.Session):
    """
    Extend requests.Session to add retry logic on both error statuses and certain exceptions.
    """

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
                    status_forcelist=[500, 502, 503, 504],
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

                if self.verbose:
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

    @staticmethod
    def _get_retry_methods(retry_on_post=False):
        return (
            # setting to False in order to retry on all methods, otherwise every method except POST.
            False
            if retry_on_post
            else urllib3.util.retry.Retry.DEFAULT_METHOD_WHITELIST
        )
