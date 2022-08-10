import time

import requests
import requests.adapters
import urllib3.util.retry

from ..config import config
from . import logger

HTTP_RETRY_COUNT = 3
HTTP_RETRY_BACKOFF_FACTOR = 1

# make sure to only add exceptions that are raised early in the request. For example, ConnectionError can be raised
# during the handling of a request, and therefore should not be retried, as the request might not be idempotent.
HTTP_RETRYABLE_EXCEPTIONS = {
    # ConnectionResetError is raised when the server closes the connection prematurely during TCP handshake.
    ConnectionResetError: ["Connection reset by peer", "Connection aborted"],
    # "Connection aborted" and "Connection refused" happen when the server doesn't respond at all.
    ConnectionError: ["Connection aborted", "Connection refused"],
    ConnectionRefusedError: ["Connection refused"],
}

http_adapter = requests.adapters.HTTPAdapter(
    max_retries=urllib3.util.retry.Retry(
        total=HTTP_RETRY_COUNT,
        backoff_factor=HTTP_RETRY_BACKOFF_FACTOR,
        status_forcelist=[500, 502, 503, 504],
        # we want to retry but not to raise since we do want that last response (to parse details on the
        # error from response body) we'll handle raising ourselves
        raise_on_status=False,
    ),
    pool_maxsize=int(config.httpdb.max_workers),
)


class SessionWithRetry(requests.Session):
    def __init__(self):
        super().__init__()
        self.mount("http://", http_adapter)
        self.mount("https://", http_adapter)

    def request(self, method, url, **kwargs):
        max_retries = (
            HTTP_RETRY_COUNT
            if config.httpdb.retry_api_call_on_exception == "enabled"
            else 0
        )
        retry_count = 0
        while True:
            try:
                response = super().request(method, url, **kwargs)
                return response
            except tuple(HTTP_RETRYABLE_EXCEPTIONS.keys()) as exc:
                if retry_count >= max_retries:
                    logger.warning(
                        f"Maximum retries exhausted for {method} {url} request",
                        exception_type=type(exc),
                        exception_message=str(exc),
                        retry_interval=HTTP_RETRY_BACKOFF_FACTOR,
                        retry_count=retry_count,
                        max_retries=max_retries,
                    )
                    raise exc

                # only retry on exceptions with the right message
                exception_is_retryable = any(
                    [msg in str(exc) for msg in HTTP_RETRYABLE_EXCEPTIONS[type(exc)]]
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
                    f"retrying in {HTTP_RETRY_BACKOFF_FACTOR} seconds",
                    exception_type=type(exc),
                    exception_message=str(exc),
                    retry_interval=HTTP_RETRY_BACKOFF_FACTOR,
                    retry_count=retry_count,
                    max_retries=max_retries,
                )
                retry_count += 1
                time.sleep(HTTP_RETRY_BACKOFF_FACTOR)
