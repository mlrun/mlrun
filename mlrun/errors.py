import requests
from fastapi import status


class MLRunSDKError(Exception):
    pass


class MLRunHTTPError(MLRunSDKError, requests.HTTPError):
    def __init__(
        self, message: str, response: requests.Response = None, status_code: int = None
    ):

        # because response object is probably with an error, it returns False, so we
        # should use 'is None' specifically
        if response is None:
            response = requests.Response()
        if status_code:
            response.status_code = status_code

        requests.HTTPError.__init__(self, message, response=response)


class MLRunDataStoreError(MLRunHTTPError):
    error_status_code = None

    def __init__(self, message: str, response: requests.Response = None):
        super(MLRunDataStoreError, self).__init__(
            message, response=response, status_code=self.error_status_code
        )


class MLRunDBError(MLRunHTTPError):
    pass


def raise_for_status(response: requests.Response):
    """
    Raise a specific MLRunSDK error depending on the given response status code.
    If no specific error exists, raises an MLRunHTTPError
    """
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        if response.status_code in STATUS_ERRORS:
            raise STATUS_ERRORS[response.status_code](str(e), response=response) from e
        raise MLRunHTTPError(str(e), response=response)


# Specific Errors


class UnauthorizedError(MLRunDataStoreError):
    error_status_code = status.HTTP_401_UNAUTHORIZED


class AccessForbiddenError(MLRunDataStoreError):
    error_status_code = status.HTTP_403_FORBIDDEN


class NotFoundError(MLRunDataStoreError):
    error_status_code = status.HTTP_404_NOT_FOUND


STATUS_ERRORS = {
    status.HTTP_401_UNAUTHORIZED: UnauthorizedError,
    status.HTTP_403_FORBIDDEN: AccessForbiddenError,
    status.HTTP_404_NOT_FOUND: NotFoundError,
}
