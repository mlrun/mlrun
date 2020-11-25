from http import HTTPStatus

import requests


class MLRunBaseError(Exception):
    """
    A base class from which all other exceptions inherit.
    If you want to catch all errors that the MLRun SDK might raise,
    catch this base exception.
    """

    pass


class MLRunHTTPError(MLRunBaseError, requests.HTTPError):
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


class MLRunHTTPStatusError(MLRunHTTPError):
    """
    When an error has a matching http status code it is "HTTP statusable"
    HTTP Status errors should inherit from this class and set the right status code in the
    error_status_code attribute
    """

    error_status_code = None

    def __init__(self, message: str, response: requests.Response = None):
        super(MLRunHTTPStatusError, self).__init__(
            message, response=response, status_code=self.error_status_code
        )


def raise_for_status(response: requests.Response):
    """
    Raise a specific MLRunSDK error depending on the given response status code.
    If no specific error exists, raises an MLRunHTTPError
    """
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            raise STATUS_ERRORS[response.status_code](
                str(exc), response=response
            ) from exc
        except KeyError:
            raise MLRunHTTPError(str(exc), response=response) from exc


# Specific Errors
class MLRunUnauthorizedError(MLRunHTTPStatusError):
    error_status_code = HTTPStatus.UNAUTHORIZED.value


class MLRunAccessDeniedError(MLRunHTTPStatusError):
    error_status_code = HTTPStatus.FORBIDDEN.value


class MLRunNotFoundError(MLRunHTTPStatusError):
    error_status_code = HTTPStatus.NOT_FOUND.value


class MLRunBadRequestError(MLRunHTTPStatusError):
    error_status_code = HTTPStatus.BAD_REQUEST.value


class MLRunInvalidArgumentError(MLRunHTTPStatusError, ValueError):
    error_status_code = HTTPStatus.BAD_REQUEST.value


class MLRunIncompatibleVersionError(MLRunHTTPStatusError):
    error_status_code = HTTPStatus.BAD_REQUEST.value


STATUS_ERRORS = {
    HTTPStatus.BAD_REQUEST.value: MLRunBadRequestError,
    HTTPStatus.UNAUTHORIZED.value: MLRunUnauthorizedError,
    HTTPStatus.FORBIDDEN.value: MLRunAccessDeniedError,
    HTTPStatus.NOT_FOUND.value: MLRunNotFoundError,
}
