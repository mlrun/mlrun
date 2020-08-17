from http import HTTPStatus

import requests


class BaseError(Exception):
    """
    A base class from which all other exceptions inherit.
    If you want to catch all errors that the MLRun SDK might raise,
    catch this base exception.
    """

    pass


class HTTPError(BaseError, requests.HTTPError):
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


class HTTPStatusableError(HTTPError):
    """
    When an error has a matching http status code it is "HTTP statusable"
    HTTP Statusable errors should inherit from this class and set the right status code in the
    error_status_code attribute
    """

    error_status_code = None

    def __init__(self, message: str, response: requests.Response = None):
        super(HTTPStatusableError, self).__init__(
            message, response=response, status_code=self.error_status_code
        )


def raise_for_status(response: requests.Response):
    """
    Raise a specific MLRunSDK error depending on the given response status code.
    If no specific error exists, raises an HTTPError
    """
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            raise STATUS_ERRORS[response.status_code](
                str(exc), response=response
            ) from exc
        except KeyError:
            raise HTTPError(
                str(exc), response=response, status_code=response.status_code
            ) from exc


# Specific Errors
class UnauthorizedError(HTTPStatusableError):
    error_status_code = HTTPStatus.UNAUTHORIZED.value


class AccessDeniedError(HTTPStatusableError):
    error_status_code = HTTPStatus.FORBIDDEN.value


class NotFoundError(HTTPStatusableError):
    error_status_code = HTTPStatus.NOT_FOUND.value


class BadRequestError(HTTPStatusableError):
    error_status_code = HTTPStatus.BAD_REQUEST.value


class InvalidArgumentError(HTTPStatusableError, ValueError):
    error_status_code = HTTPStatus.BAD_REQUEST.value


STATUS_ERRORS = {
    HTTPStatus.BAD_REQUEST.value: BadRequestError,
    HTTPStatus.UNAUTHORIZED.value: UnauthorizedError,
    HTTPStatus.FORBIDDEN.value: AccessDeniedError,
    HTTPStatus.NOT_FOUND.value: NotFoundError,
}
