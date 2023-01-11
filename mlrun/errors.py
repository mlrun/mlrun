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
import typing
from http import HTTPStatus

import aiohttp
import requests


class MLRunBaseError(Exception):
    """
    A base class from which all other exceptions inherit.
    If you want to catch all errors that the MLRun SDK might raise,
    catch this base exception.
    """

    pass


class MLRunTaskNotReady(MLRunBaseError):
    """indicate we are trying to read a value which is not ready
    or need to come from a job which is in progress"""


class MLRunHTTPError(MLRunBaseError, requests.HTTPError):
    def __init__(
        self,
        *args,
        response: typing.Optional[
            typing.Union[requests.Response, aiohttp.ClientResponse]
        ] = None,
        status_code: typing.Optional[int] = None,
        **kwargs,
    ):

        # because response object is probably with an error, it returns False, so we
        # should use 'is None' specifically
        if response is None:
            response = requests.Response()
        if status_code:
            response.status_code = status_code

        if isinstance(response, aiohttp.ClientResponse):
            if "request" not in kwargs:
                kwargs["request"] = response.request_info

            # consolidate the response object to be a requests.Response object-like
            setattr(
                response, "status_code", status_code if status_code else response.status
            )

        requests.HTTPError.__init__(self, *args, response=response, **kwargs)


class MLRunHTTPStatusError(MLRunHTTPError):
    """
    When an error has a matching http status code it is "HTTP statusable"
    HTTP Status errors should inherit from this class and set the right status code in the
    error_status_code attribute
    """

    error_status_code = None

    def __init__(self, *args, response: requests.Response = None, **kwargs):
        super(MLRunHTTPStatusError, self).__init__(
            *args, response=response, status_code=self.error_status_code, **kwargs
        )


def raise_for_status(
    response: typing.Union[
        requests.Response,
        aiohttp.ClientResponse,
    ],
    message: str = None,
):
    """
    Raise a specific MLRunSDK error depending on the given response status code.
    If no specific error exists, raises an MLRunHTTPError
    """
    try:
        response.raise_for_status()
    except (requests.HTTPError, aiohttp.ClientResponseError) as exc:
        error_message = err_to_str(exc)
        if message:
            error_message = f"{error_message}: {message}"
        status_code = (
            response.status_code
            if hasattr(response, "status_code")
            else response.status
        )
        try:
            raise STATUS_ERRORS[status_code](error_message, response=response) from exc
        except KeyError:
            raise MLRunHTTPError(error_message, response=response) from exc


def raise_for_status_code(status_code: int, message: str = None):
    """
    Raise a specific MLRunSDK error depending on the given response status code.
    If no specific error exists, raises an MLRunHTTPError
    """
    try:
        raise STATUS_ERRORS[status_code](message)
    except KeyError:
        raise MLRunHTTPError(message)


def err_to_str(err):
    if not err:
        return ""

    if isinstance(err, str):
        return err

    errors = []
    error_strings = []
    while err and err not in errors:
        errors.append(err)
        error_strings.append(str(err))
        err = err.__cause__

    return ", caused by: ".join(error_strings)


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


class MLRunInvalidArgumentTypeError(MLRunHTTPStatusError, TypeError):
    error_status_code = HTTPStatus.BAD_REQUEST.value


class MLRunConflictError(MLRunHTTPStatusError):
    error_status_code = HTTPStatus.CONFLICT.value


class MLRunPreconditionFailedError(MLRunHTTPStatusError):
    error_status_code = HTTPStatus.PRECONDITION_FAILED.value


class MLRunIncompatibleVersionError(MLRunHTTPStatusError):
    error_status_code = HTTPStatus.BAD_REQUEST.value


class MLRunInternalServerError(MLRunHTTPStatusError):
    error_status_code = HTTPStatus.INTERNAL_SERVER_ERROR.value


class MLRunRuntimeError(MLRunHTTPStatusError, RuntimeError):
    error_status_code = HTTPStatus.INTERNAL_SERVER_ERROR.value


class MLRunMissingDependencyError(MLRunInternalServerError):
    pass


class MLRunTimeoutError(MLRunHTTPStatusError, TimeoutError):
    error_status_code = HTTPStatus.GATEWAY_TIMEOUT.value


class MLRunFatalFailureError(Exception):
    """
    Internal exception meant to be used inside mlrun.utils.helpers.retry_until_successful to signal the loop not to
    retry
    Allowing to pass to original exception that will be raised from the loop (instead of this exception)
    """

    def __init__(
        self, *args, original_exception: typing.Optional[Exception] = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.original_exception = original_exception


STATUS_ERRORS = {
    HTTPStatus.BAD_REQUEST.value: MLRunBadRequestError,
    HTTPStatus.UNAUTHORIZED.value: MLRunUnauthorizedError,
    HTTPStatus.FORBIDDEN.value: MLRunAccessDeniedError,
    HTTPStatus.NOT_FOUND.value: MLRunNotFoundError,
    HTTPStatus.CONFLICT.value: MLRunConflictError,
    HTTPStatus.PRECONDITION_FAILED.value: MLRunPreconditionFailedError,
    HTTPStatus.INTERNAL_SERVER_ERROR.value: MLRunInternalServerError,
}
