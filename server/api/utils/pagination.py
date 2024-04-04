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
#
import typing

import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.singleton
import server.api.crud
from mlrun.utils import logger


class PaginatedMethods:
    _methods: list[typing.Callable] = [
        # TODO: add methods when they implement pagination
        # server.api.crud.Runs().list_runs,
    ]
    _method_map = {method.__name__: method for method in _methods}

    @classmethod
    def method_is_supported(cls, method: typing.Union[str, typing.Callable]) -> bool:
        if isinstance(method, str):
            return method in cls._method_map
        return method in cls._methods

    @classmethod
    def get_method(cls, method_name: str) -> typing.Callable:
        return cls._method_map[method_name]


class Paginator(metaclass=mlrun.utils.singleton.Singleton):
    def __init__(self):
        self._logger = logger.get_child("paginator")
        self._pagination_cache = server.api.crud.PaginationCache()

    async def paginate_permission_filtered_request(
        self,
        session: sqlalchemy.orm.Session,
        method: typing.Callable,
        filter_: typing.Callable,
        auth_info: typing.Optional[mlrun.common.schemas.AuthInfo] = None,
        token: typing.Optional[str] = None,
        page: typing.Optional[int] = None,
        page_size: typing.Optional[int] = None,
        **method_kwargs,
    ) -> tuple[typing.Any, dict[str, typing.Union[str, int]]]:
        """
        Paginate a request and filter the results based on the provided filter function.
        If the result of the filter has fewer items than the page size, the pagination will request more items until
        the page size is reached.
        There is an option here to overflow and to receive more items than the page size.
        And actually the maximum number of items that can be returned is page_size * 2 - 1.
        """
        last_pagination_info = {}
        current_page = page or 1
        result = []

        while len(result) < page_size:
            new_result, pagination_info = self.paginate_request(
                session,
                method,
                auth_info,
                token,
                current_page,
                page_size,
                **method_kwargs,
            )
            if not pagination_info:
                # no more results
                break
            last_pagination_info = pagination_info
            new_result = await filter_(new_result)
            result.extend(new_result)
            current_page += 1

        return result, last_pagination_info

    def paginate_request(
        self,
        session: sqlalchemy.orm.Session,
        method: typing.Callable,
        auth_info: typing.Optional[mlrun.common.schemas.AuthInfo] = None,
        token: typing.Optional[str] = None,
        page: typing.Optional[int] = None,
        page_size: typing.Optional[int] = None,
        **method_kwargs,
    ) -> tuple[typing.Any, dict[str, typing.Union[str, int]]]:
        if not PaginatedMethods.method_is_supported(method):
            raise NotImplementedError(
                f"Pagination is not supported for method {method}"
            )

        if page_size is None and token is None:
            self._logger.debug("No token or page size provided, returning all records")
            return method(**method_kwargs), {}

        token, page, page_size, method, method_kwargs = (
            self._create_or_update_pagination_cache_record(
                session,
                method,
                auth_info,
                token,
                page,
                page_size,
                **method_kwargs,
            )
        )

        try:
            self._logger.debug(
                "Retrieving page",
                page=page,
                page_size=page_size,
                method=method.__name__,
            )
            return method(**method_kwargs, page=page, page_size=page_size), {
                "token": token,
                "page": page,
                "page_size": page_size,
            }
        except StopIteration:
            self._logger.debug("End of pagination", token=token, method=method.__name__)
            self._pagination_cache.delete_pagination_cache_record(session, key=token)
            return [], {}

    def _create_or_update_pagination_cache_record(
        self,
        session: sqlalchemy.orm.Session,
        method: typing.Callable,
        auth_info: typing.Optional[mlrun.common.schemas.AuthInfo] = None,
        token: typing.Optional[str] = None,
        page: typing.Optional[int] = None,
        page_size: typing.Optional[int] = None,
        **method_kwargs,
    ) -> tuple[str, int, int, typing.Callable, dict]:
        if token:
            self._logger.debug(
                "Token provided, updating pagination cache record", token=token
            )
            pagination_cache_record = (
                self._pagination_cache.get_pagination_cache_record(session, key=token)
            )
            if pagination_cache_record is None:
                raise mlrun.errors.MLRunNotFoundError(
                    f"Token {token} not found in pagination cache"
                )
            method = PaginatedMethods.get_method(pagination_cache_record.function)
            method_kwargs = pagination_cache_record.kwargs
            page = page or pagination_cache_record.current_page + 1
            page_size = pagination_cache_record.page_size
            user = pagination_cache_record.user

            if user and (not auth_info or auth_info.user_id != user):
                raise mlrun.errors.MLRunAccessDeniedError(
                    "User is not allowed to access this token"
                )

        # upsert pagination cache record to update last_accessed time or create a new record
        self._logger.debug(
            "Storing pagination cache record",
            method=method.__name__,
            page=page,
            page_size=page_size,
        )
        token = self._pagination_cache.store_pagination_cache_record(
            session,
            user=auth_info.user_id if auth_info else None,
            method=method,
            current_page=page,
            page_size=page_size,
            kwargs=method_kwargs,
        )
        return token, page, page_size, method, method_kwargs
