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

import pydantic
import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.singleton
import server.api.crud
import server.api.utils.asyncio
from mlrun import mlconf
from mlrun.utils import logger


class PaginatedMethods:
    _method_schemas: dict[typing.Callable, pydantic.BaseModel] = {
        # TODO: add methods when they implement pagination
        server.api.crud.Runs().list_runs: mlrun.common.schemas.runs.ListRunsRequest,
    }
    _method_map = {
        method.__name__: {
            "method": method,
            "schema": schema,
        }
        for method, schema in _method_schemas.items()
    }

    @classmethod
    def method_is_supported(cls, method: typing.Union[str, typing.Callable]) -> bool:
        method_name = method if isinstance(method, str) else method.__name__
        return method_name in cls._method_map

    @classmethod
    def get_method(cls, method_name: str) -> typing.Callable:
        return cls._method_map[method_name]["method"]

    @classmethod
    def get_method_schema(cls, method_name: str) -> pydantic.BaseModel:
        return cls._method_map[method_name]["schema"]


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
        current_page = page
        result = []

        while not page_size or len(result) < page_size:
            new_result, pagination_info = await self.paginate_request(
                session,
                method,
                auth_info,
                token,
                current_page,
                page_size,
                **method_kwargs,
            )
            new_result = await server.api.utils.asyncio.await_or_call_in_threadpool(
                filter_, new_result
            )
            result.extend(new_result)

            if not pagination_info:
                # no more results
                break

            last_pagination_info = pagination_info
            current_page = last_pagination_info["page"] + 1
            page_size = last_pagination_info["page-size"]

        return result, last_pagination_info

    async def paginate_request(
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
            return await server.api.utils.asyncio.await_or_call_in_threadpool(
                method, session, **method_kwargs
            ), {}

        page_size = page_size or mlconf.httpdb.pagination.default_page_size

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
            return await server.api.utils.asyncio.await_or_call_in_threadpool(
                method, session, **method_kwargs, page=page, page_size=page_size
            ), {
                "token": token,
                "page": page,
                "page-size": page_size,
            }
        except (RuntimeError, StopIteration) as exc:
            if isinstance(exc, StopIteration) or "StopIteration" in str(exc):
                self._logger.debug(
                    "End of pagination", token=token, method=method.__name__
                )
                self._pagination_cache.delete_pagination_cache_record(
                    session, key=token
                )
                return [], {}
            raise

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
        method_schema = PaginatedMethods.get_method_schema(method.__name__)
        serialized_kwargs = method_schema(**method_kwargs).dict()
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
            kwargs=serialized_kwargs,
        )
        return token, page, page_size, method, serialized_kwargs
