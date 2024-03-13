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
import asyncio
from http import HTTPStatus
from typing import Union

from fastapi import APIRouter, Depends, Request, Response
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.common.schemas.model_monitoring.grafana
import server.py.services.api.crud.model_monitoring.grafana
import server.py.services.api.crud.model_monitoring.helpers
from server.py.services.api.api import deps

router = APIRouter(prefix="/grafana-proxy/model-endpoints")

NAME_TO_SEARCH_FUNCTION_DICTIONARY = {
    "list_projects": server.py.services.api.crud.model_monitoring.grafana.grafana_list_projects,
}
NAME_TO_QUERY_FUNCTION_DICTIONARY = {
    "list_endpoints": server.py.services.api.crud.model_monitoring.grafana.grafana_list_endpoints,
    "individual_feature_analysis": server.py.services.api.crud.model_monitoring.grafana.grafana_individual_feature_analysis,
    "overall_feature_analysis": server.py.services.api.crud.model_monitoring.grafana.grafana_overall_feature_analysis,
    "incoming_features": server.py.services.api.crud.model_monitoring.grafana.grafana_incoming_features,
}

SUPPORTED_QUERY_FUNCTIONS = set(NAME_TO_QUERY_FUNCTION_DICTIONARY.keys())
SUPPORTED_SEARCH_FUNCTIONS = set(NAME_TO_SEARCH_FUNCTION_DICTIONARY)


@router.get("", status_code=HTTPStatus.OK.value)
def grafana_proxy_model_endpoints_check_connection(
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
):
    """
    Root of grafana proxy for the model-endpoints API, used for validating the model-endpoints data source
    connectivity.
    """
    if not mlrun.mlconf.is_ce_mode():
        server.py.services.api.crud.model_monitoring.helpers.get_access_key(auth_info)
    return Response(status_code=HTTPStatus.OK.value)


@router.post("/search", response_model=list[str])
async def grafana_proxy_model_endpoints_search(
    request: Request,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
) -> list[str]:
    """
    Search route for model-endpoints grafana proxy API, used for creating an interface between grafana queries and
    model-endpoints logic.

    This implementation requires passing target_endpoint query parameter in order to dispatch different
    model-endpoint monitoring functions.

    :param request:    An api request with the required target and parameters.
    :param auth_info:  The auth info of the request.
    :param db_session: A session that manages the current dialog with the database.

    :return: List of results. e.g. list of available project names.
    """
    if not mlrun.mlconf.is_ce_mode():
        server.py.services.api.crud.model_monitoring.helpers.get_access_key(auth_info)
    body = await request.json()
    query_parameters = (
        server.py.services.api.crud.model_monitoring.grafana.parse_search_parameters(
            body
        )
    )
    server.py.services.api.crud.model_monitoring.grafana.validate_query_parameters(
        query_parameters, SUPPORTED_SEARCH_FUNCTIONS
    )

    # At this point everything is validated and we can access everything that is needed without performing all previous
    # checks again.
    target_endpoint = query_parameters["target_endpoint"]
    function = NAME_TO_SEARCH_FUNCTION_DICTIONARY[target_endpoint]

    if asyncio.iscoroutinefunction(function):
        result = await function(db_session, auth_info, query_parameters)
    else:
        result = await run_in_threadpool(
            function, db_session, auth_info, query_parameters
        )
    return result


@router.post(
    "/query",
    response_model=list[
        Union[
            mlrun.common.schemas.model_monitoring.grafana.GrafanaTable,
            mlrun.common.schemas.model_monitoring.grafana.GrafanaTimeSeriesTarget,
        ]
    ],
)
async def grafana_proxy_model_endpoints_query(
    request: Request,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
) -> list[
    Union[
        mlrun.common.schemas.model_monitoring.grafana.GrafanaTable,
        mlrun.common.schemas.model_monitoring.grafana.GrafanaTimeSeriesTarget,
    ]
]:
    """
    Query route for model-endpoints grafana proxy API, used for creating an interface between grafana queries and
    model-endpoints logic.

    This implementation requires passing target_endpoint query parameter in order to dispatch different
    model-endpoint monitoring functions.
    """

    body = await request.json()
    query_parameters = (
        server.py.services.api.crud.model_monitoring.grafana.parse_query_parameters(
            body
        )
    )
    server.py.services.api.crud.model_monitoring.grafana.validate_query_parameters(
        query_parameters, SUPPORTED_QUERY_FUNCTIONS
    )
    query_parameters = (
        server.py.services.api.crud.model_monitoring.grafana.drop_grafana_escape_chars(
            query_parameters
        )
    )

    # At this point everything is validated and we can access everything that is needed without performing all previous
    # checks again.
    target_endpoint = query_parameters["target_endpoint"]
    function = NAME_TO_QUERY_FUNCTION_DICTIONARY[target_endpoint]
    if asyncio.iscoroutinefunction(function):
        return await function(body, query_parameters, auth_info)
    result = await run_in_threadpool(function, body, query_parameters, auth_info)
    return result
