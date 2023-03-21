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
import json
import os
import warnings
from http import HTTPStatus
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.api.deps
import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
from mlrun.errors import MLRunConflictError

router = APIRouter()


@router.put(
    "/projects/{project}/model-endpoints/{endpoint_id}",
    status_code=HTTPStatus.NO_CONTENT.value,
)
async def create_or_patch(
    project: str,
    endpoint_id: str,
    model_endpoint: mlrun.api.schemas.ModelEndpoint,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
):
    """
    Either create or updates the record of a given ModelEndpoint object.
    Leaving here for backwards compatibility.
    """

    warnings.warn(
        "This PUT call is deprecated, please use POST for create or PATCH for update"
        "This will be removed in 1.5.0",
        # TODO: Remove this API in 1.5.0
        FutureWarning,
    )

    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.model_endpoint,
        project,
        endpoint_id,
        mlrun.api.schemas.AuthorizationAction.store,
        auth_info,
    )
    # get_access_key will validate the needed auth (which is used later) exists in the request
    mlrun.api.crud.ModelEndpoints().get_access_key(auth_info)
    if project != model_endpoint.metadata.project:
        raise MLRunConflictError(
            f"Can't store endpoint of project {model_endpoint.metadata.project} into project {project}"
        )
    if endpoint_id != model_endpoint.metadata.uid:
        raise MLRunConflictError(
            f"Mismatch between endpoint_id {endpoint_id} and ModelEndpoint.metadata.uid {model_endpoint.metadata.uid}."
            f"\nMake sure the supplied function_uri, and model are configured as intended"
        )
    # Since the endpoint records are created automatically, at point of serving function deployment, we need to use
    # V3IO_ACCESS_KEY here
    await run_in_threadpool(
        mlrun.api.crud.ModelEndpoints().create_or_patch,
        db_session=db_session,
        access_key=os.environ.get("V3IO_ACCESS_KEY"),
        model_endpoint=model_endpoint,
        auth_info=auth_info,
    )


@router.post(
    "/projects/{project}/model-endpoints/{endpoint_id}",
    response_model=mlrun.api.schemas.ModelEndpoint,
)
async def create_model_endpoint(
    project: str,
    endpoint_id: str,
    model_endpoint: mlrun.api.schemas.ModelEndpoint,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
) -> mlrun.api.schemas.ModelEndpoint:
    """
    Create a DB record of a given ModelEndpoint object.

    :param project:         The name of the project.
    :param endpoint_id:     The unique id of the model endpoint.
    :param model_endpoint:  Model endpoint object to record in DB.
    :param auth_info:       The auth info of the request.
    :param db_session:      A session that manages the current dialog with the database. When creating a new model
                            endpoint id record, we need to use the db session for getting information from an existing
                            model artifact and also for storing the new model monitoring feature set.

    :return: A Model endpoint object.
    """
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        resource_type=mlrun.api.schemas.AuthorizationResourceTypes.model_endpoint,
        project_name=project,
        resource_name=endpoint_id,
        action=mlrun.api.schemas.AuthorizationAction.store,
        auth_info=auth_info,
    )

    if project != model_endpoint.metadata.project:
        raise MLRunConflictError(
            f"Can't store endpoint of project {model_endpoint.metadata.project} into project {project}"
        )
    if endpoint_id != model_endpoint.metadata.uid:
        raise MLRunConflictError(
            f"Mismatch between endpoint_id {endpoint_id} and ModelEndpoint.metadata.uid {model_endpoint.metadata.uid}."
            f"\nMake sure the supplied function_uri, and model are configured as intended"
        )

    return await run_in_threadpool(
        mlrun.api.crud.ModelEndpoints().create_model_endpoint,
        db_session=db_session,
        model_endpoint=model_endpoint,
    )


@router.patch(
    "/projects/{project}/model-endpoints/{endpoint_id}",
    response_model=mlrun.api.schemas.ModelEndpoint,
)
async def patch_model_endpoint(
    project: str,
    endpoint_id: str,
    attributes: str = None,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
) -> mlrun.api.schemas.ModelEndpoint:
    """
    Update a DB record of a given ModelEndpoint object.

    :param project:       The name of the project.
    :param endpoint_id:   The unique id of the model endpoint.
    :param attributes:    Attributes that will be updated. The input is provided in a json structure that will be
                          converted into a dictionary before applying the patch process. Note that the keys of
                          dictionary should exist in the DB target. More details about the model endpoint available
                          attributes can be found under :py:class:`~mlrun.api.schemas.ModelEndpoint`.

                          example::

                          attributes = {"drift_status": "POSSIBLE_DRIFT", "state": "new_state"}

    :param auth_info:     The auth info of the request.

    :return: A Model endpoint object.
    """

    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        resource_type=mlrun.api.schemas.AuthorizationResourceTypes.model_endpoint,
        project_name=project,
        resource_name=endpoint_id,
        action=mlrun.api.schemas.AuthorizationAction.update,
        auth_info=auth_info,
    )

    if not attributes:
        raise mlrun.errors.MLRunNotFoundError(
            f"No attributes provided for patching the model endpoint {endpoint_id}",
        )
    return await run_in_threadpool(
        mlrun.api.crud.ModelEndpoints().patch_model_endpoint,
        project=project,
        endpoint_id=endpoint_id,
        attributes=json.loads(attributes),
    )


@router.delete(
    "/projects/{project}/model-endpoints/{endpoint_id}",
    status_code=HTTPStatus.NO_CONTENT.value,
)
async def delete_model_endpoint(
    project: str,
    endpoint_id: str,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    """
    Clears endpoint record from the DB based on endpoint_id.

    :param project:       The name of the project.
    :param endpoint_id:   The unique id of the model endpoint.
    :param auth_info:     The auth info of the request.

    """

    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        resource_type=mlrun.api.schemas.AuthorizationResourceTypes.model_endpoint,
        project_name=project,
        resource_name=endpoint_id,
        action=mlrun.api.schemas.AuthorizationAction.delete,
        auth_info=auth_info,
    )

    await run_in_threadpool(
        mlrun.api.crud.ModelEndpoints().delete_model_endpoint,
        project=project,
        endpoint_id=endpoint_id,
    )


@router.get(
    "/projects/{project}/model-endpoints",
    response_model=mlrun.api.schemas.ModelEndpointList,
)
async def list_model_endpoints(
    project: str,
    model: Optional[str] = Query(None),
    function: Optional[str] = Query(None),
    labels: List[str] = Query([], alias="label"),
    start: str = Query(default="now-1h"),
    end: str = Query(default="now"),
    metrics: List[str] = Query([], alias="metric"),
    top_level: bool = Query(False, alias="top-level"),
    uids: List[str] = Query(None, alias="uid"),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
) -> mlrun.api.schemas.ModelEndpointList:
    """
    Returns a list of endpoints of type 'ModelEndpoint', supports filtering by model, function, tag,
    labels or top level. By default, when no filters are applied, all available endpoints for the given project will be
    listed.

    If uids are passed: will return ModelEndpointList of endpoints with uid in uids
    Labels can be used to filter on the existence of a label:
    api/projects/{project}/model-endpoints/?label=mylabel

    Or on the value of a given label:
    api/projects/{project}/model-endpoints/?label=mylabel=1

    Multiple labels can be queried in a single request by either using "&" separator:
    api/projects/{project}/model-endpoints/?label=mylabel=1&label=myotherlabel=2

    Or by using a "," (comma) separator:
    api/projects/{project}/model-endpoints/?label=mylabel=1,myotherlabel=2
    Top level: if true will return only routers and endpoint that are NOT children of any router

    :param auth_info: The auth info of the request.
    :param project:   The name of the project.
    :param model:     The name of the model to filter by.
    :param function:  The name of the function to filter by.
    :param labels:    A list of labels to filter by. Label filters work by either filtering a specific value of a label
                      (i.e. list("key==value")) or by looking for the existence of a given key (i.e. "key").
    :param metrics:   A list of metrics to return for each endpoint. There are pre-defined metrics for model endpoints
                      such as predictions_per_second and latency_avg_5m but also custom metrics defined by the user.
                      Please note that these metrics are stored in the time series DB and the results will be appeared
                      under model_endpoint.spec.metrics of each endpoint.
    :param start:     The start time of the metrics. Can be represented by a string containing an RFC 3339
                      time, a Unix timestamp in milliseconds, a relative time (`'now'` or `'now-[0-9]+[mhd]'`, where
                      `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the earliest time.
    :param end:       The end time of the metrics. Can be represented by a string containing an RFC 3339
                      time, a Unix timestamp in milliseconds, a relative time (`'now'` or `'now-[0-9]+[mhd]'`, where
                      `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the earliest time.
    :param top_level: If True will return only routers and endpoint that are NOT children of any router.
    :param uids:      Will return ModelEndpointList of endpoints with uid in uids.

    :return: An object of ModelEndpointList which is literally a list of model endpoints along with some metadata. To
             get a standard list of model endpoints use ModelEndpointList.endpoints.
    """

    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project_name=project,
        action=mlrun.api.schemas.AuthorizationAction.read,
        auth_info=auth_info,
    )

    endpoints = await run_in_threadpool(
        mlrun.api.crud.ModelEndpoints().list_model_endpoints,
        auth_info=auth_info,
        project=project,
        model=model,
        function=function,
        labels=labels,
        metrics=metrics,
        start=start,
        end=end,
        top_level=top_level,
        uids=uids,
    )
    allowed_endpoints = await mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.model_endpoint,
        endpoints.endpoints,
        lambda _endpoint: (
            _endpoint.metadata.project,
            _endpoint.metadata.uid,
        ),
        auth_info,
    )

    endpoints.endpoints = allowed_endpoints
    return endpoints


@router.get(
    "/projects/{project}/model-endpoints/{endpoint_id}",
    response_model=mlrun.api.schemas.ModelEndpoint,
)
async def get_model_endpoint(
    project: str,
    endpoint_id: str,
    start: str = Query(default="now-1h"),
    end: str = Query(default="now"),
    metrics: List[str] = Query([], alias="metric"),
    feature_analysis: bool = Query(default=False),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
) -> mlrun.api.schemas.ModelEndpoint:
    """Get a single model endpoint object. You can apply different time series metrics that will be added to the
       result.

    :param project:          The name of the project.
    :param endpoint_id:      The unique id of the model endpoint.
    :param start:            The start time of the metrics. Can be represented by a string containing an RFC 3339
                             time, a Unix timestamp in milliseconds, a relative time (`'now'` or `'now-[0-9]+[mhd]'`,
                             where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the earliest time.
    :param end:              The end time of the metrics. Can be represented by a string containing an RFC 3339
                             time, a Unix timestamp in milliseconds, a relative time (`'now'` or `'now-[0-9]+[mhd]'`,
                             where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the earliest time.
    :param metrics:          A list of metrics to return for the model endpoint. There are pre-defined metrics for model
                             endpoints such as predictions_per_second and latency_avg_5m but also custom metrics
                             defined by the user. Please note that these metrics are stored in the time series DB and
                             the results will be appeared under model_endpoint.spec.metrics.
    :param feature_analysis: When True, the base feature statistics and current feature statistics will be added to
                             the output of the resulting object.
    :param auth_info:        The auth info of the request.

    :return: A ModelEndpoint object.
    """
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.model_endpoint,
        project,
        endpoint_id,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )

    return await run_in_threadpool(
        mlrun.api.crud.ModelEndpoints().get_model_endpoint,
        auth_info=auth_info,
        project=project,
        endpoint_id=endpoint_id,
        metrics=metrics,
        start=start,
        end=end,
        feature_analysis=feature_analysis,
    )
