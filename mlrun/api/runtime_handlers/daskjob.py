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
from typing import Dict, List, Optional, Union

from kubernetes.client.rest import ApiException
from sqlalchemy.orm import Session

import mlrun.common.schemas
import mlrun.errors
import mlrun.k8s_utils
import mlrun.utils
import mlrun.utils.regex
from mlrun.api.db.base import DBInterface
from mlrun.api.runtime_handlers.base import BaseRuntimeHandler
from mlrun.config import config
from mlrun.runtimes.base import RuntimeClassMode
from mlrun.runtimes.utils import get_k8s
from mlrun.utils import logger


class DaskRuntimeHandler(BaseRuntimeHandler):
    kind = "dask"
    class_modes = {RuntimeClassMode.run: "dask"}

    # Dask runtime resources are per function (and not per run).
    # It means that monitoring runtime resources state doesn't say anything about the run state.
    # Therefore dask run monitoring is done completely by the SDK, so overriding the monitoring method with no logic
    def monitor_runs(
        self, db: DBInterface, db_session: Session, leader_session: Optional[str] = None
    ):
        return

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f"mlrun/function={object_id}"

    @staticmethod
    def resolve_object_id(
        run: dict,
    ) -> typing.Optional[str]:
        """
        Resolves the object ID from the run object.
        In dask runtime, the object ID is the function name.
        :param run: run object
        :return: function name
        """

        function = run.get("spec", {}).get("function", None)
        if function:

            # a dask run's function field is in the format <project-name>/<function-name>@<run-uid>
            # we only want the function name
            project_and_function = function.split("@")[0]
            return project_and_function.split("/")[-1]

        return None

    def _enrich_list_resources_response(
        self,
        response: Union[
            mlrun.common.schemas.RuntimeResources,
            mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
            mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
        ],
        namespace: str,
        label_selector: str = None,
        group_by: Optional[
            mlrun.common.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ) -> Union[
        mlrun.common.schemas.RuntimeResources,
        mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
        mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
    ]:
        """
        Handling listing service resources
        """
        enrich_needed = self._validate_if_enrich_is_needed_by_group_by(group_by)
        if not enrich_needed:
            return response
        services = get_k8s().v1api.list_namespaced_service(
            namespace, label_selector=label_selector
        )
        service_resources = []
        for service in services.items:
            service_resources.append(
                mlrun.common.schemas.RuntimeResource(
                    name=service.metadata.name, labels=service.metadata.labels
                )
            )
        return self._enrich_service_resources_in_response(
            response, service_resources, group_by
        )

    def _build_output_from_runtime_resources(
        self,
        response: Union[
            mlrun.common.schemas.RuntimeResources,
            mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
            mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
        ],
        runtime_resources_list: List[mlrun.common.schemas.RuntimeResources],
        group_by: Optional[
            mlrun.common.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ):
        enrich_needed = self._validate_if_enrich_is_needed_by_group_by(group_by)
        if not enrich_needed:
            return response
        service_resources = []
        for runtime_resources in runtime_resources_list:
            if runtime_resources.service_resources:
                service_resources += runtime_resources.service_resources
        return self._enrich_service_resources_in_response(
            response, service_resources, group_by
        )

    def _validate_if_enrich_is_needed_by_group_by(
        self,
        group_by: Optional[
            mlrun.common.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ) -> bool:
        # Dask runtime resources are per function (and not per job) therefore, when grouping by job we're simply
        # omitting the dask runtime resources
        if group_by == mlrun.common.schemas.ListRuntimeResourcesGroupByField.job:
            return False
        elif group_by == mlrun.common.schemas.ListRuntimeResourcesGroupByField.project:
            return True
        elif group_by is not None:
            raise NotImplementedError(
                f"Provided group by field is not supported. group_by={group_by}"
            )
        return True

    def _enrich_service_resources_in_response(
        self,
        response: Union[
            mlrun.common.schemas.RuntimeResources,
            mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
            mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
        ],
        service_resources: List[mlrun.common.schemas.RuntimeResource],
        group_by: Optional[
            mlrun.common.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ):
        if group_by == mlrun.common.schemas.ListRuntimeResourcesGroupByField.project:
            for service_resource in service_resources:
                self._add_resource_to_grouped_by_project_resources_response(
                    response, "service_resources", service_resource
                )
        else:
            response.service_resources = service_resources
        return response

    def _delete_extra_resources(
        self,
        db: DBInterface,
        db_session: Session,
        namespace: str,
        deleted_resources: List[Dict],
        label_selector: str = None,
        force: bool = False,
        grace_period: int = None,
    ):
        """
        Handling services deletion
        """
        if grace_period is None:
            grace_period = config.runtime_resources_deletion_grace_period
        service_names = []
        for pod_dict in deleted_resources:
            dask_component = (
                pod_dict["metadata"].get("labels", {}).get("dask.org/component")
            )
            cluster_name = (
                pod_dict["metadata"].get("labels", {}).get("dask.org/cluster-name")
            )
            if dask_component == "scheduler" and cluster_name:
                service_names.append(cluster_name)

        services = get_k8s().v1api.list_namespaced_service(
            namespace, label_selector=label_selector
        )
        for service in services.items:
            try:
                if force or service.metadata.name in service_names:
                    get_k8s().v1api.delete_namespaced_service(
                        service.metadata.name, namespace
                    )
                    logger.info(f"Deleted service: {service.metadata.name}")
            except ApiException as exc:
                # ignore error if service is already removed
                if exc.status != 404:
                    raise
