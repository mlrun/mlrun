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

import mergedeep
import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.config
import mlrun.errors
import mlrun.runtimes
import mlrun.utils.singleton
import server.api.runtime_handlers
import server.api.utils.singletons.db


class RuntimeResources(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def list_runtime_resources(
        self,
        project: str,
        kind: typing.Optional[str] = None,
        object_id: typing.Optional[str] = None,
        label_selector: typing.Optional[str] = None,
        group_by: typing.Optional[
            mlrun.common.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ) -> typing.Union[
        mlrun.common.schemas.RuntimeResourcesOutput,
        mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
        mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
    ]:
        response = [] if group_by is None else {}
        kinds = mlrun.runtimes.RuntimeKinds.runtime_with_handlers()
        if kind is not None:
            self.validate_runtime_resources_kind(kind)
            kinds = [kind]
        for kind in kinds:
            runtime_handler = server.api.runtime_handlers.get_runtime_handler(kind)
            resources = runtime_handler.list_resources(
                project, object_id, label_selector, group_by
            )
            if group_by is None:
                response.append(
                    mlrun.common.schemas.KindRuntimeResources(
                        kind=kind, resources=resources
                    )
                )
            else:
                mergedeep.merge(response, resources)
        return response

    def filter_and_format_grouped_by_project_runtime_resources_output(
        self,
        grouped_by_project_runtime_resources_output: mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
        allowed_projects: list[str],
        group_by: typing.Optional[
            mlrun.common.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ) -> typing.Union[
        mlrun.common.schemas.RuntimeResourcesOutput,
        mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
        mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
    ]:
        runtime_resources_by_kind = {}
        for (
            project,
            kind_runtime_resources_map,
        ) in grouped_by_project_runtime_resources_output.items():
            for kind, runtime_resources in kind_runtime_resources_map.items():
                if project in allowed_projects:
                    runtime_resources_by_kind.setdefault(kind, []).append(
                        runtime_resources
                    )
        runtimes_resources_output = [] if group_by is None else {}
        for kind, runtime_resources_list in runtime_resources_by_kind.items():
            runtime_handler = server.api.runtime_handlers.get_runtime_handler(kind)
            resources = runtime_handler.build_output_from_runtime_resources(
                runtime_resources_list, group_by
            )
            if group_by is None:
                runtimes_resources_output.append(
                    mlrun.common.schemas.KindRuntimeResources(
                        kind=kind, resources=resources
                    )
                )
            else:
                mergedeep.merge(runtimes_resources_output, resources)
        return runtimes_resources_output

    def validate_runtime_resources_kind(self, kind: str):
        if kind not in mlrun.runtimes.RuntimeKinds.runtime_with_handlers():
            raise mlrun.errors.MLRunBadRequestError(
                f"Invalid runtime kind {kind}. Must be one of: {mlrun.runtimes.RuntimeKinds.runtime_with_handlers()}"
            )

    def delete_runtime_resources(
        self,
        db_session: sqlalchemy.orm.Session,
        kind: typing.Optional[str] = None,
        object_id: typing.Optional[str] = None,
        label_selector: typing.Optional[str] = None,
        force: bool = False,
        grace_period: typing.Optional[int] = None,
    ):
        kinds = mlrun.runtimes.RuntimeKinds.runtime_with_handlers()
        if kind is not None:
            self.validate_runtime_resources_kind(kind)
            kinds = [kind]
        for kind in kinds:
            runtime_handler = server.api.runtime_handlers.get_runtime_handler(kind)
            if object_id:
                runtime_handler.delete_runtime_object_resources(
                    server.api.utils.singletons.db.get_db(),
                    db_session,
                    object_id,
                    label_selector,
                    force,
                    grace_period,
                )
            else:
                runtime_handler.delete_resources(
                    server.api.utils.singletons.db.get_db(),
                    db_session,
                    label_selector,
                    force,
                    grace_period,
                )
