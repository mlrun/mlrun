import typing

import mergedeep
import sqlalchemy.orm

import mlrun.api.api.utils
import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.singletons.db
import mlrun.config
import mlrun.errors
import mlrun.runtimes
import mlrun.utils.singleton


class RuntimeResources(metaclass=mlrun.utils.singleton.Singleton,):
    def list_runtime_resources(
        self,
        project: str,
        kind: typing.Optional[str] = None,
        object_id: typing.Optional[str] = None,
        label_selector: typing.Optional[str] = None,
        group_by: typing.Optional[
            mlrun.api.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ) -> typing.Union[
        mlrun.api.schemas.RuntimeResourcesOutput,
        mlrun.api.schemas.GroupedByJobRuntimeResourcesOutput,
        mlrun.api.schemas.GroupedByProjectRuntimeResourcesOutput,
    ]:
        response = [] if group_by is None else {}
        kinds = mlrun.runtimes.RuntimeKinds.runtime_with_handlers()
        if kind is not None:
            self.validate_runtime_resources_kind(kind)
            kinds = [kind]
        for kind in kinds:
            runtime_handler = mlrun.runtimes.get_runtime_handler(kind)
            resources = runtime_handler.list_resources(
                project, object_id, label_selector, group_by
            )
            if group_by is None:
                response.append(
                    mlrun.api.schemas.KindRuntimeResources(
                        kind=kind, resources=resources
                    )
                )
            else:
                mergedeep.merge(response, resources)
        return response

    def filter_and_format_grouped_by_project_runtime_resources_output(
        self,
        grouped_by_project_runtime_resources_output: mlrun.api.schemas.GroupedByProjectRuntimeResourcesOutput,
        allowed_projects: typing.List[str],
        group_by: typing.Optional[
            mlrun.api.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ) -> typing.Union[
        mlrun.api.schemas.RuntimeResourcesOutput,
        mlrun.api.schemas.GroupedByJobRuntimeResourcesOutput,
        mlrun.api.schemas.GroupedByProjectRuntimeResourcesOutput,
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
            runtime_handler = mlrun.runtimes.get_runtime_handler(kind)
            resources = runtime_handler.build_output_from_runtime_resources(
                runtime_resources_list, group_by
            )
            if group_by is None:
                runtimes_resources_output.append(
                    mlrun.api.schemas.KindRuntimeResources(
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
        grace_period: int = mlrun.config.config.runtime_resources_deletion_grace_period,
    ):
        kinds = mlrun.runtimes.RuntimeKinds.runtime_with_handlers()
        if kind is not None:
            self.validate_runtime_resources_kind(kind)
            kinds = [kind]
        for kind in kinds:
            runtime_handler = mlrun.runtimes.get_runtime_handler(kind)
            if object_id:
                runtime_handler.delete_runtime_object_resources(
                    mlrun.api.utils.singletons.db.get_db(),
                    db_session,
                    object_id,
                    label_selector,
                    force,
                    grace_period,
                )
            else:
                runtime_handler.delete_resources(
                    mlrun.api.utils.singletons.db.get_db(),
                    db_session,
                    label_selector,
                    force,
                    grace_period,
                )
