import http
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


class Runtimes(metaclass=mlrun.utils.singleton.Singleton,):
    def list_runtimes(
        self,
        project: str,
        label_selector: str = None,
        group_by: typing.Optional[
            mlrun.api.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ) -> typing.Union[typing.Dict, mlrun.api.schemas.GroupedRuntimeResourcesOutput]:
        runtimes = [] if group_by is None else {}
        for kind in mlrun.runtimes.RuntimeKinds.runtime_with_handlers():
            runtime_handler = mlrun.runtimes.get_runtime_handler(kind)
            resources = runtime_handler.list_resources(
                project, label_selector, group_by
            )
            if group_by is None:
                runtimes.append({"kind": kind, "resources": resources})
            else:
                mergedeep.merge(runtimes, resources)
        return runtimes

    def get_runtime(self, kind: str, label_selector: str = None):
        if kind not in mlrun.runtimes.RuntimeKinds.runtime_with_handlers():
            mlrun.api.api.utils.log_and_raise(
                http.HTTPStatus.BAD_REQUEST.value, kind=kind, err="Invalid runtime kind"
            )
        runtime_handler = mlrun.runtimes.get_runtime_handler(kind)
        resources = runtime_handler.list_resources("*", label_selector)
        return {
            "kind": kind,
            "resources": resources,
        }

    def delete_runtimes(
        self,
        db_session: sqlalchemy.orm.Session,
        label_selector: str = None,
        force: bool = False,
        grace_period: int = mlrun.config.config.runtime_resources_deletion_grace_period,
        leader_session: typing.Optional[str] = None,
    ):
        for kind in mlrun.runtimes.RuntimeKinds.runtime_with_handlers():
            runtime_handler = mlrun.runtimes.get_runtime_handler(kind)
            runtime_handler.delete_resources(
                mlrun.api.utils.singletons.db.get_db(),
                db_session,
                label_selector,
                force,
                grace_period,
                leader_session,
            )

    def delete_runtime(
        self,
        db_session: sqlalchemy.orm.Session,
        kind: str,
        label_selector: str = None,
        force: bool = False,
        grace_period: int = mlrun.config.config.runtime_resources_deletion_grace_period,
        leader_session: typing.Optional[str] = None,
    ):
        if kind not in mlrun.runtimes.RuntimeKinds.runtime_with_handlers():
            mlrun.api.api.utils.log_and_raise(
                http.HTTPStatus.BAD_REQUEST.value, kind=kind, err="Invalid runtime kind"
            )
        runtime_handler = mlrun.runtimes.get_runtime_handler(kind)
        runtime_handler.delete_resources(
            mlrun.api.utils.singletons.db.get_db(),
            db_session,
            label_selector,
            force,
            grace_period,
            leader_session,
        )

    def delete_runtime_object(
        self,
        db_session: sqlalchemy.orm.Session,
        kind: str,
        object_id: str,
        label_selector: str = None,
        force: bool = False,
        grace_period: int = mlrun.config.config.runtime_resources_deletion_grace_period,
        leader_session: typing.Optional[str] = None,
    ):
        if kind not in mlrun.runtimes.RuntimeKinds.runtime_with_handlers():
            mlrun.api.api.utils.log_and_raise(
                http.HTTPStatus.BAD_REQUEST.value, kind=kind, err="Invalid runtime kind"
            )
        runtime_handler = mlrun.runtimes.get_runtime_handler(kind)
        runtime_handler.delete_runtime_object_resources(
            mlrun.api.utils.singletons.db.get_db(),
            db_session,
            object_id,
            label_selector,
            force,
            grace_period,
            leader_session,
        )
