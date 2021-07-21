import typing

import sqlalchemy.orm

import mlrun.api.api.utils
import mlrun.api.schemas
import mlrun.api.utils.clients.opa
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.project_member
import mlrun.config
import mlrun.errors
import mlrun.runtimes
import mlrun.utils.singleton


class FeatureStore(metaclass=mlrun.utils.singleton.Singleton,):
    def create_feature_set(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        feature_set: mlrun.api.schemas.FeatureSet,
        versioned: bool = True,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> str:
        self._validate_and_enrich_identity_for_object_creation(project, feature_set)
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project(
            db_session, project, leader_session=auth_info.session
        )
        mlrun.api.utils.clients.opa.Client().query_resource_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.feature_set,
            project,
            feature_set.metadata.name,
            mlrun.api.schemas.AuthorizationAction.create,
            auth_info,
        )
        return mlrun.api.utils.singletons.db.get_db().create_feature_set(
            db_session, project, feature_set, versioned
        )

    def store_feature_set(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        feature_set: mlrun.api.schemas.FeatureSet,
        tag: typing.Optional[str] = None,
        uid: typing.Optional[str] = None,
        versioned: bool = True,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> str:
        self._validate_and_enrich_identity_for_object_store(
            mlrun.api.schemas.FeatureSet, project, name, tag, uid
        )
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project(
            db_session, project, leader_session=auth_info.session
        )
        mlrun.api.utils.clients.opa.Client().query_resource_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.feature_set,
            project,
            name,
            mlrun.api.schemas.AuthorizationAction.store,
            auth_info,
        )
        return mlrun.api.utils.singletons.db.get_db().store_feature_set(
            db_session, project, name, feature_set, tag, uid, versioned,
        )

    def patch_feature_set(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        feature_set_patch: dict,
        tag: typing.Optional[str] = None,
        uid: typing.Optional[str] = None,
        patch_mode: mlrun.api.schemas.PatchMode = mlrun.api.schemas.PatchMode.replace,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> str:
        self._validate_identity_for_object_patch(
            mlrun.api.schemas.FeatureSet.__class__.__name__,
            feature_set_patch,
            project,
            name,
            tag,
            uid,
        )
        mlrun.api.utils.clients.opa.Client().query_resource_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.feature_set,
            project,
            name,
            mlrun.api.schemas.AuthorizationAction.update,
            auth_info,
        )
        return mlrun.api.utils.singletons.db.get_db().patch_feature_set(
            db_session, project, name, feature_set_patch, tag, uid, patch_mode,
        )

    def get_feature_set(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        tag: typing.Optional[str] = None,
        uid: typing.Optional[str] = None,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> mlrun.api.schemas.FeatureSet:
        mlrun.api.utils.clients.opa.Client().query_resource_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.feature_set,
            project,
            name,
            mlrun.api.schemas.AuthorizationAction.read,
            auth_info,
        )
        return mlrun.api.utils.singletons.db.get_db().get_feature_set(
            db_session, project, name, tag, uid
        )

    def list_feature_sets(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        tag: typing.Optional[str] = None,
        state: str = None,
        entities: typing.List[str] = None,
        features: typing.List[str] = None,
        labels: typing.List[str] = None,
        partition_by: mlrun.api.schemas.FeatureStorePartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort: mlrun.api.schemas.SortField = None,
        partition_order: mlrun.api.schemas.OrderType = mlrun.api.schemas.OrderType.desc,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> mlrun.api.schemas.FeatureSetsOutput:
        feature_sets = mlrun.api.utils.singletons.db.get_db().list_feature_sets(
            db_session,
            project,
            name,
            tag,
            state,
            entities,
            features,
            labels,
            partition_by,
            rows_per_partition,
            partition_sort,
            partition_order,
        )
        feature_sets = mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.feature_set,
            feature_sets.feature_sets,
            lambda feature_set: (
                feature_sets.metadata.project,
                feature_sets.metadata.name,
            ),
            auth_info,
        )
        return mlrun.api.schemas.FeatureSetsOutput(feature_sets=feature_sets)

    def delete_feature_set(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        tag: typing.Optional[str] = None,
        uid: typing.Optional[str] = None,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ):
        mlrun.api.utils.clients.opa.Client().query_resource_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.feature_set,
            project,
            name,
            mlrun.api.schemas.AuthorizationAction.delete,
            auth_info,
        )
        return mlrun.api.utils.singletons.db.get_db().delete_feature_set(
            db_session, project, name, tag, uid
        )

    @staticmethod
    def _validate_identity_for_object_patch(
        object_type, object_patch, project, name, tag, uid
    ):
        if not tag and not uid:
            raise ValueError(
                f"cannot store {object_type} without reference (tag or uid)"
            )

        object_project = object_patch.get("metadata", {}).get("project")
        if object_project and object_project != project:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"{object_type} object with conflicting project name - {object_project}"
            )

        object_name = object_patch.get("metadata", {}).get("name")
        if object_name and object_name != name:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Changing name for an existing {object_type}"
            )

    @staticmethod
    def _validate_and_enrich_identity_for_object_store(object, project, name, tag, uid):
        object_type = object.__class__.__name__

        if not tag and not uid:
            raise ValueError(
                f"cannot store {object_type} without reference (tag or uid)"
            )

        object_project = object.metadata.project
        if object_project and object_project != project:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"{object_type} object with conflicting project name - {object_project}"
            )

        object.metadata.project = project

        if object.metadata.name != name:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Changing name for an existing {object_type}"
            )

    @staticmethod
    def _validate_and_enrich_identity_for_object_creation(
        project: str, object: typing.Union[mlrun.api.schemas.FeatureSet],
    ):
        object_type = object.__class__.__name__
        if not object.metadata.name or not project:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"{object_type} missing name or project"
            )

        object_project = object.metadata.project
        if object_project and object_project != project:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"{object_type} object with conflicting project name - {object_project}"
            )

        object.metadata.project = project
        object.metadata.tag = object.metadata.tag or "latest"
