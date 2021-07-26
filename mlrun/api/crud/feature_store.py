import typing

import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.clients.opa
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.project_member
import mlrun.config
import mlrun.errors
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
        return self._create_object(
            db_session, project, feature_set, versioned, auth_info
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
        return self._store_object(
            db_session, project, name, feature_set, tag, uid, versioned, auth_info,
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
        return self._patch_object(
            db_session,
            mlrun.api.schemas.FeatureSet,
            project,
            name,
            feature_set_patch,
            tag,
            uid,
            patch_mode,
            auth_info,
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
        return self._get_object(
            db_session, mlrun.api.schemas.FeatureSet, project, name, tag, uid, auth_info
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
        project = project or mlrun.mlconf.default_project
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
                feature_set.metadata.project,
                feature_set.metadata.name,
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
        self._delete_object(
            db_session,
            mlrun.api.schemas.FeatureSet,
            project,
            name,
            tag,
            uid,
            auth_info,
        )

    def list_features(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        tag: typing.Optional[str] = None,
        entities: typing.List[str] = None,
        labels: typing.List[str] = None,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> mlrun.api.schemas.FeaturesOutput:
        project = project or mlrun.mlconf.default_project
        features = mlrun.api.utils.singletons.db.get_db().list_features(
            db_session, project, name, tag, entities, labels,
        )
        features = mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.feature,
            features.features,
            lambda feature_list_output: (
                feature_list_output.feature.name,
                feature_list_output.feature_set_digest.metadata.project,
            ),
            auth_info,
        )
        return mlrun.api.schemas.FeaturesOutput(features=features)

    def list_entities(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        tag: typing.Optional[str] = None,
        labels: typing.List[str] = None,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> mlrun.api.schemas.EntitiesOutput:
        project = project or mlrun.mlconf.default_project
        entities = mlrun.api.utils.singletons.db.get_db().list_entities(
            db_session, project, name, tag, labels,
        )
        entities = mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.entity,
            entities.entities,
            lambda entity_list_output: (
                entity_list_output.entity.name,
                entity_list_output.feature_set_digest.metadata.project,
            ),
            auth_info,
        )
        return mlrun.api.schemas.EntitiesOutput(entities=entities)

    def create_feature_vector(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        feature_vector: mlrun.api.schemas.FeatureVector,
        versioned: bool = True,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> str:
        return self._create_object(
            db_session, project, feature_vector, versioned, auth_info
        )

    def store_feature_vector(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        feature_vector: mlrun.api.schemas.FeatureVector,
        tag: typing.Optional[str] = None,
        uid: typing.Optional[str] = None,
        versioned: bool = True,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> str:
        return self._store_object(
            db_session, project, name, feature_vector, tag, uid, versioned, auth_info,
        )

    def patch_feature_vector(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        feature_vector_patch: dict,
        tag: typing.Optional[str] = None,
        uid: typing.Optional[str] = None,
        patch_mode: mlrun.api.schemas.PatchMode = mlrun.api.schemas.PatchMode.replace,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> str:
        return self._patch_object(
            db_session,
            mlrun.api.schemas.FeatureVector,
            project,
            name,
            feature_vector_patch,
            tag,
            uid,
            patch_mode,
            auth_info,
        )

    def get_feature_vector(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        tag: typing.Optional[str] = None,
        uid: typing.Optional[str] = None,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> mlrun.api.schemas.FeatureVector:
        return self._get_object(
            db_session,
            mlrun.api.schemas.FeatureVector,
            project,
            name,
            tag,
            uid,
            auth_info,
        )

    def list_feature_vectors(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        tag: typing.Optional[str] = None,
        state: str = None,
        labels: typing.List[str] = None,
        partition_by: mlrun.api.schemas.FeatureStorePartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort: mlrun.api.schemas.SortField = None,
        partition_order: mlrun.api.schemas.OrderType = mlrun.api.schemas.OrderType.desc,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> mlrun.api.schemas.FeatureVectorsOutput:
        project = project or mlrun.mlconf.default_project
        feature_vectors = mlrun.api.utils.singletons.db.get_db().list_feature_vectors(
            db_session,
            project,
            name,
            tag,
            state,
            labels,
            partition_by,
            rows_per_partition,
            partition_sort,
            partition_order,
        )
        feature_vectors = mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.feature_vector,
            feature_vectors.feature_vectors,
            lambda feature_vector: (
                feature_vector.metadata.project,
                feature_vector.metadata.name,
            ),
            auth_info,
        )
        return mlrun.api.schemas.FeatureVectorsOutput(feature_vectors=feature_vectors)

    def delete_feature_vector(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        tag: typing.Optional[str] = None,
        uid: typing.Optional[str] = None,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ):
        self._delete_object(
            db_session,
            mlrun.api.schemas.FeatureVector,
            project,
            name,
            tag,
            uid,
            auth_info,
        )

    def _create_object(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        object_: typing.Union[
            mlrun.api.schemas.FeatureSet, mlrun.api.schemas.FeatureVector
        ],
        versioned: bool = True,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> str:
        project = project or mlrun.mlconf.default_project
        self._validate_and_enrich_identity_for_object_creation(project, object_)
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project(
            db_session, project, leader_session=auth_info.session
        )
        mlrun.api.utils.clients.opa.Client().query_resource_permissions(
            object_.get_authorization_resource_type(),
            project,
            object_.metadata.name,
            mlrun.api.schemas.AuthorizationAction.create,
            auth_info,
        )
        if isinstance(object_, mlrun.api.schemas.FeatureSet):
            return mlrun.api.utils.singletons.db.get_db().create_feature_set(
                db_session, project, object_, versioned
            )
        elif isinstance(object_, mlrun.api.schemas.FeatureVector):
            return mlrun.api.utils.singletons.db.get_db().create_feature_vector(
                db_session, project, object_, versioned
            )
        else:
            raise NotImplementedError(
                f"Provided object type is not supported. object_type={type(object_)}"
            )

    def _store_object(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        object_: typing.Union[
            mlrun.api.schemas.FeatureSet, mlrun.api.schemas.FeatureVector
        ],
        tag: typing.Optional[str] = None,
        uid: typing.Optional[str] = None,
        versioned: bool = True,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> str:
        project = project or mlrun.mlconf.default_project
        self._validate_and_enrich_identity_for_object_store(
            object_, project, name, tag, uid
        )
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project(
            db_session, project, leader_session=auth_info.session
        )
        mlrun.api.utils.clients.opa.Client().query_resource_permissions(
            object_.get_authorization_resource_type(),
            project,
            name,
            mlrun.api.schemas.AuthorizationAction.store,
            auth_info,
        )
        if isinstance(object_, mlrun.api.schemas.FeatureSet):
            return mlrun.api.utils.singletons.db.get_db().store_feature_set(
                db_session, project, name, object_, tag, uid, versioned,
            )
        elif isinstance(object_, mlrun.api.schemas.FeatureVector):
            return mlrun.api.utils.singletons.db.get_db().store_feature_vector(
                db_session, project, name, object_, tag, uid, versioned,
            )
        else:
            raise NotImplementedError(
                f"Provided object type is not supported. object_type={type(object_)}"
            )

    def _patch_object(
        self,
        db_session: sqlalchemy.orm.Session,
        object_schema: typing.ClassVar,
        project: str,
        name: str,
        object_patch: dict,
        tag: typing.Optional[str] = None,
        uid: typing.Optional[str] = None,
        patch_mode: mlrun.api.schemas.PatchMode = mlrun.api.schemas.PatchMode.replace,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> str:
        project = project or mlrun.mlconf.default_project
        self._validate_identity_for_object_patch(
            object_schema.__class__.__name__, object_patch, project, name, tag, uid,
        )
        mlrun.api.utils.clients.opa.Client().query_resource_permissions(
            object_schema.get_authorization_resource_type(),
            project,
            name,
            mlrun.api.schemas.AuthorizationAction.update,
            auth_info,
        )
        if object_schema.__name__ == mlrun.api.schemas.FeatureSet.__name__:
            return mlrun.api.utils.singletons.db.get_db().patch_feature_set(
                db_session, project, name, object_patch, tag, uid, patch_mode,
            )
        elif object_schema.__name__ == mlrun.api.schemas.FeatureVector.__name__:
            return mlrun.api.utils.singletons.db.get_db().patch_feature_vector(
                db_session, project, name, object_patch, tag, uid, patch_mode,
            )
        else:
            raise NotImplementedError(
                f"Provided object type is not supported. object_type={object_schema.__class__.__name__}"
            )

    def _get_object(
        self,
        db_session: sqlalchemy.orm.Session,
        object_schema: typing.ClassVar,
        project: str,
        name: str,
        tag: typing.Optional[str] = None,
        uid: typing.Optional[str] = None,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> typing.Union[mlrun.api.schemas.FeatureSet, mlrun.api.schemas.FeatureVector]:
        project = project or mlrun.mlconf.default_project
        mlrun.api.utils.clients.opa.Client().query_resource_permissions(
            object_schema.get_authorization_resource_type(),
            project,
            name,
            mlrun.api.schemas.AuthorizationAction.read,
            auth_info,
        )
        if object_schema.__name__ == mlrun.api.schemas.FeatureSet.__name__:
            return mlrun.api.utils.singletons.db.get_db().get_feature_set(
                db_session, project, name, tag, uid
            )
        elif object_schema.__name__ == mlrun.api.schemas.FeatureVector.__name__:
            return mlrun.api.utils.singletons.db.get_db().get_feature_vector(
                db_session, project, name, tag, uid
            )
        else:
            raise NotImplementedError(
                f"Provided object type is not supported. object_type={object_schema.__class__.__name__}"
            )

    def _delete_object(
        self,
        db_session: sqlalchemy.orm.Session,
        object_schema: typing.ClassVar,
        project: str,
        name: str,
        tag: typing.Optional[str] = None,
        uid: typing.Optional[str] = None,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ):
        project = project or mlrun.mlconf.default_project
        mlrun.api.utils.clients.opa.Client().query_resource_permissions(
            object_schema.get_authorization_resource_type(),
            project,
            name,
            mlrun.api.schemas.AuthorizationAction.delete,
            auth_info,
        )
        if object_schema.__name__ == mlrun.api.schemas.FeatureSet.__name__:
            mlrun.api.utils.singletons.db.get_db().delete_feature_set(
                db_session, project, name, tag, uid
            )
        elif object_schema.__name__ == mlrun.api.schemas.FeatureVector.__name__:
            mlrun.api.utils.singletons.db.get_db().delete_feature_vector(
                db_session, project, name, tag, uid
            )
        else:
            raise NotImplementedError(
                f"Provided object type is not supported. object_type={object_schema.__class__.__name__}"
            )

    @staticmethod
    def _validate_identity_for_object_patch(
        object_type: str,
        object_patch: dict,
        project: str,
        name: str,
        tag: str,
        uid: str,
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
    def _validate_and_enrich_identity_for_object_store(
        object_: typing.Union[
            mlrun.api.schemas.FeatureSet, mlrun.api.schemas.FeatureVector
        ],
        project: str,
        name: str,
        tag: str,
        uid: str,
    ):
        object_type = object_.__class__.__name__

        if not tag and not uid:
            raise ValueError(
                f"cannot store {object_type} without reference (tag or uid)"
            )

        object_project = object_.metadata.project
        if object_project and object_project != project:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"{object_type} object with conflicting project name - {object_project}"
            )

        object_.metadata.project = project

        if object_.metadata.name != name:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Changing name for an existing {object_type}"
            )

    @staticmethod
    def _validate_and_enrich_identity_for_object_creation(
        project: str,
        object_: typing.Union[
            mlrun.api.schemas.FeatureSet, mlrun.api.schemas.FeatureVector
        ],
    ):
        object_type = object_.__class__.__name__
        if not object_.metadata.name or not project:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"{object_type} missing name or project"
            )

        object_project = object_.metadata.project
        if object_project and object_project != project:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"{object_type} object with conflicting project name - {object_project}"
            )

        object_.metadata.project = project
        object_.metadata.tag = object_.metadata.tag or "latest"
