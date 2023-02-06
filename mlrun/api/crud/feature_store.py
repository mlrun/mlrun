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
import typing

import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.project_member
import mlrun.config
import mlrun.errors
import mlrun.utils.singleton


class FeatureStore(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def create_feature_set(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        feature_set: mlrun.api.schemas.FeatureSet,
        versioned: bool = True,
    ) -> str:
        return self._create_object(
            db_session,
            project,
            feature_set,
            versioned,
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
    ) -> str:
        return self._store_object(
            db_session,
            project,
            name,
            feature_set,
            tag,
            uid,
            versioned,
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
        )

    def get_feature_set(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        tag: typing.Optional[str] = None,
        uid: typing.Optional[str] = None,
    ) -> mlrun.api.schemas.FeatureSet:
        return self._get_object(
            db_session, mlrun.api.schemas.FeatureSet, project, name, tag, uid
        )

    def list_feature_sets_tags(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
    ) -> typing.List[typing.Tuple[str, str, str]]:
        """
        :return: a list of Tuple of (project, feature_set.name, tag)
        """
        return self._list_object_type_tags(
            db_session, mlrun.api.schemas.FeatureSet, project
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
        partition_sort_by: mlrun.api.schemas.SortField = None,
        partition_order: mlrun.api.schemas.OrderType = mlrun.api.schemas.OrderType.desc,
    ) -> mlrun.api.schemas.FeatureSetsOutput:
        project = project or mlrun.mlconf.default_project
        return mlrun.api.utils.singletons.db.get_db().list_feature_sets(
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
            partition_sort_by,
            partition_order,
        )

    def delete_feature_set(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        tag: typing.Optional[str] = None,
        uid: typing.Optional[str] = None,
    ):
        self._delete_object(
            db_session,
            mlrun.api.schemas.FeatureSet,
            project,
            name,
            tag,
            uid,
        )

    def list_features(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        tag: typing.Optional[str] = None,
        entities: typing.List[str] = None,
        labels: typing.List[str] = None,
    ) -> mlrun.api.schemas.FeaturesOutput:
        project = project or mlrun.mlconf.default_project
        return mlrun.api.utils.singletons.db.get_db().list_features(
            db_session,
            project,
            name,
            tag,
            entities,
            labels,
        )

    def list_entities(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        tag: typing.Optional[str] = None,
        labels: typing.List[str] = None,
    ) -> mlrun.api.schemas.EntitiesOutput:
        project = project or mlrun.mlconf.default_project
        return mlrun.api.utils.singletons.db.get_db().list_entities(
            db_session,
            project,
            name,
            tag,
            labels,
        )

    def create_feature_vector(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        feature_vector: mlrun.api.schemas.FeatureVector,
        versioned: bool = True,
    ) -> str:
        return self._create_object(db_session, project, feature_vector, versioned)

    def store_feature_vector(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        feature_vector: mlrun.api.schemas.FeatureVector,
        tag: typing.Optional[str] = None,
        uid: typing.Optional[str] = None,
        versioned: bool = True,
    ) -> str:
        return self._store_object(
            db_session,
            project,
            name,
            feature_vector,
            tag,
            uid,
            versioned,
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
        )

    def get_feature_vector(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        tag: typing.Optional[str] = None,
        uid: typing.Optional[str] = None,
    ) -> mlrun.api.schemas.FeatureVector:
        return self._get_object(
            db_session,
            mlrun.api.schemas.FeatureVector,
            project,
            name,
            tag,
            uid,
        )

    def list_feature_vectors_tags(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
    ) -> typing.List[typing.Tuple[str, str, str]]:
        """
        :return: a list of Tuple of (project, feature_vector.name, tag)
        """
        return self._list_object_type_tags(
            db_session, mlrun.api.schemas.FeatureVector, project
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
        partition_sort_by: mlrun.api.schemas.SortField = None,
        partition_order: mlrun.api.schemas.OrderType = mlrun.api.schemas.OrderType.desc,
    ) -> mlrun.api.schemas.FeatureVectorsOutput:
        project = project or mlrun.mlconf.default_project
        return mlrun.api.utils.singletons.db.get_db().list_feature_vectors(
            db_session,
            project,
            name,
            tag,
            state,
            labels,
            partition_by,
            rows_per_partition,
            partition_sort_by,
            partition_order,
        )

    def delete_feature_vector(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        tag: typing.Optional[str] = None,
        uid: typing.Optional[str] = None,
    ):
        self._delete_object(
            db_session,
            mlrun.api.schemas.FeatureVector,
            project,
            name,
            tag,
            uid,
        )

    def _create_object(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        object_: typing.Union[
            mlrun.api.schemas.FeatureSet, mlrun.api.schemas.FeatureVector
        ],
        versioned: bool = True,
    ) -> str:
        project = project or mlrun.mlconf.default_project
        self._validate_and_enrich_identity_for_object_creation(project, object_)
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
    ) -> str:
        project = project or mlrun.mlconf.default_project
        self._validate_and_enrich_identity_for_object_store(
            object_, project, name, tag, uid
        )
        if isinstance(object_, mlrun.api.schemas.FeatureSet):
            return mlrun.api.utils.singletons.db.get_db().store_feature_set(
                db_session,
                project,
                name,
                object_,
                tag,
                uid,
                versioned,
            )
        elif isinstance(object_, mlrun.api.schemas.FeatureVector):
            return mlrun.api.utils.singletons.db.get_db().store_feature_vector(
                db_session,
                project,
                name,
                object_,
                tag,
                uid,
                versioned,
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
    ) -> str:
        project = project or mlrun.mlconf.default_project
        self._validate_identity_for_object_patch(
            object_schema.__class__.__name__,
            object_patch,
            project,
            name,
            tag,
            uid,
        )
        if object_schema.__name__ == mlrun.api.schemas.FeatureSet.__name__:
            return mlrun.api.utils.singletons.db.get_db().patch_feature_set(
                db_session,
                project,
                name,
                object_patch,
                tag,
                uid,
                patch_mode,
            )
        elif object_schema.__name__ == mlrun.api.schemas.FeatureVector.__name__:
            return mlrun.api.utils.singletons.db.get_db().patch_feature_vector(
                db_session,
                project,
                name,
                object_patch,
                tag,
                uid,
                patch_mode,
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
    ) -> typing.Union[mlrun.api.schemas.FeatureSet, mlrun.api.schemas.FeatureVector]:
        project = project or mlrun.mlconf.default_project
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

    def _list_object_type_tags(
        self,
        db_session: sqlalchemy.orm.Session,
        object_schema: typing.ClassVar,
        project: str,
    ) -> typing.List[typing.Tuple[str, str, str]]:
        project = project or mlrun.mlconf.default_project
        if object_schema.__name__ == mlrun.api.schemas.FeatureSet.__name__:
            return mlrun.api.utils.singletons.db.get_db().list_feature_sets_tags(
                db_session, project
            )
        elif object_schema.__name__ == mlrun.api.schemas.FeatureVector.__name__:
            return mlrun.api.utils.singletons.db.get_db().list_feature_vectors_tags(
                db_session, project
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
    ):
        project = project or mlrun.mlconf.default_project
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
