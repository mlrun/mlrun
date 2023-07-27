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

import datetime
from typing import List, Optional, Union

from dependency_injector import containers, providers

import mlrun.api.crud
import mlrun.common.schemas
import mlrun.db.factory
import mlrun.model_monitoring.model_endpoint
from mlrun.api.db.base import DBError
from mlrun.api.db.sqldb.db import SQLDB
from mlrun.common.db.sql_session import create_session
from mlrun.db import RunDBInterface

# This class is a proxy for the real implementation that sits under mlrun.api.db.sqldb
# The runtime objects (which manages the resources that do the real logic, like Nuclio functions, Dask jobs, etc...)
# require a RunDB to manage their state, when a user run them locally this db will be a remote httpdb.
# When the user submits something to run (task, function etc...) this runtime managers actually runs inside the api
# service, in order to prevent the api from calling itself several times for each submission request (since the runDB
# will be httpdb to that same api service) we have this class which is kind of a proxy between the RunDB interface to
# the api service's DB interface


class SQLRunDB(RunDBInterface):
    def __init__(
        self,
        dsn,
        session=None,
    ):
        self.session = session
        self.dsn = dsn
        self.db = None

    def connect(self, secrets=None):
        if not self.session:
            self.session = create_session()
        self.db = SQLDB(self.dsn)
        return self

    def store_log(self, uid, project="", body=b"", append=False):
        return self._transform_db_error(
            mlrun.api.crud.Logs().store_log,
            body,
            project,
            uid,
            append,
        )

    def get_log(self, uid, project="", offset=0, size=0):
        # TODO: this is method which is not being called through the API (only through the SDK), but due to changes in
        #  the API we changed the get_log method to async so we cannot call it here, and in this PR we won't change the
        #  SDK to run async, we will use the legacy method for now, and later when we will have a better solution
        #  we will change it.
        raise NotImplementedError(
            "This should be changed to async call, if you are running in the API, use `crud.get_log`"
            " method directly instead and not through the get_db().get_log() method"
            "This will be removed in 1.5.0",
        )

    def store_run(self, struct, uid, project="", iter=0):
        return self._transform_db_error(
            mlrun.api.crud.Runs().store_run,
            self.session,
            struct,
            uid,
            iter,
            project,
        )

    def update_run(self, updates: dict, uid, project="", iter=0):
        return self._transform_db_error(
            mlrun.api.crud.Runs().update_run,
            self.session,
            project,
            uid,
            iter,
            updates,
        )

    def abort_run(self, uid, project="", iter=0, timeout=45):
        raise NotImplementedError()

    def read_run(self, uid, project=None, iter=None):
        return self._transform_db_error(
            mlrun.api.crud.Runs().get_run,
            self.session,
            uid,
            iter,
            project,
        )

    def list_runs(
        self,
        name=None,
        uid: Optional[Union[str, List[str]]] = None,
        project=None,
        labels=None,
        state=None,
        sort=True,
        last=0,
        iter=None,
        start_time_from: datetime.datetime = None,
        start_time_to: datetime.datetime = None,
        last_update_time_from: datetime.datetime = None,
        last_update_time_to: datetime.datetime = None,
        partition_by: Union[mlrun.common.schemas.RunPartitionByField, str] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Union[mlrun.common.schemas.SortField, str] = None,
        partition_order: Union[
            mlrun.common.schemas.OrderType, str
        ] = mlrun.common.schemas.OrderType.desc,
        max_partitions: int = 0,
        with_notifications: bool = False,
    ):
        return self._transform_db_error(
            mlrun.api.crud.Runs().list_runs,
            self.session,
            name,
            uid,
            project,
            labels,
            mlrun.utils.helpers.as_list(state) if state is not None else None,
            sort,
            last,
            iter,
            start_time_from,
            start_time_to,
            last_update_time_from,
            last_update_time_to,
            partition_by,
            rows_per_partition,
            partition_sort_by,
            partition_order,
            max_partitions,
            with_notifications,
        )

    def del_run(self, uid, project=None, iter=None):
        return self._transform_db_error(
            mlrun.api.crud.Runs().delete_run,
            self.session,
            uid,
            iter,
            project,
        )

    def del_runs(self, name=None, project=None, labels=None, state=None, days_ago=0):
        return self._transform_db_error(
            mlrun.api.crud.Runs().delete_runs,
            self.session,
            name,
            project,
            labels,
            state,
            days_ago,
        )

    def store_artifact(self, key, artifact, uid, iter=None, tag="", project=""):
        return self._transform_db_error(
            mlrun.api.crud.Artifacts().store_artifact,
            self.session,
            key,
            artifact,
            uid,
            iter,
            tag,
            project,
        )

    def read_artifact(self, key, tag="", iter=None, project=""):
        return self._transform_db_error(
            mlrun.api.crud.Artifacts().get_artifact,
            self.session,
            key,
            tag,
            iter,
            project,
        )

    def list_artifacts(
        self,
        name=None,
        project=None,
        tag=None,
        labels=None,
        since=None,
        until=None,
        iter: int = None,
        best_iteration: bool = False,
        kind: str = None,
        category: Union[str, mlrun.common.schemas.ArtifactCategories] = None,
    ):
        if category and isinstance(category, str):
            category = mlrun.common.schemas.ArtifactCategories(category)

        return self._transform_db_error(
            mlrun.api.crud.Artifacts().list_artifacts,
            self.session,
            project,
            name,
            tag,
            labels,
            since,
            until,
            iter=iter,
            best_iteration=best_iteration,
            kind=kind,
            category=category,
        )

    def del_artifact(self, key, tag="", project=""):
        return self._transform_db_error(
            mlrun.api.crud.Artifacts().delete_artifact,
            self.session,
            key,
            tag,
            project,
        )

    def del_artifacts(self, name="", project="", tag="", labels=None):
        return self._transform_db_error(
            mlrun.api.crud.Artifacts().delete_artifacts,
            self.session,
            project,
            name,
            tag,
            labels,
        )

    def store_function(self, function, name, project="", tag="", versioned=False):
        return self._transform_db_error(
            mlrun.api.crud.Functions().store_function,
            self.session,
            function,
            name,
            project,
            tag,
            versioned,
        )

    def get_function(self, name, project="", tag="", hash_key=""):
        return self._transform_db_error(
            mlrun.api.crud.Functions().get_function,
            self.session,
            name,
            project,
            tag,
            hash_key,
        )

    def delete_function(self, name: str, project: str = ""):
        return self._transform_db_error(
            mlrun.api.crud.Functions().delete_function,
            self.session,
            project,
            name,
        )

    def list_functions(self, name=None, project=None, tag=None, labels=None):
        return self._transform_db_error(
            mlrun.api.crud.Functions().list_functions,
            db_session=self.session,
            project=project,
            name=name,
            tag=tag,
            labels=labels,
        )

    def list_artifact_tags(
        self,
        project=None,
        category: Union[str, mlrun.common.schemas.ArtifactCategories] = None,
    ):
        return self._transform_db_error(
            self.db.list_artifact_tags, self.session, project
        )

    def tag_objects(
        self,
        project: str,
        tag_name: str,
        tag_objects: mlrun.common.schemas.TagObjects,
        replace: bool = False,
    ):
        if replace:
            return self._transform_db_error(
                mlrun.api.crud.Tags().overwrite_object_tags_with_tag,
                self.session,
                project,
                tag_name,
                tag_objects,
            )

        return self._transform_db_error(
            mlrun.api.crud.Tags().append_tag_to_objects,
            self.session,
            project,
            tag_name,
            tag_objects,
        )

    def delete_objects_tag(
        self,
        project: str,
        tag_name: str,
        tag_objects: mlrun.common.schemas.TagObjects,
    ):
        return self._transform_db_error(
            mlrun.api.crud.Tags().delete_tag_from_objects,
            self.session,
            project,
            tag_name,
            tag_objects,
        )

    def tag_artifacts(
        self,
        artifacts,
        project: str,
        tag_name: str,
        replace: bool = False,
    ):
        tag_objects = self._resolve_artifacts_to_tag_objects(artifacts)

        return self._transform_db_error(
            self.db.tag_objects, project, tag_name, tag_objects, replace
        )

    def delete_artifacts_tags(
        self,
        artifacts,
        project: str,
        tag_name: str,
    ):
        tag_objects = self._resolve_artifacts_to_tag_objects(artifacts)

        return self._transform_db_error(
            self.db.delete_objects_tag,
            project,
            tag_name,
            tag_objects,
        )

    def store_schedule(self, data):
        return self._transform_db_error(self.db.store_schedule, self.session, data)

    def list_schedules(self):
        return self._transform_db_error(self.db.list_schedules, self.session)

    def store_project(
        self,
        name: str,
        project: mlrun.common.schemas.Project,
    ) -> mlrun.common.schemas.Project:
        if isinstance(project, dict):
            project = mlrun.common.schemas.Project(**project)

        return self._transform_db_error(
            mlrun.api.crud.Projects().store_project,
            self.session,
            name=name,
            project=project,
        )

    def patch_project(
        self,
        name: str,
        project: dict,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
    ) -> mlrun.common.schemas.Project:
        return self._transform_db_error(
            mlrun.api.crud.Projects().patch_project,
            self.session,
            name=name,
            project=project,
            patch_mode=patch_mode,
        )

    def create_project(
        self,
        project: mlrun.common.schemas.Project,
    ) -> mlrun.common.schemas.Project:
        return self._transform_db_error(
            mlrun.api.crud.Projects().create_project,
            self.session,
            project=project,
        )

    def delete_project(
        self,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
    ):
        return self._transform_db_error(
            mlrun.api.crud.Projects().delete_project,
            self.session,
            name=name,
            deletion_strategy=deletion_strategy,
        )

    def get_project(
        self, name: str = None, project_id: int = None
    ) -> mlrun.common.schemas.Project:
        return self._transform_db_error(
            mlrun.api.crud.Projects().get_project,
            self.session,
            name=name,
        )

    def list_projects(
        self,
        owner: str = None,
        format_: mlrun.common.schemas.ProjectsFormat = mlrun.common.schemas.ProjectsFormat.full,
        labels: List[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        return self._transform_db_error(
            mlrun.api.crud.Projects().list_projects,
            self.session,
            owner=owner,
            format_=format_,
            labels=labels,
            state=state,
        )

    @staticmethod
    def _transform_db_error(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DBError as exc:
            raise mlrun.db.RunDBError(exc.args)

    def create_feature_set(self, feature_set, project="", versioned=True):
        return self._transform_db_error(
            mlrun.api.crud.FeatureStore().create_feature_set,
            self.session,
            project,
            feature_set,
            versioned,
        )

    def get_feature_set(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ):
        feature_set = self._transform_db_error(
            mlrun.api.crud.FeatureStore().get_feature_set,
            self.session,
            project,
            name,
            tag,
            uid,
        )
        return feature_set.dict()

    def list_features(
        self,
        project: str,
        name: str = None,
        tag: str = None,
        entities: List[str] = None,
        labels: List[str] = None,
    ):
        return self._transform_db_error(
            mlrun.api.crud.FeatureStore().list_features,
            self.session,
            project,
            name,
            tag,
            entities,
            labels,
        )

    def list_entities(
        self,
        project: str,
        name: str = None,
        tag: str = None,
        labels: List[str] = None,
    ):
        return self._transform_db_error(
            mlrun.api.crud.FeatureStore().list_entities,
            self.session,
            project,
            name,
            tag,
            labels,
        )

    def list_feature_sets(
        self,
        project: str = "",
        name: str = None,
        tag: str = None,
        state: str = None,
        entities: List[str] = None,
        features: List[str] = None,
        labels: List[str] = None,
        partition_by: mlrun.common.schemas.FeatureStorePartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: mlrun.common.schemas.SortField = None,
        partition_order: mlrun.common.schemas.OrderType = mlrun.common.schemas.OrderType.desc,
    ):
        return self._transform_db_error(
            mlrun.api.crud.FeatureStore().list_feature_sets,
            self.session,
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

    def store_feature_set(
        self,
        feature_set: Union[dict, mlrun.common.schemas.FeatureSet],
        name=None,
        project="",
        tag=None,
        uid=None,
        versioned=True,
    ):
        if isinstance(feature_set, dict):
            feature_set = mlrun.common.schemas.FeatureSet(**feature_set)

        name = name or feature_set.metadata.name
        project = project or feature_set.metadata.project
        return self._transform_db_error(
            mlrun.api.crud.FeatureStore().store_feature_set,
            self.session,
            project,
            name,
            feature_set,
            tag,
            uid,
            versioned,
        )

    def patch_feature_set(
        self, name, feature_set, project="", tag=None, uid=None, patch_mode="replace"
    ):
        return self._transform_db_error(
            mlrun.api.crud.FeatureStore().patch_feature_set,
            self.session,
            project,
            name,
            feature_set,
            tag,
            uid,
            patch_mode,
        )

    def delete_feature_set(self, name, project="", tag=None, uid=None):
        return self._transform_db_error(
            mlrun.api.crud.FeatureStore().delete_feature_set,
            self.session,
            project,
            name,
            tag,
            uid,
        )

    def create_feature_vector(self, feature_vector, project="", versioned=True):
        return self._transform_db_error(
            mlrun.api.crud.FeatureStore().create_feature_vector,
            self.session,
            project,
            feature_vector,
            versioned,
        )

    def get_feature_vector(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ):
        return self._transform_db_error(
            mlrun.api.crud.FeatureStore().get_feature_vector,
            self.session,
            project,
            name,
            tag,
            uid,
        )

    def list_feature_vectors(
        self,
        project: str = "",
        name: str = None,
        tag: str = None,
        state: str = None,
        labels: List[str] = None,
        partition_by: mlrun.common.schemas.FeatureStorePartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: mlrun.common.schemas.SortField = None,
        partition_order: mlrun.common.schemas.OrderType = mlrun.common.schemas.OrderType.desc,
    ):
        return self._transform_db_error(
            mlrun.api.crud.FeatureStore().list_feature_vectors,
            self.session,
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

    def store_feature_vector(
        self,
        feature_vector,
        name=None,
        project="",
        tag=None,
        uid=None,
        versioned=True,
    ):
        return self._transform_db_error(
            mlrun.api.crud.FeatureStore().store_feature_vector,
            self.session,
            project,
            name,
            feature_vector,
            tag,
            uid,
            versioned,
        )

    def patch_feature_vector(
        self,
        name,
        feature_vector_update: dict,
        project="",
        tag=None,
        uid=None,
        patch_mode="replace",
    ):
        return self._transform_db_error(
            mlrun.api.crud.FeatureStore().patch_feature_vector,
            self.session,
            project,
            name,
            feature_vector_update,
            tag,
            uid,
            patch_mode,
        )

    def delete_feature_vector(self, name, project="", tag=None, uid=None):
        return self._transform_db_error(
            mlrun.api.crud.FeatureStore().delete_feature_vector,
            self.session,
            project,
            name,
            tag,
            uid,
        )

    def list_pipelines(
        self,
        project: str,
        namespace: str = None,
        sort_by: str = "",
        page_token: str = "",
        filter_: str = "",
        format_: Union[
            str, mlrun.common.schemas.PipelinesFormat
        ] = mlrun.common.schemas.PipelinesFormat.metadata_only,
        page_size: int = None,
    ) -> mlrun.common.schemas.PipelinesOutput:
        raise NotImplementedError()

    def create_project_secrets(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: dict = None,
    ):
        raise NotImplementedError()

    def list_project_secrets(
        self,
        project: str,
        token: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: List[str] = None,
    ) -> mlrun.common.schemas.SecretsData:
        raise NotImplementedError()

    def list_project_secret_keys(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        token: str = None,
    ) -> mlrun.common.schemas.SecretKeysData:
        raise NotImplementedError()

    def delete_project_secrets(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: List[str] = None,
    ):
        raise NotImplementedError()

    def create_user_secrets(
        self,
        user: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.vault,
        secrets: dict = None,
    ):
        raise NotImplementedError()

    def create_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        model_endpoint: Union[
            mlrun.model_monitoring.model_endpoint.ModelEndpoint, dict
        ],
    ):
        raise NotImplementedError()

    def delete_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
    ):
        raise NotImplementedError()

    def list_model_endpoints(
        self,
        project: str,
        model: Optional[str] = None,
        function: Optional[str] = None,
        labels: List[str] = None,
        start: str = "now-1h",
        end: str = "now",
        metrics: Optional[List[str]] = None,
    ):
        raise NotImplementedError()

    def get_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        features: bool = False,
    ):
        raise NotImplementedError()

    def patch_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        attributes: dict,
    ):
        raise NotImplementedError()

    def create_hub_source(
        self, source: Union[dict, mlrun.common.schemas.IndexedHubSource]
    ):
        raise NotImplementedError()

    def store_hub_source(
        self,
        source_name: str,
        source: Union[dict, mlrun.common.schemas.IndexedHubSource],
    ):
        raise NotImplementedError()

    def list_hub_sources(self):
        raise NotImplementedError()

    def get_hub_source(self, source_name: str):
        raise NotImplementedError()

    def delete_hub_source(self, source_name: str):
        raise NotImplementedError()

    def get_hub_catalog(
        self,
        source_name: str,
        version: str = None,
        tag: str = None,
        force_refresh: bool = False,
    ):
        raise NotImplementedError()

    def get_hub_item(
        self,
        source_name: str,
        item_name: str,
        version: str = None,
        tag: str = "latest",
        force_refresh: bool = False,
    ):
        raise NotImplementedError()

    def verify_authorization(
        self,
        authorization_verification_input: mlrun.common.schemas.AuthorizationVerificationInput,
    ):
        # on server side authorization is done in endpoint anyway, so for server side we can "pass" on check
        # done from ingest()
        pass

    def watch_log(self, uid, project="", watch=True, offset=0):
        raise NotImplementedError("Watching logs is not supported on the server")


# Once this file is imported it will override the default RunDB implementation (RunDBContainer)
@containers.override(mlrun.db.factory.RunDBContainer)
class SQLRunDBContainer(containers.DeclarativeContainer):
    run_db = providers.Factory(SQLRunDB)
