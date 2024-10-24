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
from typing import Optional, Union

from dependency_injector import containers, providers
from sqlalchemy.exc import SQLAlchemyError

import mlrun.alerts
import mlrun.common.formatters
import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.common.schemas.artifact
import mlrun.db.factory
import mlrun.model_monitoring.model_endpoint
import server.api.crud
import server.api.db.session
from mlrun.common.db.sql_session import create_session
from mlrun.db import RunDBInterface
from server.api.db.base import DBError
from server.api.db.sqldb.db import SQLDB

# This class is a proxy for the real implementation that sits under server.api.db.sqldb
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
            server.api.crud.Logs().store_log,
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
            "This should be changed to async call, if you are running in the API, use `server.api.crud.get_log`"
            " method directly instead and not through the get_db().get_log() method. "
            "This will be removed in 1.5.0",
        )

    def store_run(self, struct, uid, project="", iter=0):
        return self._transform_db_error(
            server.api.crud.Runs().store_run,
            self.session,
            struct,
            uid,
            iter,
            project,
        )

    def update_run(self, updates: dict, uid, project="", iter=0):
        return self._transform_db_error(
            server.api.crud.Runs().update_run,
            self.session,
            project,
            uid,
            iter,
            updates,
        )

    def abort_run(self, uid, project="", iter=0, timeout=45, status_text=""):
        raise NotImplementedError()

    def read_run(
        self,
        uid: str,
        project: str = None,
        iter: int = None,
        format_: mlrun.common.formatters.RunFormat = mlrun.common.formatters.RunFormat.full,
    ):
        return self._transform_db_error(
            server.api.crud.Runs().get_run,
            self.session,
            uid,
            iter,
            project,
            format_,
        )

    def list_runs(
        self,
        name: Optional[str] = None,
        uid: Optional[Union[str, list[str]]] = None,
        project: Optional[str] = None,
        labels: Optional[Union[str, list[str]]] = None,
        state: Optional[mlrun.common.runtimes.constants.RunStates] = None,
        states: Optional[list[mlrun.common.runtimes.constants.RunStates]] = None,
        sort: bool = True,
        last: int = 0,
        iter: bool = False,
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
            server.api.db.session.run_function_with_new_db_session,
            server.api.crud.Runs().list_runs,
            name=name,
            uid=uid,
            project=project,
            labels=labels,
            states=mlrun.utils.helpers.as_list(state)
            if state is not None
            else states or None,
            sort=sort,
            last=last,
            iter=iter,
            start_time_from=start_time_from,
            start_time_to=start_time_to,
            last_update_time_from=last_update_time_from,
            last_update_time_to=last_update_time_to,
            partition_by=partition_by,
            rows_per_partition=rows_per_partition,
            partition_sort_by=partition_sort_by,
            partition_order=partition_order,
            max_partitions=max_partitions,
            with_notifications=with_notifications,
        )

    async def del_run(self, uid, project=None, iter=None):
        return await self._transform_db_error(
            server.api.crud.Runs().delete_run,
            self.session,
            uid,
            iter,
            project,
        )

    async def del_runs(
        self, name=None, project=None, labels=None, state=None, days_ago=0
    ):
        return await self._transform_db_error(
            server.api.crud.Runs().delete_runs,
            self.session,
            name,
            project,
            labels,
            state,
            days_ago,
        )

    def store_artifact(
        self, key, artifact, uid=None, iter=None, tag="", project="", tree=None
    ):
        return self._transform_db_error(
            server.api.crud.Artifacts().store_artifact,
            self.session,
            key,
            artifact,
            uid,
            iter,
            tag,
            project,
            tree,
        )

    def read_artifact(
        self,
        key,
        tag="",
        iter=None,
        project="",
        tree=None,
        uid=None,
        format_: mlrun.common.formatters.ArtifactFormat = mlrun.common.formatters.ArtifactFormat.full,
    ):
        return self._transform_db_error(
            server.api.crud.Artifacts().get_artifact,
            self.session,
            key,
            tag=tag,
            iter=iter,
            project=project,
            producer_id=tree,
            object_uid=uid,
            format_=format_,
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
        tree: str = None,
        format_: mlrun.common.formatters.ArtifactFormat = mlrun.common.formatters.ArtifactFormat.full,
        limit: int = None,
    ):
        if category and isinstance(category, str):
            category = mlrun.common.schemas.ArtifactCategories(category)

        return self._transform_db_error(
            server.api.crud.Artifacts().list_artifacts,
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
            producer_id=tree,
            format_=format_,
            limit=limit,
        )

    def del_artifact(
        self,
        key,
        tag="",
        project="",
        tree=None,
        uid=None,
        deletion_strategy: mlrun.common.schemas.artifact.ArtifactsDeletionStrategies = (
            mlrun.common.schemas.artifact.ArtifactsDeletionStrategies.metadata_only
        ),
        secrets: dict = None,
        iter=None,
    ):
        return self._transform_db_error(
            server.api.crud.Artifacts().delete_artifact,
            self.session,
            key,
            tag,
            project,
            deletion_strategy=deletion_strategy,
            secrets=secrets,
            iteration=iter,
        )

    def del_artifacts(self, name="", project="", tag="", labels=None):
        return self._transform_db_error(
            server.api.crud.Artifacts().delete_artifacts,
            self.session,
            project,
            name,
            tag,
            labels,
        )

    def store_function(self, function, name, project="", tag="", versioned=False):
        return self._transform_db_error(
            server.api.crud.Functions().store_function,
            self.session,
            function,
            name,
            project,
            tag,
            versioned,
        )

    def get_function(self, name, project="", tag="", hash_key=""):
        return self._transform_db_error(
            server.api.crud.Functions().get_function,
            self.session,
            name,
            project,
            tag,
            hash_key,
        )

    def delete_function(self, name: str, project: str = ""):
        return self._transform_db_error(
            server.api.crud.Functions().delete_function,
            self.session,
            project,
            name,
        )

    def list_functions(
        self, name=None, project=None, tag=None, labels=None, since=None, until=None
    ):
        return self._transform_db_error(
            server.api.crud.Functions().list_functions,
            db_session=self.session,
            project=project,
            name=name,
            tag=tag,
            labels=labels,
            since=since,
            until=until,
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
                server.api.crud.Tags().overwrite_object_tags_with_tag,
                self.session,
                project,
                tag_name,
                tag_objects,
            )

        return self._transform_db_error(
            server.api.crud.Tags().append_tag_to_objects,
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
            server.api.crud.Tags().delete_tag_from_objects,
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
            server.api.crud.Projects().store_project,
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
            server.api.crud.Projects().patch_project,
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
            server.api.crud.Projects().create_project,
            self.session,
            project=project,
        )

    def delete_project(
        self,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
    ):
        return self._transform_db_error(
            server.api.crud.Projects().delete_project,
            self.session,
            name=name,
            deletion_strategy=deletion_strategy,
        )

    def get_project(
        self, name: str = None, project_id: int = None
    ) -> mlrun.common.schemas.Project:
        return self._transform_db_error(
            server.api.crud.Projects().get_project,
            self.session,
            name=name,
        )

    def list_projects(
        self,
        owner: str = None,
        format_: mlrun.common.formatters.ProjectFormat = mlrun.common.formatters.ProjectFormat.name_only,
        labels: list[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        return self._transform_db_error(
            server.api.crud.Projects().list_projects,
            self.session,
            owner=owner,
            format_=format_,
            labels=labels,
            state=state,
        )

    def create_feature_set(self, feature_set, project="", versioned=True):
        return self._transform_db_error(
            server.api.crud.FeatureStore().create_feature_set,
            self.session,
            project,
            feature_set,
            versioned,
        )

    def get_feature_set(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ):
        feature_set = self._transform_db_error(
            server.api.crud.FeatureStore().get_feature_set,
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
        entities: list[str] = None,
        labels: list[str] = None,
    ):
        return self._transform_db_error(
            server.api.crud.FeatureStore().list_features,
            self.session,
            project,
            name,
            tag,
            entities,
            labels,
        )

    def list_features_v2(
        self,
        project: str,
        name: str = None,
        tag: str = None,
        entities: list[str] = None,
        labels: list[str] = None,
    ):
        return self._transform_db_error(
            server.api.crud.FeatureStore().list_features_v2,
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
        labels: list[str] = None,
    ):
        return self._transform_db_error(
            server.api.crud.FeatureStore().list_entities,
            self.session,
            project,
            name,
            tag,
            labels,
        )

    def list_entities_v2(
        self,
        project: str,
        name: str = None,
        tag: str = None,
        labels: list[str] = None,
    ):
        return self._transform_db_error(
            server.api.crud.FeatureStore().list_entities_v2,
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
        entities: list[str] = None,
        features: list[str] = None,
        labels: list[str] = None,
        partition_by: mlrun.common.schemas.FeatureStorePartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: mlrun.common.schemas.SortField = None,
        partition_order: mlrun.common.schemas.OrderType = mlrun.common.schemas.OrderType.desc,
        format_: Union[
            str, mlrun.common.formatters.FeatureSetFormat
        ] = mlrun.common.formatters.FeatureSetFormat.full,
    ):
        return self._transform_db_error(
            server.api.crud.FeatureStore().list_feature_sets,
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
            format_=format_,
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
            server.api.crud.FeatureStore().store_feature_set,
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
            server.api.crud.FeatureStore().patch_feature_set,
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
            server.api.crud.FeatureStore().delete_feature_set,
            self.session,
            project,
            name,
            tag,
            uid,
        )

    def create_feature_vector(self, feature_vector, project="", versioned=True):
        return self._transform_db_error(
            server.api.crud.FeatureStore().create_feature_vector,
            self.session,
            project,
            feature_vector,
            versioned,
        )

    def get_feature_vector(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ):
        return self._transform_db_error(
            server.api.crud.FeatureStore().get_feature_vector,
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
        labels: list[str] = None,
        partition_by: mlrun.common.schemas.FeatureStorePartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: mlrun.common.schemas.SortField = None,
        partition_order: mlrun.common.schemas.OrderType = mlrun.common.schemas.OrderType.desc,
    ):
        return self._transform_db_error(
            server.api.crud.FeatureStore().list_feature_vectors,
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
            server.api.crud.FeatureStore().store_feature_vector,
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
            server.api.crud.FeatureStore().patch_feature_vector,
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
            server.api.crud.FeatureStore().delete_feature_vector,
            self.session,
            project,
            name,
            tag,
            uid,
        )

    def store_run_notifications(
        self,
        notification_objects: list[mlrun.model.Notification],
        run_uid: str,
        project: str = None,
        mask_params: bool = True,
    ):
        # We run this function with a new session because it may run concurrently.
        # Older sessions will not be able to see the changes made by this function until they are committed.
        return self._transform_db_error(
            server.api.db.session.run_function_with_new_db_session,
            server.api.crud.Notifications().store_run_notifications,
            notification_objects,
            run_uid,
            project,
            mask_params,
        )

    def store_alert_notifications(
        self,
        session,
        notification_objects: list[mlrun.model.Notification],
        alert_id: str,
        project: str = None,
        mask_params: bool = True,
    ):
        # We run this function with a new session because it may run concurrently.
        # Older sessions will not be able to see the changes made by this function until they are committed.
        return self._transform_db_error(
            server.api.db.session.run_function_with_new_db_session,
            server.api.crud.Notifications().store_alerts_notifications,
            notification_objects,
            alert_id,
            project,
            mask_params,
        )

    def function_status(self, project, name, kind, selector):
        """Retrieve status of a function being executed remotely (relevant to ``dask`` functions).

        :param project:     The project of the function, not needed here.
        :param name:        The name of the function, not needed here.
        :param kind:        The kind of the function, currently ``dask`` is supported.
        :param selector:    Selector clause to be applied to the Kubernetes status query to filter the results.
        """
        return self._transform_db_error(
            server.api.crud.Functions().get_function_status,
            kind,
            selector,
        )

    def start_function(
        self, func_url: str = None, function: "mlrun.runtimes.BaseRuntime" = None
    ):
        """Execute a function remotely, Used for ``dask`` functions.

        :param func_url: URL to the function to be executed.
        :param function: The function object to start.
        :returns: A BackgroundTask object, with details on execution process and its status.
        """
        return self._transform_db_error(
            server.api.crud.Functions().start_function,
            function,
        )

    def list_hub_sources(
        self,
        item_name: Optional[str] = None,
        tag: Optional[str] = None,
        version: Optional[str] = None,
    ):
        return self._transform_db_error(
            server.api.db.session.run_function_with_new_db_session,
            server.api.crud.Hub().list_hub_sources,
            item_name,
            tag,
            version,
        )

    def get_pipeline(
        self,
        run_id: str,
        namespace: str = None,
        timeout: int = 30,
        format_: Union[
            str, mlrun.common.formatters.PipelineFormat
        ] = mlrun.common.formatters.PipelineFormat.summary,
        project: str = None,
    ):
        raise NotImplementedError()

    def list_pipelines(
        self,
        project: str,
        namespace: str = None,
        sort_by: str = "",
        page_token: str = "",
        filter_: str = "",
        format_: Union[
            str, mlrun.common.formatters.PipelineFormat
        ] = mlrun.common.formatters.PipelineFormat.metadata_only,
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

    def store_api_gateway(
        self,
        api_gateway: Union[
            mlrun.common.schemas.APIGateway,
            mlrun.runtimes.nuclio.api_gateway.APIGateway,
        ],
        project: Optional[str] = None,
    ):
        raise NotImplementedError()

    def list_api_gateways(self, project=None) -> mlrun.common.schemas.APIGatewaysOutput:
        raise NotImplementedError()

    def get_api_gateway(self, name, project=None) -> mlrun.common.schemas.APIGateway:
        raise NotImplementedError()

    def delete_api_gateway(self, name, project=None):
        raise NotImplementedError()

    def list_project_secrets(
        self,
        project: str,
        token: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: list[str] = None,
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
        secrets: list[str] = None,
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
        labels: list[str] = None,
        start: str = "now-1h",
        end: str = "now",
        metrics: Optional[list[str]] = None,
    ):
        raise NotImplementedError()

    def get_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        metrics: Optional[list[str]] = None,
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

    def get_log_size(self, uid, project=""):
        raise NotImplementedError("Getting log size is not supported on the server")

    def watch_log(self, uid, project="", watch=True, offset=0):
        raise NotImplementedError("Watching logs is not supported on the server")

    def get_datastore_profile(
        self, name: str, project: str
    ) -> Optional[mlrun.common.schemas.DatastoreProfile]:
        return self._transform_db_error(
            server.api.db.session.run_function_with_new_db_session,
            server.api.crud.DatastoreProfiles().get_datastore_profile,
            name,
            project,
        )

    def delete_datastore_profile(self, name: str, project: str):
        raise NotImplementedError()

    def list_datastore_profiles(
        self, project: str
    ) -> list[mlrun.common.schemas.DatastoreProfile]:
        raise NotImplementedError()

    def store_datastore_profile(
        self, profile: mlrun.common.schemas.DatastoreProfile, project: str
    ):
        raise NotImplementedError()

    def submit_workflow(
        self,
        project: str,
        name: str,
        workflow_spec: Union[
            mlrun.projects.pipelines.WorkflowSpec,
            mlrun.common.schemas.WorkflowSpec,
            dict,
        ],
        arguments: Optional[dict] = None,
        artifact_path: Optional[str] = None,
        source: Optional[str] = None,
        run_name: Optional[str] = None,
        namespace: Optional[str] = None,
        notifications: list[mlrun.model.Notification] = None,
    ) -> "mlrun.common.schemas.WorkflowResponse":
        raise NotImplementedError()

    def remote_builder(
        self,
        func: "mlrun.runtimes.BaseRuntime",
        with_mlrun: bool,
        mlrun_version_specifier: Optional[str] = None,
        skip_deployed: bool = False,
        builder_env: Optional[dict] = None,
        force_build: bool = False,
    ):
        raise NotImplementedError()

    def deploy_nuclio_function(
        self,
        func: mlrun.runtimes.RemoteRuntime,
        builder_env: Optional[dict] = None,
    ):
        raise NotImplementedError()

    def get_builder_status(
        self,
        func: "mlrun.runtimes.BaseRuntime",
        offset: int = 0,
        logs: bool = True,
        last_log_timestamp: float = 0.0,
        verbose: bool = False,
    ):
        raise NotImplementedError()

    def get_nuclio_deploy_status(
        self,
        func: "mlrun.runtimes.RemoteRuntime",
        last_log_timestamp: float = 0.0,
        verbose: bool = False,
    ):
        raise NotImplementedError()

    def set_run_notifications(
        self,
        project: str,
        runs: list[mlrun.model.RunObject],
        notifications: list[mlrun.model.Notification],
    ):
        raise NotImplementedError()

    def update_model_monitoring_controller(
        self,
        project: str,
        base_period: int = 10,
        image: str = "mlrun/mlrun",
    ):
        raise NotImplementedError

    def enable_model_monitoring(
        self,
        project: str,
        base_period: int = 10,
        image: str = "mlrun/mlrun",
        deploy_histogram_data_drift_app: bool = True,
        rebuild_images: bool = False,
        fetch_credentials_from_sys_config: bool = False,
    ) -> None:
        raise NotImplementedError

    def disable_model_monitoring(
        self,
        project: str,
        delete_resources: bool = True,
        delete_stream_function: bool = False,
        delete_histogram_data_drift_app: bool = True,
        delete_user_applications: bool = False,
        user_application_list: list[str] = None,
    ) -> bool:
        raise NotImplementedError

    def delete_model_monitoring_function(
        self, project: str, functions: list[str]
    ) -> bool:
        raise NotImplementedError

    def deploy_histogram_data_drift_app(
        self, project: str, image: str = "mlrun/mlrun"
    ) -> None:
        raise NotImplementedError

    def set_model_monitoring_credentials(
        self,
        project: str,
        credentials: dict[str, str],
        replace_creds: bool = False,
    ) -> None:
        raise NotImplementedError

    def _transform_db_error(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)

        except SQLAlchemyError as exc:
            # If we got a SQLAlchemyError, it means the error was not handled by the SQLDB and we need to rollback
            # to make the session usable again
            self.session.rollback()
            raise mlrun.db.RunDBError(exc.args) from exc

        except DBError as exc:
            raise mlrun.db.RunDBError(exc.args) from exc

    def generate_event(
        self, name: str, event_data: Union[dict, mlrun.common.schemas.Event], project=""
    ):
        pass

    def store_alert_config(
        self,
        alert_name: str,
        alert_data: Union[dict, mlrun.alerts.alert.AlertConfig],
        project="",
    ):
        pass

    def get_alert_config(self, alert_name: str, project=""):
        pass

    def list_alerts_configs(self, project=""):
        pass

    def delete_alert_config(self, alert_name, project=""):
        pass

    def reset_alert_config(self, alert_name, project=""):
        pass

    def get_alert_template(self, template_name: str):
        pass

    def list_alert_templates(self):
        pass


# Once this file is imported it will override the default RunDB implementation (RunDBContainer)
@containers.override(mlrun.db.factory.RunDBContainer)
class SQLRunDBContainer(containers.DeclarativeContainer):
    run_db = providers.Factory(SQLRunDB)
