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
import typing
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from deprecated import deprecated

import mlrun.alerts
import mlrun.common.formatters
import mlrun.common.schemas
import mlrun.common.types
import mlrun.lists
import mlrun.model


class DBError(Exception):
    pass


class DBInterface(ABC):
    @abstractmethod
    def initialize(self, session):
        pass

    @abstractmethod
    def store_log(
        self,
        session,
        uid,
        project="",
        body=None,
        append=False,
    ):
        pass

    @abstractmethod
    def get_log(self, session, uid, project="", offset=0, size=0):
        pass

    @abstractmethod
    def store_run(
        self,
        session,
        run_data,
        uid,
        project="",
        iter=0,
    ):
        pass

    @abstractmethod
    def update_run(self, session, updates: dict, uid, project="", iter=0):
        pass

    @abstractmethod
    def list_distinct_runs_uids(
        self,
        session,
        project: str = None,
        requested_logs_modes: list[bool] = None,
        only_uids: bool = False,
        last_update_time_from: datetime.datetime = None,
        states: list[str] = None,
        specific_uids: list[str] = None,
    ):
        pass

    @abstractmethod
    def update_runs_requested_logs(
        self, session, uids: list[str], requested_logs: bool = True
    ):
        pass

    @abstractmethod
    def read_run(self, session, uid, project="", iter=0):
        pass

    @abstractmethod
    def list_runs(
        self,
        session,
        name: Optional[str] = None,
        uid: Optional[Union[str, list[str]]] = None,
        project: str = "",
        labels: Optional[Union[str, list[str]]] = None,
        states: Optional[list[str]] = None,
        sort: bool = True,
        last: int = 0,
        iter: bool = False,
        start_time_from: datetime.datetime = None,
        start_time_to: datetime.datetime = None,
        last_update_time_from: datetime.datetime = None,
        last_update_time_to: datetime.datetime = None,
        partition_by: mlrun.common.schemas.RunPartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: mlrun.common.schemas.SortField = None,
        partition_order: mlrun.common.schemas.OrderType = mlrun.common.schemas.OrderType.desc,
        max_partitions: int = 0,
        requested_logs: bool = None,
        return_as_run_structs: bool = True,
        with_notifications: bool = False,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> mlrun.lists.RunList:
        pass

    @abstractmethod
    def del_run(self, session, uid, project="", iter=0):
        pass

    @abstractmethod
    def del_runs(self, session, name="", project="", labels=None, state="", days_ago=0):
        pass

    def overwrite_artifacts_with_tag(
        self,
        session,
        project: str,
        tag: str,
        identifiers: list[mlrun.common.schemas.ArtifactIdentifier],
    ):
        pass

    def append_tag_to_artifacts(
        self,
        session,
        project: str,
        tag: str,
        identifiers: list[mlrun.common.schemas.ArtifactIdentifier],
    ):
        pass

    def delete_tag_from_artifacts(
        self,
        session,
        project: str,
        tag: str,
        identifiers: list[mlrun.common.schemas.ArtifactIdentifier],
    ):
        pass

    @abstractmethod
    def store_artifact(
        self,
        session,
        key,
        artifact,
        uid=None,
        iter=None,
        tag="",
        project="",
        producer_id=None,
        best_iteration=False,
        always_overwrite=False,
    ):
        pass

    @abstractmethod
    def create_artifact(
        self,
        session,
        project,
        artifact,
        key,
        tag="",
        uid=None,
        iteration=None,
        producer_id="",
        best_iteration=False,
    ):
        pass

    @abstractmethod
    def read_artifact(
        self,
        session,
        key,
        tag="",
        iter=None,
        project="",
        producer_id: str = None,
        uid: str = None,
        raise_on_not_found: bool = True,
        format_: mlrun.common.formatters.ArtifactFormat = mlrun.common.formatters.ArtifactFormat.full,
    ):
        pass

    @abstractmethod
    def list_artifacts(
        self,
        session,
        name="",
        project="",
        tag="",
        labels=None,
        since: datetime.datetime = None,
        until: datetime.datetime = None,
        kind=None,
        category: mlrun.common.schemas.ArtifactCategories = None,
        iter: int = None,
        best_iteration: bool = False,
        as_records: bool = False,
        uid: str = None,
        producer_id: str = None,
        producer_uri: str = None,
        format_: mlrun.common.formatters.ArtifactFormat = mlrun.common.formatters.ArtifactFormat.full,
        limit: int = None,
    ):
        pass

    @abstractmethod
    def list_artifacts_for_producer_id(
        self,
        session,
        producer_id: str,
        project: str,
        key_tag_iteration_pairs: list[tuple] = "",
    ):
        pass

    @abstractmethod
    def del_artifact(
        self, session, key, tag="", project="", uid=None, producer_id=None, iter=None
    ):
        pass

    @abstractmethod
    def del_artifacts(
        self,
        session,
        name="",
        project="",
        tag="*",
        labels=None,
        ids=None,
        producer_id=None,
    ):
        pass

    def list_artifact_tags(
        self, session, project, category: mlrun.common.schemas.ArtifactCategories = None
    ):
        return []

    # TODO: remove in 1.8.0
    @deprecated(
        version="1.8.0",
        reason="'store_artifact_v1' will be removed from this file in 1.8.0, use "
        "'store_artifact' instead",
        category=FutureWarning,
    )
    def store_artifact_v1(
        self,
        session,
        key,
        artifact,
        uid,
        iter=None,
        tag="",
        project="",
        tag_artifact=True,
    ):
        """
        Store artifact v1 in the DB, this is the deprecated legacy artifact format
        and is only left for testing purposes
        """
        pass

    # TODO: remove in 1.8.0
    @deprecated(
        version="1.8.0",
        reason="'read_artifact_v1' will be removed from this file in 1.8.0, use "
        "'read_artifact' instead",
        category=FutureWarning,
    )
    def read_artifact_v1(self, session, key, tag="", iter=None, project=""):
        """
        Read artifact v1 from the DB, this is the deprecated legacy artifact format
        and is only left for testing purposes
        """
        pass

    @abstractmethod
    def store_function(
        self,
        session,
        function,
        name,
        project="",
        tag="",
        versioned=False,
    ) -> str:
        pass

    @abstractmethod
    def get_function(
        self,
        session,
        name: str = None,
        project: str = None,
        tag: str = None,
        hash_key: str = None,
        format_: str = None,
    ):
        pass

    @abstractmethod
    def delete_function(self, session, project: str, name: str):
        pass

    @abstractmethod
    def delete_functions(
        self, session, project: str, names: typing.Union[str, list[str]]
    ) -> None:
        pass

    @abstractmethod
    def list_functions(
        self,
        session,
        name: str = None,
        project: str = None,
        tag: str = None,
        labels: list[str] = None,
        hash_key: str = None,
        format_: str = None,
        page: int = None,
        page_size: int = None,
        since: datetime.datetime = None,
        until: datetime.datetime = None,
    ):
        pass

    @abstractmethod
    def update_function(
        self,
        session,
        name,
        updates: dict,
        project: str = None,
        tag: str = None,
        hash_key: str = None,
    ):
        pass

    @abstractmethod
    def update_function_external_invocation_url(
        self,
        session,
        name: str,
        url: str,
        project: str = "",
        tag: str = "",
        hash_key: str = "",
        operation: mlrun.common.types.Operation = mlrun.common.types.Operation.ADD,
    ):
        pass

    @abstractmethod
    def create_schedule(
        self,
        session,
        project: str,
        name: str,
        kind: mlrun.common.schemas.ScheduleKinds,
        scheduled_object: Any,
        cron_trigger: mlrun.common.schemas.ScheduleCronTrigger,
        concurrency_limit: int,
        labels: dict = None,
        next_run_time: datetime.datetime = None,
    ):
        pass

    @abstractmethod
    def update_schedule(
        self,
        session,
        project: str,
        name: str,
        scheduled_object: Any = None,
        cron_trigger: mlrun.common.schemas.ScheduleCronTrigger = None,
        labels: dict = None,
        last_run_uri: str = None,
        concurrency_limit: int = None,
        next_run_time: datetime.datetime = None,
    ):
        pass

    def store_schedule(
        self,
        session,
        project: str,
        name: str,
        kind: mlrun.common.schemas.ScheduleKinds = None,
        scheduled_object: Any = None,
        cron_trigger: mlrun.common.schemas.ScheduleCronTrigger = None,
        labels: dict = None,
        last_run_uri: str = None,
        concurrency_limit: int = None,
        next_run_time: datetime = None,
    ):
        pass

    @abstractmethod
    def list_schedules(
        self,
        session,
        project: str = None,
        name: str = None,
        labels: list[str] = None,
        kind: mlrun.common.schemas.ScheduleKinds = None,
    ) -> list[mlrun.common.schemas.ScheduleRecord]:
        pass

    @abstractmethod
    def get_schedule(
        self, session, project: str, name: str, raise_on_not_found: bool = True
    ) -> mlrun.common.schemas.ScheduleRecord:
        pass

    @abstractmethod
    def delete_schedule(self, session, project: str, name: str):
        pass

    @abstractmethod
    def delete_project_schedules(self, session, project: str):
        pass

    @abstractmethod
    def delete_schedules(
        self, session, project: str, names: typing.Union[str, list[str]]
    ) -> None:
        pass

    @abstractmethod
    def delete_project_related_resources(self, session, name: str):
        pass

    @abstractmethod
    def verify_project_has_no_related_resources(self, session, name: str):
        pass

    @abstractmethod
    def is_project_exists(self, session, name: str):
        pass

    @abstractmethod
    def list_projects(
        self,
        session,
        owner: str = None,
        format_: mlrun.common.formatters.ProjectFormat = mlrun.common.formatters.ProjectFormat.full,
        labels: list[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: Optional[list[str]] = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        pass

    @abstractmethod
    def get_project(
        self,
        session,
        name: str = None,
        project_id: int = None,
    ) -> mlrun.common.schemas.Project:
        pass

    @abstractmethod
    async def get_project_resources_counters(
        self,
    ) -> tuple[
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
    ]:
        pass

    @abstractmethod
    def create_project(self, session, project: mlrun.common.schemas.Project):
        pass

    @abstractmethod
    def store_project(self, session, name: str, project: mlrun.common.schemas.Project):
        pass

    @abstractmethod
    def patch_project(
        self,
        session,
        name: str,
        project: dict,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
    ):
        pass

    @abstractmethod
    def delete_project(
        self,
        session,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
    ):
        pass

    def get_project_summary(
        self,
        session,
        project: str,
    ) -> mlrun.common.schemas.ProjectSummary:
        pass

    def list_project_summaries(
        self,
        session,
        owner: str = None,
        labels: list[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: list[str] = None,
    ):
        pass

    def refresh_project_summaries(
        self, session, project_summaries: list[mlrun.common.schemas.ProjectSummary]
    ):
        pass

    @abstractmethod
    def create_feature_set(
        self,
        session,
        project,
        feature_set: mlrun.common.schemas.FeatureSet,
        versioned=True,
    ) -> str:
        pass

    @abstractmethod
    def store_feature_set(
        self,
        session,
        project,
        name,
        feature_set: mlrun.common.schemas.FeatureSet,
        tag=None,
        uid=None,
        versioned=True,
        always_overwrite=False,
    ) -> str:
        pass

    @abstractmethod
    def get_feature_set(
        self, session, project: str, name: str, tag: str = None, uid: str = None
    ) -> mlrun.common.schemas.FeatureSet:
        pass

    # TODO: remove in 1.9.0
    @deprecated(
        version="1.9.0",
        reason="'list_features' will be removed in 1.9.0, use 'list_features_v2' instead",
        category=FutureWarning,
    )
    @abstractmethod
    def list_features(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        entities: list[str] = None,
        labels: list[str] = None,
    ) -> mlrun.common.schemas.FeaturesOutput:
        pass

    @abstractmethod
    def list_features_v2(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        entities: list[str] = None,
        labels: list[str] = None,
    ) -> mlrun.common.schemas.FeaturesOutputV2:
        pass

    # TODO: remove in 1.9.0
    @deprecated(
        version="1.9.0",
        reason="'list_entities' will be removed in 1.9.0, use 'list_entities_v2' instead",
        category=FutureWarning,
    )
    @abstractmethod
    def list_entities(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        labels: list[str] = None,
    ) -> mlrun.common.schemas.EntitiesOutput:
        pass

    @abstractmethod
    def list_entities_v2(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        labels: list[str] = None,
    ) -> mlrun.common.schemas.EntitiesOutputV2:
        pass

    @abstractmethod
    def list_feature_sets(
        self,
        session,
        project: str,
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
        format_: mlrun.common.formatters.FeatureSetFormat = mlrun.common.formatters.FeatureSetFormat.full,
    ) -> mlrun.common.schemas.FeatureSetsOutput:
        pass

    @abstractmethod
    def list_feature_sets_tags(
        self,
        session,
        project: str,
    ) -> list[tuple[str, str, str]]:
        """
        :return: a list of Tuple of (project, feature_set.name, tag)
        """
        pass

    @abstractmethod
    def patch_feature_set(
        self,
        session,
        project,
        name,
        feature_set_patch: dict,
        tag=None,
        uid=None,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
    ) -> str:
        pass

    @abstractmethod
    def delete_feature_set(self, session, project, name, tag=None, uid=None):
        pass

    @abstractmethod
    def create_feature_vector(
        self,
        session,
        project,
        feature_vector: mlrun.common.schemas.FeatureVector,
        versioned=True,
    ) -> str:
        pass

    @abstractmethod
    def get_feature_vector(
        self, session, project: str, name: str, tag: str = None, uid: str = None
    ) -> mlrun.common.schemas.FeatureVector:
        pass

    @abstractmethod
    def list_feature_vectors(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        state: str = None,
        labels: list[str] = None,
        partition_by: mlrun.common.schemas.FeatureStorePartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: mlrun.common.schemas.SortField = None,
        partition_order: mlrun.common.schemas.OrderType = mlrun.common.schemas.OrderType.desc,
    ) -> mlrun.common.schemas.FeatureVectorsOutput:
        pass

    @abstractmethod
    def list_feature_vectors_tags(
        self,
        session,
        project: str,
    ) -> list[tuple[str, str, str]]:
        """
        :return: a list of Tuple of (project, feature_vector.name, tag)
        """
        pass

    @abstractmethod
    def store_feature_vector(
        self,
        session,
        project,
        name,
        feature_vector: mlrun.common.schemas.FeatureVector,
        tag=None,
        uid=None,
        versioned=True,
        always_overwrite=False,
    ) -> str:
        pass

    @abstractmethod
    def patch_feature_vector(
        self,
        session,
        project,
        name,
        feature_vector_update: dict,
        tag=None,
        uid=None,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
    ) -> str:
        pass

    @abstractmethod
    def delete_feature_vector(
        self,
        session,
        project,
        name,
        tag=None,
        uid=None,
    ):
        pass

    def create_hub_source(
        self, session, ordered_source: mlrun.common.schemas.IndexedHubSource
    ):
        pass

    def store_hub_source(
        self,
        session,
        name,
        ordered_source: mlrun.common.schemas.IndexedHubSource,
    ):
        pass

    def list_hub_sources(self, session) -> list[mlrun.common.schemas.IndexedHubSource]:
        pass

    def delete_hub_source(self, session, name):
        pass

    def get_hub_source(
        self, session, name=None, index=None
    ) -> mlrun.common.schemas.IndexedHubSource:
        pass

    def store_background_task(
        self,
        session,
        name: str,
        project: str,
        state: str = mlrun.common.schemas.BackgroundTaskState.running,
        timeout: int = None,
        error: str = None,
    ):
        pass

    def get_background_task(
        self, session, name: str, project: str, background_task_exceeded_timeout_func
    ) -> mlrun.common.schemas.BackgroundTask:
        pass

    def list_background_tasks(
        self,
        session,
        project: str,
        background_task_exceeded_timeout_func,
        states: Optional[list[str]] = None,
        created_from: datetime.datetime = None,
        created_to: datetime.datetime = None,
        last_update_time_from: datetime.datetime = None,
        last_update_time_to: datetime.datetime = None,
    ) -> list[mlrun.common.schemas.BackgroundTask]:
        pass

    def delete_background_task(self, session, name: str, project: str):
        pass

    @abstractmethod
    def store_alert_template(
        self, session, template: mlrun.common.schemas.AlertTemplate
    ) -> mlrun.common.schemas.AlertTemplate:
        pass

    @abstractmethod
    def get_alert_template(
        self, session, name: str
    ) -> mlrun.common.schemas.AlertTemplate:
        pass

    @abstractmethod
    def delete_alert_template(self, session, name: str):
        pass

    @abstractmethod
    def list_alert_templates(self, session) -> list[mlrun.common.schemas.AlertTemplate]:
        pass

    @abstractmethod
    def store_alert(
        self, session, alert: mlrun.common.schemas.AlertConfig
    ) -> mlrun.common.schemas.AlertConfig:
        pass

    @abstractmethod
    def get_all_alerts(self, session) -> list[mlrun.common.schemas.AlertConfig]:
        pass

    @abstractmethod
    def list_alerts(
        self, session, project: str = None
    ) -> list[mlrun.common.schemas.AlertConfig]:
        pass

    @abstractmethod
    def get_alert(
        self, session, project: str, name: str
    ) -> mlrun.common.schemas.AlertConfig:
        pass

    @abstractmethod
    def get_alert_by_id(
        self, session, alert_id: int
    ) -> mlrun.common.schemas.AlertConfig:
        pass

    @abstractmethod
    def enrich_alert(self, session, alert: mlrun.common.schemas.AlertConfig):
        pass

    @abstractmethod
    def delete_alert(self, session, project: str, name: str):
        pass

    @abstractmethod
    def store_alert_state(
        self,
        session,
        project: str,
        name: str,
        last_updated: datetime,
        count: typing.Optional[int] = None,
        active: bool = False,
        obj: typing.Optional[dict] = None,
    ):
        pass

    @abstractmethod
    def get_alert_state_dict(self, session, alert_id: int) -> dict:
        pass

    @abstractmethod
    def get_num_configured_alerts(self, session) -> int:
        pass

    @abstractmethod
    def store_alert_notifications(
        self,
        session,
        notification_objects: list[mlrun.model.Notification],
        alert_id: str,
        project: str,
    ):
        pass

    @abstractmethod
    def store_run_notifications(
        self,
        session,
        notification_objects: list[mlrun.model.Notification],
        run_uid: str,
        project: str,
    ):
        pass

    @abstractmethod
    def list_run_notifications(
        self,
        session,
        run_uid: str,
        project: str,
    ) -> list[mlrun.model.Notification]:
        pass

    def delete_run_notifications(
        self,
        session,
        name: str = None,
        run_uid: str = None,
        project: str = None,
        commit: bool = True,
    ):
        pass

    def set_run_notifications(
        self,
        session,
        project: str,
        notifications: list[mlrun.model.Notification],
        identifiers: list[mlrun.common.schemas.RunIdentifier],
        **kwargs,
    ):
        pass

    def store_datastore_profile(
        self,
        session,
        profile: mlrun.common.schemas.DatastoreProfile,
    ) -> str:
        pass

    def get_datastore_profile(
        self,
        session,
        profile: str,
        project: str,
    ) -> Optional[mlrun.common.schemas.DatastoreProfile]:
        pass

    def delete_datastore_profile(
        self,
        session,
        profile: str,
        project: str,
    ):
        pass

    def list_datastore_profiles(
        self,
        session,
        project: str,
    ) -> list[mlrun.common.schemas.DatastoreProfile]:
        pass

    # Pagination Cache Methods
    # They are not abstract methods because they are not required for all DBs.
    # However, they do raise NotImplementedError for DBs that do not implement them.
    def store_paginated_query_cache_record(
        self,
        session,
        user: str,
        function: str,
        current_page: int,
        page_size: int,
        kwargs: dict,
    ):
        raise NotImplementedError

    def get_paginated_query_cache_record(
        self,
        session,
        key: str,
    ):
        raise NotImplementedError

    def list_paginated_query_cache_record(
        self,
        session,
        key: str = None,
        user: str = None,
        function: str = None,
        last_accessed_before: datetime = None,
        order_by: Optional[mlrun.common.schemas.OrderType] = None,
        as_query: bool = False,
    ):
        raise NotImplementedError

    def delete_paginated_query_cache_record(
        self,
        session,
        key: str,
    ):
        raise NotImplementedError

    # EO Pagination Section
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

    def delete_alert_config(self, alert_name: str, project=""):
        pass

    def reset_alert_config(self, alert_name: str, project=""):
        pass

    def store_time_window_tracker_record(
        self,
        session,
        key: str,
        timestamp: typing.Optional[datetime.datetime] = None,
        max_window_size_seconds: typing.Optional[int] = None,
    ):
        pass

    def get_time_window_tracker_record(
        self, session, key: str, raise_on_not_found: bool = True
    ):
        pass
