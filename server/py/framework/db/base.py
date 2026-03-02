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

import sqlalchemy
from sqlalchemy.orm import Session

import mlrun.alerts
import mlrun.common.formatters
import mlrun.common.schemas
import mlrun.common.types
import mlrun.lists
import mlrun.model

if typing.TYPE_CHECKING:
    import framework.db.sqldb.models


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

    def create_or_get_run(
        self,
        session,
        run_data: dict,
        uid: str,
        project: str = "",
        iter: int = 0,
    ):
        pass

    @abstractmethod
    def update_run(self, session, updates: dict, uid, project, iter=0):
        pass

    @abstractmethod
    def set_run_retrying_status(
        self,
        session: Session,
        project: str,
        uid: str,
        retrying: bool,
    ) -> dict:
        pass

    @abstractmethod
    def list_distinct_runs_uids(
        self,
        session,
        project: str | None = None,
        requested_logs_modes: list[bool] | None = None,
        only_uids: bool = False,
        last_update_time_from: datetime.datetime | None = None,
        states: list[str] | None = None,
        specific_uids: list[str] | None = None,
    ):
        pass

    @abstractmethod
    def update_runs_requested_logs(
        self, session, uids: list[str], requested_logs: bool = True
    ):
        pass

    @abstractmethod
    def read_run(
        self,
        session,
        uid: str,
        project: str,
        iter: int = 0,
        with_notifications: bool = False,
        populate_existing: bool = False,
    ):
        pass

    @abstractmethod
    def list_runs(
        self,
        session,
        project: typing.Union[str, list[str]],
        name: str | None = None,
        uid: Union[str, list[str]] | None = None,
        labels: Union[str, list[str]] | None = None,
        states: list[str] | None = None,
        sort: bool = True,
        iter: bool = False,
        start_time_from: datetime.datetime | None = None,
        start_time_to: datetime.datetime | None = None,
        last_update_time_from: datetime.datetime | None = None,
        last_update_time_to: datetime.datetime | None = None,
        end_time_from: datetime.datetime | None = None,
        end_time_to: datetime.datetime | None = None,
        partition_by: mlrun.common.schemas.RunPartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: mlrun.common.schemas.SortField = None,
        partition_order: mlrun.common.schemas.OrderType = mlrun.common.schemas.OrderType.desc,
        max_partitions: int = 0,
        requested_logs: bool | None = None,
        return_as_run_structs: bool = True,
        with_notifications: bool = False,
        offset: int | None = None,
        limit: int | None = None,
    ) -> mlrun.lists.RunList:
        pass

    @abstractmethod
    def del_run(self, session, uid, project, iter=0):
        pass

    @abstractmethod
    def del_runs(
        self, session, project, name="", labels=None, state="", days_ago=0, uids=None
    ):
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
        project,
        uid=None,
        iter=None,
        tag="",
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
        project,
        tag="",
        iter=None,
        producer_id: str | None = None,
        uid: str | None = None,
        raise_on_not_found: bool = True,
        format_: mlrun.common.formatters.ArtifactFormat = mlrun.common.formatters.ArtifactFormat.full,
    ):
        pass

    @abstractmethod
    def list_artifacts(
        self,
        session,
        project,
        name="",
        tag="",
        labels=None,
        since: datetime.datetime | None = None,
        until: datetime.datetime | None = None,
        kind=None,
        category: mlrun.common.schemas.ArtifactCategories = None,
        iter: int | None = None,
        best_iteration: bool = False,
        as_records: bool = False,
        uid: str | None = None,
        producer_id: str | None = None,
        producer_uri: str | None = None,
        most_recent: bool = False,
        parent_uri: str | None = None,
        format_: mlrun.common.formatters.ArtifactFormat = mlrun.common.formatters.ArtifactFormat.full,
        offset: int | None = None,
        limit: int | None = None,
        partition_by: mlrun.common.schemas.ArtifactPartitionByField | None = None,
        rows_per_partition: int | None = 1,
        partition_sort_by: mlrun.common.schemas.SortField
        | None = mlrun.common.schemas.SortField.updated,
        partition_order: mlrun.common.schemas.OrderType
        | None = mlrun.common.schemas.OrderType.desc,
    ) -> typing.Union[list, mlrun.lists.ArtifactList]:
        pass

    @abstractmethod
    def list_artifacts_for_producer_id(
        self,
        session,
        project: str,
        producer_id: str,
        artifact_identifiers: list[tuple] = "",
    ):
        pass

    @abstractmethod
    def del_artifact(
        self,
        session,
        key,
        project,
        tag="",
        uid=None,
        producer_id=None,
        iter=None,
    ):
        pass

    @abstractmethod
    def del_artifacts(
        self,
        session,
        project,
        name="",
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

    def validate_artifact_removal_preconditions(
        self,
        session,
        key: str,
        project: str,
        tag: str = "",
        iter: str | None = None,
        producer_id: str | None = None,
        uid: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Validate whether an artifact can be safely removed from the system.

        This method checks if the specified artifact is currently in use by other resources,
        such as model endpoints. If it is, the deletion will be blocked, and an appropriate
        exception should be raised (MLRunConflictError).

        :param session:     Active SQLAlchemy DB session for querying.
        :param key:         Artifact key.
        :param tag:         Specific tag for the artifact.
        :param iter:        Artifact iteration number, if applicable.
        :param project:     Project to which the artifact belongs.
        :param producer_id: Identifier of the artifact's producer.
        :param uid:         UID of the artifact object.

        :return: An artifact dictionary.
        :raises MLRunConflictError: If the artifact is in use and cannot be deleted.
        """
        pass

    @abstractmethod
    def store_function(
        self,
        session,
        function,
        name,
        project,
        tag="",
        versioned=False,
    ) -> str:
        pass

    @abstractmethod
    def get_function(
        self,
        session,
        project: str,
        name: str | None = None,
        tag: str | None = None,
        hash_key: str | None = None,
        format_: str | None = None,
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
        project: Union[str, list[str]],
        name: str | None = None,
        tag: str | None = None,
        kind: str | None = None,
        labels: list[str] | None = None,
        states: list[mlrun.common.schemas.FunctionState] | None = None,
        hash_key: str | None = None,
        format_: mlrun.common.formatters.FunctionFormat = mlrun.common.formatters.FunctionFormat.full,
        offset: int | None = None,
        limit: int | None = None,
        since: datetime.datetime | None = None,
        until: datetime.datetime | None = None,
    ):
        pass

    @abstractmethod
    def update_function(
        self,
        session,
        name,
        updates: dict,
        project: str,
        tag: str | None = None,
        hash_key: str | None = None,
    ):
        pass

    @abstractmethod
    def update_function_external_invocation_url(
        self,
        session,
        name: str,
        url: str,
        project: str,
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
        labels: dict | None = None,
        next_run_time: datetime.datetime | None = None,
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
        labels: dict | None = None,
        last_run_uri: str | None = None,
        concurrency_limit: int | None = None,
        next_run_time: datetime.datetime | None = None,
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
        labels: dict | None = None,
        last_run_uri: str | None = None,
        concurrency_limit: int | None = None,
        next_run_time: datetime.datetime | None = None,
    ):
        pass

    @abstractmethod
    def list_schedules(
        self,
        session,
        project: Union[str, list[str]] | None = None,
        name: str | None = None,
        labels: list[str] | None = None,
        kind: mlrun.common.schemas.ScheduleKinds = None,
        next_run_time_since: datetime.datetime | None = None,
        next_run_time_until: datetime.datetime | None = None,
        limit: int | None = None,
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
        owner: str | None = None,
        format_: mlrun.common.formatters.ProjectFormat = mlrun.common.formatters.ProjectFormat.full,
        labels: list[str] | None = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: list[str] | None = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        pass

    @abstractmethod
    def get_project(
        self,
        session,
        name: str | None = None,
        project_id: int | None = None,
    ) -> mlrun.common.schemas.Project:
        pass

    @abstractmethod
    async def get_project_resources_counters(
        self,
        projects_with_creation_time: list[tuple[str, datetime]],
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
        dict[str, int],
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
        owner: str | None = None,
        labels: list[str] | None = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: list[str] | None = None,
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
        self,
        session,
        project: str,
        name: str,
        tag: str | None = None,
        uid: str | None = None,
    ) -> mlrun.common.schemas.FeatureSet:
        pass

    @abstractmethod
    def list_features_v2(
        self,
        session,
        project: str,
        name: str | None = None,
        tag: str | None = None,
        entities: list[str] | None = None,
        labels: list[str] | None = None,
    ) -> mlrun.common.schemas.FeaturesOutputV2:
        pass

    @abstractmethod
    def list_entities_v2(
        self,
        session,
        project: str,
        name: str | None = None,
        tag: str | None = None,
        labels: list[str] | None = None,
    ) -> mlrun.common.schemas.EntitiesOutputV2:
        pass

    @abstractmethod
    def list_feature_sets(
        self,
        session,
        project: str,
        name: str | None = None,
        tag: str | None = None,
        state: str | None = None,
        entities: list[str] | None = None,
        features: list[str] | None = None,
        labels: list[str] | None = None,
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
        self,
        session,
        project: str,
        name: str,
        tag: str | None = None,
        uid: str | None = None,
    ) -> mlrun.common.schemas.FeatureVector:
        pass

    @abstractmethod
    def list_feature_vectors(
        self,
        session,
        project: str,
        name: str | None = None,
        tag: str | None = None,
        state: str | None = None,
        labels: list[str] | None = None,
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
        timeout: int | None = None,
        error: str | None = None,
        labels: dict[str, str] | None = None,
    ):
        pass

    def get_background_task(
        self, session, name: str, project: str, background_task_exceeded_timeout_func
    ) -> mlrun.common.schemas.BackgroundTask:
        pass

    def get_background_task_by_state_and_labels(
        self,
        session,
        status: mlrun.common.schemas.BackgroundTaskState,
        labels: dict[str, str],
    ) -> mlrun.common.schemas.BackgroundTask:
        """
        Get a background task by its status and labels.
        :param session: The database session.
        :param status: The status of the background task to filter by.
        :param labels: A dictionary of labels to filter the background task.
        :return: The background task matching the labels.
        """
        pass

    def list_background_tasks(
        self,
        session,
        project: str,
        background_task_exceeded_timeout_func,
        states: list[str] | None = None,
        created_from: datetime.datetime | None = None,
        created_to: datetime.datetime | None = None,
        last_update_time_from: datetime.datetime | None = None,
        last_update_time_to: datetime.datetime | None = None,
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
        self,
        session,
        project: typing.Union[str, list[str]] | None = None,
        exclude_updated: bool = False,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[mlrun.common.schemas.AlertConfig]:
        pass

    @abstractmethod
    def delete_project_alerts(
        self,
        session,
        project: str,
        chunk_size: int | None = None,
    ) -> list[int]:
        pass

    @abstractmethod
    def get_alert(
        self,
        session,
        project: str,
        name: str,
        with_state=False,
    ) -> mlrun.common.schemas.AlertConfig:
        pass

    @abstractmethod
    def get_alert_by_id(
        self, session, alert_id: int
    ) -> mlrun.common.schemas.AlertConfig:
        pass

    @abstractmethod
    def enrich_alert(
        self,
        session,
        alert: mlrun.common.schemas.AlertConfig,
        state: Optional["framework.db.sqldb.models.AlertState"] = None,
    ):
        pass

    @staticmethod
    def create_partitions(
        session,
        table_name: str,
        partitioning_information_list: list[tuple[str, str]],
    ):
        pass

    @staticmethod
    def drop_partitions(
        session,
        table_name: str,
        cutoff_partition_name: str,
    ):
        pass

    @staticmethod
    def table_exists(
        session,
        table_name: str,
    ) -> bool:
        pass

    @abstractmethod
    def create_alert(
        self,
        session,
        alert: mlrun.common.schemas.AlertConfig,
    ) -> mlrun.common.schemas.AlertConfig:
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
        last_updated: datetime.datetime | None,
        count: int | None = None,
        active: bool = False,
        obj: dict | None = None,
        alert_id: int | None = None,
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
    def store_alert_activation(
        self,
        session,
        alert_data: mlrun.common.schemas.AlertConfig,
        event_data: mlrun.common.schemas.Event,
    ):
        pass

    @abstractmethod
    def update_alert_activation(
        self,
        session,
        activation_id: int,
        activation_time: datetime.datetime,
        number_of_events: int | None = None,
        notifications_states: list[mlrun.common.schemas.NotificationState]
        | None = None,
        update_reset_time: bool = False,
    ):
        pass

    @abstractmethod
    def list_alert_activations(
        self,
        session,
        projects_with_creation_time: list[tuple[str, datetime.datetime]],
        name: str | None = None,
        since: datetime.datetime | None = None,
        until: datetime.datetime | None = None,
        entity: str | None = None,
        severity: list[Union[mlrun.common.schemas.alert.AlertSeverity, str]]
        | None = None,
        entity_kind: Union[mlrun.common.schemas.alert.EventEntityKind, str]
        | None = None,
        event_kind: Union[mlrun.common.schemas.alert.EventKind, str] | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[mlrun.common.schemas.AlertActivation]:
        pass

    @abstractmethod
    def get_alert_activation(
        self,
        session,
        activation_id: int,
    ) -> mlrun.common.schemas.AlertActivation:
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
        project: str,
        name: str | None = None,
        run_uid: str | None = None,
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
    ) -> mlrun.common.schemas.DatastoreProfile | None:
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
        pagination_cache_record: typing.Optional[
            "framework.db.sqldb.models.PaginationCache"
        ] = None,
    ):
        raise NotImplementedError

    def get_paginated_query_cache_record(
        self,
        session,
        key: str,
        for_update: bool = False,
    ):
        raise NotImplementedError

    def list_paginated_query_cache_record(
        self,
        session,
        key: str | None = None,
        user: str | None = None,
        function: str | None = None,
        last_accessed_before: datetime.datetime | None = None,
        order_by: mlrun.common.schemas.OrderType | None = None,
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
        force_reset: bool = False,
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
        timestamp: datetime.datetime | None = None,
        max_window_size_seconds: int | None = None,
    ):
        pass

    def get_time_window_tracker_record(
        self, session, key: str, raise_on_not_found: bool = True
    ):
        pass

    def store_model_endpoint(
        self,
        session,
        model_endpoint: mlrun.common.schemas.ModelEndpoint,
    ) -> str:
        """
        Store a model endpoint in the DB.

        :param session:         The database session.
        :param model_endpoint:  The model endpoint object.
        :return:                The created model endpoint uid.
        """
        pass

    def store_model_endpoints(
        self,
        session,
        model_endpoints: list[mlrun.common.schemas.ModelEndpoint],
        function_name: str,
        function_tag: str,
        project: str,
    ) -> None:
        """
        Store list of model endpoints in the DB.
        Note all the model endpoints should have the same function name and tag.

        :param session:         The database session.
        :param model_endpoints: Model endpoints object to store.
        :param project:         The project name.
        :param function_name:   The function name.
        :param function_tag:    The function tag.
        """
        pass

    def get_model_endpoint(
        self,
        session,
        project: str,
        name: str,
        function_name: str | None = None,
        function_tag: str | None = None,
        uid: str | None = None,
    ) -> mlrun.common.schemas.ModelEndpoint:
        """
        Get a model endpoint by project, name and uid.
        If uid is not provided, the latest model endpoint with the provided name and project will be returned.

        :param session:       The database session.
        :param project:       The project name.
        :param name:          The model endpoint name.
        :param function_name: The function name.
        :param function_tag:  The function tag.
        :param uid:           The model endpoint uid.
        :return:              The model endpoint object.
        """
        pass

    def update_model_endpoint(
        self,
        session,
        project: str,
        name: str,
        attributes: dict,
        function_name: str | None = None,
        function_tag: str | None = None,
        uid: str | None = None,
    ) -> str:
        """
        Update a model endpoint by project, name and uid.
        If uid is not provided, the latest model endpoint with the provided name and project will be updated.
        The attributes parameter is a flatten dictionary which should contain the fields that need to be update.

        :param session:         The database session.
        :param project:         The project name.
        :param name:            The model endpoint name.
        :param attributes:      The attributes to update.
        :param function_name:   The function name.
        :param function_tag:    The function tag.
        :param uid:             The model endpoint uid.
        :return:                The updated model endpoint uid.
        """
        pass

    def update_model_endpoints(
        self,
        session,
        project: str,
        attributes: dict[str, dict[str, Any]],
    ) -> None:
        """
        Update a model endpoint by project, name and uid.
        If uid is not provided, the latest model endpoint with the provided name and project will be updated.
        The attributes parameter is a flatten dictionary which should contain the fields that need to be update.

        :param session:         The database session.
        :param project:         The project name.
        :param attributes:      Dictionary where the key is the model endpoint uids to update
                                and the value are the attribute to update in.
        """
        pass

    def list_model_endpoints(
        self,
        session,
        project: str,
        names: list[str] | None = None,
        function_name: str | None = None,
        function_tag: str | None = None,
        model_name: str | None = None,
        model_tag: str | None = None,
        top_level: bool | None = None,
        modes: list[mlrun.common.schemas.EndpointMode] | None = None,
        labels: list[str] | None = None,
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
        uids: list[str] | None = None,
        latest_only: bool = False,
        offset: int | None = None,
        limit: int | None = None,
        order_by: str | None = None,
        as_dict: bool = False,
    ) -> Union[
        mlrun.common.schemas.ModelEndpointList,
        dict[str, "framework.db.sqldb.models.ModelEndpoint"],
    ]:
        """
        List model endpoints by project and optional filters.

        :param session:         The database session.
        :param project:         The project name.
        :param names:           The model endpoint list of names.
        :param function_name:   The function name.
        :param function_tag:    The function tag.
        :param model_name:      The model name.
        :param model_tag:       The model tag.
        :param top_level:       Whether to return only top level model endpoints (1,2,4).
        :param mode:            Specifies the mode of the model endpoint. Can be "real-time" (0), "batch" (1), or
                                both if set to None.
        :param labels:          The labels to filter by.
        :param start:           The start time to filter by.
        :param end:             The end time to filter by.
        :param uids:            The model endpoint uids to filter by.
        :param latest_only:     Whether to return only the latest model endpoint for each name.
        :param offset:          SQL query offset.
        :param limit:           SQL query limit.
        :param order_by:        Name of column to order by it (in ascending order).
        :param as_dict:         When True, the result will be returned as a dictionary of str in the structure of
                                "<project name>-<function_name>-<function_tag>-<endpoint_name>" map to model
                                endpoint uid.
        :return:                A list of model endpoints.
        """
        pass

    def delete_model_endpoint(
        self,
        session,
        project: str,
        name: str,
        function_name: str | None = None,
        function_tag: str | None = None,
        uid: str | None = None,
    ) -> None:
        """
        Delete a model endpoint by project, name and uid.
        In order to delete all the model endpoints with the same name and project pass uid=*.

        :param session:         The database session.
        :param project:         The project name.
        :param name:            The model endpoint name.
        :param function_name:   The function name.
        :param function_tag:    The function tag.
        :param uid:             The model endpoint uid.
        """
        pass

    def delete_model_endpoints(
        self,
        session,
        project: str,
        uids: list[str] | None = None,
    ) -> None:
        """
        Delete model endpoints across projects and names.

        :param session: The database session.
        :param project: The project name.
        """
        pass

    def delete_feature_sets(
        self,
        session,
        project: str,
        uids: list[str] | None = None,
    ) -> None:
        """
        Delete multiple feature sets.
        :param session: The database session.
        :param project: The project name.
        :param uids:    The feature set uids to delete.
        """
        pass

    def cleanup_old_background_tasks(self, db_session: Session, max_age_seconds: int):
        """
        Cleanup old background tasks that are older than the specified age.
        """
        pass

    def get_partition_interval_for_table(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
    ) -> mlrun.common.schemas.partition_interval.PartitionInterval | None:
        """
        Retrieve the partition interval registered for a specific table, if any.

        :param session: The active SQLAlchemy session used for querying metadata.
        :param table_name: The name of the table to look up.
        :return: The partition interval assigned to the table, or None if not configured.
        """
        pass

    def set_partition_interval_for_table(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        partition_interval: mlrun.common.schemas.partition_interval.PartitionInterval,
    ) -> None:
        """
        Register a partition interval for a table, or validate it if already set.

        If the table already has a different interval registered, an exception is raised
        to prevent inconsistent metadata.

        :param session: The active SQLAlchemy session used for storing metadata.
        :param table_name: The name of the table to update.
        :param partition_interval: The partition interval to set or validate.
        :raises MLRunInvalidArgumentError: If the table already has a conflicting interval.
        """
        pass
