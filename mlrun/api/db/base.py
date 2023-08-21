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
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import mlrun.common.schemas
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
        struct,
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
        requested_logs_modes: List[bool] = None,
        only_uids: bool = False,
        last_update_time_from: datetime.datetime = None,
        states: List[str] = None,
    ):
        pass

    @abstractmethod
    def update_runs_requested_logs(
        self, session, uids: List[str], requested_logs: bool = True
    ):
        pass

    @abstractmethod
    def read_run(self, session, uid, project="", iter=0):
        pass

    @abstractmethod
    def list_runs(
        self,
        session,
        name="",
        uid: Optional[Union[str, List[str]]] = None,
        project="",
        labels=None,
        states=None,
        sort=True,
        last=0,
        iter=False,
        start_time_from=None,
        start_time_to=None,
        last_update_time_from=None,
        last_update_time_to=None,
        partition_by: mlrun.common.schemas.RunPartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: mlrun.common.schemas.SortField = None,
        partition_order: mlrun.common.schemas.OrderType = mlrun.common.schemas.OrderType.desc,
        max_partitions: int = 0,
        requested_logs: bool = None,
        return_as_run_structs: bool = True,
        with_notifications: bool = False,
    ):
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
        identifiers: List[mlrun.common.schemas.ArtifactIdentifier],
    ):
        pass

    def append_tag_to_artifacts(
        self,
        session,
        project: str,
        tag: str,
        identifiers: List[mlrun.common.schemas.ArtifactIdentifier],
    ):
        pass

    def delete_tag_from_artifacts(
        self,
        session,
        project: str,
        tag: str,
        identifiers: List[mlrun.common.schemas.ArtifactIdentifier],
    ):
        pass

    @abstractmethod
    def store_artifact(
        self,
        session,
        key,
        artifact,
        uid,
        iter=None,
        tag="",
        project="",
    ):
        pass

    @abstractmethod
    def read_artifact(self, session, key, tag="", iter=None, project=""):
        pass

    @abstractmethod
    def list_artifacts(
        self,
        session,
        name="",
        project="",
        tag="",
        labels=None,
        since=None,
        until=None,
        kind=None,
        category: mlrun.common.schemas.ArtifactCategories = None,
        iter: int = None,
        best_iteration: bool = False,
        as_records: bool = False,
        use_tag_as_uid: bool = None,
    ):
        pass

    @abstractmethod
    def del_artifact(self, session, key, tag="", project=""):
        pass

    @abstractmethod
    def del_artifacts(self, session, name="", project="", tag="", labels=None):
        pass

    # TODO: Make these abstract once filedb implements them
    def store_metric(
        self, session, uid, project="", keyvals=None, timestamp=None, labels=None
    ):
        warnings.warn("store_metric not implemented yet")

    def read_metric(self, session, keys, project="", query=""):
        warnings.warn("store_metric not implemented yet")

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
    def get_function(self, session, name, project="", tag="", hash_key=""):
        pass

    @abstractmethod
    def delete_function(self, session, project: str, name: str):
        pass

    @abstractmethod
    def list_functions(
        self,
        session,
        name: str = None,
        project: str = None,
        tag: str = None,
        labels: List[str] = None,
        hash_key: str = None,
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
        labels: Dict = None,
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
        labels: Dict = None,
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
        labels: Dict = None,
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
        labels: str = None,
        kind: mlrun.common.schemas.ScheduleKinds = None,
    ) -> List[mlrun.common.schemas.ScheduleRecord]:
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
    def delete_schedules(self, session, project: str):
        pass

    @abstractmethod
    def generate_projects_summaries(
        self, session, projects: List[str]
    ) -> List[mlrun.common.schemas.ProjectSummary]:
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
        format_: mlrun.common.schemas.ProjectsFormat = mlrun.common.schemas.ProjectsFormat.full,
        labels: List[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: Optional[List[str]] = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        pass

    @abstractmethod
    def get_project(
        self, session, name: str = None, project_id: int = None
    ) -> mlrun.common.schemas.Project:
        pass

    @abstractmethod
    async def get_project_resources_counters(
        self,
    ) -> Tuple[
        Dict[str, int],
        Dict[str, int],
        Dict[str, int],
        Dict[str, int],
        Dict[str, int],
        Dict[str, int],
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

    @abstractmethod
    def list_features(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        entities: List[str] = None,
        labels: List[str] = None,
    ) -> mlrun.common.schemas.FeaturesOutput:
        pass

    @abstractmethod
    def list_entities(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        labels: List[str] = None,
    ) -> mlrun.common.schemas.EntitiesOutput:
        pass

    @abstractmethod
    def list_feature_sets(
        self,
        session,
        project: str,
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
    ) -> mlrun.common.schemas.FeatureSetsOutput:
        pass

    @abstractmethod
    def list_feature_sets_tags(
        self,
        session,
        project: str,
    ) -> List[Tuple[str, str, str]]:
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
        labels: List[str] = None,
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
    ) -> List[Tuple[str, str, str]]:
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

    def list_artifact_tags(
        self,
        session,
        project,
        category: Union[str, mlrun.common.schemas.ArtifactCategories] = None,
    ):
        return []

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

    def list_hub_sources(self, session) -> List[mlrun.common.schemas.IndexedHubSource]:
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
    ):
        pass

    def get_background_task(
        self, session, name: str, project: str
    ) -> mlrun.common.schemas.BackgroundTask:
        pass

    @abstractmethod
    def store_run_notifications(
        self,
        session,
        notification_objects: typing.List[mlrun.model.Notification],
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
    ) -> typing.List[mlrun.model.Notification]:
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
        notifications: typing.List[mlrun.model.Notification],
        identifiers: typing.List[mlrun.common.schemas.RunIdentifier],
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
    ) -> List[mlrun.common.schemas.DatastoreProfile]:
        pass
