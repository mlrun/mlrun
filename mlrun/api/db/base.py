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

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from mlrun.api import schemas


class DBError(Exception):
    pass


class DBInterface(ABC):
    @abstractmethod
    def initialize(self, session):
        pass

    @abstractmethod
    def store_log(
        self, session, uid, project="", body=None, append=False,
    ):
        pass

    @abstractmethod
    def get_log(self, session, uid, project="", offset=0, size=0):
        pass

    @abstractmethod
    def store_run(
        self, session, struct, uid, project="", iter=0,
    ):
        pass

    @abstractmethod
    def update_run(self, session, updates: dict, uid, project="", iter=0):
        pass

    @abstractmethod
    def read_run(self, session, uid, project="", iter=0):
        pass

    @abstractmethod
    def list_runs(
        self,
        session,
        name="",
        uid=None,
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
        partition_by: schemas.RunPartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: schemas.SortField = None,
        partition_order: schemas.OrderType = schemas.OrderType.desc,
    ):
        pass

    @abstractmethod
    def del_run(self, session, uid, project="", iter=0):
        pass

    @abstractmethod
    def del_runs(self, session, name="", project="", labels=None, state="", days_ago=0):
        pass

    @abstractmethod
    def store_artifact(
        self, session, key, artifact, uid, iter=None, tag="", project="",
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
        category: schemas.ArtifactCategories = None,
        iter: int = None,
        best_iteration: bool = False,
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
        self, session, function, name, project="", tag="", versioned=False,
    ) -> str:
        pass

    @abstractmethod
    def get_function(self, session, name, project="", tag="", hash_key=""):
        pass

    @abstractmethod
    def delete_function(self, session, project: str, name: str):
        pass

    @abstractmethod
    def list_functions(self, session, name=None, project="", tag="", labels=None):
        pass

    @abstractmethod
    def create_schedule(
        self,
        session,
        project: str,
        name: str,
        kind: schemas.ScheduleKinds,
        scheduled_object: Any,
        cron_trigger: schemas.ScheduleCronTrigger,
        concurrency_limit: int,
        labels: Dict = None,
    ):
        pass

    @abstractmethod
    def update_schedule(
        self,
        session,
        project: str,
        name: str,
        scheduled_object: Any = None,
        cron_trigger: schemas.ScheduleCronTrigger = None,
        labels: Dict = None,
        last_run_uri: str = None,
        concurrency_limit: int = None,
    ):
        pass

    @abstractmethod
    def list_schedules(
        self,
        session,
        project: str = None,
        name: str = None,
        labels: str = None,
        kind: schemas.ScheduleKinds = None,
    ) -> List[schemas.ScheduleRecord]:
        pass

    @abstractmethod
    def get_schedule(self, session, project: str, name: str) -> schemas.ScheduleRecord:
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
    ) -> List[schemas.ProjectSummary]:
        pass

    @abstractmethod
    def delete_project_related_resources(self, session, name: str):
        pass

    @abstractmethod
    def verify_project_has_no_related_resources(self, session, name: str):
        pass

    @abstractmethod
    # adding **kwargs to leave room for other projects store implementations see mlrun.api.crud.projects.delete_project
    # for explanations
    def is_project_exists(self, session, name: str, **kwargs):
        pass

    @abstractmethod
    def list_projects(
        self,
        session,
        owner: str = None,
        format_: schemas.ProjectsFormat = schemas.ProjectsFormat.full,
        labels: List[str] = None,
        state: schemas.ProjectState = None,
        names: Optional[List[str]] = None,
    ) -> schemas.ProjectsOutput:
        pass

    @abstractmethod
    def get_project(
        self, session, name: str = None, project_id: int = None
    ) -> schemas.Project:
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
    def create_project(self, session, project: schemas.Project):
        pass

    @abstractmethod
    def store_project(self, session, name: str, project: schemas.Project):
        pass

    @abstractmethod
    def patch_project(
        self,
        session,
        name: str,
        project: dict,
        patch_mode: schemas.PatchMode = schemas.PatchMode.replace,
    ):
        pass

    @abstractmethod
    def delete_project(
        self,
        session,
        name: str,
        deletion_strategy: schemas.DeletionStrategy = schemas.DeletionStrategy.default(),
    ):
        pass

    @abstractmethod
    def create_feature_set(
        self, session, project, feature_set: schemas.FeatureSet, versioned=True,
    ) -> str:
        pass

    @abstractmethod
    def store_feature_set(
        self,
        session,
        project,
        name,
        feature_set: schemas.FeatureSet,
        tag=None,
        uid=None,
        versioned=True,
        always_overwrite=False,
    ) -> str:
        pass

    @abstractmethod
    def get_feature_set(
        self, session, project: str, name: str, tag: str = None, uid: str = None
    ) -> schemas.FeatureSet:
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
    ) -> schemas.FeaturesOutput:
        pass

    @abstractmethod
    def list_entities(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        labels: List[str] = None,
    ) -> schemas.EntitiesOutput:
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
        partition_by: schemas.FeatureStorePartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: schemas.SortField = None,
        partition_order: schemas.OrderType = schemas.OrderType.desc,
    ) -> schemas.FeatureSetsOutput:
        pass

    @abstractmethod
    def list_feature_sets_tags(
        self, session, project: str,
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
        patch_mode: schemas.PatchMode = schemas.PatchMode.replace,
    ) -> str:
        pass

    @abstractmethod
    def delete_feature_set(self, session, project, name, tag=None, uid=None):
        pass

    @abstractmethod
    def create_feature_vector(
        self, session, project, feature_vector: schemas.FeatureVector, versioned=True,
    ) -> str:
        pass

    @abstractmethod
    def get_feature_vector(
        self, session, project: str, name: str, tag: str = None, uid: str = None
    ) -> schemas.FeatureVector:
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
        partition_by: schemas.FeatureStorePartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: schemas.SortField = None,
        partition_order: schemas.OrderType = schemas.OrderType.desc,
    ) -> schemas.FeatureVectorsOutput:
        pass

    @abstractmethod
    def list_feature_vectors_tags(
        self, session, project: str,
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
        feature_vector: schemas.FeatureVector,
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
        patch_mode: schemas.PatchMode = schemas.PatchMode.replace,
    ) -> str:
        pass

    @abstractmethod
    def delete_feature_vector(
        self, session, project, name, tag=None, uid=None,
    ):
        pass

    def list_artifact_tags(self, session, project):
        return []

    def create_marketplace_source(
        self, session, ordered_source: schemas.IndexedMarketplaceSource
    ):
        pass

    def store_marketplace_source(
        self, session, name, ordered_source: schemas.IndexedMarketplaceSource
    ):
        pass

    def list_marketplace_sources(
        self, session
    ) -> List[schemas.IndexedMarketplaceSource]:
        pass

    def delete_marketplace_source(self, session, name):
        pass

    def get_marketplace_source(self, session, name) -> schemas.IndexedMarketplaceSource:
        pass
