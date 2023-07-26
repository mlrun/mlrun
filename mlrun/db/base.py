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
from typing import List, Optional, Union

import mlrun.common.schemas
import mlrun.model_monitoring.model_endpoint


class RunDBError(Exception):
    pass


class RunDBInterface(ABC):
    kind = ""

    @abstractmethod
    def connect(self, secrets=None):
        return self

    @abstractmethod
    def store_log(self, uid, project="", body=None, append=False):
        pass

    @abstractmethod
    def get_log(self, uid, project="", offset=0, size=0):
        pass

    @abstractmethod
    def store_run(self, struct, uid, project="", iter=0):
        pass

    @abstractmethod
    def update_run(self, updates: dict, uid, project="", iter=0):
        pass

    @abstractmethod
    def abort_run(self, uid, project="", iter=0, timeout=45):
        pass

    @abstractmethod
    def read_run(self, uid, project="", iter=0):
        pass

    @abstractmethod
    def list_runs(
        self,
        name="",
        uid: Optional[Union[str, List[str]]] = None,
        project="",
        labels=None,
        state="",
        sort=True,
        last=0,
        iter=False,
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
        pass

    @abstractmethod
    def del_run(self, uid, project="", iter=0):
        pass

    @abstractmethod
    def del_runs(self, name="", project="", labels=None, state="", days_ago=0):
        pass

    @abstractmethod
    def store_artifact(self, key, artifact, uid, iter=None, tag="", project=""):
        pass

    @abstractmethod
    def read_artifact(self, key, tag="", iter=None, project=""):
        pass

    @abstractmethod
    def list_artifacts(
        self,
        name="",
        project="",
        tag="",
        labels=None,
        since=None,
        until=None,
        iter: int = None,
        best_iteration: bool = False,
        kind: str = None,
        category: Union[str, mlrun.common.schemas.ArtifactCategories] = None,
    ):
        pass

    @abstractmethod
    def del_artifact(self, key, tag="", project=""):
        pass

    @abstractmethod
    def del_artifacts(self, name="", project="", tag="", labels=None):
        pass

    # TODO: Make these abstract once filedb implements them
    def store_metric(self, uid, project="", keyvals=None, timestamp=None, labels=None):
        warnings.warn("store_metric not implemented yet")

    def read_metric(self, keys, project="", query=""):
        warnings.warn("store_metric not implemented yet")

    @abstractmethod
    def store_function(self, function, name, project="", tag="", versioned=False):
        pass

    @abstractmethod
    def get_function(self, name, project="", tag="", hash_key=""):
        pass

    @abstractmethod
    def delete_function(self, name: str, project: str = ""):
        pass

    @abstractmethod
    def list_functions(self, name=None, project="", tag="", labels=None):
        pass

    @abstractmethod
    def tag_objects(
        self,
        project: str,
        tag_name: str,
        tag_objects: mlrun.common.schemas.TagObjects,
        replace: bool = False,
    ):
        pass

    @abstractmethod
    def delete_objects_tag(
        self,
        project: str,
        tag_name: str,
        tag_objects: mlrun.common.schemas.TagObjects,
    ):
        pass

    @abstractmethod
    def tag_artifacts(
        self,
        artifacts,
        project: str,
        tag_name: str,
        replace: bool = False,
    ):
        pass

    @abstractmethod
    def delete_artifacts_tags(
        self,
        artifacts,
        project: str,
        tag_name: str,
    ):
        pass

    @staticmethod
    def _resolve_artifacts_to_tag_objects(
        artifacts,
    ) -> mlrun.common.schemas.TagObjects:
        """
        :param artifacts: Can be a list of :py:class:`~mlrun.artifacts.Artifact` objects or
            dictionaries, or a single object.
        :return: :py:class:`~mlrun.common.schemas.TagObjects`
        """
        # to avoid circular imports we import here
        import mlrun.artifacts.base

        if not isinstance(artifacts, list):
            artifacts = [artifacts]

        artifact_identifiers = []
        for artifact in artifacts:
            artifact_obj = (
                artifact.to_dict()
                if isinstance(artifact, mlrun.artifacts.base.Artifact)
                else artifact
            )
            artifact_identifiers.append(
                mlrun.common.schemas.ArtifactIdentifier(
                    key=mlrun.utils.get_in_artifact(artifact_obj, "key"),
                    # we are passing tree as uid when storing an artifact, so if uid is not defined,
                    # pass the tree as uid
                    uid=mlrun.utils.get_in_artifact(artifact_obj, "uid")
                    or mlrun.utils.get_in_artifact(artifact_obj, "tree"),
                    kind=mlrun.utils.get_in_artifact(artifact_obj, "kind"),
                    iter=mlrun.utils.get_in_artifact(artifact_obj, "iter"),
                )
            )
        return mlrun.common.schemas.TagObjects(
            kind="artifact", identifiers=artifact_identifiers
        )

    @abstractmethod
    def delete_project(
        self,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
    ):
        pass

    @abstractmethod
    def store_project(
        self,
        name: str,
        project: mlrun.common.schemas.Project,
    ) -> mlrun.common.schemas.Project:
        pass

    @abstractmethod
    def patch_project(
        self,
        name: str,
        project: dict,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
    ) -> mlrun.common.schemas.Project:
        pass

    @abstractmethod
    def create_project(
        self,
        project: mlrun.common.schemas.Project,
    ) -> mlrun.common.schemas.Project:
        pass

    @abstractmethod
    def list_projects(
        self,
        owner: str = None,
        format_: mlrun.common.schemas.ProjectsFormat = mlrun.common.schemas.ProjectsFormat.full,
        labels: List[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        pass

    @abstractmethod
    def get_project(self, name: str) -> mlrun.common.schemas.Project:
        pass

    @abstractmethod
    def list_artifact_tags(
        self,
        project=None,
        category: Union[str, mlrun.common.schemas.ArtifactCategories] = None,
    ):
        pass

    @abstractmethod
    def create_feature_set(
        self,
        feature_set: Union[dict, mlrun.common.schemas.FeatureSet],
        project="",
        versioned=True,
    ) -> dict:
        pass

    @abstractmethod
    def get_feature_set(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ) -> dict:
        pass

    @abstractmethod
    def list_features(
        self,
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
        project: str,
        name: str = None,
        tag: str = None,
        labels: List[str] = None,
    ) -> mlrun.common.schemas.EntitiesOutput:
        pass

    @abstractmethod
    def list_feature_sets(
        self,
        project: str = "",
        name: str = None,
        tag: str = None,
        state: str = None,
        entities: List[str] = None,
        features: List[str] = None,
        labels: List[str] = None,
        partition_by: Union[
            mlrun.common.schemas.FeatureStorePartitionByField, str
        ] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Union[mlrun.common.schemas.SortField, str] = None,
        partition_order: Union[
            mlrun.common.schemas.OrderType, str
        ] = mlrun.common.schemas.OrderType.desc,
    ) -> List[dict]:
        pass

    @abstractmethod
    def store_feature_set(
        self,
        feature_set: Union[dict, mlrun.common.schemas.FeatureSet],
        name=None,
        project="",
        tag=None,
        uid=None,
        versioned=True,
    ):
        pass

    @abstractmethod
    def patch_feature_set(
        self,
        name,
        feature_set: dict,
        project="",
        tag=None,
        uid=None,
        patch_mode: Union[
            str, mlrun.common.schemas.PatchMode
        ] = mlrun.common.schemas.PatchMode.replace,
    ):
        pass

    @abstractmethod
    def delete_feature_set(self, name, project="", tag=None, uid=None):
        pass

    @abstractmethod
    def create_feature_vector(
        self,
        feature_vector: Union[dict, mlrun.common.schemas.FeatureVector],
        project="",
        versioned=True,
    ) -> dict:
        pass

    @abstractmethod
    def get_feature_vector(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ) -> dict:
        pass

    @abstractmethod
    def list_feature_vectors(
        self,
        project: str = "",
        name: str = None,
        tag: str = None,
        state: str = None,
        labels: List[str] = None,
        partition_by: Union[
            mlrun.common.schemas.FeatureStorePartitionByField, str
        ] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Union[mlrun.common.schemas.SortField, str] = None,
        partition_order: Union[
            mlrun.common.schemas.OrderType, str
        ] = mlrun.common.schemas.OrderType.desc,
    ) -> List[dict]:
        pass

    @abstractmethod
    def store_feature_vector(
        self,
        feature_vector: Union[dict, mlrun.common.schemas.FeatureVector],
        name=None,
        project="",
        tag=None,
        uid=None,
        versioned=True,
    ):
        pass

    @abstractmethod
    def patch_feature_vector(
        self,
        name,
        feature_vector_update: dict,
        project="",
        tag=None,
        uid=None,
        patch_mode: Union[
            str, mlrun.common.schemas.PatchMode
        ] = mlrun.common.schemas.PatchMode.replace,
    ):
        pass

    @abstractmethod
    def delete_feature_vector(self, name, project="", tag=None, uid=None):
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def create_project_secrets(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: dict = None,
    ):
        pass

    @abstractmethod
    def list_project_secrets(
        self,
        project: str,
        token: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: List[str] = None,
    ) -> mlrun.common.schemas.SecretsData:
        pass

    @abstractmethod
    def list_project_secret_keys(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        token: str = None,
    ) -> mlrun.common.schemas.SecretKeysData:
        pass

    @abstractmethod
    def delete_project_secrets(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: List[str] = None,
    ):
        pass

    @abstractmethod
    def create_user_secrets(
        self,
        user: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.vault,
        secrets: dict = None,
    ):
        pass

    @abstractmethod
    def create_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        model_endpoint: Union[
            mlrun.model_monitoring.model_endpoint.ModelEndpoint, dict
        ],
    ):
        pass

    @abstractmethod
    def delete_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
    ):
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        features: bool = False,
    ):
        pass

    @abstractmethod
    def patch_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        attributes: dict,
    ):
        pass

    @abstractmethod
    def create_hub_source(
        self, source: Union[dict, mlrun.common.schemas.IndexedHubSource]
    ):
        pass

    @abstractmethod
    def store_hub_source(
        self,
        source_name: str,
        source: Union[dict, mlrun.common.schemas.IndexedHubSource],
    ):
        pass

    @abstractmethod
    def list_hub_sources(self):
        pass

    @abstractmethod
    def get_hub_source(self, source_name: str):
        pass

    @abstractmethod
    def delete_hub_source(self, source_name: str):
        pass

    @abstractmethod
    def get_hub_catalog(
        self,
        source_name: str,
        version: str = None,
        tag: str = None,
        force_refresh: bool = False,
    ):
        pass

    @abstractmethod
    def get_hub_item(
        self,
        source_name: str,
        item_name: str,
        version: str = None,
        tag: str = "latest",
        force_refresh: bool = False,
    ):
        pass

    @abstractmethod
    def verify_authorization(
        self,
        authorization_verification_input: mlrun.common.schemas.AuthorizationVerificationInput,
    ):
        pass

    def get_builder_status(
        self,
        func: "mlrun.runtimes.BaseRuntime",
        offset: int = 0,
        logs: bool = True,
        last_log_timestamp: float = 0.0,
        verbose: bool = False,
    ):
        pass

    def set_run_notifications(
        self,
        project: str,
        runs: typing.List[mlrun.model.RunObject],
        notifications: typing.List[mlrun.model.Notification],
    ):
        pass

    def watch_log(self, uid, project="", watch=True, offset=0):
        pass
