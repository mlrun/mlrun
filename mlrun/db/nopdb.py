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

import mlrun.common.schemas
import mlrun.errors

from ..config import config
from ..utils import logger
from .base import RunDBInterface


class NopDB(RunDBInterface):
    def __init__(self, url=None, *args, **kwargs):
        self.url = url

    def __getattribute__(self, attr):
        def nop(*args, **kwargs):
            env_var_message = (
                "MLRUN_DBPATH is misconfigured. Set this environment variable to the URL of the API "
                "server in order to connect"
            )
            if config.httpdb.nop_db.raise_error:
                raise mlrun.errors.MLRunBadRequestError(env_var_message)

            if config.httpdb.nop_db.verbose:
                logger.warning(
                    "Could not detect path to API server, not connected to API server!"
                )
                logger.warning(env_var_message)

            return

        # ignore __class__ because __getattribute__ overrides the parent class's method and it spams logs
        if attr in ["connect", "__class__"]:
            return super().__getattribute__(attr)
        else:
            nop()
            return super().__getattribute__(attr)

    def connect(self, secrets=None):
        pass

    def store_log(self, uid, project="", body=None, append=False):
        pass

    def get_log(self, uid, project="", offset=0, size=0):
        pass

    def store_run(self, struct, uid, project="", iter=0):
        pass

    def update_run(self, updates: dict, uid, project="", iter=0):
        pass

    def abort_run(self, uid, project="", iter=0, timeout=45, status_text=""):
        pass

    def read_run(self, uid, project="", iter=0):
        pass

    def list_runs(
        self,
        name: Optional[str] = None,
        uid: Optional[Union[str, List[str]]] = None,
        project: Optional[str] = None,
        labels: Optional[Union[str, List[str]]] = None,
        state: Optional[str] = None,
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
        pass

    def del_run(self, uid, project="", iter=0):
        pass

    def del_runs(self, name="", project="", labels=None, state="", days_ago=0):
        pass

    def store_artifact(
        self, key, artifact, uid=None, iter=None, tag="", project="", tree=None
    ):
        pass

    def read_artifact(self, key, tag="", iter=None, project="", tree=None, uid=None):
        pass

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
        tree: str = None,
    ):
        pass

    def del_artifact(self, key, tag="", project="", tree=None, uid=None):
        pass

    def del_artifacts(self, name="", project="", tag="", labels=None):
        pass

    def store_function(self, function, name, project="", tag="", versioned=False):
        pass

    def get_function(self, name, project="", tag="", hash_key=""):
        pass

    def delete_function(self, name: str, project: str = ""):
        pass

    def list_functions(self, name=None, project="", tag="", labels=None):
        pass

    def tag_objects(
        self,
        project: str,
        tag_name: str,
        tag_objects: mlrun.common.schemas.TagObjects,
        replace: bool = False,
    ):
        pass

    def delete_objects_tag(
        self, project: str, tag_name: str, tag_objects: mlrun.common.schemas.TagObjects
    ):
        pass

    def tag_artifacts(
        self, artifacts, project: str, tag_name: str, replace: bool = False
    ):
        pass

    def delete_artifacts_tags(self, artifacts, project: str, tag_name: str):
        pass

    def delete_project(
        self,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
    ):
        pass

    def store_project(
        self, name: str, project: mlrun.common.schemas.Project
    ) -> mlrun.common.schemas.Project:
        pass

    def patch_project(
        self,
        name: str,
        project: dict,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
    ) -> mlrun.common.schemas.Project:
        pass

    def create_project(
        self, project: mlrun.common.schemas.Project
    ) -> mlrun.common.schemas.Project:
        pass

    def list_projects(
        self,
        owner: str = None,
        format_: mlrun.common.schemas.ProjectsFormat = mlrun.common.schemas.ProjectsFormat.name_only,
        labels: List[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        pass

    def get_project(self, name: str) -> mlrun.common.schemas.Project:
        pass

    def list_artifact_tags(
        self,
        project=None,
        category: Union[str, mlrun.common.schemas.ArtifactCategories] = None,
    ):
        pass

    def create_feature_set(
        self,
        feature_set: Union[dict, mlrun.common.schemas.FeatureSet],
        project="",
        versioned=True,
    ) -> dict:
        pass

    def get_feature_set(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ) -> dict:
        pass

    def list_features(
        self,
        project: str,
        name: str = None,
        tag: str = None,
        entities: List[str] = None,
        labels: List[str] = None,
    ) -> mlrun.common.schemas.FeaturesOutput:
        pass

    def list_entities(
        self, project: str, name: str = None, tag: str = None, labels: List[str] = None
    ) -> mlrun.common.schemas.EntitiesOutput:
        pass

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

    def delete_feature_set(self, name, project="", tag=None, uid=None):
        pass

    def create_feature_vector(
        self,
        feature_vector: Union[dict, mlrun.common.schemas.FeatureVector],
        project="",
        versioned=True,
    ) -> dict:
        pass

    def get_feature_vector(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ) -> dict:
        pass

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

    def delete_feature_vector(self, name, project="", tag=None, uid=None):
        pass

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

    def create_project_secrets(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: dict = None,
    ):
        pass

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

    def list_project_secret_keys(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        token: str = None,
    ) -> mlrun.common.schemas.SecretKeysData:
        pass

    def delete_project_secrets(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: List[str] = None,
    ):
        pass

    def create_user_secrets(
        self,
        user: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.vault,
        secrets: dict = None,
    ):
        pass

    def create_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        model_endpoint: mlrun.common.schemas.ModelEndpoint,
    ):
        pass

    def delete_model_endpoint(self, project: str, endpoint_id: str):
        pass

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

    def patch_model_endpoint(self, project: str, endpoint_id: str, attributes: dict):
        pass

    def create_hub_source(
        self, source: Union[dict, mlrun.common.schemas.IndexedHubSource]
    ):
        pass

    def store_hub_source(
        self,
        source_name: str,
        source: Union[dict, mlrun.common.schemas.IndexedHubSource],
    ):
        pass

    def list_hub_sources(
        self,
        item_name: Optional[str] = None,
        tag: Optional[str] = None,
        version: Optional[str] = None,
    ):
        pass

    def get_hub_source(self, source_name: str):
        pass

    def delete_hub_source(self, source_name: str):
        pass

    def get_hub_catalog(
        self,
        source_name: str,
        channel: str = None,
        version: str = None,
        tag: str = None,
        force_refresh: bool = False,
    ):
        pass

    def get_hub_item(
        self,
        source_name: str,
        item_name: str,
        channel: str = "development",
        version: str = None,
        tag: str = "latest",
        force_refresh: bool = False,
    ):
        pass

    def verify_authorization(
        self,
        authorization_verification_input: mlrun.common.schemas.AuthorizationVerificationInput,
    ):
        pass

    def get_datastore_profile(
        self, name: str, project: str
    ) -> Optional[mlrun.common.schemas.DatastoreProfile]:
        pass

    def delete_datastore_profile(self, name: str, project: str):
        pass

    def list_datastore_profiles(
        self, project: str
    ) -> List[mlrun.common.schemas.DatastoreProfile]:
        pass

    def store_datastore_profile(
        self, profile: mlrun.common.schemas.DatastoreProfile, project: str
    ):
        pass
