# Copyright 2022 Iguazio
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

import mlrun.errors

from ..api import schemas
from ..api.schemas import ModelEndpoint
from ..config import config
from ..utils import logger
from .base import RunDBInterface


class NopDB(RunDBInterface):
    def __init__(self, url=None, *args, **kwargs):
        self.url = url

    def __getattribute__(self, attr):
        def nop(*args, **kwargs):
            if config.httpdb.nop_db.raise_error:
                raise mlrun.errors.MLRunBadRequestError(
                    "MLRUN_DB_PATH is not set. Please set this environment variable to the URL of the API server in"
                    " order to connect"
                )

            if config.httpdb.nop_db.verbose:
                logger.warning(
                    "Could not detect path to API server, not connected to API server!"
                )
                logger.warning(
                    "MLRUN_DB_PATH is not set. Please set this environment variable to the URL of the API server in"
                    " order to connect"
                )

            return

        if attr == "connect":
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

    def abort_run(self, uid, project="", iter=0):
        pass

    def read_run(self, uid, project="", iter=0):
        pass

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
        partition_by: Union[schemas.RunPartitionByField, str] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Union[schemas.SortField, str] = None,
        partition_order: Union[schemas.OrderType, str] = schemas.OrderType.desc,
        max_partitions: int = 0,
    ):
        pass

    def del_run(self, uid, project="", iter=0):
        pass

    def del_runs(self, name="", project="", labels=None, state="", days_ago=0):
        pass

    def store_artifact(self, key, artifact, uid, iter=None, tag="", project=""):
        pass

    def read_artifact(self, key, tag="", iter=None, project=""):
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
        category: Union[str, schemas.ArtifactCategories] = None,
    ):
        pass

    def del_artifact(self, key, tag="", project=""):
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
        tag_objects: schemas.TagObjects,
        replace: bool = False,
    ):
        pass

    def delete_objects_tag(
        self, project: str, tag_name: str, tag_objects: schemas.TagObjects
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
        deletion_strategy: schemas.DeletionStrategy = schemas.DeletionStrategy.default(),
    ):
        pass

    def store_project(self, name: str, project: schemas.Project) -> schemas.Project:
        pass

    def patch_project(
        self,
        name: str,
        project: dict,
        patch_mode: schemas.PatchMode = schemas.PatchMode.replace,
    ) -> schemas.Project:
        pass

    def create_project(self, project: schemas.Project) -> schemas.Project:
        pass

    def list_projects(
        self,
        owner: str = None,
        format_: schemas.ProjectsFormat = schemas.ProjectsFormat.full,
        labels: List[str] = None,
        state: schemas.ProjectState = None,
    ) -> schemas.ProjectsOutput:
        pass

    def get_project(self, name: str) -> schemas.Project:
        pass

    def list_artifact_tags(
        self, project=None, category: Union[str, schemas.ArtifactCategories] = None
    ):
        pass

    def create_feature_set(
        self, feature_set: Union[dict, schemas.FeatureSet], project="", versioned=True
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
    ) -> schemas.FeaturesOutput:
        pass

    def list_entities(
        self, project: str, name: str = None, tag: str = None, labels: List[str] = None
    ) -> schemas.EntitiesOutput:
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
        partition_by: Union[schemas.FeatureStorePartitionByField, str] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Union[schemas.SortField, str] = None,
        partition_order: Union[schemas.OrderType, str] = schemas.OrderType.desc,
    ) -> List[dict]:
        pass

    def store_feature_set(
        self,
        feature_set: Union[dict, schemas.FeatureSet],
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
        patch_mode: Union[str, schemas.PatchMode] = schemas.PatchMode.replace,
    ):
        pass

    def delete_feature_set(self, name, project="", tag=None, uid=None):
        pass

    def create_feature_vector(
        self,
        feature_vector: Union[dict, schemas.FeatureVector],
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
        partition_by: Union[schemas.FeatureStorePartitionByField, str] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Union[schemas.SortField, str] = None,
        partition_order: Union[schemas.OrderType, str] = schemas.OrderType.desc,
    ) -> List[dict]:
        pass

    def store_feature_vector(
        self,
        feature_vector: Union[dict, schemas.FeatureVector],
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
        patch_mode: Union[str, schemas.PatchMode] = schemas.PatchMode.replace,
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
            str, schemas.PipelinesFormat
        ] = schemas.PipelinesFormat.metadata_only,
        page_size: int = None,
    ) -> schemas.PipelinesOutput:
        pass

    def create_project_secrets(
        self,
        project: str,
        provider: Union[
            str, schemas.SecretProviderName
        ] = schemas.SecretProviderName.kubernetes,
        secrets: dict = None,
    ):
        pass

    def list_project_secrets(
        self,
        project: str,
        token: str,
        provider: Union[
            str, schemas.SecretProviderName
        ] = schemas.SecretProviderName.kubernetes,
        secrets: List[str] = None,
    ) -> schemas.SecretsData:
        pass

    def list_project_secret_keys(
        self,
        project: str,
        provider: Union[
            str, schemas.SecretProviderName
        ] = schemas.SecretProviderName.kubernetes,
        token: str = None,
    ) -> schemas.SecretKeysData:
        pass

    def delete_project_secrets(
        self,
        project: str,
        provider: Union[
            str, schemas.SecretProviderName
        ] = schemas.SecretProviderName.kubernetes,
        secrets: List[str] = None,
    ):
        pass

    def create_user_secrets(
        self,
        user: str,
        provider: Union[
            str, schemas.SecretProviderName
        ] = schemas.SecretProviderName.vault,
        secrets: dict = None,
    ):
        pass

    def create_model_endpoint(
        self, project: str, endpoint_id: str, model_endpoint: ModelEndpoint
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

    def create_marketplace_source(
        self, source: Union[dict, schemas.IndexedMarketplaceSource]
    ):
        pass

    def store_marketplace_source(
        self, source_name: str, source: Union[dict, schemas.IndexedMarketplaceSource]
    ):
        pass

    def list_marketplace_sources(self):
        pass

    def get_marketplace_source(self, source_name: str):
        pass

    def delete_marketplace_source(self, source_name: str):
        pass

    def get_marketplace_catalog(
        self,
        source_name: str,
        channel: str = None,
        version: str = None,
        tag: str = None,
        force_refresh: bool = False,
    ):
        pass

    def get_marketplace_item(
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
        self, authorization_verification_input: schemas.AuthorizationVerificationInput
    ):
        pass
