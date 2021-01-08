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
from typing import List, Union

from mlrun.api import schemas


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
    def read_run(self, uid, project="", iter=0):
        pass

    @abstractmethod
    def list_runs(
        self,
        name="",
        uid=None,
        project="",
        labels=None,
        state="",
        sort=True,
        last=0,
        iter=False,
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
        self, name="", project="", tag="", labels=None, since=None, until=None
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
    def delete_project(
        self,
        name: str,
        deletion_strategy: schemas.DeletionStrategy = schemas.DeletionStrategy.default(),
    ):
        pass

    @abstractmethod
    def store_project(self, name: str, project: schemas.Project,) -> schemas.Project:
        pass

    @abstractmethod
    def patch_project(
        self,
        name: str,
        project: dict,
        patch_mode: schemas.PatchMode = schemas.PatchMode.replace,
    ) -> schemas.Project:
        pass

    @abstractmethod
    def create_project(self, project: schemas.Project,) -> schemas.Project:
        pass

    @abstractmethod
    def list_projects(
        self,
        owner: str = None,
        format_: schemas.Format = schemas.Format.full,
        labels: List[str] = None,
        state: schemas.ProjectState = None,
    ) -> schemas.ProjectsOutput:
        pass

    @abstractmethod
    def get_project(self, name: str) -> schemas.Project:
        pass

    @abstractmethod
    def list_artifact_tags(self, project=None):
        pass

    @abstractmethod
    def create_feature_set(
        self, feature_set: Union[dict, schemas.FeatureSet], project="", versioned=True
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
    ) -> schemas.FeaturesOutput:
        pass

    @abstractmethod
    def list_entities(
        self, project: str, name: str = None, tag: str = None, labels: List[str] = None,
    ) -> schemas.EntitiesOutput:
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
    ) -> List[dict]:
        pass

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def delete_feature_set(self, name, project=""):
        pass

    @abstractmethod
    def create_feature_vector(
        self,
        feature_vector: Union[dict, schemas.FeatureVector],
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
    ) -> List[dict]:
        pass

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def delete_feature_vector(self, name, project=""):
        pass

    @abstractmethod
    def list_pipelines(
        self,
        project: str,
        namespace: str = None,
        sort_by: str = "",
        page_token: str = "",
        filter_: str = "",
        format_: Union[str, schemas.Format] = schemas.Format.metadata_only,
        page_size: int = None,
    ) -> schemas.PipelinesOutput:
        pass

    @abstractmethod
    def create_project_secrets(
        self,
        project: str,
        provider: Union[
            str, schemas.SecretProviderName
        ] = schemas.SecretProviderName.vault,
        secrets: dict = None,
    ):
        pass

    def get_project_secrets(
        self,
        project: str,
        token: str,
        provider: Union[
            str, schemas.SecretProviderName
        ] = schemas.SecretProviderName.vault,
        secrets: List[str] = None,
    ) -> schemas.SecretsData:
        pass

    @abstractmethod
    def create_user_secrets(
        self,
        user: str,
        provider: Union[
            str, schemas.SecretProviderName
        ] = schemas.SecretProviderName.vault,
        secrets: dict = None,
    ):
        pass
