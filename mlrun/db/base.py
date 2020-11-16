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
    def delete_project(self, name: str):
        pass

    def list_projects(self):
        return []

    def list_artifact_tags(self, project):
        return []

    @abstractmethod
    def create_feature_set(
        self, feature_set: Union[dict, schemas.FeatureSet], project="", versioned=True
    ) -> schemas.FeatureSet:
        pass

    @abstractmethod
    def get_feature_set(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ) -> schemas.FeatureSet:
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
    def list_feature_sets(
        self,
        project: str = "",
        name: str = None,
        tag: str = None,
        state: str = None,
        entities: List[str] = None,
        features: List[str] = None,
        labels: List[str] = None,
    ) -> schemas.FeatureSetsOutput:
        pass

    @abstractmethod
    def store_feature_set(
        self,
        name,
        feature_set: Union[dict, schemas.FeatureSet],
        project="",
        tag=None,
        uid=None,
        versioned=True,
    ):
        pass

    @abstractmethod
    def update_feature_set(
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
