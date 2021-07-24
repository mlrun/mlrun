# Copyright 2019 Iguazio
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

from typing import List, Optional, Union

import mlrun.api.schemas
from mlrun.api.db.base import DBError
from mlrun.api.db.sqldb.db import SQLDB as SQLAPIDB
from mlrun.api.db.sqldb.session import create_session

# This class is a proxy for the real implementation that sits under mlrun.api.db.sqldb
# The runtime objects (which manages the resources that do the real logic, like Nuclio functions, Dask jobs, etc...)
# require a RunDB to manage their state, when a user run them locally this db will either be the
# local filedb or the remote httpdb (we decided that we don't want to support SQLDB as an optional RunDB).
# When the user submits something to run (task, function etc...) this runtime managers actually runs inside the api
# service, in order to prevent the api from calling itself several times for each submission request (since the runDB
# will be httpdb to that same api service) we have this class which is kind of a proxy between the RunDB interface to
# the api service's DB interface
from ..api.schemas import ModelEndpoint
from .base import RunDBError, RunDBInterface


class SQLDB(RunDBInterface):
    def __init__(self, dsn, session=None, leader_session: Optional[str] = None):
        self.session = session
        self.leader_session = leader_session
        self.dsn = dsn
        self.db = None

    def connect(self, secrets=None):
        if not self.session:
            self.session = create_session()
        self.db = SQLAPIDB(self.dsn)
        return self

    def store_log(self, uid, project="", body=b"", append=False):
        return self._transform_db_error(
            self.db.store_log,
            self.session,
            uid,
            project,
            body,
            append,
            self.leader_session,
        )

    def get_log(self, uid, project="", offset=0, size=0):
        return self._transform_db_error(
            self.db.get_log, self.session, uid, project, offset, size
        )

    def store_run(self, struct, uid, project="", iter=0):
        return self._transform_db_error(
            self.db.store_run,
            self.session,
            struct,
            uid,
            project,
            iter,
            self.leader_session,
        )

    def update_run(self, updates: dict, uid, project="", iter=0):
        return self._transform_db_error(
            self.db.update_run, self.session, updates, uid, project, iter
        )

    def abort_run(self, uid, project="", iter=0):
        raise NotImplementedError()

    def read_run(self, uid, project=None, iter=None):
        return self._transform_db_error(
            self.db.read_run, self.session, uid, project, iter
        )

    def list_runs(
        self,
        name=None,
        uid=None,
        project=None,
        labels=None,
        state=None,
        sort=True,
        last=0,
        iter=None,
    ):
        return self._transform_db_error(
            self.db.list_runs,
            self.session,
            name,
            uid,
            project,
            labels,
            state,
            sort,
            last,
            iter,
        )

    def del_run(self, uid, project=None, iter=None):
        return self._transform_db_error(
            self.db.del_run, self.session, uid, project, iter
        )

    def del_runs(self, name=None, project=None, labels=None, state=None, days_ago=0):
        return self._transform_db_error(
            self.db.del_runs, self.session, name, project, labels, state, days_ago
        )

    def store_artifact(self, key, artifact, uid, iter=None, tag="", project=""):
        return self._transform_db_error(
            self.db.store_artifact,
            self.session,
            key,
            artifact,
            uid,
            iter,
            tag,
            project,
            self.leader_session,
        )

    def read_artifact(self, key, tag="", iter=None, project=""):
        return self._transform_db_error(
            self.db.read_artifact, self.session, key, tag, iter, project
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
    ):
        return self._transform_db_error(
            self.db.list_artifacts,
            self.session,
            name,
            project,
            tag,
            labels,
            since,
            until,
            iter=iter,
            best_iteration=best_iteration,
        )

    def del_artifact(self, key, tag="", project=""):
        return self._transform_db_error(
            self.db.del_artifact, self.session, key, tag, project
        )

    def del_artifacts(self, name="", project="", tag="", labels=None):
        return self._transform_db_error(
            self.db.del_artifacts, self.session, name, project, tag, labels
        )

    def store_function(self, function, name, project="", tag="", versioned=False):
        return self._transform_db_error(
            self.db.store_function,
            self.session,
            function,
            name,
            project,
            tag,
            versioned,
            self.leader_session,
        )

    def get_function(self, name, project="", tag="", hash_key=""):
        return self._transform_db_error(
            self.db.get_function, self.session, name, project, tag, hash_key
        )

    def delete_function(self, name: str, project: str = ""):
        return self._transform_db_error(
            self.db.delete_function, self.session, project, name
        )

    def list_functions(self, name=None, project=None, tag=None, labels=None):
        return self._transform_db_error(
            self.db.list_functions, self.session, name, project, tag, labels
        )

    def list_artifact_tags(self, project=None):
        return self._transform_db_error(
            self.db.list_artifact_tags, self.session, project
        )

    def store_schedule(self, data):
        return self._transform_db_error(self.db.store_schedule, self.session, data)

    def list_schedules(self):
        return self._transform_db_error(self.db.list_schedules, self.session)

    def tag_objects(self, objs, project: str, name: str):
        return self._transform_db_error(
            self.db.tag_objects, self.session, objs, project, name
        )

    def del_tag(self, project: str, name: str):
        return self._transform_db_error(self.db.del_tag, self.session, project, name)

    def find_tagged(self, project: str, name: str):
        return self._transform_db_error(
            self.db.find_tagged, self.session, project, name
        )

    def list_tags(self, project: str):
        return self._transform_db_error(self.db.list_tags, self.session, project)

    def store_project(
        self, name: str, project: mlrun.api.schemas.Project,
    ) -> mlrun.api.schemas.Project:
        raise NotImplementedError()

    def patch_project(
        self,
        name: str,
        project: dict,
        patch_mode: mlrun.api.schemas.PatchMode = mlrun.api.schemas.PatchMode.replace,
    ) -> mlrun.api.schemas.Project:
        raise NotImplementedError()

    def create_project(
        self, project: mlrun.api.schemas.Project,
    ) -> mlrun.api.schemas.Project:
        raise NotImplementedError()

    def delete_project(
        self,
        name: str,
        deletion_strategy: mlrun.api.schemas.DeletionStrategy = mlrun.api.schemas.DeletionStrategy.default(),
    ):
        raise NotImplementedError()

    def get_project(
        self, name: str = None, project_id: int = None
    ) -> mlrun.api.schemas.Project:
        raise NotImplementedError()

    def list_projects(
        self,
        owner: str = None,
        format_: mlrun.api.schemas.Format = mlrun.api.schemas.Format.full,
        labels: List[str] = None,
        state: mlrun.api.schemas.ProjectState = None,
    ) -> mlrun.api.schemas.ProjectsOutput:
        raise NotImplementedError()

    @staticmethod
    def _transform_db_error(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DBError as exc:
            raise RunDBError(exc.args)

    def create_feature_set(self, feature_set, project="", versioned=True):
        return self._transform_db_error(
            self.db.create_feature_set,
            self.session,
            project,
            feature_set,
            versioned,
            self.leader_session,
        )

    def get_feature_set(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ):
        feature_set = self._transform_db_error(
            self.db.get_feature_set, self.session, project, name, tag, uid
        )
        return feature_set.dict()

    def list_features(
        self,
        project: str,
        name: str = None,
        tag: str = None,
        entities: List[str] = None,
        labels: List[str] = None,
    ):
        return self._transform_db_error(
            self.db.list_features, self.session, project, name, tag, entities, labels,
        )

    def list_entities(
        self, project: str, name: str = None, tag: str = None, labels: List[str] = None,
    ):
        return self._transform_db_error(
            self.db.list_entities, self.session, project, name, tag, labels,
        )

    def list_feature_sets(
        self,
        project: str = "",
        name: str = None,
        tag: str = None,
        state: str = None,
        entities: List[str] = None,
        features: List[str] = None,
        labels: List[str] = None,
        partition_by: mlrun.api.schemas.FeatureStorePartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: mlrun.api.schemas.SortField = None,
        partition_order: mlrun.api.schemas.OrderType = mlrun.api.schemas.OrderType.desc,
    ):
        return self._transform_db_error(
            self.db.list_feature_sets,
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
        )

    def store_feature_set(
        self,
        feature_set: Union[dict, mlrun.api.schemas.FeatureSet],
        name=None,
        project="",
        tag=None,
        uid=None,
        versioned=True,
    ):
        if isinstance(feature_set, dict):
            feature_set = mlrun.api.schemas.FeatureSet(**feature_set)

        name = name or feature_set.metadata.name
        project = project or feature_set.metadata.project

        return self._transform_db_error(
            self.db.store_feature_set,
            self.session,
            project,
            name,
            feature_set,
            tag,
            uid,
            versioned,
            self.leader_session,
        )

    def patch_feature_set(
        self, name, feature_set, project="", tag=None, uid=None, patch_mode="replace"
    ):
        return self._transform_db_error(
            self.db.patch_feature_set,
            self.session,
            project,
            name,
            feature_set,
            tag,
            uid,
            patch_mode,
            self.leader_session,
        )

    def delete_feature_set(self, name, project="", tag=None, uid=None):
        return self._transform_db_error(
            self.db.delete_feature_set, self.session, project, name, tag, uid
        )

    def create_feature_vector(self, feature_vector, project="", versioned=True):
        return self._transform_db_error(
            self.db.create_feature_vector,
            self.session,
            project,
            feature_vector,
            versioned,
            self.leader_session,
        )

    def get_feature_vector(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ):
        return self._transform_db_error(
            self.db.get_feature_vector, self.session, project, name, tag, uid,
        )

    def list_feature_vectors(
        self,
        project: str = "",
        name: str = None,
        tag: str = None,
        state: str = None,
        labels: List[str] = None,
        partition_by: mlrun.api.schemas.FeatureStorePartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: mlrun.api.schemas.SortField = None,
        partition_order: mlrun.api.schemas.OrderType = mlrun.api.schemas.OrderType.desc,
    ):
        return self._transform_db_error(
            self.db.list_feature_vectors,
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
        self, feature_vector, name=None, project="", tag=None, uid=None, versioned=True,
    ):
        return self._transform_db_error(
            self.db.store_feature_vector,
            self.session,
            project,
            name,
            feature_vector,
            tag,
            uid,
            versioned,
            self.leader_session,
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
            self.db.patch_feature_vector,
            self.session,
            project,
            name,
            feature_vector_update,
            tag,
            uid,
            patch_mode,
            self.leader_session,
        )

    def delete_feature_vector(self, name, project="", tag=None, uid=None):
        return self._transform_db_error(
            self.db.delete_feature_vector, self.session, project, name, tag, uid
        )

    def list_pipelines(
        self,
        project: str,
        namespace: str = None,
        sort_by: str = "",
        page_token: str = "",
        filter_: str = "",
        format_: Union[
            str, mlrun.api.schemas.Format
        ] = mlrun.api.schemas.Format.metadata_only,
        page_size: int = None,
    ) -> mlrun.api.schemas.PipelinesOutput:
        raise NotImplementedError()

    def create_project_secrets(
        self,
        project: str,
        provider: Union[
            str, mlrun.api.schemas.SecretProviderName
        ] = mlrun.api.schemas.SecretProviderName.vault,
        secrets: dict = None,
    ):
        raise NotImplementedError()

    def list_project_secrets(
        self,
        project: str,
        token: str,
        provider: Union[
            str, mlrun.api.schemas.SecretProviderName
        ] = mlrun.api.schemas.SecretProviderName.vault,
        secrets: List[str] = None,
    ) -> mlrun.api.schemas.SecretsData:
        raise NotImplementedError()

    def list_project_secret_keys(
        self,
        project: str,
        provider: Union[
            str, mlrun.api.schemas.SecretProviderName
        ] = mlrun.api.schemas.SecretProviderName.vault,
        token: str = None,
    ) -> mlrun.api.schemas.SecretKeysData:
        raise NotImplementedError()

    def delete_project_secrets(
        self,
        project: str,
        provider: Union[
            str, mlrun.api.schemas.SecretProviderName
        ] = mlrun.api.schemas.SecretProviderName.vault,
        secrets: List[str] = None,
    ):
        raise NotImplementedError()

    def create_user_secrets(
        self,
        user: str,
        provider: Union[
            str, mlrun.api.schemas.SecretProviderName
        ] = mlrun.api.schemas.SecretProviderName.vault,
        secrets: dict = None,
    ):
        raise NotImplementedError()

    def create_or_patch(
        self,
        project: str,
        endpoint_id: str,
        model_endpoint: ModelEndpoint,
        access_key=None,
    ):
        raise NotImplementedError()

    def delete_endpoint_record(self, project: str, endpoint_id: str, access_key=None):
        raise NotImplementedError()

    def list_endpoints(
        self,
        project: str,
        model: Optional[str] = None,
        function: Optional[str] = None,
        labels: List[str] = None,
        start: str = "now-1h",
        end: str = "now",
        metrics: Optional[List[str]] = None,
        access_key=None,
    ):
        raise NotImplementedError()

    def get_endpoint(
        self,
        project: str,
        endpoint_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        features: bool = False,
        access_key=None,
    ):
        raise NotImplementedError()
