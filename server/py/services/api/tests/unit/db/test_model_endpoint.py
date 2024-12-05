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
#

from datetime import datetime

import pytest

import mlrun
import mlrun.common.schemas
from mlrun.common.schemas import EndpointType, ModelMonitoringMode

from framework.db.sqldb.db import unversioned_tagged_object_uid_prefix
from framework.db.sqldb.models import ModelEndpoint
from framework.tests.unit.db.common_fixtures import TestDatabaseBase


class TestModelEndpoint(TestDatabaseBase):
    @staticmethod
    def _generate_function(
        function_name: str = "function_name_1",
        project: str = "project_name",
        tag: str = "latest",
    ):
        return mlrun.new_function(
            name=function_name,
            project=project,
            tag=tag,
        )

    def _store_function(
        self,
        function_name: str = "function-1",
        project: str = "project-1",
    ) -> str:
        function = self._generate_function(function_name=function_name, project=project)
        function_hash_key = self._db.store_function(
            self._db_session,
            function.to_dict(),
            function.metadata.name,
            function.metadata.project,
        )
        return function_hash_key

    def _store_artifact(self, key: str) -> str:
        artifact = {
            "metadata": {"tree": "artifact_tree", "tag": "latest"},
            "spec": {"src_path": "/some/path"},
            "kind": "model",
            "status": {"bla": "blabla"},
        }
        model_uid = self._db.store_artifact(
            self._db_session,
            key,
            artifact,
            tag="latest",
            project="project-1",
        )
        return model_uid

    def test_sanity(self) -> None:
        uids = []
        model_uids = []
        # store artifact
        for i in range(2):
            model_uids.append(self._store_artifact(f"model-{i}"))
        # store function
        function_hash_key = self._store_function()
        model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={"name": "model-endpoint-1", "project": "project-1"},
            spec={
                "function_name": "function-1",
                "function_uid": f"{unversioned_tagged_object_uid_prefix}latest",
                "model_uid": model_uids[1],
                "model_name": "model-1",
            },
            status={"monitoring_mode": "enabled", "last_request": datetime.now()},
        )
        for i in range(2):
            mep = self._db.store_model_endpoint(
                self._db_session,
                model_endpoint,
                name=model_endpoint.metadata.name,
                project=model_endpoint.metadata.project,
                function_name="function-1",
            )
            model_endpoint_from_db = self._db.get_model_endpoint(
                self._db_session,
                name=model_endpoint.metadata.name,
                project=model_endpoint.metadata.project,
                function_name="function-1",
            )
            assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
            assert model_endpoint_from_db.metadata.project == "project-1"
            assert model_endpoint_from_db.metadata.uid == mep.metadata.uid
            assert (
                model_endpoint_from_db.spec.function_uri
                == f"project-1/function-1@{function_hash_key}"
            )
            assert model_endpoint_from_db.spec.model_name == "model-1"
            uids.append(mep.metadata.uid)

        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            uid=uids[0],
            function_name="function-1",
        )

        assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
        assert model_endpoint_from_db.metadata.project == "project-1"
        assert model_endpoint_from_db.metadata.uid == uids[0]

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
        )
        assert len(list_mep.endpoints) == 2

        self._db.delete_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
            uid="*",
        )
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self._db.get_model_endpoint(
                self._db_session,
                name=model_endpoint.metadata.name,
                project=model_endpoint.metadata.project,
                function_name="function-1",
            )
        for uid in uids:
            with pytest.raises(mlrun.errors.MLRunNotFoundError):
                self._db.get_model_endpoint(
                    self._db_session,
                    name=model_endpoint.metadata.name,
                    project=model_endpoint.metadata.project,
                    uid=uid,
                    function_name="function-1",
                )

    def test_list_filters(self) -> None:
        uids = []
        model_uids = []
        # store artifact
        for i in range(2):
            model_uids.append(self._store_artifact(f"model-{i}"))
        # store function
        _ = self._store_function()
        model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={"name": "model-endpoint-1", "project": "project-1"},
            spec={
                "function_name": "function-1",
                "function_uid": f"{unversioned_tagged_object_uid_prefix}latest",
                "model_uid": model_uids[1],
                "model_name": "model-1",
            },
            status={"monitoring_mode": "enabled"},
        )
        for i in range(2):
            model_endpoint.metadata.labels = {
                "label1": f"value_{i}",
                "label2": f"value_{i+1}",
                "label": "value",
            }
            mep = self._db.store_model_endpoint(
                self._db_session,
                model_endpoint,
                name=model_endpoint.metadata.name,
                project=model_endpoint.metadata.project,
                function_name="function-1",
            )
            uids.append(mep.metadata.uid)

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            model_name="model-1",
        ).endpoints
        assert len(list_mep) == 2

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            model_name="model-2",
        ).endpoints
        assert len(list_mep) == 0

        list_mep = self._db.list_model_endpoints(
            self._db_session, project=model_endpoint.metadata.project, latest_only=True
        ).endpoints
        assert len(list_mep) == 1

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            labels=["label=value"],
        ).endpoints
        assert len(list_mep) == 2

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            labels=["label1=value_0"],
        ).endpoints
        assert len(list_mep) == 1

        list_mep = self._db.list_model_endpoints(
            self._db_session, project=model_endpoint.metadata.project, uids=uids
        ).endpoints
        assert len(list_mep) == 2

        list_mep = self._db.list_model_endpoints(
            self._db_session, project=model_endpoint.metadata.project, uids=["uids"]
        ).endpoints
        assert len(list_mep) == 0

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            function_name="function-1",
        ).endpoints
        assert len(list_mep) == 2

        model_endpoint.metadata.endpoint_type = EndpointType.LEAF_EP
        last_stored_mep = self._db.store_model_endpoint(
            self._db_session,
            model_endpoint,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
        )

        list_mep = self._db.list_model_endpoints(
            self._db_session, project=model_endpoint.metadata.project, top_level=True
        ).endpoints

        assert len(list_mep) == 2

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            start=last_stored_mep.metadata.created,
        ).endpoints
        assert len(list_mep) == 1

        self._db.delete_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
            uid="*",
        )

    def test_update_automatically_after_function_update(self) -> None:
        model_uids = []
        # store artifact
        for i in range(2):
            model_uids.append(self._store_artifact(f"model-{i}"))
        # store function
        function_hash_key = self._store_function()
        model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={"name": "model-endpoint-1", "project": "project-1"},
            spec={
                "function_name": "function-1",
                "function_uid": f"{unversioned_tagged_object_uid_prefix}latest",
                "model_uid": model_uids[1],
                "model_name": "model-1",
            },
            status={"monitoring_mode": "enabled"},
        )
        for i in range(2):
            self._db.store_model_endpoint(
                self._db_session,
                model_endpoint,
                name=model_endpoint.metadata.name,
                project=model_endpoint.metadata.project,
                function_name="function-1",
            )
            self._db.update_function(
                self._db_session,
                "function-1",
                updates={"status": {"state": "error"}},
                project="project-1",
                tag="latest",
            )
            model_endpoint_from_db = self._db.get_model_endpoint(
                self._db_session,
                name=model_endpoint.metadata.name,
                project=model_endpoint.metadata.project,
                function_name="function-1",
            )
            assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
            assert model_endpoint_from_db.metadata.project == "project-1"
            assert (
                model_endpoint_from_db.metadata.labels == model_endpoint.metadata.labels
            )
            assert (
                model_endpoint_from_db.spec.function_uri
                == f"project-1/function-1@{function_hash_key}"
            )
            assert model_endpoint_from_db.spec.model_name == "model-1"
            assert model_endpoint_from_db.status.state == "error"
        self._db.delete_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
            uid="*",
        )

    def test_update_automatically_after_model_update(self) -> None:
        model_uids = []
        # store artifact
        for i in range(2):
            model_uids.append(self._store_artifact(f"model-{i}"))
        # store function
        self._store_function()
        model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={"name": "model-endpoint-1", "project": "project-1"},
            spec={
                "function_name": "function-1",
                "function_uid": f"{unversioned_tagged_object_uid_prefix}latest",
                "model_uid": model_uids[1],
                "model_name": "model-1",
                "model_tag": "latest",
            },
            status={"monitoring_mode": "enabled"},
        )

        self._db.store_model_endpoint(
            self._db_session,
            model_endpoint,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
        )
        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
        )
        assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
        assert model_endpoint_from_db.metadata.project == "project-1"
        assert model_endpoint_from_db.spec.model_name == "model-1"
        assert model_endpoint_from_db.spec.model_tag == "latest"

        artifact = {
            "metadata": {"tree": "artifact_tree"},
            "spec": {"src_path": "/some/new/path"},
            "kind": "model",
            "status": {"bla": "blablasdvcfs"},
        }
        self._db.store_artifact(
            self._db_session,
            f"model-{1}",
            artifact,
            project="project-1",
        )

        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
        )
        assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
        assert model_endpoint_from_db.metadata.project == "project-1"
        assert model_endpoint_from_db.spec.model_name == "model-1"
        assert model_endpoint_from_db.spec.model_tag is None

        self._db.delete_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
            uid="*",
        )

    def test_update(self) -> None:
        model_uids = []
        # store artifact
        for i in range(2):
            model_uids.append(self._store_artifact(f"model-{i}"))
        # store function
        self._store_function()
        model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={"name": "model-endpoint-1", "project": "project-1"},
            spec={
                "function_name": "function-1",
                "function_uid": f"{unversioned_tagged_object_uid_prefix}latest",
                "model_uid": model_uids[1],
                "model_name": "model-1",
            },
            status={"monitoring_mode": "enabled"},
        )
        uids = []
        for i in range(2):
            mep = self._db.store_model_endpoint(
                self._db_session,
                model_endpoint,
                name=model_endpoint.metadata.name,
                project=model_endpoint.metadata.project,
                function_name="function-1",
            )
            uids.append(mep.metadata.uid)

        self._db.update_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
            attributes={"monitoring_mode": ModelMonitoringMode.disabled},
        )

        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
        )
        # check that the monitoring mode was updated for the latest model endpoint
        assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
        assert model_endpoint_from_db.metadata.project == "project-1"
        assert model_endpoint_from_db.metadata.uid == uids[1]
        assert model_endpoint_from_db.status.monitoring_mode == "disabled"

        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            uid=uids[0],
            function_name="function-1",
        )
        # check that the monitoring mode was not updated for the old model endpoint
        assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
        assert model_endpoint_from_db.metadata.project == "project-1"
        assert model_endpoint_from_db.metadata.uid == uids[0]
        assert model_endpoint_from_db.status.monitoring_mode == "enabled"

        self._db.update_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            uid=uids[0],
            attributes={"feature_names": ["a", "b"], "function_uid": "111"},
            function_name="function-1",
        )

        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
        )
        # check that the feature_names value was not updated for the latest model endpoint
        assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
        assert model_endpoint_from_db.metadata.project == "project-1"
        assert model_endpoint_from_db.metadata.uid == uids[1]
        assert model_endpoint_from_db.spec.feature_names == []

        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
            uid=uids[0],
        )
        # check that the feature_names value was updated for the old model endpoint
        assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
        assert model_endpoint_from_db.metadata.project == "project-1"
        assert model_endpoint_from_db.metadata.uid == uids[0]
        assert model_endpoint_from_db.spec.feature_names == ["a", "b"]
        assert model_endpoint_from_db.spec.function_uid == "111"

    def test_delete_model_endpoints(self) -> None:
        model_uids = []
        # store artifact
        for i in range(2):
            model_uids.append(self._store_artifact(f"model-{i}"))
        # store function
        self._store_function()
        model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={"name": "model-endpoint-1", "project": "project-1"},
            spec={
                "function_name": "function-1",
                "function_uid": f"{unversioned_tagged_object_uid_prefix}latest",
                "model_uid": model_uids[1],
                "model_name": "model-1",
            },
            status={"monitoring_mode": "enabled"},
        )
        for i in range(4):
            self._db.store_model_endpoint(
                self._db_session,
                model_endpoint,
                name=model_endpoint.metadata.name,
                project=model_endpoint.metadata.project,
                function_name="function-1",
            )

        assert self._db_session.query(ModelEndpoint.Label).count() == 0
        assert self._db_session.query(ModelEndpoint.Tag).count() == 1
        assert self._db_session.query(ModelEndpoint).count() == 4

        self._db.delete_model_endpoints(
            session=self._db_session, project=model_endpoint.metadata.project
        )

        assert self._db_session.query(ModelEndpoint.Label).count() == 0
        assert self._db_session.query(ModelEndpoint.Tag).count() == 0
        assert self._db_session.query(ModelEndpoint).count() == 0

    def test_insert_without_model(self) -> None:
        function_hash_key = self._store_function()
        model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={"name": "model-endpoint-1", "project": "project-1"},
            spec={
                "function_name": "function-1",
                "function_uid": f"{unversioned_tagged_object_uid_prefix}latest",
            },
            status={"monitoring_mode": "enabled", "last_request": datetime.now()},
        )
        mep = self._db.store_model_endpoint(
            self._db_session,
            model_endpoint,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
        )
        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
        )
        assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
        assert model_endpoint_from_db.metadata.project == "project-1"
        assert model_endpoint_from_db.metadata.uid == mep.metadata.uid
        assert (
            model_endpoint_from_db.spec.function_uri
            == f"project-1/function-1@{function_hash_key}"
        )
        assert model_endpoint_from_db.spec.model_name == ""

    def test_insert_without_function(self) -> None:
        model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={"name": "model-endpoint-1", "project": "project-1"},
            spec={
                "function_name": None,
                "function_uid": None,
            },
            status={"monitoring_mode": "enabled", "last_request": datetime.now()},
        )
        mep = self._db.store_model_endpoint(
            self._db_session,
            model_endpoint,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name=None,
        )
        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name=None,
        )
        assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
        assert model_endpoint_from_db.metadata.project == "project-1"
        assert model_endpoint_from_db.metadata.uid == mep.metadata.uid
        assert model_endpoint_from_db.spec.model_name == ""
        assert model_endpoint_from_db.spec.function_name is None

    def test_2_functions(self) -> None:
        for i in range(2):
            model_endpoint = mlrun.common.schemas.ModelEndpoint(
                metadata={"name": "model-endpoint-1", "project": "project-1"},
                spec={
                    "function_name": f"f-{i}",
                    "function_uid": None,
                },
                status={"monitoring_mode": "enabled", "last_request": datetime.now()},
            )
            self._db.store_model_endpoint(
                self._db_session,
                model_endpoint,
                name=model_endpoint.metadata.name,
                project=model_endpoint.metadata.project,
                function_name=f"f-{i}",
            )

        endpoints = self._db.list_model_endpoints(
            self._db_session, project="project-1", latest_only=True
        ).endpoints
        assert len(endpoints) == 2

        endpoints = self._db.list_model_endpoints(
            self._db_session, project="project-1", function_name="f-0"
        ).endpoints
        assert len(endpoints) == 1
