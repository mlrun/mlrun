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
import uuid
from datetime import datetime
from typing import Optional

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
        tag: Optional[str] = None,
    ) -> None:
        function = self._generate_function(
            function_name=function_name, project=project, tag=tag or "latest"
        )
        self._db.store_function(
            self._db_session,
            function.to_dict(),
            function.metadata.name,
            function.metadata.project,
            function.metadata.tag,
            versioned=False,
        )

    def _store_artifact(
        self, key: str, uid: Optional[str] = None, status: Optional[dict] = None
    ) -> str:
        artifact = {
            "metadata": {"tree": "artifact_tree", "tag": "latest"},
            "spec": {"src_path": "/some/path"},
            "kind": "model",
            "status": status or {"bla": "blabla"},
        }
        return self._db.store_artifact(
            self._db_session,
            key,
            artifact,
            tag="latest",
            project="project-1",
            uid=uid,
        )

    def test_sanity(self) -> None:
        uids = []
        # store artifact
        for i in range(2):
            self._store_artifact(f"model-{i}")
        # store function
        self._store_function()
        model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={"name": "model-endpoint-1", "project": "project-1"},
            spec={
                "function_name": "function-1",
                "function_tag": "latest",
            },
            status={"monitoring_mode": "enabled", "last_request": datetime.now()},
        )
        for i in range(2):
            model_endpoint.spec._model_id = i + 1
            uid = self._db.store_model_endpoint(
                self._db_session,
                model_endpoint,
            )
            self._db.list_model_endpoints(self._db_session, "project-1")
            model_endpoint_from_db = self._db.get_model_endpoint(
                self._db_session,
                name=model_endpoint.metadata.name,
                project=model_endpoint.metadata.project,
                function_name="function-1",
                function_tag="latest",
            )
            assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
            assert model_endpoint_from_db.metadata.project == "project-1"
            assert model_endpoint_from_db.metadata.uid == uid
            assert (
                model_endpoint_from_db.spec.function_uri
                == f"project-1/function-1@{unversioned_tagged_object_uid_prefix}latest"
            )
            assert model_endpoint_from_db.spec.model_name == f"model-{i}"
            assert is_hex(
                model_endpoint_from_db.metadata.uid
            ), "expected uid as hex value"
            uids.append(uid)

        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            uid=uids[0],
            function_name="function-1",
            function_tag="latest",
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
            uid="*",
            function_name="function-1",
            function_tag="latest",
        )
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self._db.get_model_endpoint(
                self._db_session,
                name=model_endpoint.metadata.name,
                project=model_endpoint.metadata.project,
                function_name="function-1",
                function_tag="latest",
            )
        for uid in uids:
            with pytest.raises(mlrun.errors.MLRunNotFoundError):
                self._db.get_model_endpoint(
                    self._db_session,
                    name=model_endpoint.metadata.name,
                    project=model_endpoint.metadata.project,
                    uid=uid,
                )

    def test_batch_insert_and_update(self) -> None:
        # store artifact
        for i in range(2):
            self._store_artifact(f"model-{i}")
        # store function
        self._store_function()
        model_endpoint_1 = mlrun.common.schemas.ModelEndpoint(
            metadata={
                "name": "model-endpoint-1",
                "project": "project-1",
                "uid": "5cfeed6672cc4d978ff9b7b06ebe77f2",
            },
            spec={
                "function_name": "function-1",
                "function_tag": "latest",
                "_model_id": 2,
            },
            status={"monitoring_mode": "enabled", "last_request": datetime.now()},
        )

        model_endpoint_2 = mlrun.common.schemas.ModelEndpoint(
            metadata={
                "name": "model-endpoint-2",
                "project": "project-1",
                "uid": "2127986e91f544af9be31250295f03b6",
            },
            spec={
                "function_name": "function-1",
                "function_tag": "latest",
                "_model_id": 2,
            },
            status={"monitoring_mode": "enabled", "last_request": datetime.now()},
        )

        self._db.store_model_endpoints(
            self._db_session,
            [model_endpoint_1, model_endpoint_2],
            "function-1",
            "latest",
            "project-1",
        )

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project="project-1",
        )
        assert len(list_mep.endpoints) == 2

        self._db.update_model_endpoints(
            self._db_session,
            "project-1",
            {
                uuid.UUID("5cfeed6672cc4d978ff9b7b06ebe77f2").hex: {
                    "monitoring_mode": ModelMonitoringMode.disabled
                },
                uuid.UUID("2127986e91f544af9be31250295f03b6").hex: {
                    "model_class": "new_class"
                },
            },
        )

        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint_1.metadata.name,
            project=model_endpoint_1.metadata.project,
            function_name="function-1",
            function_tag="latest",
        )
        assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
        assert model_endpoint_from_db.metadata.project == "project-1"
        assert model_endpoint_from_db.metadata.uid == "5cfeed6672cc4d978ff9b7b06ebe77f2"

        # assert model_endpoint_from_db.status.monitoring_mode == "disabled"

        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint_2.metadata.name,
            project=model_endpoint_2.metadata.project,
            function_name="function-1",
            function_tag="latest",
        )
        assert model_endpoint_from_db.metadata.name == "model-endpoint-2"
        assert model_endpoint_from_db.metadata.project == "project-1"
        assert model_endpoint_from_db.metadata.uid == "2127986e91f544af9be31250295f03b6"

        assert model_endpoint_from_db.spec.model_class == "new_class"

    def test_list_filters(self) -> None:
        uids = []
        # store artifact
        for i in range(3):
            self._store_artifact(f"model-{i}")
        # store function
        self._store_function()
        self._store_function(tag="v1")
        model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={
                "name": "model-endpoint-1",
                "project": "project-1",
                "mode": mlrun.common.schemas.EndpointMode.REAL_TIME,
            },
            spec={
                "function_name": "function-1",
                "function_tag": "latest",
                "_model_id": 2,
            },
            status={"monitoring_mode": "enabled"},
        )
        different_name_model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={
                "name": "model-endpoint-2",
                "project": "project-1",
                "mode": mlrun.common.schemas.EndpointMode.BATCH,
                "endpoint_type": EndpointType.BATCH_EP,
            },
            spec={
                "function_name": "function-1",
                "function_tag": "latest",
                "_model_id": 2,
            },
            status={"monitoring_mode": "enabled"},
        )
        for i in range(2):
            model_endpoint.metadata.labels = {
                "label1": f"value_{i}",
                "label2": f"value_{i+1}",
                "label": "value",
            }
            uid = self._db.store_model_endpoint(
                self._db_session,
                model_endpoint,
            )
            uids.append(uid)

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
            self._db_session,
            project=model_endpoint.metadata.project,
            uids=[uuid.UUID("f65cf291-2829-46f3-a5ba-a04c1cfefc19")],
        ).endpoints
        assert len(list_mep) == 0

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            latest_only=True,
            names=["model-endpoint-1"],
        ).endpoints
        assert len(list_mep) == 1

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            names=["model-endpoint-1"],
        ).endpoints
        assert len(list_mep) == 2

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            function_name="function-1",
        ).endpoints
        assert len(list_mep) == 2

        model_endpoint.metadata.endpoint_type = EndpointType.LEAF_EP
        model_endpoint.spec.function_tag = "v1"
        last_stored_mep_uid = self._db.store_model_endpoint(
            self._db_session,
            model_endpoint,
        )
        last_stored_mep = self._db.get_model_endpoint(
            self._db_session,
            uid=last_stored_mep_uid,
            project="project-1",
            function_name="function-1",
            function_tag="v1",
            name="model-endpoint-1",
        )

        list_mep = self._db.list_model_endpoints(
            self._db_session, project=model_endpoint.metadata.project, top_level=True
        ).endpoints

        assert len(list_mep) == 2

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            latest_only=True,
        ).endpoints

        assert len(list_mep) == 2

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            function_name="function-1",
            function_tag="v1",
        ).endpoints

        assert len(list_mep) == 1

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
            function_tag="v1",
            uid="*",
        )

        self._db.delete_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            uid=uids[0],
        )

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            function_name="function-1",
            function_tag="v1",
        ).endpoints
        assert len(list_mep) == 0

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            function_name="function-1",
        ).endpoints
        assert len(list_mep) == 1

        self._db.store_model_endpoint(
            self._db_session,
            different_name_model_endpoint,
        )
        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            latest_only=True,
            names=["model-endpoint-1", "model-endpoint-2"],
        ).endpoints
        assert len(list_mep) == 2

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            latest_only=True,
            project=model_endpoint.metadata.project,
            names=["model-endpoint-1"],
        ).endpoints
        assert len(list_mep) == 1

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            modes=[
                mlrun.common.schemas.EndpointMode.REAL_TIME,
                mlrun.common.schemas.EndpointMode.BATCH,
            ],
        ).endpoints
        assert len(list_mep) == 2

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            modes=[mlrun.common.schemas.EndpointMode.REAL_TIME],
        ).endpoints
        assert len(list_mep) == 1

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            modes=[mlrun.common.schemas.EndpointMode.BATCH],
        ).endpoints
        assert len(list_mep) == 1

    def test_latest_only(self) -> None:
        # store artifact
        for i in range(3):
            self._store_artifact(f"model-{i}")
        # store functions
        self._store_function(function_name="function-1")
        self._store_function(function_name="function-2", tag="v2")
        model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={
                "name": "model-endpoint-1",
                "project": "project-1",
                "uid": "5cfeed66-72cc-4d97-8ff9-b7b06ebe77f2",
            },
            spec={
                "function_name": "function-1",
                "function_tag": "latest",
                "_model_id": 2,
            },
            status={"monitoring_mode": "enabled"},
        )

        batch_model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={
                "name": "model-endpoint-2",
                "project": "project-1",
                "endpoint_type": EndpointType.BATCH_EP,
            },
            spec={
                "_model_id": 2,
                "function_name": "function-2",
                "function_tag": "v2",
            },
            status={"monitoring_mode": "enabled"},
        )

        self._db.store_model_endpoint(
            self._db_session,
            model_endpoint,
        )

        self._db.store_model_endpoint(
            self._db_session,
            batch_model_endpoint,
        )

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            latest_only=True,
            project=model_endpoint.metadata.project,
        ).endpoints

        # expecting two model endpoints that are the latest
        assert len(list_mep) == 2
        assert list_mep[0].metadata.uid == "5cfeed6672cc4d978ff9b7b06ebe77f2"

        # store another model endpoint with the same name but different uid
        model_endpoint.metadata.uid = "2127986e91f544af9be31250295f03b6"
        self._db.store_model_endpoint(
            self._db_session,
            model_endpoint,
        )

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
        ).endpoints

        # expecting 3 model endpoints because we don't filter by latest
        assert len(list_mep) == 3

        # expecting 2 model endpoints that are the latest
        list_mep = self._db.list_model_endpoints(
            self._db_session,
            latest_only=True,
            project=model_endpoint.metadata.project,
        ).endpoints

        # expecting two model endpoints that are the latest
        assert len(list_mep) == 2
        assert list_mep[0].metadata.uid == "2127986e91f544af9be31250295f03b6"

        list_mep = self._db.list_model_endpoints(
            self._db_session,
            project=model_endpoint.metadata.project,
            names=["model-endpoint-2"],
        ).endpoints

        # expecting a single model endpoint with the name model-endpoint-2
        assert len(list_mep) == 1
        assert list_mep[0].metadata.name == "model-endpoint-2"
        assert list_mep[0].metadata.endpoint_type == EndpointType.BATCH_EP

    def test_update_automatically_after_function_update(self) -> None:
        # store artifact
        for i in range(2):
            self._store_artifact(f"model-{i}")
        # store function
        self._store_function()
        model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={"name": "model-endpoint-1", "project": "project-1"},
            spec={
                "function_name": "function-1",
                "function_tag": "latest",
                "_model_id": 1,
            },
            status={"monitoring_mode": "enabled"},
        )
        for i in range(2):
            self._db.store_model_endpoint(
                self._db_session,
                model_endpoint,
            )
            if i == 0:
                self._db.update_function(
                    self._db_session,
                    name="function-1",
                    updates={"status": {"state": "error"}},
                    project="project-1",
                    tag="latest",
                )
                model_endpoint_from_db = self._db.get_model_endpoint(
                    self._db_session,
                    name=model_endpoint.metadata.name,
                    project=model_endpoint.metadata.project,
                    function_name="function-1",
                    function_tag="latest",
                )
                assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
                assert model_endpoint_from_db.metadata.project == "project-1"
                assert (
                    model_endpoint_from_db.metadata.labels
                    == model_endpoint.metadata.labels
                )
                assert (
                    model_endpoint_from_db.spec.function_uri
                    == f"project-1/function-1@{unversioned_tagged_object_uid_prefix}latest"
                )
                assert model_endpoint_from_db.spec.model_name == "model-0"
                assert model_endpoint_from_db.status.state == "error"
                model_endpoint.spec._model_id = 2
        mep_list = self._db.list_model_endpoints(
            session=self._db_session, project="project-1"
        ).endpoints
        assert len(mep_list) == 2
        for mep in mep_list:
            if mep.spec.model_name == "model-1":
                assert (
                    mep.spec.function_uri
                    == f"project-1/function-1@{unversioned_tagged_object_uid_prefix}latest"
                )
            else:
                # archived model endpoint should not have function_uri
                assert mep.spec.function_uri is None

    def test_update_automatically_after_model_update(self) -> None:
        # store artifact
        for i in range(2):
            uid = self._store_artifact(f"model-{i}")
        # store function
        self._store_function()
        model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={"name": "model-endpoint-1", "project": "project-1"},
            spec={
                "function_name": "function-1",
                "function_tag": "latest",
                "_model_id": 2,
            },
            status={"monitoring_mode": "enabled"},
        )

        self._db.store_model_endpoint(
            self._db_session,
            model_endpoint,
        )
        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
            function_tag="latest",
        )
        assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
        assert model_endpoint_from_db.metadata.project == "project-1"
        assert model_endpoint_from_db.spec.model_name == "model-1"
        assert model_endpoint_from_db.spec.model_tags == ["latest"]
        identifier = mlrun.common.schemas.ArtifactIdentifier(key="model-1", uid=uid)

        self._db.append_tag_to_artifacts(
            self._db_session, "project-1", "v3", [identifier]
        )

        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
            function_tag="latest",
        )
        assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
        assert model_endpoint_from_db.metadata.project == "project-1"
        assert model_endpoint_from_db.spec.model_name == "model-1"
        assert model_endpoint_from_db.spec.model_tags == ["latest", "v3"]

    def test_update(self) -> None:
        # store artifact
        for i in range(2):
            self._store_artifact(f"model-{i}")
        # store function
        self._store_function()
        model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={"name": "model-endpoint-1", "project": "project-1"},
            spec={
                "function_name": "function-1",
                "function_tag": "latest",
                "_model_id": 1,
            },
            status={"monitoring_mode": "enabled"},
        )
        uids = []
        for i in range(2):
            uid = self._db.store_model_endpoint(
                self._db_session,
                model_endpoint,
            )
            uids.append(uid)

        self._db.update_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
            function_tag="latest",
            attributes={"monitoring_mode": ModelMonitoringMode.disabled},
        )

        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
            function_tag="latest",
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
            attributes={"feature_names": ["a", "b"]},
        )

        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
            function_tag="latest",
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
            uid=uids[0],
        )
        # check that the feature_names value was updated for the old model endpoint
        assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
        assert model_endpoint_from_db.metadata.project == "project-1"
        assert model_endpoint_from_db.metadata.uid == uids[0]
        assert model_endpoint_from_db.spec.feature_names == ["a", "b"]

    def test_delete_model_endpoints(self) -> None:
        # store artifact
        for i in range(2):
            self._store_artifact(f"model-{i}")
        # store function
        self._store_function()
        model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={"name": "model-endpoint-1", "project": "project-1"},
            spec={
                "function_name": "function-1",
                "function_tag": "latest",
                "_model_id": 1,
            },
            status={"monitoring_mode": "enabled"},
        )
        for i in range(4):
            self._db.store_model_endpoint(
                self._db_session,
                model_endpoint,
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
        self._store_function()
        model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={"name": "model-endpoint-1", "project": "project-1"},
            spec={
                "function_name": "function-1",
                "function_tag": "latest",
            },
            status={"monitoring_mode": "enabled", "last_request": datetime.now()},
        )
        uid = self._db.store_model_endpoint(
            self._db_session,
            model_endpoint,
        )
        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name="function-1",
            function_tag="latest",
        )
        assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
        assert model_endpoint_from_db.metadata.project == "project-1"
        assert model_endpoint_from_db.metadata.uid == uid
        assert (
            model_endpoint_from_db.spec.function_uri
            == f"project-1/function-1@{unversioned_tagged_object_uid_prefix}latest"
        )
        assert model_endpoint_from_db.spec.model_name == ""

    def test_insert_without_function(self) -> None:
        model_endpoint = mlrun.common.schemas.ModelEndpoint(
            metadata={
                "name": "model-endpoint-1",
                "project": "project-1",
                "labels": {"K": 57, "V": 44, "f": 43, "v": 4},
            },
            spec={
                "function_name": "some-non-mlrun-function",
            },
            status={"monitoring_mode": "enabled", "last_request": datetime.now()},
        )
        uid = self._db.store_model_endpoint(
            self._db_session,
            model_endpoint,
        )
        model_endpoint_from_db = self._db.get_model_endpoint(
            self._db_session,
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            uid=uid,
        )
        assert model_endpoint_from_db.metadata.name == "model-endpoint-1"
        assert model_endpoint_from_db.metadata.project == "project-1"
        assert model_endpoint_from_db.metadata.uid == uid
        assert model_endpoint_from_db.spec.model_name == ""
        assert model_endpoint_from_db.spec.function_name == ""
        assert model_endpoint_from_db.metadata.labels == {
            "K": 57,
            "V": 44,
            "f": 43,
            "v": 4,
        }

    def test_2_functions(self) -> None:
        self._store_function()
        for i in range(2):
            self._store_function(function_name=f"f-{i}")
            model_endpoint = mlrun.common.schemas.ModelEndpoint(
                metadata={"name": "model-endpoint-1", "project": "project-1"},
                spec={
                    "function_name": f"f-{i}",
                    "function_tag": "latest",
                },
                status={"monitoring_mode": "enabled", "last_request": datetime.now()},
            )
            self._db.store_model_endpoint(
                self._db_session,
                model_endpoint,
            )

        endpoints = self._db.list_model_endpoints(
            self._db_session, project="project-1", latest_only=True
        ).endpoints
        assert len(endpoints) == 2

        endpoints = self._db.list_model_endpoints(
            self._db_session, project="project-1", function_name="f-0"
        ).endpoints
        assert len(endpoints) == 1

        endpoints = self._db.list_model_endpoints(
            self._db_session, project="project-1", function_tag="v2"
        ).endpoints
        assert len(endpoints) == 0

    def test_delete_multi_by_uids(self):
        uids = []
        for i in range(4):
            model_endpoint = mlrun.common.schemas.ModelEndpoint(
                metadata={"name": "model-endpoint-1", "project": "project-1"},
                spec={
                    "function_name": "func",
                    "function_tag": None,
                },
                status={"monitoring_mode": "enabled", "last_request": datetime.now()},
            )
            uid = self._db.store_model_endpoint(
                self._db_session,
                model_endpoint,
            )
            uids.append(uid)

        endpoints = self._db.list_model_endpoints(
            self._db_session, project="project-1"
        ).endpoints

        assert len(endpoints) == 4

        self._db.delete_model_endpoints(
            session=self._db_session, project="project-1", uids=uids
        )

        endpoints = self._db.list_model_endpoints(
            self._db_session, project="project-1"
        ).endpoints

        assert len(endpoints) == 0


def is_hex(s: str):
    try:
        int(s, 16)
        return True
    except (ValueError, TypeError):
        return False
