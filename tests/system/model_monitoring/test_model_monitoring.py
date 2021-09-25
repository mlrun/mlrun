import json
import os
import string
from datetime import datetime, timedelta
from random import choice, randint, uniform
from time import monotonic, sleep
from typing import Optional

import pandas as pd
import pytest
from sklearn.datasets import load_iris
from v3io.dataplane import RaiseForStatus
from v3io_frames import frames_pb2 as fpb2
from v3io_frames.errors import CreateError

import mlrun
import mlrun.api.schemas
from mlrun.api.schemas import (
    ModelEndpoint,
    ModelEndpointMetadata,
    ModelEndpointSpec,
    ModelEndpointStatus,
)
from mlrun.config import config
from mlrun.errors import MLRunNotFoundError
from mlrun.utils.model_monitoring import parse_model_endpoint_store_prefix
from mlrun.utils.v3io_clients import get_frames_client, get_v3io_client
from tests.system.base import TestMLRunSystem


# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestModelMonitoringAPI(TestMLRunSystem):
    project_name = "model-monitoring-system-test-project"

    def test_clear_endpoint(self):
        endpoint = self._mock_random_endpoint()
        db = mlrun.get_run_db()

        db.create_or_patch_model_endpoint(
            endpoint.metadata.project, endpoint.metadata.uid, endpoint
        )

        endpoint_response = db.get_model_endpoint(
            endpoint.metadata.project, endpoint.metadata.uid
        )
        assert endpoint_response
        assert endpoint_response.metadata.uid == endpoint.metadata.uid

        db.delete_model_endpoint_record(
            endpoint.metadata.project, endpoint.metadata.uid
        )

        # test for existence with "underlying layers" functions
        with pytest.raises(MLRunNotFoundError):
            endpoint = db.get_model_endpoint(
                endpoint.metadata.project, endpoint.metadata.uid
            )

    def test_store_endpoint_update_existing(self):
        endpoint = self._mock_random_endpoint()
        db = mlrun.get_run_db()

        db.create_or_patch_model_endpoint(
            endpoint.metadata.project, endpoint.metadata.uid, endpoint
        )

        endpoint_before_update = db.get_model_endpoint(
            endpoint.metadata.project, endpoint.metadata.uid
        )

        assert endpoint_before_update.status.state is None

        updated_state = "testing...testing...1 2 1 2"
        endpoint_before_update.status.state = updated_state

        db.create_or_patch_model_endpoint(
            endpoint_before_update.metadata.project,
            endpoint_before_update.metadata.uid,
            endpoint_before_update,
        )

        endpoint_after_update = db.get_model_endpoint(
            endpoint.metadata.project, endpoint.metadata.uid
        )

        assert endpoint_after_update.status.state == updated_state

    def test_list_endpoints_on_empty_project(self):
        endpoints_out = mlrun.get_run_db().list_model_endpoints(self.project_name)
        assert len(endpoints_out.endpoints) == 0

    def test_list_endpoints(self):
        db = mlrun.get_run_db()

        number_of_endpoints = 5
        endpoints_in = [
            self._mock_random_endpoint("testing") for _ in range(number_of_endpoints)
        ]

        for endpoint in endpoints_in:
            db.create_or_patch_model_endpoint(
                endpoint.metadata.project, endpoint.metadata.uid, endpoint
            )

        endpoints_out = db.list_model_endpoints(self.project_name)

        in_endpoint_ids = set(map(lambda e: e.metadata.uid, endpoints_in))
        out_endpoint_ids = set(map(lambda e: e.metadata.uid, endpoints_out.endpoints))

        endpoints_intersect = in_endpoint_ids.intersection(out_endpoint_ids)
        assert len(endpoints_intersect) == number_of_endpoints

    def test_list_endpoints_filter(self):
        number_of_endpoints = 5
        db = mlrun.get_run_db()

        # access_key = auth_info.data_session
        for i in range(number_of_endpoints):
            endpoint_details = self._mock_random_endpoint()

            if i < 1:
                endpoint_details.spec.model = "filterme"

            if i < 2:
                endpoint_details.spec.function_uri = "test/filterme"

            if i < 4:
                endpoint_details.metadata.labels = {"filtermex": "1", "filtermey": "2"}

            db.create_or_patch_model_endpoint(
                endpoint_details.metadata.project,
                endpoint_details.metadata.uid,
                endpoint_details,
            )

        filter_model = db.list_model_endpoints(self.project_name, model="filterme")
        assert len(filter_model.endpoints) == 1

        filter_labels = db.list_model_endpoints(
            self.project_name, labels=["filtermex=1"]
        )
        assert len(filter_labels.endpoints) == 4

        filter_labels = db.list_model_endpoints(
            self.project_name, labels=["filtermex=1", "filtermey=2"]
        )
        assert len(filter_labels.endpoints) == 4

        filter_labels = db.list_model_endpoints(
            self.project_name, labels=["filtermey=2"]
        )
        assert len(filter_labels.endpoints) == 4

    def test_get_endpoint_metrics(self):
        auth_info = self._get_auth_info()
        access_key = auth_info.data_session
        db = mlrun.get_run_db()

        path = config.model_endpoint_monitoring.store_prefixes.default.format(
            project=self.project_name,
            kind=mlrun.api.schemas.ModelMonitoringStoreKinds.EVENTS,
        )
        _, container, path = parse_model_endpoint_store_prefix(path)

        frames = get_frames_client(
            token=access_key, container=container, address=config.v3io_framesd,
        )

        start = datetime.utcnow()

        for i in range(5):
            endpoint = self._mock_random_endpoint()
            db.create_or_patch_model_endpoint(
                endpoint.metadata.project, endpoint.metadata.uid, endpoint
            )
            frames.create(backend="tsdb", table=path, rate="10/m", if_exists=1)

            total = 0

            dfs = []

            for j in range(10):
                count = randint(1, 10)
                total += count
                data = {
                    "predictions_per_second_count_1s": count,
                    "endpoint_id": endpoint.metadata.uid,
                    "timestamp": start - timedelta(minutes=10 - j),
                }
                df = pd.DataFrame(data=[data])
                dfs.append(df)

            frames.write(
                backend="tsdb",
                table=path,
                dfs=dfs,
                index_cols=["timestamp", "endpoint_id"],
            )

            endpoint = db.get_model_endpoint(
                self.project_name,
                endpoint.metadata.uid,
                metrics=["predictions_per_second_count_1s"],
            )
            assert len(endpoint.status.metrics) > 0

            predictions_per_second = endpoint.status.metrics[
                "predictions_per_second_count_1s"
            ]

            assert predictions_per_second.name == "predictions_per_second_count_1s"

            response_total = sum((m[1] for m in predictions_per_second.values))

            assert total == response_total

    @pytest.mark.timeout(200)
    def test_basic_model_monitoring(self):
        simulation_time = 20  # 20 seconds
        # Deploy Model Servers
        project = mlrun.get_run_db().get_project(self.project_name)
        project.set_model_monitoring_credentials(os.environ.get("V3IO_ACCESS_KEY"))

        iris = load_iris()
        train_set = pd.DataFrame(
            iris["data"],
            columns=[
                "sepal_length_cm",
                "sepal_width_cm",
                "petal_length_cm",
                "petal_width_cm",
            ],
        )

        # Import the serving function from the function hub
        serving_fn = mlrun.import_function(
            "hub://v2_model_server", project=self.project_name
        ).apply(mlrun.auto_mount())
        # enable model monitoring
        serving_fn.set_tracking()

        model_name = "sklearn_RandomForestClassifier"

        # Upload the model through the projects API so that it is available to the serving function
        project.log_model(
            model_name,
            model_dir=os.path.relpath(self.assets_path),
            model_file="model.pkl",
            training_set=train_set,
            artifact_path=f"v3io:///projects/{project.metadata.name}",
        )
        # Add the model to the serving function's routing spec
        serving_fn.add_model(
            model_name,
            model_path=f"store://models/{self.project_name}/{model_name}:latest",
        )

        # Deploy the function
        serving_fn.deploy()

        # Simulating Requests
        iris_data = iris["data"].tolist()

        t_end = monotonic() + simulation_time
        while monotonic() < t_end:
            data_point = choice(iris_data)
            serving_fn.invoke(
                f"v2/models/{model_name}/infer", json.dumps({"inputs": [data_point]})
            )
            sleep(uniform(0.2, 1.7))

    @pytest.mark.timeout(200)
    def test_model_monitoring_voting_ensemble(self):
        simulation_time = 20  # 20 seconds
        project = mlrun.get_run_db().get_project(self.project_name)
        project.set_model_monitoring_credentials(os.environ.get("V3IO_ACCESS_KEY"))

        iris = load_iris()
        train_set = pd.DataFrame(
            iris["data"],
            columns=[
                "sepal_length_cm",
                "sepal_width_cm",
                "petal_length_cm",
                "petal_width_cm",
            ],
        )

        # Deploy Model Servers
        # Use the following code to deploy a model server in the Iguazio instance.

        # Import the serving function from the function hub
        serving_fn = mlrun.import_function(
            "hub://v2_model_server", project=self.project_name
        ).apply(mlrun.auto_mount())

        serving_fn.set_topology(
            "router", "mlrun.serving.VotingEnsemble", name="VotingEnsemble"
        )
        serving_fn.set_tracking()

        model_names = [
            "sklearn_RandomForestClassifier",
            "sklearn_LogisticRegression",
            "sklearn_AdaBoostClassifier",
        ]

        for name in model_names:
            # Log the model through the projects API so that it is available through the feature store API
            project.log_model(
                name,
                model_dir=os.path.relpath(self.assets_path),
                model_file="model.pkl",
                training_set=train_set,
                artifact_path=f"v3io:///projects/{project.metadata.name}",
            )
            # Add the model to the serving function's routing spec
            serving_fn.add_model(
                name, model_path=f"store://models/{self.project_name}/{name}:latest"
            )

        # Enable model monitoring
        serving_fn.deploy()

        iris_data = iris["data"].tolist()

        t_end = monotonic() + simulation_time
        while monotonic() < t_end:
            data_point = choice(iris_data)
            serving_fn.invoke(
                "v2/models/VotingEnsemble/infer", json.dumps({"inputs": [data_point]})
            )
            sleep(uniform(0.2, 1.7))

    @staticmethod
    def _get_auth_info() -> mlrun.api.schemas.AuthInfo:
        return mlrun.api.schemas.AuthInfo(
            data_session=os.environ.get("V3IO_ACCESS_KEY")
        )

    @pytest.fixture(autouse=True)
    def cleanup_endpoints(self):
        db = mlrun.get_run_db()

        endpoints = db.list_model_endpoints(self.project_name)
        for endpoint in endpoints.endpoints:
            db.delete_model_endpoint_record(
                endpoint.metadata.project, endpoint.metadata.uid
            )

        v3io = get_v3io_client(
            endpoint=config.v3io_api, access_key=self._get_auth_info().data_session
        )

        path = config.model_endpoint_monitoring.store_prefixes.default.format(
            project=self.project_name,
            kind=mlrun.api.schemas.ModelMonitoringStoreKinds.ENDPOINTS,
        )
        _, container, path = parse_model_endpoint_store_prefix(path)

        frames = get_frames_client(
            token=self._get_auth_info().data_session,
            container=container,
            address=config.v3io_framesd,
        )
        try:
            all_records = v3io.kv.new_cursor(
                container=container,
                table_path=path,
                raise_for_status=RaiseForStatus.never,
            ).all()

            all_records = [r["__name"] for r in all_records]

            # Cleanup KV
            for record in all_records:
                v3io.kv.delete(
                    container=container,
                    table_path=path,
                    key=record,
                    raise_for_status=RaiseForStatus.never,
                )
        except RuntimeError:
            pass

        try:
            # Cleanup TSDB
            frames.delete(
                backend="tsdb", table=path, if_missing=fpb2.IGNORE,
            )
        except CreateError:
            pass

    def _mock_random_endpoint(self, state: Optional[str] = None) -> ModelEndpoint:
        def random_labels():
            return {
                f"{choice(string.ascii_letters)}": randint(0, 100) for _ in range(1, 5)
            }

        return ModelEndpoint(
            metadata=ModelEndpointMetadata(
                project=self.project_name, labels=random_labels()
            ),
            spec=ModelEndpointSpec(
                function_uri=f"test/function_{randint(0, 100)}:v{randint(0, 100)}",
                model=f"model_{randint(0, 100)}:v{randint(0, 100)}",
                model_class="classifier",
            ),
            status=ModelEndpointStatus(state=state),
        )
