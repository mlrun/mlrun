import json
import os
import string
from random import choice, randint, uniform
from time import monotonic, sleep
from typing import Optional

import pandas as pd
import pytest
from sklearn.datasets import load_iris

import mlrun
import mlrun.api.schemas
from mlrun.api.schemas import (
    ModelEndpoint,
    ModelEndpointMetadata,
    ModelEndpointSpec,
    ModelEndpointStatus,
)
from mlrun.errors import MLRunNotFoundError
from mlrun.model import BaseMetadata
from mlrun.runtimes import BaseRuntime
from tests.system.base import TestMLRunSystem


# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestModelMonitoringAPI(TestMLRunSystem):
    project_name = "model-monitor-sys-test"

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

    @pytest.mark.timeout(270)
    def test_basic_model_monitoring(self):
        simulation_time = 90  # 90 seconds
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
            model_path=project.get_artifact_uri(
                key=f"{model_name}:latest", category="model"
            ),
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
            sleep(uniform(0.2, 1.1))

        # test metrics
        endpoints_list = mlrun.get_run_db().list_model_endpoints(
            self.project_name, metrics=["predictions_per_second"]
        )
        assert len(endpoints_list.endpoints) == 1

        endpoint = endpoints_list.endpoints[0]
        assert len(endpoint.status.metrics) > 0

        predictions_per_second = endpoint.status.metrics["predictions_per_second"]
        assert predictions_per_second.name == "predictions_per_second"

        total = sum((m[1] for m in predictions_per_second.values))
        assert total > 0

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
                name,
                model_path=project.get_artifact_uri(
                    key=f"{name}:latest", category="model"
                ),
            )

        # Enable model monitoring
        serving_fn.deploy()

        # checking that stream processing and batch monitoring were successfully deployed
        mlrun.get_run_db().get_schedule(self.project_name, "model-monitoring-batch")
        metadta = BaseMetadata(
            name="model-monitoring-stream", project=self.project_name, tag=""
        )
        mlrun.get_run_db().get_builder_status(BaseRuntime(metadata=metadta))

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
