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
import json
import os
import pickle
import string
from datetime import datetime, timedelta, timezone
from random import choice, randint, uniform
from time import monotonic, sleep
from typing import Optional, Union

import fsspec
import numpy as np
import pandas as pd
import pytest
import v3iofs
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import mlrun.artifacts.model
import mlrun.common.schemas.model_monitoring
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.feature_store
import mlrun.model_monitoring.api
import mlrun.runtimes.utils
import mlrun.serving.routers
import mlrun.utils
from mlrun.errors import MLRunNotFoundError
from mlrun.model import BaseMetadata
from mlrun.model_monitoring.writer import _TSDB_BE, _TSDB_TABLE, ModelMonitoringWriter
from mlrun.runtimes import BaseRuntime
from mlrun.utils.v3io_clients import get_frames_client
from tests.system.base import TestMLRunSystem


# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
class TestModelEndpointsOperations(TestMLRunSystem):
    """Applying basic model endpoint CRUD operations through MLRun API"""

    project_name = "pr-endpoints-operations"

    def test_clear_endpoint(self):
        """Validates the process of create and delete a basic model endpoint"""

        endpoint = self._mock_random_endpoint()
        db = mlrun.get_run_db()

        db.create_model_endpoint(
            endpoint.metadata.project, endpoint.metadata.uid, endpoint.dict()
        )

        endpoint_response = db.get_model_endpoint(
            endpoint.metadata.project, endpoint.metadata.uid
        )
        assert endpoint_response
        assert endpoint_response.metadata.uid == endpoint.metadata.uid

        db.delete_model_endpoint(endpoint.metadata.project, endpoint.metadata.uid)

        # test for existence with "underlying layers" functions
        with pytest.raises(MLRunNotFoundError):
            endpoint = db.get_model_endpoint(
                endpoint.metadata.project, endpoint.metadata.uid
            )

    def test_store_endpoint_update_existing(self):
        """Validates the process of create and update a basic model endpoint"""

        endpoint = self._mock_random_endpoint()
        db = mlrun.get_run_db()

        db.create_model_endpoint(
            project=endpoint.metadata.project,
            endpoint_id=endpoint.metadata.uid,
            model_endpoint=endpoint.dict(),
        )

        endpoint_before_update = db.get_model_endpoint(
            project=endpoint.metadata.project, endpoint_id=endpoint.metadata.uid
        )

        assert endpoint_before_update.status.state == "null"

        # Check default drift thresholds
        assert endpoint_before_update.spec.monitor_configuration == {
            mlrun.common.schemas.EventFieldType.DRIFT_DETECTED_THRESHOLD: (
                mlrun.mlconf.model_endpoint_monitoring.drift_thresholds.default.drift_detected
            ),
            mlrun.common.schemas.EventFieldType.POSSIBLE_DRIFT_THRESHOLD: (
                mlrun.mlconf.model_endpoint_monitoring.drift_thresholds.default.possible_drift
            ),
        }

        updated_state = "testing...testing...1 2 1 2"
        drift_status = "DRIFT_DETECTED"
        current_stats = {
            "tvd_sum": 2.2,
            "tvd_mean": 0.5,
            "hellinger_sum": 3.6,
            "hellinger_mean": 0.9,
            "kld_sum": 24.2,
            "kld_mean": 6.0,
            "f1": {"tvd": 0.5, "hellinger": 1.0, "kld": 6.4},
            "f2": {"tvd": 0.5, "hellinger": 1.0, "kld": 6.5},
        }

        # Create attributes dictionary according to the required format
        attributes = {
            "state": updated_state,
            "drift_status": drift_status,
            "current_stats": json.dumps(current_stats),
        }

        db.patch_model_endpoint(
            project=endpoint_before_update.metadata.project,
            endpoint_id=endpoint_before_update.metadata.uid,
            attributes=attributes,
        )

        endpoint_after_update = db.get_model_endpoint(
            project=endpoint.metadata.project, endpoint_id=endpoint.metadata.uid
        )

        assert endpoint_after_update.status.state == updated_state
        assert endpoint_after_update.status.drift_status == drift_status
        assert endpoint_after_update.status.current_stats == current_stats

    def test_list_endpoints_on_empty_project(self):
        endpoints_out = mlrun.get_run_db().list_model_endpoints(self.project_name)
        assert len(endpoints_out) == 0

    def test_list_endpoints(self):
        db = mlrun.get_run_db()

        number_of_endpoints = 5
        endpoints_in = [
            self._mock_random_endpoint("testing") for _ in range(number_of_endpoints)
        ]

        for endpoint in endpoints_in:
            db.create_model_endpoint(
                endpoint.metadata.project, endpoint.metadata.uid, endpoint.dict()
            )

        endpoints_out = db.list_model_endpoints(self.project_name)

        in_endpoint_ids = set(map(lambda e: e.metadata.uid, endpoints_in))
        out_endpoint_ids = set(map(lambda e: e.metadata.uid, endpoints_out))

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

            db.create_model_endpoint(
                endpoint_details.metadata.project,
                endpoint_details.metadata.uid,
                endpoint_details.dict(),
            )

        filter_model = db.list_model_endpoints(self.project_name, model="filterme")
        assert len(filter_model) == 1

        # TODO: Uncomment the following assertions once the KV labels filters is fixed.
        #       Following the implementation of supporting SQL store for model endpoints records, this table
        #       has static schema. That means, in order to keep the schema logic for both SQL and KV,
        #       it is not possible to add new label columns dynamically to the KV table. Therefore, the label filtering
        #       process for the KV should be updated accordingly.
        #

        # filter_labels = db.list_model_endpoints(
        #     self.project_name, labels=["filtermex=1"]
        # )
        # assert len(filter_labels) == 4
        #
        # filter_labels = db.list_model_endpoints(
        #     self.project_name, labels=["filtermex=1", "filtermey=2"]
        # )
        # assert len(filter_labels) == 4
        #
        # filter_labels = db.list_model_endpoints(
        #     self.project_name, labels=["filtermey=2"]
        # )
        # assert len(filter_labels) == 4

    def _mock_random_endpoint(
        self, state: Optional[str] = None
    ) -> mlrun.common.schemas.model_monitoring.ModelEndpoint:
        def random_labels():
            return {
                f"{choice(string.ascii_letters)}": randint(0, 100) for _ in range(1, 5)
            }

        return mlrun.common.schemas.model_monitoring.ModelEndpoint(
            metadata=mlrun.common.schemas.model_monitoring.ModelEndpointMetadata(
                project=self.project_name,
                labels=random_labels(),
                uid=str(randint(1000, 5000)),
            ),
            spec=mlrun.common.schemas.model_monitoring.ModelEndpointSpec(
                function_uri=f"test/function_{randint(0, 100)}:v{randint(0, 100)}",
                model=f"model_{randint(0, 100)}:v{randint(0, 100)}",
                model_class="classifier",
                active=True,
            ),
            status=mlrun.common.schemas.model_monitoring.ModelEndpointStatus(
                state=state
            ),
        )


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestBasicModelMonitoring(TestMLRunSystem):
    """Deploy and apply monitoring on a basic pre-trained model"""

    project_name = "pr-basic-model-monitoring"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: Optional[str] = None

    @pytest.mark.timeout(270)
    @pytest.mark.parametrize("engine", ["sync", "async"])
    def test_basic_model_monitoring(self, engine) -> None:
        # Main validations:
        # 1 - a single model endpoint is created
        # 2 - stream metrics are recorded as expected under the model endpoint

        # Deploy Model Servers
        project = self.project

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
            "hub://v2-model-server", project=self.project_name
        ).apply(mlrun.auto_mount())

        serving_fn.set_topology(
            "router",
            engine=engine,
        )

        # enable model monitoring
        serving_fn.set_tracking()
        project.enable_model_monitoring(
            deploy_histogram_data_drift_app=False,
            **({} if self.image is None else {"image": self.image}),
        )

        model_name = "sklearn_RandomForestClassifier"

        # Upload the model through the projects API so that it is available to the serving function
        project.log_model(
            model_name,
            model_dir=str(self.assets_path),
            model_file="model.pkl",
            training_set=train_set,
            artifact_path=f"v3io:///projects/{project.metadata.name}",
        )
        # Add the model to the serving function's routing spec
        serving_fn.add_model(
            model_name,
            model_path=project.get_artifact_uri(
                key=model_name, category="model", tag="latest"
            ),
        )
        if self.image is not None:
            serving_fn.spec.image = serving_fn.spec.build.image = self.image

        # Deploy the function
        serving_fn.deploy()

        # Simulating valid requests
        iris_data = iris["data"].tolist()

        for _ in range(102):
            data_point = choice(iris_data)
            serving_fn.invoke(
                f"v2/models/{model_name}/infer", json.dumps({"inputs": [data_point]})
            )
            sleep(choice([0.01, 0.04]))

        # Test metrics
        sleep(5)
        self._assert_model_endpoint_metrics()

    def _assert_model_endpoint_metrics(self) -> None:
        endpoints_list = mlrun.get_run_db().list_model_endpoints(
            self.project_name, metrics=["predictions_per_second"]
        )
        assert len(endpoints_list) == 1

        endpoint = endpoints_list[0]

        assert len(endpoint.status.metrics) > 0
        self._logger.debug("Model endpoint metrics", endpoint.status.metrics)

        assert endpoint.status.metrics["generic"]["predictions_count_5m"] == 102

        predictions_per_second = endpoint.status.metrics["real_time"][
            "predictions_per_second"
        ]
        total = sum(m[1] for m in predictions_per_second)
        assert total > 0


@pytest.mark.skip(reason="Chronically fails, see ML-5820")
@TestMLRunSystem.skip_test_if_env_not_configured
class TestModelMonitoringRegression(TestMLRunSystem):
    """Train, deploy and apply monitoring on a regression model"""

    project_name = "pr-regression-model-monitoring-v4"

    # TODO: Temporary skip this test on open source until fixed
    @pytest.mark.enterprise
    @pytest.mark.timeout(200)
    def test_model_monitoring_with_regression(self):
        # Main validations:
        # 1 - model monitoring feature is created according to the feature vector instead of a model object when
        #     inputs are missing
        # 2 - access key secret within the model monitoring batch job
        # 3 - scheduling policy of the batch job

        # Load boston housing pricing dataset
        diabetes_data = load_diabetes()
        train_set = pd.DataFrame(
            diabetes_data.data, columns=diabetes_data.feature_names
        ).reset_index()
        train_set.rename({"index": "patient_id"}, axis=1, inplace=True)

        # Load target dataset
        target_set = pd.DataFrame(
            diabetes_data.target, columns=["target"]
        ).reset_index()
        target_set.rename({"index": "patient_id"}, axis=1, inplace=True)

        # Create feature sets for both the features and the labels
        diabetes_set = mlrun.feature_store.FeatureSet(
            "diabetes-set", entities=["patient_id"]
        )
        label_set = mlrun.feature_store.FeatureSet(
            "target-set", entities=["patient_id"]
        )

        # Ingest data
        diabetes_set.ingest(train_set)
        label_set.ingest(target_set, targets=[mlrun.datastore.targets.ParquetTarget()])

        # Define feature vector and save it to MLRun's feature store DB
        fv = mlrun.feature_store.FeatureVector(
            "diabetes-features",
            features=["diabetes-set.*"],
            label_feature="target-set.target",
        )
        fv.save()

        assert (
            fv.uri == f"store://feature-vectors/{self.project_name}/diabetes-features"
        )

        # Request (get or create) the offline dataset from the feature store and save to a parquet target
        mlrun.feature_store.get_offline_features(
            fv, target=mlrun.datastore.targets.ParquetTarget()
        )

        # Train the model using the auto trainer from the hub
        train = mlrun.import_function("hub://auto-trainer", new_name="train")
        train.deploy()
        model_class = "sklearn.linear_model.LinearRegression"
        model_name = "diabetes_model"
        label_columns = "target"

        train_run = train.run(
            inputs={"dataset": fv.uri},
            params={
                "model_class": model_class,
                "model_name": model_name,
                "label_columns": label_columns,
                "train_test_split_size": 0.2,
            },
            handler="train",
        )

        # Remove features from model obj and set feature vector uri
        db = mlrun.get_run_db()
        model_obj: mlrun.artifacts.ModelArtifact = (
            mlrun.datastore.store_resources.get_store_resource(
                train_run.outputs["model"], db=db
            )
        )
        model_obj.inputs = []
        model_obj.feature_vector = fv.uri + ":latest"
        mlrun.artifacts.model.update_model(model_obj)

        # Set the serving topology to simple model routing
        # with data enrichment and imputing from the feature vector
        serving_fn = mlrun.import_function("hub://v2-model-server", new_name="serving")
        serving_fn.set_topology(
            "router",
            mlrun.serving.routers.EnrichmentModelRouter(
                feature_vector_uri=str(fv.uri), impute_policy={"*": "$mean"}
            ),
        )
        serving_fn.add_model("diabetes_model", model_path=train_run.outputs["model"])

        # Enable model monitoring
        serving_fn.set_tracking()

        # Deploy the serving function
        serving_fn.deploy()

        # Validate that the model monitoring batch access key is replaced with an internal secret
        batch_function = mlrun.get_run_db().get_function(
            name="model-monitoring-batch", project=self.project_name
        )
        batch_access_key = batch_function["metadata"]["credentials"]["access_key"]
        auth_secret = mlrun.mlconf.secret_stores.kubernetes.auth_secret_name.format(
            hashed_access_key=""
        )
        assert batch_access_key.startswith(
            mlrun.model.Credentials.secret_reference_prefix + auth_secret
        )

        # Validate a single endpoint
        endpoints_list = mlrun.get_run_db().list_model_endpoints(self.project_name)
        assert len(endpoints_list) == 1

        # Validate monitoring mode
        model_endpoint = endpoints_list[0]
        assert (
            model_endpoint.spec.monitoring_mode
            == mlrun.common.schemas.model_monitoring.ModelMonitoringMode.enabled.value
        )

        # Validate tracking policy
        batch_job = db.get_schedule(
            project=self.project_name, name="model-monitoring-batch"
        )
        assert batch_job.cron_trigger.hour == "*/3"

        # TODO: uncomment the following assertion once the auto trainer function
        #  from mlrun hub is upgraded to 1.0.8
        # assert len(model_obj.spec.feature_stats) == len(
        #     model_endpoint.spec.feature_names
        # ) + len(model_endpoint.spec.label_names)

        # Validate monitoring feature set URI
        monitoring_feature_set = mlrun.feature_store.get_feature_set(
            model_endpoint.status.monitoring_feature_set_uri
        )

        expected_uri = (
            f"store://feature-sets/{self.project_name}/monitoring-"
            f"{serving_fn.metadata.name}-{model_name}-latest:{model_endpoint.metadata.uid}_"
        )
        assert expected_uri == monitoring_feature_set.uri


@pytest.mark.skip(reason="Chronically fails, see ML-5820")
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestVotingModelMonitoring(TestMLRunSystem):
    """Train, deploy and apply monitoring on a voting ensemble router with 3 models"""

    project_name = "pr-voting-model-monitoring"

    @pytest.mark.timeout(300)
    def test_model_monitoring_voting_ensemble(self):
        # Main validations:
        # 1 - model monitoring feature set is created with the relevant features and target
        # 2 - deployment status of monitoring stream nuclio function
        # 3 - model endpoints types for both children and router
        # 4 - metrics and drift status per model endpoint
        # 5 - invalid records are considered in the aggregated error count value
        # 6 - KV schema file is generated as expected

        simulation_time = 120  # 120 seconds to allow tsdb batching

        iris = load_iris()
        columns = [
            "sepal_length_cm",
            "sepal_width_cm",
            "petal_length_cm",
            "petal_width_cm",
        ]

        label_column = "label"

        # preparing training set
        train_set = pd.DataFrame(
            iris["data"],
            columns=columns,
        )

        train_set[label_column] = iris["target"]
        # store training set as parquet which will be used in the training function
        path = "v3io:///bigdata/bla.parquet"
        fsys = fsspec.filesystem(v3iofs.fs.V3ioFS.protocol)
        train_set.to_parquet(path=path, filesystem=fsys)

        # Deploy Model Servers
        # Use the following code to deploy a model server in the Iguazio instance.

        # Import the serving function from the function hub
        serving_fn = mlrun.import_function(
            "hub://v2-model-server", project=self.project_name
        ).apply(mlrun.auto_mount())

        serving_fn.set_topology(
            "router", "mlrun.serving.VotingEnsemble", name="VotingEnsemble"
        )

        # enable model monitoring
        serving_fn.set_tracking()

        # define different models
        model_names = {
            "sklearn_RandomForestClassifier": "sklearn.ensemble.RandomForestClassifier",
            "sklearn_LogisticRegression": "sklearn.linear_model.LogisticRegression",
            "sklearn_AdaBoostClassifier": "sklearn.ensemble.AdaBoostClassifier",
        }

        # Import the auto trainer function from the hub (hub://)
        train = mlrun.import_function("hub://auto-trainer")

        for name, pkg in model_names.items():
            # Run the function and specify input dataset path and some parameters (algorithm and label column name)
            train_run = train.run(
                name=name,
                inputs={"dataset": path},
                params={"model_class": pkg, "label_columns": label_column},
            )

            # Add the model to the serving function's routing spec
            serving_fn.add_model(name, model_path=train_run.outputs["model"])

        # Enable model monitoring
        serving_fn.deploy()

        # checking that monitoring feature sets were created
        fs_list = mlrun.get_run_db().list_feature_sets()
        assert len(fs_list) == 3

        # validate monitoring feature set features and target
        m_fs = fs_list[0]
        assert list(m_fs.spec.features.keys()) == columns + ["label"]
        assert m_fs.status.to_dict()["targets"][0]["kind"] == "parquet"

        # checking that stream processing and batch monitoring were successfully deployed
        mlrun.get_run_db().get_schedule(self.project_name, "model-monitoring-batch")

        # get the runtime object and check the build process of the monitoring stream
        base_runtime = BaseRuntime(
            BaseMetadata(
                name="model-monitoring-stream", project=self.project_name, tag=""
            )
        )

        # Wait 20 sec if model monitoring stream function is still in building process
        mlrun.utils.helpers.retry_until_successful(
            2,
            20,
            self._logger,
            False,
            self._check_monitoring_building_state,
            base_runtime=base_runtime,
        )

        # invoke the model before running the model monitoring batch job
        iris_data = iris["data"].tolist()

        # Simulating invalid request
        invalid_input = ["n", "s", "o", "-"]
        with pytest.raises(RuntimeError):
            serving_fn.invoke(
                "v2/models/VotingEnsemble/infer",
                json.dumps({"inputs": [invalid_input]}),
            )

        # Simulating valid requests
        t_end = monotonic() + simulation_time
        start_time = datetime.now(timezone.utc)
        data_sent = 0
        while monotonic() < t_end:
            data_point = choice(iris_data)
            serving_fn.invoke(
                "v2/models/VotingEnsemble/infer", json.dumps({"inputs": [data_point]})
            )
            sleep(uniform(0.2, 0.3))
            data_sent += 1

        # sleep to allow TSDB to be written (10/m)
        sleep(20)

        mlrun.get_run_db().invoke_schedule(self.project_name, "model-monitoring-batch")
        # it can take ~1 minute for the batch pod to finish running
        sleep(60)

        # Check that the KV schema has been generated as expected
        self._check_kv_schema_file()

        tsdb_path = f"/pipelines/{self.project_name}/model-endpoints/events/"
        client = get_frames_client(
            token=os.environ.get("V3IO_ACCESS_KEY"),
            address=os.environ.get("V3IO_FRAMESD"),
            container="users",
        )

        # checking top level methods
        top_level_endpoints = mlrun.get_run_db().list_model_endpoints(
            self.project_name, top_level=True
        )

        assert len(top_level_endpoints) == 1
        assert (
            top_level_endpoints[0].status.endpoint_type
            == mlrun.common.schemas.model_monitoring.EndpointType.ROUTER
        )

        children_list = top_level_endpoints[0].status.children_uids
        assert len(children_list) == len(model_names)

        endpoints_children_list = mlrun.get_run_db().list_model_endpoints(
            self.project_name, uids=children_list
        )
        assert len(endpoints_children_list) == len(model_names)
        for child in endpoints_children_list:
            assert (
                child.status.endpoint_type
                == mlrun.common.schemas.model_monitoring.EndpointType.LEAF_EP
            )

        # list model endpoints and perform analysis for each endpoint
        endpoints_list = mlrun.get_run_db().list_model_endpoints(self.project_name)

        for endpoint in endpoints_list:
            # Validate that the model endpoint record has been updated through the stream process
            assert endpoint.status.first_request != endpoint.status.last_request
            data = client.read(
                backend="tsdb",
                table=tsdb_path,
                filter=f"endpoint_id=='{endpoint.metadata.uid}'",
            )
            assert data.empty is False

            if (
                endpoint.status.endpoint_type
                == mlrun.common.schemas.model_monitoring.EndpointType.LEAF_EP
            ):
                assert (
                    datetime.fromisoformat(endpoint.status.first_request) >= start_time
                )
                assert datetime.fromisoformat(
                    endpoint.status.last_request
                ) <= start_time + timedelta(0, simulation_time)
                assert endpoint.status.drift_status == "NO_DRIFT"
                endpoint_with_details = mlrun.get_run_db().get_model_endpoint(
                    self.project_name, endpoint.metadata.uid, feature_analysis=True
                )
                drift_measures = endpoint_with_details.status.drift_measures
                measures = [
                    "tvd_sum",
                    "tvd_mean",
                    "hellinger_sum",
                    "hellinger_mean",
                    "kld_sum",
                    "kld_mean",
                ]
                stuff_for_each_column = ["tvd", "hellinger", "kld"]
                # feature analysis (details dashboard)
                for feature in columns:
                    assert feature in drift_measures
                    calcs = drift_measures[feature]
                    for calc in stuff_for_each_column:
                        assert calc in calcs
                        assert isinstance(calcs[calc], float)
                expected = endpoint_with_details.status.feature_stats
                for feature in columns:
                    assert feature in expected
                    assert (
                        expected[feature]["min"]
                        <= expected[feature]["mean"]
                        <= expected[feature]["max"]
                    )
                actual = endpoint_with_details.status.current_stats
                for feature in columns:
                    assert feature in actual
                    assert (
                        actual[feature]["min"]
                        <= actual[feature]["mean"]
                        <= actual[feature]["max"]
                    )
                # overall drift analysis (details dashboard)
                for measure in measures:
                    assert measure in drift_measures
                    assert isinstance(drift_measures[measure], float)

                # Validate error count value
                assert endpoint.status.error_count == 1

    def _check_monitoring_building_state(self, base_runtime):
        # Check if model monitoring stream function is ready
        stat = mlrun.get_run_db().get_builder_status(base_runtime)
        assert base_runtime.status.state == "ready", stat

    def _check_kv_schema_file(self):
        """Check that the KV schema has been generated as expected"""

        # Initialize V3IO client object that will be used to retrieve the KV schema
        client = mlrun.utils.v3io_clients.get_v3io_client(
            endpoint=mlrun.mlconf.v3io_api
        )

        # Get the schema raw object
        schema_raw = client.object.get(
            container="users",
            path=f"pipelines/{self.project_name}/model-endpoints/endpoints/.#schema",
            access_key=os.environ.get("V3IO_ACCESS_KEY"),
        )

        # Convert the content into a dict
        schema = json.loads(schema_raw.body)

        # Validate the schema key value
        assert schema["key"] == mlrun.common.schemas.model_monitoring.EventFieldType.UID

        # Create a new dictionary of field_name:field_type out of the schema dictionary
        fields_dict = {item["name"]: item["type"] for item in schema["fields"]}

        # Validate the type of several keys
        assert fields_dict["error_count"] == "long"
        assert fields_dict["function_uri"] == "string"
        assert fields_dict["endpoint_type"] == "string"
        assert fields_dict["active"] == "boolean"


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestBatchDrift(TestMLRunSystem):
    """Record monitoring parquet results and trigger the monitoring batch drift job analysis. This flow tests
    the monitoring process of the batch infer job function that can be imported from the functions hub.
    """

    project_name = "pr-batch-drift"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: Optional[str] = None

    def custom_setup(self):
        mlrun.runtimes.utils.global_context.set(None)

    def test_batch_drift(self):
        # Main validations:
        # 1 - Generate new model endpoint record through get_or_create_model_endpoint() within MLRun SDK
        # 2 - Write monitoring parquet result to the relevant context
        # 3 - Register and trigger monitoring batch drift job
        # 4 - Log monitoring artifacts

        # Generate project and context (context will be used for logging the artifacts)
        project = self.project
        context = mlrun.get_or_create_ctx(name="batch-drift-context")

        # Log a model artifact
        iris = load_iris()
        train_set = pd.DataFrame(
            data=np.c_[iris["data"], iris["target"]],
            columns=(
                [
                    "sepal_length_cm",
                    "sepal_width_cm",
                    "petal_length_cm",
                    "petal_width_cm",
                    "p0",
                ]
            ),
        )
        model_name = "sklearn_RandomForestClassifier"
        # Upload the model through the projects API so that it is available to the serving function
        project.log_model(
            model_name,
            model_dir=os.path.relpath(self.assets_path),
            model_file="model.pkl",
            training_set=train_set,
            artifact_path=f"v3io:///projects/{project.name}",
            label_column="p0",
        )

        # Deploy model monitoring infra
        project.enable_model_monitoring(
            base_period=1,
            deploy_histogram_data_drift_app=True,
            **({} if self.image is None else {"image": self.image}),
        )

        controller = self.project.get_function(
            key=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER
        )

        controller._wait_for_function_deployment(db=controller._get_db())
        # Generate a dataframe that will be written as a monitoring parquet
        # This dataframe is basically replacing the result set that is being generated through the batch infer function
        infer_results_df = pd.DataFrame(
            {
                "sepal_length_cm": [-500, -500],
                "sepal_width_cm": [-500, -500],
                "petal_length_cm": [-500, -500],
                "petal_width_cm": [-500, -500],
                "p0": [0, 0],
            }
        )
        infer_results_df[mlrun.common.schemas.EventFieldType.TIMESTAMP] = (
            mlrun.utils.datetime_now()
        )

        # Record results and trigger the monitoring batch job
        endpoint_id = "123123123123"
        mlrun.model_monitoring.api.record_results(
            project=project.metadata.name,
            endpoint_id=endpoint_id,
            model_path=project.get_artifact_uri(
                key=model_name, category="model", tag="latest"
            ),
            model_endpoint_name="batch-drift-test",
            function_name="batch-drift-function",
            context=context,
            infer_results_df=infer_results_df,
            # TODO: activate ad-hoc mode when ML-5792 is done
        )

        # Wait for the controller, app and writer to complete
        sleep(130)

        model_endpoint = mlrun.model_monitoring.api.get_or_create_model_endpoint(
            project=project.name, endpoint_id=endpoint_id
        )

        # Validate that model_uri is based on models prefix
        assert (
            model_endpoint.spec.model_uri
            == f"store://models/{project.name}/{model_name}:latest"
        )

        # Test the drift results
        # TODO: comment out when ML-5767 is done
        # assert model_endpoint.status.feature_stats
        # assert model_endpoint.status.current_stats
        # assert model_endpoint.status.drift_status == "DRIFT_DETECTED"

        # Validate that the artifacts were logged under the generated context
        assert len(project.list_artifacts(name="~drift_table_plot")) == 1
        assert len(project.list_artifacts(name="~features_drift_results")) == 1
        # TODO: take the artifacts from the original context when ML-5792 is done
        # artifacts = context.artifacts


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestModelMonitoringKafka(TestMLRunSystem):
    """Deploy a basic iris model configured with kafka stream"""

    brokers = (
        os.environ["MLRUN_SYSTEM_TESTS_KAFKA_BROKERS"]
        if "MLRUN_SYSTEM_TESTS_KAFKA_BROKERS" in os.environ
        and os.environ["MLRUN_SYSTEM_TESTS_KAFKA_BROKERS"]
        else None
    )

    project_name = "pr-kafka-model-monitoring"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: Optional[str] = None

    @pytest.mark.timeout(300)
    @pytest.mark.skipif(
        not brokers, reason="MLRUN_SYSTEM_TESTS_KAFKA_BROKERS not defined"
    )
    def test_model_monitoring_with_kafka_stream(self):
        project = self.project

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
                key=model_name, category="model", tag="latest"
            ),
        )

        project.set_model_monitoring_credentials(stream_path=f"kafka://{self.brokers}")

        # enable model monitoring
        serving_fn.set_tracking()
        project.enable_model_monitoring(
            deploy_histogram_data_drift_app=False,
            **({} if self.image is None else {"image": self.image}),
        )
        # Deploy the function
        if self.image is not None:
            serving_fn.spec.image = serving_fn.spec.build.image = self.image
        serving_fn.deploy()

        monitoring_stream_fn = project.get_function("model-monitoring-stream")

        function_config = monitoring_stream_fn.spec.config

        # Validate kakfa stream trigger configurations
        assert function_config["spec.triggers.kafka"]
        assert (
            function_config["spec.triggers.kafka"]["attributes"]["topics"][0]
            == f"monitoring_stream_{self.project_name}"
        )
        assert (
            function_config["spec.triggers.kafka"]["attributes"]["brokers"][0]
            == self.brokers
        )

        import kafka

        # Validate that the topic exist as expected
        consumer = kafka.KafkaConsumer(bootstrap_servers=[self.brokers])
        topics = consumer.topics()
        assert f"monitoring_stream_{self.project_name}" in topics

        # Simulating Requests
        iris_data = iris["data"].tolist()

        for i in range(100):
            data_point = choice(iris_data)
            serving_fn.invoke(
                f"v2/models/{model_name}/infer", json.dumps({"inputs": [data_point]})
            )
            sleep(uniform(0.02, 0.03))

        # Validate that the model endpoint metrics were updated as indication for the sanity of the flow
        model_endpoint = mlrun.get_run_db().list_model_endpoints(
            project=self.project_name
        )[0]

        assert model_endpoint.status.metrics["generic"]["latency_avg_5m"] > 0
        assert model_endpoint.status.metrics["generic"]["predictions_count_5m"] > 0


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestInferenceWithSpecialChars(TestMLRunSystem):
    project_name = "pr-infer-special-chars"
    name_prefix = "infer-monitoring"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: Optional[str] = None

    @classmethod
    def custom_setup_class(cls) -> None:
        cls.classif = SVC()
        cls.model_name = "classif_model"
        cls.columns = ["feat 1", "b (C)", "Last   for df "]
        cls.y_name = "class (0-4) "
        cls.num_rows = 20
        cls.num_cols = len(cls.columns)
        cls.num_classes = 5
        cls.x_train, cls.x_test, cls.y_train, cls.y_test = cls._generate_data()
        cls.training_set = cls.x_train.join(cls.y_train)
        cls.test_set = cls.x_test.join(cls.y_test)
        cls.infer_results_df = cls.test_set
        cls.infer_results_df[mlrun.common.schemas.EventFieldType.TIMESTAMP] = (
            mlrun.utils.datetime_now()
        )
        cls.endpoint_id = "5d6ce0e704442c0ac59a933cb4d238baba83bb5d"
        cls.function_name = f"{cls.name_prefix}-function"
        cls._train()

    def custom_setup(self) -> None:
        mlrun.runtimes.utils.global_context.set(None)

    @classmethod
    def _generate_data(cls) -> list[Union[pd.DataFrame, pd.Series]]:
        rng = np.random.default_rng(seed=23)
        x = pd.DataFrame(rng.random((cls.num_rows, cls.num_cols)), columns=cls.columns)
        y = pd.Series(np.arange(cls.num_rows) % cls.num_classes, name=cls.y_name)
        assert cls.num_rows > cls.num_classes
        return train_test_split(x, y, train_size=0.6, random_state=4)

    @classmethod
    def _train(cls) -> None:
        cls.classif.fit(
            cls.x_train,
            cls.y_train,  # pyright: ignore[reportGeneralTypeIssues]
        )

    def _get_monitoring_feature_set(self) -> mlrun.feature_store.FeatureSet:
        model_endpoint = mlrun.get_run_db().get_model_endpoint(
            project=self.project_name,
            endpoint_id=self.endpoint_id,
        )
        return mlrun.feature_store.get_feature_set(
            model_endpoint.status.monitoring_feature_set_uri
        )

    def _test_feature_names(self) -> None:
        feature_set = self._get_monitoring_feature_set()
        features = feature_set.spec.features
        feature_names = [feat.name for feat in features]
        assert feature_names == [
            mlrun.feature_store.api.norm_column_name(feat)
            for feat in self.columns
            + [self.y_name]
            + mm_constants.FeatureSetFeatures.list()
        ]

    def test_inference_feature_set(self) -> None:
        self.project.log_model(  # pyright: ignore[reportOptionalMemberAccess]
            self.model_name,
            body=pickle.dumps(self.classif),
            model_file="classif.pkl",
            framework="sklearn",
            training_set=self.training_set,
            label_column=self.y_name,
        )

        # TODO: activate ad-hoc mode when ML-5792 is done
        # self.project.enable_model_monitoring(
        #     **({} if self.image is None else {"image": self.image}),
        # )

        mlrun.model_monitoring.api.record_results(
            project=self.project_name,
            model_path=self.project.get_artifact_uri(
                key=self.model_name, category="model", tag="latest"
            ),
            model_endpoint_name=f"{self.name_prefix}-test",
            function_name=self.function_name,
            endpoint_id=self.endpoint_id,
            context=mlrun.get_or_create_ctx(name=f"{self.name_prefix}-context"),  # pyright: ignore[reportGeneralTypeIssues]
            infer_results_df=self.infer_results_df,
            # TODO: activate ad-hoc mode when ML-5792 is done
        )

        self._test_feature_names()


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestModelInferenceTSDBRecord(TestMLRunSystem):
    """
    Test that batch inference records results to V3IO TSDB when tracking is
    enabled and the selected model does not have a serving endpoint.
    """

    project_name = "infer-model-tsdb"
    name_prefix = "infer-model-only"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: Optional[str] = None

    @classmethod
    def custom_setup_class(cls) -> None:
        dataset = load_iris()
        cls.train_set = pd.DataFrame(
            dataset.data,  # pyright: ignore[reportGeneralTypeIssues]
            columns=[
                "sepal_length_cm",
                "sepal_width_cm",
                "petal_length_cm",
                "petal_width_cm",
            ],
        )
        cls.model_name = "clf_model"

        cls.infer_results_df = cls.train_set.copy()

    def custom_setup(self) -> None:
        mlrun.runtimes.utils.global_context.set(None)

    def _log_model(self) -> str:
        model = self.project.log_model(  # pyright: ignore[reportOptionalMemberAccess]
            self.model_name,
            model_dir=os.path.relpath(self.assets_path),
            model_file="model.pkl",
            training_set=self.train_set,
            artifact_path=f"v3io:///projects/{self.project_name}",
        )
        return model.uri

    def _wait_for_deployments(self) -> None:
        for fn_name in [
            mm_constants.MonitoringFunctionNames.STREAM,
            mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
            mm_constants.MonitoringFunctionNames.WRITER,
            mm_constants.MLRUN_HISTOGRAM_DATA_DRIFT_APP_NAME,
        ]:
            fn = self.project.get_function(key=fn_name)
            fn._wait_for_function_deployment(db=fn._get_db())

    @classmethod
    def _test_v3io_tsdb_record(cls) -> None:
        tsdb_client = ModelMonitoringWriter._get_v3io_frames_client(
            v3io_container=ModelMonitoringWriter.get_v3io_container(cls.project_name)
        )
        df: pd.DataFrame = tsdb_client.read(
            backend=_TSDB_BE, table=_TSDB_TABLE, start="now-5m", end="now"
        )

        assert not df.empty, "No TSDB data"
        assert (
            len(df) == 4
        ), "Expects four results of the histogram data drift app in the TSDB"
        assert set(df.application_name) == {
            "histogram-data-drift"
        }, "The application names are different than expected"
        assert df.endpoint_id.nunique() == 1, "Expects a single model endpoint"
        assert set(df.result_name) == {
            "general_drift",
            "hellinger_mean",
            "kld_mean",
            "tvd_mean",
        }, "The results are different than expected"

    def test_record(self) -> None:
        self.project.enable_model_monitoring(
            base_period=1,
            deploy_histogram_data_drift_app=True,
            **({} if self.image is None else {"image": self.image}),
        )

        self._wait_for_deployments()

        model_uri = self._log_model()

        mlrun.model_monitoring.api.record_results(
            project=self.project_name,
            infer_results_df=self.infer_results_df,
            model_path=model_uri,
            model_endpoint_name=f"{self.name_prefix}-test",
            context=mlrun.get_or_create_ctx(name=f"{self.name_prefix}-context"),  # pyright: ignore[reportGeneralTypeIssues]
            # TODO: activate ad-hoc mode when ML-5792 is done
        )

        sleep(130)

        self._test_v3io_tsdb_record()
