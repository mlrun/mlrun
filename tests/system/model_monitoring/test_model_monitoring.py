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
from sklearn.datasets import load_diabetes, load_iris, make_classification
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import mlrun.artifacts.model
import mlrun.common.schemas.model_monitoring
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.feature_store
import mlrun.model_monitoring.api
import mlrun.runtimes.mounts
import mlrun.runtimes.utils
import mlrun.serving.routers
import mlrun.utils
from mlrun.model import BaseMetadata
from mlrun.model_monitoring.helpers import get_result_instance_fqn
from mlrun.runtimes import BaseRuntime
from mlrun.utils import generate_artifact_uri
from mlrun.utils.v3io_clients import get_frames_client
from tests.system.base import TestMLRunSystem

_MLRUN_MODEL_MONITORING_DB = "mysql+pymysql://root@mlrun-db:3306/mlrun_model_monitoring"


# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
@pytest.mark.model_monitoring
class TestModelEndpointsOperations(TestMLRunSystem):
    """Applying basic model endpoint CRUD operations through MLRun API"""

    project_name = "mm-app-project"

    def setup_method(self, method):
        super().setup_method(method)
        if method.__name__ in [
            "test_list_endpoints_without_creds",
            "test_get_model_endpoint_metrics",
        ]:
            return
        self.project.set_model_monitoring_credentials(
            stream_path=os.getenv("MLRUN_MODEL_ENDPOINT_MONITORING__STREAM_CONNECTION"),
            tsdb_connection=mlrun.mlconf.model_endpoint_monitoring.tsdb_connection,
        )

    def _generate_event(
        self,
        endpoint_id,
        event_name,
        event_kind="result",
        app_name="my_app",
    ):
        start_infer_time = datetime.isoformat(datetime(2024, 1, 1, tzinfo=timezone.utc))
        end_infer_time = datetime.isoformat(
            datetime(2024, 1, 1, second=1, tzinfo=timezone.utc)
        )
        event_value = 123
        event_name_key = f"{event_kind}_name"
        event_value_key = f"{event_kind}_value"
        if event_kind == "result":
            extra_kwargs = {
                "result_kind": 0,
                "result_status": 0,
                "result_extra_data": """{}""",
            }
        else:
            extra_kwargs = {}
        data = {
            "endpoint_id": endpoint_id,
            "application_name": app_name,
            event_name_key: event_name,
            "event_kind": event_kind,
            "start_infer_time": start_infer_time,
            "end_infer_time": end_infer_time,
            event_value_key: event_value,
            **extra_kwargs,
        }
        return data

    def test_get_model_endpoint_metrics(self):
        tsdb_client = mlrun.model_monitoring.get_tsdb_connector(
            project=self.project_name,
            tsdb_connection_string=mlrun.mlconf.model_endpoint_monitoring.tsdb_connection,
        )
        self.project.set_model_monitoring_credentials(
            stream_path=os.getenv("MLRUN_MODEL_ENDPOINT_MONITORING__STREAM_CONNECTION"),
            tsdb_connection=mlrun.mlconf.model_endpoint_monitoring.tsdb_connection,
        )
        db = mlrun.get_run_db()
        model_endpoint = self._mock_random_endpoint("testing")
        model_endpoint = db.create_model_endpoint(model_endpoint)

        model_endpoint2 = self._mock_random_endpoint("testing2")
        model_endpoint2 = db.create_model_endpoint(model_endpoint2)

        mep_uid = model_endpoint.metadata.uid
        mep2_uid = model_endpoint2.metadata.uid

        try:
            tsdb_client.create_tables()
            tsdb_client.write_application_event(
                self._generate_event(endpoint_id=mep_uid, event_name="result1")
            )
            tsdb_client.write_application_event(
                self._generate_event(endpoint_id=mep_uid, event_name="result2")
            )
            tsdb_client.write_application_event(
                self._generate_event(endpoint_id=mep_uid, event_name="result3")
            )
            tsdb_client.write_application_event(
                self._generate_event(
                    endpoint_id=mep_uid, event_name="metric1", event_kind="metric"
                ),
                kind=mm_constants.WriterEventKind.METRIC,
            )
            tsdb_client.write_application_event(
                self._generate_event(endpoint_id=mep2_uid, event_name="result3")
            )
            tsdb_client.write_application_event(
                self._generate_event(endpoint_id=mep2_uid, event_name="result4")
            )
            tsdb_client.write_application_event(
                self._generate_event(
                    endpoint_id=mep2_uid, event_name="metric1", event_kind="metric"
                ),
                kind=mm_constants.WriterEventKind.METRIC,
            )
            expected_for_mep1 = [
                "invocations",
                "metric1",
                "result1",
                "result2",
                "result3",
            ]
            expected_for_mep2 = ["invocations", "metric1", "result3", "result4"]

            income_events_mep1 = self._run_db.get_model_endpoint_monitoring_metrics(
                project=self.project.name, endpoint_id=mep_uid
            )
            assert expected_for_mep1 == sorted(
                [event.name for event in income_events_mep1]
            )

            # separation:
            income_events_by_endpoint = self._run_db.get_metrics_by_multiple_endpoints(
                project=self.project.name, endpoint_ids=[mep_uid, mep2_uid]
            )

            result_for_mep1 = [
                event.name for event in income_events_by_endpoint[mep_uid]
            ]
            assert expected_for_mep1 == sorted(result_for_mep1)

            result_for_mep2 = [
                event.name for event in income_events_by_endpoint[mep2_uid]
            ]
            assert expected_for_mep2 == sorted(result_for_mep2)

            # intersection:
            intersection_events_by_type = (
                self._run_db.get_metrics_by_multiple_endpoints(
                    project=self.project.name,
                    endpoint_ids=[mep_uid, mep2_uid],
                    events_format=mm_constants.GetEventsFormat.INTERSECTION,
                )
            )
            metrics_key = mm_constants.INTERSECT_DICT_KEYS[
                mm_constants.ModelEndpointMonitoringMetricType.METRIC
            ]
            results_key = mm_constants.INTERSECT_DICT_KEYS[
                mm_constants.ModelEndpointMonitoringMetricType.RESULT
            ]
            assert ["invocations", "metric1"] == sorted(
                [metric.name for metric in intersection_events_by_type[metrics_key]]
            )
            assert ["result3"] == sorted(
                [result.name for result in intersection_events_by_type[results_key]]
            )

            # get nonexistent MEP IDs:
            result_for_non_exist = self._run_db.get_model_endpoint_monitoring_metrics(
                project=self.project.name, endpoint_id="not_exist", type="results"
            )
            assert result_for_non_exist == []

            result_for_non_exist = self._run_db.get_metrics_by_multiple_endpoints(
                project=self.project.name, endpoint_ids=["not_exist"], type="results"
            )
            assert result_for_non_exist == {"not_exist": []}

            intersection_results_for_non_exist = (
                self._run_db.get_metrics_by_multiple_endpoints(
                    project=self.project.name,
                    endpoint_ids=["not_exist", "not_exist2"],
                    events_format=mm_constants.GetEventsFormat.INTERSECTION,
                    type="results",
                )
            )
            assert intersection_results_for_non_exist[results_key] == []

        finally:
            tsdb_client.delete_tsdb_resources()

    @pytest.mark.parametrize("by_uid", [True, False])
    def test_clear_endpoint(self, by_uid):
        """Validates the process of create and delete a basic model endpoint"""
        db = mlrun.get_run_db()
        model_endpoint = self._mock_random_endpoint("testing")
        db.create_model_endpoint(model_endpoint)
        endpoint_response = db.get_model_endpoint(
            name=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name=model_endpoint.spec.function_name,
            function_tag=model_endpoint.spec.function_tag,
        )
        assert endpoint_response
        assert endpoint_response.metadata.name == model_endpoint.metadata.name

        if by_uid:
            attributes = {"endpoint_id": endpoint_response.metadata.uid}
        else:
            attributes = {
                "function_name": endpoint_response.spec.function_name,
                "function_tag": endpoint_response.spec.function_tag,
            }

        db.delete_model_endpoint(
            name=endpoint_response.metadata.name,
            project=endpoint_response.metadata.project,
            **attributes,
        )

        # test for existence with "underlying layers" functions
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            db.get_model_endpoint(
                name=endpoint_response.metadata.name,
                project=endpoint_response.metadata.project,
                **attributes,
            )

    def test_store_endpoint_update_existing(self):
        """Validates the process of create and update a basic model endpoint"""

        model_endpoint = self._mock_random_endpoint(
            "testing",
            function_name="function1",
            function_tag=None,  # latest is the default
        )
        db = mlrun.get_run_db()

        db.create_model_endpoint(model_endpoint=model_endpoint)

        endpoint_before_update = db.get_model_endpoint(
            project=model_endpoint.metadata.project,
            name=model_endpoint.metadata.name,
            function_name=model_endpoint.spec.function_name,
            function_tag="latest",
        )

        assert endpoint_before_update.status.monitoring_mode == "enabled"
        assert endpoint_before_update.spec.model_class == "modelcc"

        # Create attributes dictionary according to the required format
        attributes = {
            "monitoring_mode": "disabled",
            "model_class": "modelcc-2",
        }
        db.patch_model_endpoint(
            name=endpoint_before_update.metadata.name,
            project=endpoint_before_update.metadata.project,
            endpoint_id=endpoint_before_update.metadata.uid,
            attributes=attributes,
        )
        endpoint_after_update = db.get_model_endpoint(
            project=endpoint_before_update.metadata.project,
            endpoint_id=endpoint_before_update.metadata.uid,
            name=endpoint_before_update.metadata.name,
        )
        assert endpoint_after_update.status.monitoring_mode == "disabled"
        assert endpoint_after_update.spec.model_class == "modelcc-2"

        attributes = {
            "monitoring_mode": "enabled",
            "model_class": "modelcc-3",
        }
        db.patch_model_endpoint(
            name=endpoint_before_update.metadata.name,
            project=endpoint_before_update.metadata.project,
            attributes=attributes,
            function_name="function1",
            function_tag="latest",
        )
        endpoint_after_update = db.get_model_endpoint(
            project=endpoint_before_update.metadata.project,
            endpoint_id=endpoint_before_update.metadata.uid,
            name=endpoint_before_update.metadata.name,
        )
        assert endpoint_after_update.status.monitoring_mode == "enabled"
        assert endpoint_after_update.spec.model_class == "modelcc-3"

    def test_list_endpoints_on_empty_project(self):
        endpoints_out = self.project.list_model_endpoints()
        assert len(endpoints_out.endpoints) == 0

    def test_list_endpoints_without_creds(self):
        # empty project
        endpoints_out = self.project.list_model_endpoints()
        assert len(endpoints_out.endpoints) == 0

        # add endpoint
        db = mlrun.get_run_db()
        model_endpoint = self._mock_random_endpoint("testing")
        db.create_model_endpoint(model_endpoint)

        # list endpoints without credentials
        endpoints_out = self.project.list_model_endpoints()
        assert len(endpoints_out.endpoints) == 1

    def test_list_endpoints(self):
        db = mlrun.get_run_db()

        number_of_endpoints = 5
        endpoints_in = [
            self._mock_random_endpoint(f"testing-{i}")
            for i in range(number_of_endpoints)
        ]

        for endpoint in endpoints_in:
            db.create_model_endpoint(endpoint)

        endpoints_out = self.project.list_model_endpoints().endpoints

        in_endpoint_names = set(map(lambda e: e.metadata.name, endpoints_in))
        out_endpoint_names = set(map(lambda e: e.metadata.name, endpoints_out))

        endpoints_intersect = in_endpoint_names.intersection(out_endpoint_names)
        assert len(endpoints_intersect) == number_of_endpoints

    def test_max_archive_list_endpoints(self):
        # Validates the process of listing model endpoints with max archive limitation. In this test
        # we create 5 model endpoints and then create another one. The oldest one should be deleted
        db = mlrun.get_run_db()

        number_of_endpoints = 5
        endpoints_in = [
            self._mock_random_endpoint("testing") for _ in range(number_of_endpoints)
        ]

        for endpoint in endpoints_in:
            db.create_model_endpoint(endpoint, creation_strategy="archive")

        endpoints_out = self.project.list_model_endpoints(latest_only=False).endpoints
        assert len(endpoints_out) == number_of_endpoints
        created: Optional[datetime] = None
        uid: Optional[str] = None
        for mep in endpoints_out:
            if not created or mep.metadata.created < created:
                created = mep.metadata.created
                uid = mep.metadata.uid

        db.create_model_endpoint(
            self._mock_random_endpoint("testing"), creation_strategy="archive"
        )
        endpoints_out = self.project.list_model_endpoints(latest_only=False).endpoints
        assert uid not in [mep.metadata.uid for mep in endpoints_out]
        assert len(endpoints_out) == 5  # max_archive=5

    def test_list_endpoints_filter(self):
        number_of_endpoints = 5
        db = mlrun.get_run_db()

        for i in range(number_of_endpoints):
            endpoint = self._mock_random_endpoint(
                name=f"testing-{i}", function_tag=None
            )

            if i < 1:
                endpoint.spec.model_name = "filterme"
                endpoint.spec.function_tag = "v45"
            if i < 2:
                endpoint.metadata.name = "test-filter"
                endpoint.spec.function_name = "filter_function"

            if i < 4:
                endpoint.metadata.labels = {"filtermex": "1", "filtermey": "2"}

            db.create_model_endpoint(model_endpoint=endpoint)

        all_meps = self.project.list_model_endpoints()
        assert len(all_meps.endpoints) == number_of_endpoints

        filter_model = self.project.list_model_endpoints(model_name="filterme")
        assert len(filter_model.endpoints) == 1

        filter_functions = self.project.list_model_endpoints(
            function_name="filter_function", function_tag="v45"
        )
        assert len(filter_functions.endpoints) == 1

        filter_functions = self.project.list_model_endpoints(
            function_name="filter_function", function_tag="latest"
        )
        assert len(filter_functions.endpoints) == 1

        filter_functions_latest = self.project.list_model_endpoints(
            name="test-filter", latest_only=True
        )
        assert len(filter_functions_latest.endpoints) == 2

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

    @pytest.mark.parametrize("creation_strategy", ["archive", "inplace", "overwrite"])
    def test_creation_strategy(self, creation_strategy):
        db = mlrun.get_run_db()
        model_endpoint = self._mock_random_endpoint("testing", model_name="model-1")
        db.create_model_endpoint(model_endpoint, creation_strategy)
        model_endpoint = self._mock_random_endpoint("testing", model_name="model-2")
        db.create_model_endpoint(model_endpoint, creation_strategy)

        endpoints_out = self.project.list_model_endpoints().endpoints
        if creation_strategy == mm_constants.ModelEndpointCreationStrategy.ARCHIVE:
            assert len(endpoints_out) == 2
            endpoints_out = self.project.list_model_endpoints(
                latest_only=True
            ).endpoints

        assert len(endpoints_out) == 1
        assert endpoints_out[0].spec.model_name == "model-2"

        mep = mlrun.get_run_db().get_model_endpoint(
            project=endpoints_out[0].metadata.project,
            name=endpoints_out[0].metadata.name,
            endpoint_id=endpoints_out[0].metadata.uid,
            feature_analysis=True,
        )

        assert mep.status.drift_measures_timestamp is not None
        assert mep.status.current_stats_timestamp is not None

    def _mock_random_endpoint(
        self,
        name: str,
        function_name: Optional[str] = "function-1",
        function_uid: Optional[str] = None,
        function_tag: Optional[str] = "v1",
        model_name: Optional[str] = None,
        model_uid: Optional[str] = None,
    ) -> mlrun.common.schemas.model_monitoring.ModelEndpoint:
        def random_labels():
            return {
                f"{choice(string.ascii_letters)}": randint(0, 100) for _ in range(1, 5)
            }

        return mlrun.common.schemas.model_monitoring.ModelEndpoint(
            metadata=mlrun.common.schemas.model_monitoring.ModelEndpointMetadata(
                name=name,
                project=self.project_name,
                labels=random_labels(),
            ),
            spec=mlrun.common.schemas.model_monitoring.ModelEndpointSpec(
                function_name=function_name,
                function_tag=function_tag,
                function_uid=function_uid,
                model_name=model_name,
                model_uid=model_uid,
                model_class="modelcc",
                model_tag="latest",
            ),
            status=mlrun.common.schemas.model_monitoring.ModelEndpointStatus(
                monitoring_mode=mlrun.common.schemas.model_monitoring.ModelMonitoringMode.enabled,
            ),
        )


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
@pytest.mark.model_monitoring
class TestBasicModelMonitoring(TestMLRunSystem):
    """Deploy and apply monitoring on a basic pre-trained model"""

    project_name = "pr-basic-model-monitoring"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: Optional[str] = None

    @pytest.mark.timeout(540)
    def test_basic_model_monitoring(self) -> None:
        # Main validations:
        # 1 - a single model endpoint is created
        # 2 - model name, tag and values are recorded as expected under the model endpoint
        # 3 - stream metrics are recorded as expected under the model endpoint
        # 4 - test on both SQL and KV store targets

        # Deploy Model Servers
        project = self.project

        project.set_model_monitoring_credentials(
            stream_path=os.getenv("MLRUN_MODEL_ENDPOINT_MONITORING__STREAM_CONNECTION"),
            tsdb_connection=mlrun.mlconf.model_endpoint_monitoring.tsdb_connection,
            replace_creds=True,  # remove once ML-7501 is resolved
        )

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
        ).apply(mlrun.runtimes.mounts.auto_mount())
        # enable model monitoring
        serving_fn.set_tracking()
        project.enable_model_monitoring(
            deploy_histogram_data_drift_app=False,
            **({} if self.image is None else {"image": self.image}),
        )

        model_name = "sklearn_RandomForestClassifier"
        tag = "some-tag"
        labels = {"framework": "sklearn", "mylabel": "l1"}

        # Upload the model through the projects API so that it is available to the serving function
        model_obj = project.log_model(
            model_name,
            model_dir=str(self.assets_path),
            model_file="model.pkl",
            training_set=train_set,
            artifact_path=f"v3io:///projects/{project.metadata.name}",
            tag=tag,
            labels=labels,
        )
        # Add the model to the serving function's routing spec
        serving_fn.add_model(
            model_name,
            model_path=project.get_artifact_uri(
                key=model_name, category="model", tag=tag
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

        sleep(15)
        endpoints_list = mlrun.get_run_db().list_model_endpoints(self.project_name)
        assert len(endpoints_list.endpoints) == 1

        endpoint = endpoints_list.endpoints[0]

        assert not endpoint.spec.feature_stats

        self._assert_model_endpoint_tags_and_labels(
            endpoint=endpoint,
            model_name=model_name,
            tag=["some-tag", "latest"],
            labels=labels,
        )
        self._assert_model_uri(model_obj=model_obj, endpoint=endpoint)

        metrics = mlrun.get_run_db().get_model_endpoint_monitoring_metrics(
            self.project_name, endpoint.metadata.uid
        )
        assert len(metrics) == 1
        expected_metric_fqn = f"{endpoint.metadata.uid}.mlrun-infra.result.invocations"
        metric_fqn = get_result_instance_fqn(
            model_endpoint_id=endpoint.metadata.uid,
            app_name=metrics[0].app,
            result_name=metrics[0].name,
        )
        assert metric_fqn == expected_metric_fqn

    def _assert_model_uri(
        self,
        model_obj: mlrun.artifacts.ModelArtifact,
        endpoint: mlrun.common.schemas.ModelEndpoint,
    ) -> None:
        assert endpoint.spec.model_uri == mlrun.datastore.get_store_uri(
            kind=mlrun.utils.helpers.StorePrefix.Model,
            uri=generate_artifact_uri(
                project=model_obj.project,
                key=model_obj.key,
                iter=model_obj.iter,
                tree=model_obj.tree,
            ),
        )

    def _assert_model_endpoint_tags_and_labels(
        self,
        endpoint: mlrun.common.schemas.ModelEndpoint,
        model_name: str,
        tag: list[str],
        labels: dict[str, str],
    ) -> None:
        assert endpoint.metadata.labels == labels
        assert endpoint.spec.model_name == model_name
        assert endpoint.spec.model_tag == tag


@pytest.mark.skip(reason="Chronically fails, see ML-5820")
@TestMLRunSystem.skip_test_if_env_not_configured
class TestModelMonitoringRegression(TestMLRunSystem):
    """Train, deploy and apply monitoring on a regression model"""

    project_name = "pr-regression-model-monitoring"

    # TODO: Temporary skip this test on open source until fixed
    @pytest.mark.enterprise
    @pytest.mark.model_monitoring
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
@pytest.mark.model_monitoring
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
        ).apply(mlrun.runtimes.mounts.auto_mount())

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
                    self.project_name,
                    name=endpoint.metadata.name,
                    endpoint_id=endpoint.metadata.uid,
                    feature_analysis=True,
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
@pytest.mark.model_monitoring
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
        model = project.log_model(
            model_name,
            model_dir=os.path.relpath(self.assets_path),
            model_file="model.pkl",
            training_set=train_set,
            artifact_path=f"v3io:///projects/{project.name}",
            label_column="p0",
        )

        # Deploy model monitoring infra
        project.set_model_monitoring_credentials(
            stream_path=os.getenv("MLRUN_MODEL_ENDPOINT_MONITORING__STREAM_CONNECTION"),
            tsdb_connection=mlrun.mlconf.model_endpoint_monitoring.tsdb_connection,
        )
        project.enable_model_monitoring(
            base_period=1,
            deploy_histogram_data_drift_app=True,
            **({} if self.image is None else {"image": self.image}),
            wait_for_deployment=True,
        )

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
        model_endpoint = mlrun.model_monitoring.api.record_results(
            project=project.metadata.name,
            model_path=project.get_artifact_uri(
                key=model_name, category="model", tag="latest"
            ),
            model_endpoint_name="batch-drift-test",
            function_name="batch-drift-function",
            context=context,
            infer_results_df=infer_results_df,
        )

        # Wait for the controller, app and writer to complete
        sleep(130)

        model_endpoint = mlrun.model_monitoring.api.get_or_create_model_endpoint(
            project=project.name,
            endpoint_id=model_endpoint.metadata.uid,
            model_endpoint_name="batch-drift-test",
            function_name="batch-drift-function",
        )
        # Validate that model_uri is based on models prefix
        self._validate_model_uri(model_obj=model, model_endpoint=model_endpoint)

        # Validate that the artifacts were logged in the project
        artifacts = project.list_artifacts(
            labels={
                "mlrun/producer-type": "model-monitoring-app",
                "mlrun/app-name": "histogram-data-drift",
                "mlrun/endpoint-id": model_endpoint.metadata.uid,
            }
        )
        assert len(artifacts) == 2
        assert {art["metadata"]["key"] for art in artifacts} == {
            "drift_table_plot",
            "features_drift_results",
        }

    def _validate_model_uri(self, model_obj, model_endpoint):
        model_artifact_uri = mlrun.utils.helpers.generate_artifact_uri(
            project=model_endpoint.metadata.project,
            key=model_obj.key,
            iter=model_obj.iter,
            tree=model_obj.tree,
            uid=model_obj.metadata.uid,
        )

        # Enrich the uri schema with the store prefix
        model_artifact_uri = mlrun.datastore.get_store_uri(
            kind=mlrun.utils.helpers.StorePrefix.Model, uri=model_artifact_uri
        )

        assert model_endpoint.spec.model_uri == model_artifact_uri


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
@pytest.mark.model_monitoring
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
        ).apply(mlrun.runtimes.mounts.auto_mount())

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

        project.set_model_monitoring_credentials(
            stream_path=f"kafka://{self.brokers}",
            tsdb_connection=mlrun.mlconf.model_endpoint_monitoring.tsdb_connection,
        )

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
@pytest.mark.model_monitoring
class TestInferenceWithSpecialChars(TestMLRunSystem):
    project_name = "pr-infer-special-chars"
    name_prefix = "infer-monitoring"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: Optional[str] = None

    @classmethod
    def custom_setup_class(cls) -> None:
        # todo testsssss
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
        cls.function_name = f"{cls.name_prefix}-function"
        cls.model_endpoint_name = f"{cls.name_prefix}-test"
        cls._train()

    def custom_setup(self) -> None:
        mlrun.runtimes.utils.global_context.set(None)
        # Set the model monitoring credentials
        self.project.set_model_monitoring_credentials(
            stream_path=os.getenv("MLRUN_MODEL_ENDPOINT_MONITORING__STREAM_CONNECTION"),
            tsdb_connection=mlrun.mlconf.model_endpoint_monitoring.tsdb_connection,
        )

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
            name=self.model_endpoint_name,
            function_name=self.function_name,
            function_tag="latest",
        )
        return mlrun.feature_store.get_feature_set(
            model_endpoint.spec.monitoring_feature_set_uri
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
            model_endpoint_name=self.model_endpoint_name,
            function_name=self.function_name,
            context=mlrun.get_or_create_ctx(name=f"{self.name_prefix}-context"),  # pyright: ignore[reportGeneralTypeIssues]
            infer_results_df=self.infer_results_df,
            # TODO: activate ad-hoc mode when ML-5792 is done
        )

        self._test_feature_names()


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
@pytest.mark.model_monitoring
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

    @classmethod
    def _test_v3io_tsdb_record(cls) -> None:
        tsdb_client = mlrun.model_monitoring.get_tsdb_connector(
            project=cls.project_name,
            tsdb_connection_string=mlrun.mlconf.model_endpoint_monitoring.tsdb_connection,
        )

        df: pd.DataFrame = tsdb_client._get_records(
            table=mm_constants.V3IOTSDBTables.APP_RESULTS,
            start="now-5m",
            end="now",
        )

        assert not df.empty, "No TSDB data"
        assert (
            len(df) == 1
        ), "Expects a single result from the histogram data drift app in the TSDB"
        assert set(df.application_name) == {
            "histogram-data-drift"
        }, "The application name is different than expected"
        assert df.endpoint_id.nunique() == 1, "Expects a single model endpoint"
        assert set(df.result_name) == {
            "general_drift",
        }, "The result is different than expected"

    def test_record(self) -> None:
        self.project.set_model_monitoring_credentials(
            stream_path=os.getenv("MLRUN_MODEL_ENDPOINT_MONITORING__STREAM_CONNECTION"),
            tsdb_connection=mlrun.mlconf.model_endpoint_monitoring.tsdb_connection,
        )
        self.project.enable_model_monitoring(
            base_period=1,
            deploy_histogram_data_drift_app=True,
            **({} if self.image is None else {"image": self.image}),
            wait_for_deployment=True,
        )

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


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
@pytest.mark.model_monitoring
class TestModelEndpointWithManyFeatures(TestMLRunSystem):
    """Log a model with 500 features and validate the model endpoint feature stats."""

    project_name = "pr-many-features-model-monitoring"

    def test_model_endpoint_with_many_features(self) -> None:
        project = self.project

        project.set_model_monitoring_credentials(
            stream_path=os.getenv("MLRUN_MODEL_ENDPOINT_MONITORING__STREAM_CONNECTION"),
            tsdb_connection=mlrun.mlconf.model_endpoint_monitoring.tsdb_connection,
        )

        # Generate a model with 500 features
        x, y = make_classification(n_samples=1000, n_features=500, random_state=42)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=0.8, test_size=0.2, random_state=42
        )
        model = LinearRegression()
        model.fit(x_train, y_train)
        x_test = pd.DataFrame(x_test, columns=[f"column_{i}" for i in range(500)])
        y_test = pd.DataFrame(y_test, columns=["label"])
        training_set = pd.concat([x_test, y_test], axis=1)

        model_obj = project.log_model(
            key="model",
            body=pickle.dumps(model),
            model_file="model.pkl",
            training_set=training_set,
            label_column="label",
        )

        # Generate a model endpoint
        model_endpoint = mlrun.model_monitoring.api.get_or_create_model_endpoint(
            project=project.name,
            model_path=model_obj.uri,
            function_name="dummy_func",
            model_endpoint_name="dummy_ep",
        )

        assert len(model_endpoint.spec.feature_stats) == 501
