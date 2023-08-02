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
import base64
import http
import json
import unittest.mock

import fastapi.testclient
import kubernetes
import sqlalchemy.orm

import mlrun
import mlrun.api.crud
import mlrun.api.utils.clients.iguazio
import mlrun.common.schemas
import mlrun.errors
import mlrun.runtimes
import mlrun.utils.version


def test_client_spec(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    k8s_api = kubernetes.client.ApiClient()
    overridden_ui_projects_prefix = "some-prefix"
    mlrun.mlconf.ui.projects_prefix = overridden_ui_projects_prefix
    nuclio_version = "x.x.x"
    mlrun.mlconf.nuclio_version = nuclio_version
    mlrun.mlconf.function_defaults.preemption_mode = "constrain"
    node_selector = {"label-1": "val1"}
    mlrun.mlconf.preemptible_nodes.node_selector = base64.b64encode(
        json.dumps(node_selector).encode("utf-8")
    )
    ce_mode = "some-ce-mode"
    ce_release = "y.y.y"
    mlrun.mlconf.ce.mode = ce_mode
    mlrun.mlconf.ce.release = ce_release

    feature_store_data_prefix_default = "feature-store-data-prefix-default"
    feature_store_data_prefix_nosql = "feature-store-data-prefix-nosql"
    feature_store_data_prefix_redisnosql = "feature-store-data-prefix-redisnosql"
    mlrun.mlconf.feature_store.data_prefixes.default = feature_store_data_prefix_default
    mlrun.mlconf.feature_store.data_prefixes.nosql = feature_store_data_prefix_nosql
    mlrun.mlconf.feature_store.data_prefixes.redisnosql = (
        feature_store_data_prefix_redisnosql
    )

    tolerations = [
        kubernetes.client.V1Toleration(
            effect="NoSchedule",
            key="test1",
            operator="Exists",
            toleration_seconds=3600,
        )
    ]
    serialized_tolerations = k8s_api.sanitize_for_serialization(tolerations)
    mlrun.mlconf.preemptible_nodes.tolerations = base64.b64encode(
        json.dumps(serialized_tolerations).encode("utf-8")
    )
    mlrun.mlconf.httpdb.logs.pipelines.pull_state.mode = "enabled"
    response = client.get("client-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    response_body = response.json()
    assert response_body["scrape_metrics"] is None
    assert response_body["ui_projects_prefix"] == overridden_ui_projects_prefix
    assert response_body["nuclio_version"] == nuclio_version

    # check nuclio_version cache
    mlrun.mlconf.nuclio_version = "y.y.y"
    response = client.get("client-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    response_body = response.json()
    assert response_body["nuclio_version"] == nuclio_version

    # check default_function_pod_resources when default
    assert response_body["default_function_pod_resources"] is None

    # check default_function_pod_resources when values set
    mlrun.mlconf.default_function_pod_resources = {
        "requests": {"cpu": "25m", "memory": "1M", "gpu": ""},
        "limits": {"cpu": "2", "memory": "1G", "gpu": ""},
    }
    response = client.get("client-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    response_body = response.json()
    assert (
        response_body["default_function_pod_resources"]
        == mlrun.mlconf.default_function_pod_resources.to_dict()
    )

    assert (
        response_body["default_preemption_mode"]
        == mlrun.mlconf.function_defaults.preemption_mode
    )
    assert response_body[
        "preemptible_nodes_node_selector"
    ] == mlrun.mlconf.preemptible_nodes.node_selector.decode("utf-8")
    assert response_body[
        "preemptible_nodes_tolerations"
    ] == mlrun.mlconf.preemptible_nodes.tolerations.decode("utf-8")

    assert response_body["logs"] == mlrun.mlconf.httpdb.logs.to_dict()
    assert response_body["logs"]["pipelines"]["pull_state"]["mode"] == "enabled"

    assert response_body["feature_store_data_prefixes"]["default"] == (
        feature_store_data_prefix_default
    )
    assert response_body["feature_store_data_prefixes"]["nosql"] == (
        feature_store_data_prefix_nosql
    )
    assert response_body["feature_store_data_prefixes"]["redisnosql"] == (
        feature_store_data_prefix_redisnosql
    )
    assert response_body["ce_mode"] == response_body["ce"]["mode"] == ce_mode
    assert response_body["ce"]["release"] == ce_release


def test_client_spec_response_based_on_client_version(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:

    response = client.get("client-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    response_body = response.json()
    assert response_body["kfp_image"] == "mlrun/mlrun:unstable"
    assert response_body["dask_kfp_image"] == "mlrun/ml-base:unstable"

    response = client.get(
        "client-spec",
        headers={
            mlrun.common.schemas.HeaderNames.client_version: "",
            mlrun.common.schemas.HeaderNames.python_version: "",
        },
    )
    assert response.status_code == http.HTTPStatus.OK.value
    response_body = response.json()
    assert response_body["kfp_image"] == "mlrun/mlrun:unstable"
    assert response_body["dask_kfp_image"] == "mlrun/ml-base:unstable"

    # test response when the server has a version
    with unittest.mock.patch.object(
        mlrun.utils.version.Version, "get", return_value={"version": "1.3.0-rc23"}
    ):
        response = client.get(
            "client-spec",
            headers={
                mlrun.common.schemas.HeaderNames.client_version: "",
                mlrun.common.schemas.HeaderNames.python_version: "",
            },
        )
        assert response.status_code == http.HTTPStatus.OK.value
        response_body = response.json()
        assert response_body["kfp_image"] == "mlrun/mlrun:1.3.0-rc23"
        assert response_body["dask_kfp_image"] == "mlrun/ml-base:1.3.0-rc23"

        # test clients older than 1.3.0, when client only provided client version
        response = client.get(
            "client-spec",
            headers={
                mlrun.common.schemas.HeaderNames.client_version: "1.2.0",
            },
        )
        assert response.status_code == http.HTTPStatus.OK.value
        response_body = response.json()
        assert response_body["kfp_image"] == "mlrun/mlrun:1.2.0"
        assert response_body["dask_kfp_image"] == "mlrun/ml-base:1.2.0"

        # test clients from 1.3.0+ and return based also on the client python version
        response = client.get(
            "client-spec",
            headers={
                mlrun.common.schemas.HeaderNames.client_version: "1.3.0-rc20",
                mlrun.common.schemas.HeaderNames.python_version: "3.7.13",
            },
        )
        assert response.status_code == http.HTTPStatus.OK.value
        response_body = response.json()
        assert response_body["kfp_image"] == "mlrun/mlrun:1.3.0-rc20-py37"
        assert response_body["dask_kfp_image"] == "mlrun/ml-base:1.3.0-rc20-py37"

        response = client.get(
            "client-spec",
            headers={
                mlrun.common.schemas.HeaderNames.client_version: "1.3.0-rc20",
                mlrun.common.schemas.HeaderNames.python_version: "3.9.13",
            },
        )
        assert response.status_code == http.HTTPStatus.OK.value
        response_body = response.json()
        assert response_body["kfp_image"] == "mlrun/mlrun:1.3.0-rc20"
        assert response_body["dask_kfp_image"] == "mlrun/ml-base:1.3.0-rc20"

        # verify that we are falling back to resolve only by server
        response = client.get(
            "client-spec",
            headers={
                mlrun.common.schemas.HeaderNames.client_version: "test-integration",
                mlrun.common.schemas.HeaderNames.python_version: "3.9.13",
            },
        )
        assert response.status_code == http.HTTPStatus.OK.value
        response_body = response.json()
        assert response_body["kfp_image"] == "mlrun/mlrun:1.3.0-rc23"
        assert response_body["dask_kfp_image"] == "mlrun/ml-base:1.3.0-rc23"
