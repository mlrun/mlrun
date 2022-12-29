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
#
import base64
import http
import json

import fastapi.testclient
import kubernetes
import sqlalchemy.orm

import mlrun
import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.clients.iguazio
import mlrun.errors
import mlrun.runtimes


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
    ce_version = "y.y.y"
    mlrun.mlconf.ce.mode = ce_mode
    mlrun.mlconf.ce.release = ce_version

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
    for key in ["scrape_metrics", "hub_url"]:
        assert response_body[key] is None
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
    assert response_body["ce_mode"] == ce_mode
    assert response_body["ce_version"] == ce_version
