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

import base64
import http
import json
import unittest.mock

import fastapi.testclient
import kubernetes
import pytest
import sqlalchemy.orm

import mlrun
import mlrun.common.schemas
import mlrun.errors
import mlrun.runtimes
import mlrun.utils.version

import services.api.api.endpoints.client_spec
import services.api.crud.client_spec


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
    feature_store_data_prefix_dsnosql = "feature-store-data-prefix-dsnosql"
    mlrun.mlconf.feature_store.data_prefixes.default = feature_store_data_prefix_default
    mlrun.mlconf.feature_store.data_prefixes.nosql = feature_store_data_prefix_nosql
    mlrun.mlconf.feature_store.data_prefixes.dsnosql = feature_store_data_prefix_dsnosql
    mlrun.mlconf.feature_store.data_prefixes.redisnosql = (
        feature_store_data_prefix_redisnosql
    )

    mlrun.mlconf.function.spec.security_context.enrichment_mode = (
        mlrun.common.schemas.SecurityContextEnrichmentModes.override
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
    services.api.api.endpoints.client_spec.get_cached_client_spec.cache_clear()
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
    services.api.api.endpoints.client_spec.get_cached_client_spec.cache_clear()

    mlrun.mlconf.alerts.mode = "disabled"

    mlrun.mlconf.system_id = "12345"

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
    assert response_body["feature_store_data_prefixes"]["dsnosql"] == (
        feature_store_data_prefix_dsnosql
    )
    assert response_body["ce"]["mode"] == ce_mode
    assert response_body["ce"]["release"] == ce_release
    assert response_body["function"]["spec"]["security_context"]["enrichment_mode"] == (
        mlrun.common.schemas.SecurityContextEnrichmentModes.override
    )

    assert response_body["alerts_mode"] == "disabled"
    assert response_body["system_id"] == "12345"


@pytest.mark.parametrize(
    "server_version, client_version, python_version, expected_dask_kfp",
    [
        # Server is "unstable"
        (
            "0.0.0+unstable",
            None,
            None,
            "mlrun/mlrun:unstable",
        ),
        (
            "0.0.0+unstable",
            "",
            "",
            "mlrun/mlrun:unstable",
        ),
        # Server is "1.8.0"
        ("1.8.0", "", "", "mlrun/mlrun:1.8.0"),
        ("1.8.0", "1.2.0", None, "mlrun/mlrun:1.2.0"),
        (
            "1.9.0",
            "1.9.0-rc20",
            "3.9.13",
            "mlrun/mlrun:1.9.0-rc20-py39",
        ),
        (
            "1.9.0",
            "1.9.0-rc20",
            "3.11.13",
            "mlrun/mlrun:1.9.0-rc20",
        ),
        (
            "1.8.0",
            "test-integration",
            "3.9.13",
            "mlrun/mlrun:1.8.0",
        ),
    ],
)
def test_client_spec_response_based_on_client_version(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    server_version,
    client_version,
    python_version,
    expected_dask_kfp,
):
    # Clear any cached spec to ensure fresh resolution each time
    services.api.api.endpoints.client_spec.get_cached_client_spec.cache_clear()

    # Patch server version
    with unittest.mock.patch.object(
        mlrun.utils.version.Version, "get", return_value={"version": server_version}
    ):
        # Build headers only if client_version / python_version are not None
        headers = {}
        if client_version is not None:
            headers[mlrun.common.schemas.HeaderNames.client_version] = client_version
        if python_version is not None:
            headers[mlrun.common.schemas.HeaderNames.python_version] = python_version

        # Send request
        response = client.get("client-spec", headers=headers)
        assert response.status_code == http.HTTPStatus.OK.value

        # Validate response
        response_body = response.json()
        assert response_body["dask_kfp_image"] == expected_dask_kfp


def test_get_client_spec_cached(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
    client_spec = services.api.crud.client_spec.ClientSpec().get_client_spec()
    with unittest.mock.patch.object(
        services.api.crud.client_spec.ClientSpec,
        "get_client_spec",
        return_value=client_spec,
    ) as mocked_get_client:
        response = client.get("client-spec")
        assert response.status_code == http.HTTPStatus.OK.value
        for i in range(10):
            cached_response = client.get("client-spec")
        assert response.json() == cached_response.json()
        assert mocked_get_client.call_count == 1

        # different client version -> cache miss
        invalidated_cached_response = client.get(
            "client-spec", headers={"x-mlrun-client-version": "1.2.3"}
        )
        assert invalidated_cached_response.status_code == http.HTTPStatus.OK.value
        assert mocked_get_client.call_count == 2

        import time

        # first request is still cached
        cached_response = client.get("client-spec")
        assert response.json() == cached_response.json()
        assert mocked_get_client.call_count == 2

        # extract given "ttl_seconds" from our lru-cached function
        ttl_seconds = next(
            filter(
                lambda argument: isinstance(argument, int),
                map(
                    lambda closure: closure.cell_contents,
                    services.api.api.endpoints.client_spec.get_cached_client_spec.__closure__,
                ),
            )
        )

        # invalidate first request from cache (time-based expiration)
        real_time_monotonic = time.monotonic()
        with unittest.mock.patch.object(time, "monotonic") as monotonic_time:
            monotonic_time.return_value = real_time_monotonic + ttl_seconds + 1
            cached_response = client.get("client-spec")
            assert cached_response.status_code == http.HTTPStatus.OK.value
            assert mocked_get_client.call_count == 3


def test_client_spec_includes_oauth_config_in_iguazio_v4_mode(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    mlrun.mlconf.httpdb.authentication.mode = (
        mlrun.common.types.AuthenticationMode.IGUAZIO_V4
    )
    mlrun.mlconf.iguazio_api_url = "http://x.local"
    mlrun.mlconf.iguazio_api_url_ingress = "https://x.com"
    services.api.api.endpoints.client_spec.get_cached_client_spec.cache_clear()

    response = client.get("client-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    response_body = response.json()

    assert (
        response_body["oauth_external_token_endpoint"]
        == "https://x.com/api/v1/authentication/refresh-access-token"
    )
    assert (
        response_body["oauth_internal_token_endpoint"]
        == "http://x.local/api/v1/authentication/refresh-access-token"
    )


def test_client_spec_kfp_default_workflow_timeout(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    services.api.api.endpoints.client_spec.get_cached_client_spec.cache_clear()

    response = client.get("client-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    assert response.json()["kfp_default_workflow_timeout"] is None

    mlrun.mlconf.kfp_default_workflow_timeout = "3600"
    services.api.api.endpoints.client_spec.get_cached_client_spec.cache_clear()

    response = client.get("client-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    assert response.json()["kfp_default_workflow_timeout"] == "3600"


def test_client_spec_does_not_include_oauth_config_when_not_iguazio_v4_mode(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    mlrun.mlconf.httpdb.authentication.mode = mlrun.common.types.AuthenticationMode.NONE
    services.api.api.endpoints.client_spec.get_cached_client_spec.cache_clear()

    response = client.get("client-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    response_body = response.json()

    assert response_body.get("oauth_external_token_endpoint") is None
    assert response_body.get("oauth_internal_token_endpoint") is None


def test_client_spec_includes_default_runtime_image_by_kind_when_configured(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    # Set custom image_by_kind values (different from defaults)
    custom_job_image = "custom/job-image"
    custom_serving_image = "custom/serving-image"
    mlrun.mlconf.function_defaults.image_by_kind = {
        "job": custom_job_image,
        "serving": custom_serving_image,
        # keep default
        "nuclio": "mlrun/mlrun",
    }
    services.api.api.endpoints.client_spec.get_cached_client_spec.cache_clear()

    response = client.get("client-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    response_body = response.json()

    # Should include the custom values but not the default
    assert response_body["default_runtime_image_by_kind"] == {
        "job": custom_job_image,
        "serving": custom_serving_image,
    }


def test_client_spec_excludes_default_runtime_image_by_kind_when_default(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    # Set only defaults
    mlrun.mlconf.function_defaults.image_by_kind = {
        "job": "mlrun/mlrun",
        "serving": "mlrun/mlrun",
        "nuclio": "mlrun/mlrun",
    }

    services.api.api.endpoints.client_spec.get_cached_client_spec.cache_clear()

    response = client.get("client-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    response_body = response.json()

    # Should be None when values match defaults
    assert response_body["default_runtime_image_by_kind"] is None
