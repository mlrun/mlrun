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
import http
import unittest.mock

import deepdiff
import fastapi.testclient
import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.errors
import mlrun.runtimes
import server.api.api.utils
import server.api.utils.builder
import server.api.utils.clients.iguazio
import server.api.utils.runtimes.nuclio


def test_get_frontend_spec(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    server.api.utils.clients.iguazio.Client().try_get_grafana_service_url = (
        unittest.mock.Mock()
    )
    default_function_pod_resources = {
        "requests": {"cpu": "25m", "memory": "1Mi", "gpu": ""},
        "limits": {"cpu": "2", "memory": "20Gi", "gpu": ""},
    }
    mlrun.mlconf.httpdb.builder.docker_registry = "quay.io/some-repo"
    mlrun.mlconf.default_function_pod_resources = default_function_pod_resources
    mlrun.mlconf.httpdb.allowed_file_paths = "s3://some/s3/path"
    mlrun.mlconf.httpdb.real_path = "/some/real/path"

    response = client.get("frontend-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    frontend_spec = mlrun.common.schemas.FrontendSpec(**response.json())
    assert (
        deepdiff.DeepDiff(
            frontend_spec.abortable_function_kinds,
            mlrun.runtimes.RuntimeKinds.abortable_runtimes(),
        )
        == {}
    )
    assert (
        frontend_spec.feature_flags.project_membership
        == mlrun.common.schemas.ProjectMembershipFeatureFlag.disabled
    )
    assert (
        frontend_spec.feature_flags.authentication
        == mlrun.common.schemas.AuthenticationFeatureFlag.none
    )
    assert (
        frontend_spec.feature_flags.nuclio_streams
        == mlrun.common.schemas.NuclioStreamsFeatureFlag.disabled
    )
    assert (
        frontend_spec.feature_flags.preemption_nodes
        == mlrun.common.schemas.PreemptionNodesFeatureFlag.disabled
    )
    assert frontend_spec.default_function_image_by_kind is not None
    assert frontend_spec.function_deployment_mlrun_requirement is not None
    assert frontend_spec.default_artifact_path is not None
    # fields UI expects to be in the template
    assert (
        mlrun.mlconf.httpdb.builder.docker_registry
        in frontend_spec.function_deployment_target_image_template
    )
    for expected_template_field in ["project", "name", "tag"]:
        bla = f"{{{expected_template_field}}}"
        assert bla in frontend_spec.function_deployment_target_image_template

    assert frontend_spec.default_function_pod_resources, mlrun.common.schemas.Resources(
        **default_function_pod_resources
    )
    assert (
        frontend_spec.function_deployment_target_image_name_prefix_template
        == mlrun.mlconf.httpdb.builder.function_target_image_name_prefix_template
    )
    assert (
        frontend_spec.function_deployment_target_image_registries_to_enforce_prefix
        == mlrun.runtimes.utils.resolve_function_target_image_registries_to_enforce_prefix()
    )

    assert (
        frontend_spec.default_function_preemption_mode
        == mlrun.common.schemas.PreemptionModes.prevent.value
    )
    assert (
        frontend_spec.allowed_artifact_path_prefixes_list
        == server.api.api.utils.get_allowed_path_prefixes_list()
    )
    assert (
        frontend_spec.function_deployment_mlrun_requirement
        == server.api.utils.builder.resolve_mlrun_install_command_version()
    )


def test_get_frontend_spec_jobs_dashboard_url_resolution(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    server.api.utils.clients.iguazio.Client().try_get_grafana_service_url = (
        unittest.mock.Mock()
    )
    # no cookie so no url
    response = client.get("frontend-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    frontend_spec = mlrun.common.schemas.FrontendSpec(**response.json())
    assert frontend_spec.jobs_dashboard_url is None
    server.api.utils.clients.iguazio.Client().try_get_grafana_service_url.assert_not_called()

    # no grafana (None returned) so no url
    mlrun.mlconf.httpdb.authentication.mode = "iguazio"
    server.api.utils.clients.iguazio.AsyncClient().verify_request_session = (
        unittest.mock.AsyncMock(
            return_value=(
                mlrun.common.schemas.AuthInfo(
                    username=None,
                    session="946b0749-5c40-4837-a4ac-341d295bfaf7",
                    user_id=None,
                    user_unix_id=0,
                    user_group_ids=[],
                )
            )
        )
    )
    server.api.utils.clients.iguazio.Client().try_get_grafana_service_url = (
        unittest.mock.Mock(return_value=None)
    )
    response = client.get("frontend-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    frontend_spec = mlrun.common.schemas.FrontendSpec(**response.json())
    assert frontend_spec.jobs_dashboard_url is None
    assert (
        server.api.utils.clients.iguazio.Client().try_get_grafana_service_url.call_count
        == 2
    )

    # happy scenario - grafana url found, verify returned correctly
    grafana_url = "some-url.com"
    server.api.utils.clients.iguazio.Client().try_get_grafana_service_url = (
        unittest.mock.Mock(return_value=grafana_url)
    )

    response = client.get("frontend-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    frontend_spec = mlrun.common.schemas.FrontendSpec(**response.json())
    assert (
        frontend_spec.jobs_dashboard_url
        == grafana_url
        + "/d/mlrun-jobs-monitoring/mlrun-jobs-monitoring?orgId=1&var-groupBy={filter_name}&var-filter={filter_value}"
    )
    assert (
        frontend_spec.model_monitoring_dashboard_url
        == grafana_url
        + "/d/AohIXhAMk/model-monitoring-details?var-PROJECT={project}&var-MODELENDPOINT={model_endpoint}"
    )
    assert (
        server.api.utils.clients.iguazio.Client().try_get_grafana_service_url.call_count
        == 2
    )


def test_get_frontend_spec_nuclio_streams(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    for test_case in [
        {
            "iguazio_version": "3.2.0",
            "nuclio_version": "1.6.23",
            "expected_feature_flag": mlrun.common.schemas.NuclioStreamsFeatureFlag.disabled,
        },
        {
            "iguazio_version": None,
            "nuclio_version": "1.6.23",
            "expected_feature_flag": mlrun.common.schemas.NuclioStreamsFeatureFlag.disabled,
        },
        {
            "iguazio_version": None,
            "nuclio_version": "1.7.8",
            "expected_feature_flag": mlrun.common.schemas.NuclioStreamsFeatureFlag.disabled,
        },
        {
            "iguazio_version": "3.4.0",
            "nuclio_version": "1.7.8",
            "expected_feature_flag": mlrun.common.schemas.NuclioStreamsFeatureFlag.enabled,
        },
    ]:
        # init cached value to None in the beginning of each test case
        server.api.utils.runtimes.nuclio.cached_nuclio_version = None
        mlrun.mlconf.igz_version = test_case.get("iguazio_version")
        mlrun.mlconf.nuclio_version = test_case.get("nuclio_version")

        response = client.get("frontend-spec")
        frontend_spec = mlrun.common.schemas.FrontendSpec(**response.json())
        assert response.status_code == http.HTTPStatus.OK.value
        assert frontend_spec.feature_flags.nuclio_streams == test_case.get(
            "expected_feature_flag"
        )


def test_get_frontend_spec_ce(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    ce_mode = "some-ce-mode"
    ce_release = "y.y.y"
    mlrun.mlconf.ce.mode = ce_mode
    mlrun.mlconf.ce.release = ce_release

    response = client.get("frontend-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    frontend_spec = mlrun.common.schemas.FrontendSpec(**response.json())

    assert frontend_spec.ce["release"] == ce_release
    assert frontend_spec.ce["mode"] == ce_mode


def test_get_frontend_spec_feature_store_data_prefixes(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    feature_store_data_prefix_default = "feature-store-data-prefix-default"
    feature_store_data_prefix_nosql = "feature-store-data-prefix-nosql"
    feature_store_data_prefix_redisnosql = "feature-store-data-prefix-redisnosql"
    feature_store_data_prefix_dsnosql = "feature-store-data-prefix-dsnosql"
    mlrun.mlconf.feature_store.data_prefixes.default = feature_store_data_prefix_default
    mlrun.mlconf.feature_store.data_prefixes.nosql = feature_store_data_prefix_nosql
    mlrun.mlconf.feature_store.data_prefixes.redisnosql = (
        feature_store_data_prefix_redisnosql
    )
    mlrun.mlconf.feature_store.data_prefixes.dsnosql = feature_store_data_prefix_dsnosql
    response = client.get("frontend-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    frontend_spec = mlrun.common.schemas.FrontendSpec(**response.json())
    assert (
        frontend_spec.feature_store_data_prefixes["default"]
        == feature_store_data_prefix_default
    )
    assert (
        frontend_spec.feature_store_data_prefixes["nosql"]
        == feature_store_data_prefix_nosql
    )
    assert (
        frontend_spec.feature_store_data_prefixes["redisnosql"]
        == feature_store_data_prefix_redisnosql
    )
    assert (
        frontend_spec.feature_store_data_prefixes["dsnosql"]
        == feature_store_data_prefix_dsnosql
    )
