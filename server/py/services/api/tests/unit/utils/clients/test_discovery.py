# Copyright 2024 Iguazio
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

import re

from mlrun import mlconf

import framework.utils.clients.discovery


def test_discovery_register_api_hydra():
    mlconf.services.hydra.services = "*"
    mlconf.namespace = "default"
    discovery = framework.utils.clients.discovery.Client()
    service_instance = discovery.get_service("api")
    assert service_instance.name == "api"
    assert (
        service_instance.url
        == discovery._resolve_service_url("api")
        == "http://mlrun-api.default.svc.cluster.local:8080"
    )

    services_names = list(discovery.services.keys())
    assert services_names[-1] == "api-chief"

    service_instance = discovery.get_service("api-chief")
    assert service_instance.name == "api-chief"
    assert (
        service_instance.url
        == discovery._resolve_service_url("api-chief")
        == "http://mlrun-api-chief.default.svc.cluster.local:8080"
    )


def test_star_notation_translation():
    mlconf.services.hydra.services = ""
    star_pattern = "projects/.+/alerts.*"
    discovery = framework.utils.clients.discovery.Client()

    chief_routes = discovery._service_routes("alerts")
    for methods, pattern in chief_routes:
        if pattern == star_pattern:
            assert methods == ["*"]
            break
    else:
        assert False, f"pattern {star_pattern} not found in chief routes"

    service_instance = discovery.get_service("alerts")
    route_regex = re.compile(star_pattern)
    assert route_regex in service_instance.method_routes["put"]
    assert route_regex in service_instance.method_routes["post"]
    assert route_regex in service_instance.method_routes["delete"]
    assert route_regex in service_instance.method_routes["get"]


def test_find_service():
    method, path = "get", "projects/test/alerts"
    mlconf.services.hydra.services = "*"
    discovery = framework.utils.clients.discovery.Client()
    service_instance = discovery.resolve_service_by_request(method, path)
    assert service_instance is None

    mlconf.services.hydra.services = ""
    discovery.initialize()
    service_instance = discovery.resolve_service_by_request(method, path)
    assert service_instance.name == "alerts"
