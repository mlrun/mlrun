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
import framework.utils.clients.discovery


def test_discovery_register_service_twice():
    service_name = "test-service"
    service_url = "http://mock-server"
    discovery = framework.utils.clients.discovery.Client()
    discovery.register_service(service_name, url=service_url)
    service_instance = discovery.get_service(service_name)
    assert service_instance.name == service_name
    assert service_instance.url == service_url
    assert service_instance.status == "UP"

    # register another service
    service_url_2 = "http://mock-2-server"
    discovery.register_service(service_name, service_url_2)
    # should return 1st service
    service_instance = discovery.get_service(service_name)
    assert service_instance.name == service_name
    assert service_instance.url == service_url
    assert service_instance.status == "UP"

    # remove 1st service
    discovery.deregister_service(service_name, service_url)
    # should get 2nd service
    service_instance = discovery.get_service(service_name)
    assert service_instance.name == service_name
    assert service_instance.url == service_url_2
    assert service_instance.status == "UP"

    # remove 2nd service
    discovery.deregister_service(service_name, service_url_2)
    assert not discovery.get_service(service_name)
