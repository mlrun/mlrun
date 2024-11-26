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
from mlrun import mlconf

import framework.service


def test_service_injection():
    mlconf.services.hydra.services = "*"
    service_container = framework.service.ServiceContainer()
    service = service_container.service()
    assert "services.alerts.main.Service" in str(service)

    mlconf.services.hydra.services = ""
    service = service_container.service()
    assert "services.api.main.Service" in str(service)
