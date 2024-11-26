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

from dependency_injector import containers, providers

from mlrun import mlconf

import framework.service
import services.api.main

# The alerts import is to initialize the alerts daemon so that both services will run on the same instance
# It shall be removed once they are completely split
from services.alerts.daemon import daemon as alerts_daemon


class Daemon(framework.service.Daemon):
    @property
    def mounts(self) -> list[framework.service.Service]:
        if mlconf.services.hydra.services == "*":
            # Mount the alerts application until we have proper hydra
            return [alerts_daemon.service]
        return []

    @property
    def service(self) -> services.api.main.Service:
        return self._service


daemon = Daemon(service_cls=services.api.main.Service)


# This is used to inject the alerts service when in hydra mode until we have proper hydra
def _service_selector() -> str:
    if mlconf.services.hydra.services == "*":
        return "alerts"
    return "api"


# Overriding ``ServiceContainer`` with ``APIServiceContainer``:
@containers.override(framework.service.ServiceContainer)
class APIServiceContainer(containers.DeclarativeContainer):
    service = providers.Selector(
        _service_selector,
        alerts=providers.Object(alerts_daemon.service),
        api=providers.Object(daemon.service),
    )


def app():
    daemon.initialize()
    daemon.wire()
    return daemon.app
