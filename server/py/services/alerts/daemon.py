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

import framework.service
import services.alerts.main


class Daemon(framework.service.Daemon):
    @property
    def service(self) -> services.alerts.main.Service:
        return self._service


daemon = Daemon(service_cls=services.alerts.main.Service)


# Overriding ``ServiceContainer`` with ``AlertsServiceContainer``:
@containers.override(framework.service.ServiceContainer)
class AlertsServiceContainer(containers.DeclarativeContainer):
    service = providers.Object(daemon.service)


def app():
    # Initialization must be after the service container override
    daemon.initialize()
    daemon.wire()
    return daemon.app
