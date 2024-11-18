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

import framework.service
import services.api.main

# The alerts import is to initialize the alerts daemon so that both services will run on the same instance
# It shall be removed once they are completely split
from services.alerts.daemon import daemon as alerts_daemon


class Daemon(framework.service.Daemon):
    def __init__(self, service_cls: framework.service.Service.__class__):
        self._service: framework.service.Service = service_cls()

    @property
    def mounts(self) -> dict[str, framework.service.Service]:
        # Mount the alerts application until we have service routing/tunneling
        return {"/": alerts_daemon.service}

    @property
    def service(self) -> services.api.main.Service:
        return self._service


daemon = Daemon(service_cls=services.api.main.Service)
daemon.initialize()
app = daemon.app

# TODO: Create a container, override ServiceContainer and implement forwarding requests to alerts service
