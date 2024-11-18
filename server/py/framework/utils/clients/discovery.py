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
import threading
import typing
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import mlrun.utils.singleton
from mlrun import mlconf


@dataclass
class ServiceInstance:
    service_name: str
    url: str
    metadata: Dict[str, str]
    last_heartbeat: datetime
    status: str = "UP"  # UP, DOWN, UNKNOWN


class Client(
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    def __init__(self, heartbeat_timeout_seconds: int = 30):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.lock = threading.Lock()
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self._logger = mlrun.utils.logger.get_child(__name__)
        self.initialize()

    def initialize(self):
        # TODO: resolve from config
        self.register_service(
            "alerts", url=f"http://mlrun-alerts.{mlconf.namespace}.svc.cluster.local"
        )
        self.register_service(
            "api", url=f"http://mlrun-api.{mlconf.namespace}.svc.cluster.local"
        )

    def register_service(
        self,
        service_name: str,
        url: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Register a new service instance."""
        with self.lock:
            if metadata is None:
                metadata = {}

            instance = ServiceInstance(
                service_name=service_name,
                url=url,
                metadata=metadata,
                last_heartbeat=datetime.now(),
            )

            if service_name not in self.services:
                self.services[service_name] = []

            # Check for duplicate registration
            for existing in self.services[service_name]:
                if existing.url == url:
                    self._logger.warning(
                        "Service already registered",
                        service_name=service_name,
                        url=url,
                    )
                    return False

            self.services[service_name].append(instance)
            self._logger.info("Registered service", service_name=service_name, url=url)
            return True

    def deregister_service(self, service_name: str, url: str) -> bool:
        """Deregister a service instance."""
        with self.lock:
            if service_name not in self.services:
                return False

            for instance in self.services[service_name]:
                if instance.url == url:
                    self.services[service_name].remove(instance)
                    self._logger.info(
                        "Deregistered service",
                        service_name=service_name,
                        url=url,
                    )

                    # Remove empty service list
                    if not self.services[service_name]:
                        del self.services[service_name]
                    return True
            return False

    def get_service(self, service_name: str) -> typing.Optional[ServiceInstance]:
        """Get all healthy instances of a service."""
        with self.lock:
            if service_name not in self.services:
                return None

            for instance in self.services[service_name]:
                if instance.status == "UP":
                    return instance
