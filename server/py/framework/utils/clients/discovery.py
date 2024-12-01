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
import collections
import re
import typing
from dataclasses import dataclass

import mlrun.utils.singleton
from mlrun import mlconf


@dataclass
class ServiceInstance:
    name: str
    url: str
    method_routes: dict[str, list[re.Pattern]] = None


class Client(
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    def __init__(self):
        self._logger = mlrun.utils.logger.get_child(__name__)
        self.services = None
        self.initialize()

    def initialize(self):
        # We use an ordered dict for control over service matching order.
        # This is important for services that may have overlapping routes.
        self.services: dict[str, ServiceInstance] = collections.OrderedDict()
        if mlconf.services.hydra.services != "*":
            self.register_service(service_name="alerts")
        self.register_service(service_name="api")
        # Must be last. Allowing other services to override its routes
        self.register_service(service_name="api-chief")

    def register_service(
        self,
        service_name: str,
    ):
        """Register a new service instance."""
        method_routes = self._resolve_service_method_routes(service_name)
        url = self._resolve_service_url(service_name)
        self.services[service_name] = ServiceInstance(
            name=service_name,
            url=url,
            method_routes=method_routes,
        )
        self._logger.info("Registered service", service_name=service_name, url=url)

    def deregister_service(self, service_name: str):
        """Deregister a service instance."""
        self.services.pop(service_name, None)
        self._logger.info(
            "Deregistered service",
            service_name=service_name,
        )

    def resolve_service_by_request(
        self, method: str, path: str
    ) -> typing.Optional[ServiceInstance]:
        """
        Resolve path and returns the matching service instance for the request.

        :param method: HTTP method of the request
        :param path: URL path to match against service patterns
        :return: ServiceInstance matching the request or None if no match
        """
        method = method.lower()
        if service_name := self._find_service(method, path):
            return self.get_service(service_name)
        return None

    def get_service(self, service_name: str) -> typing.Optional[ServiceInstance]:
        """Get the registered instance of a service."""
        return self.services.get(service_name, None)

    def _resolve_service_method_routes(
        self, service_name: str
    ) -> dict[str, list[re.Pattern]]:
        """Resolve service routes per method for a service"""
        method_routes = {
            "get": [],
            "post": [],
            "put": [],
            "delete": [],
        }
        routes = self._service_routes(service_name)
        for methods, path in routes:
            if methods == ["*"]:
                for method in method_routes:
                    method_routes[method].append(re.compile(path))
                continue

            for method in methods:
                method_routes[method].append(re.compile(path))

        return method_routes

    def _find_service(self, method: str, path: str):
        """
        Find first service matching the given request URL

        :param method: HTTP method of the request
        :param path: URL path to match against service patterns
        :return: Name of matching service or None if no match
        """
        for service_name, service_instance in self.services.items():
            routes_patterns = service_instance.method_routes[method]
            for route_pattern in routes_patterns:
                if route_pattern.match(path):
                    return service_name
        return None

    @staticmethod
    def _resolve_service_url(service_name: str) -> str:
        """Resolve service URL by service name."""
        return f"http://mlrun-{service_name}.{mlconf.namespace}.svc.cluster.local:8080"

    @staticmethod
    def _service_routes(service_name: str) -> list:
        """Get all routes for a service."""
        return {
            "api-chief": [],
            "api": [],
            "alerts": [
                (["put", "get", "delete"], "alert-templates.*"),
                (["*"], "projects/.+/alerts.*"),
                (["post"], "projects/.+/events.*"),
            ],
        }[service_name]
