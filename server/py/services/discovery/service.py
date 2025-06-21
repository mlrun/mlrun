# Copyright 2025 Iguazio
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

import asyncio
from typing import Optional

import httpx
from kubernetes import client, config

import mlrun.errors
import mlrun.utils

SERVICE_PORTS = {
    "mlrun-api-chief": 8080,
    "mlrun-api": 8080,
    "mlrun-alerts": 8080,
}


class K8sServiceDiscovery:
    def __init__(
        self,
        namespace: str,
    ):
        if not namespace:
            raise ValueError("Namespace must be provided for K8sServiceDiscovery")
        self.namespace = namespace
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        self._core_api = client.CoreV1Api()
        self._discovery_api = client.DiscoveryV1Api()

    async def broadcast(
        self,
        excluded_services: list[str],
        path: str,
        json_payload: Optional[dict] = None,
        timeout: float = 10.0,
        headers: Optional[dict] = None,
    ):
        async with httpx.AsyncClient(timeout=timeout) as session:
            tasks = []
            for service, port in SERVICE_PORTS.items():
                if service in excluded_services:
                    continue
                try:
                    urls = await self._discover_urls(
                        service=service,
                        port=port,
                        path=path,
                    )
                except Exception as exc:
                    mlrun.utils.logger.exc(
                        "service discovery failed",
                        service=service,
                        exc=mlrun.errors.err_to_str(exc),
                    )
                    continue
                for url in urls:
                    tasks.append(
                        session.post(url=url, json=json_payload, headers=headers)
                    )
            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

    async def _discover_urls(
        self,
        service: str,
        port: int,
        path: str,
    ) -> list[str]:
        """
        Return every  http://IP:port/<path>  belonging to <svc>.

        1. Try EndpointSlice (modern clusters, K8s ≥ 1.21).
        2. Fall back to legacy Endpoints if no URLs found.
        """
        urls = []

        try:
            slices = self._discovery_api.list_namespaced_endpoint_slice(
                namespace=self.namespace,
                label_selector=f"kubernetes.io/service-name={service}",
            ).items
        except client.exceptions.ApiException as exc:
            if exc.status in (403, 404):
                mlrun.utils.logger.debug(
                    "EndpointSlice unavailable in service discovery",
                    exc=mlrun.errors.err_to_str(exc),
                )
                slices = []
            else:
                raise

        for slice_ in slices:
            target_port = next(
                (
                    slice_port.port
                    for slice_port in slice_.ports or []
                    if slice_port.port == port
                ),
                port,
            )
            for endpoint in slice_.endpoints:
                urls.extend(
                    f"http://{addr}:{target_port}{path}" for addr in endpoint.addresses
                )

        if urls:
            return urls

        try:
            endpoints = self._core_api.read_namespaced_endpoints(
                name=service,
                namespace=self.namespace,
            )
        except client.exceptions.ApiException as exc:
            if exc.status in (403, 404):
                endpoints = None
                mlrun.utils.logger.debug(
                    "Endpoints unavailable in service discovery",
                    exc=mlrun.errors.err_to_str(exc),
                )
            else:
                raise
        if endpoints:
            for subset in endpoints.subsets or []:
                target_port = next(
                    (
                        subset_port.port
                        for subset_port in subset.ports or []
                        if subset_port.port == port
                    ),
                    port,
                )
                for addr in subset.addresses or []:
                    urls.append(f"http://{addr.ip}:{target_port}{path}")

        return urls
