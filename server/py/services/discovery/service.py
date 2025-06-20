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

import httpx
from kubernetes import client, config

SERVICE_PORTS = {
    "mlrun-api-chief": 8080,
    "mlrun-worker": 8080,
    "mlrun-alerts": 8080,
}


class K8sServiceDiscovery:
    def __init__(
        self,
        namespace,
    ):
        self.ns = namespace
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        self._core_api = client.CoreV1Api()
        self._discovery_api = client.DiscoveryV1Api()

    async def broadcast(
        self,
        excluded_services,
        path,
        json_payload,
        timeout=10.0,
        headers=None,
    ):
        async with httpx.AsyncClient(timeout=timeout) as session:
            tasks = []
            for service, port in SERVICE_PORTS.items():
                if service in excluded_services:
                    continue
                for url in await self._discover_urls(
                    service=service,
                    port=port,
                    path=path,
                ):
                    tasks.append(
                        session.post(
                            url=url,
                            json=json_payload,
                            headers=headers,
                        )
                    )
            await asyncio.gather(*tasks, return_exceptions=True)

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

        slices = self._discovery_api.list_namespaced_endpoint_slice(
            self.ns,
            label_selector=f"kubernetes.io/service-name={service}",
        ).items

        for slc in slices:
            target_port = next(
                (p.port for p in slc.ports or [] if p.port == port), port
            )
            for ep in slc.endpoints:
                urls.extend(
                    f"http://{addr}:{target_port}{path}" for addr in ep.addresses
                )

        if urls:
            return urls

        eps = self._core_api.read_namespaced_endpoints(service, self.ns)
        for subset in eps.subsets or []:
            tgt_port = next(
                (p.port for p in subset.ports or [] if p.port == port), port
            )
            for addr in subset.addresses or []:
                urls.append(f"http://{addr.ip}:{tgt_port}{path}")

        return urls
