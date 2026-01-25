# Copyright 2026 Iguazio
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

import time

import kubernetes

import framework.utils.singletons.k8s


def wait_for_pod_phase(
    k8s: framework.utils.singletons.k8s.K8sHelper,
    name: str,
    namespace: str,
    desired_phases: set[str],
    timeout_seconds: int = 300,
    sleep_seconds: float = 2.0,
) -> str:
    deadline = time.time() + timeout_seconds
    last_phase = None
    while time.time() < deadline:
        status = k8s.get_pod_status(name=name, namespace=namespace)
        last_phase = (status.phase or "").lower()
        if last_phase in desired_phases:
            return last_phase
        time.sleep(sleep_seconds)
    raise TimeoutError(
        f"Timed out waiting for pod {namespace}/{name} to reach {desired_phases}. Last phase: {last_phase}"
    )


def dump_pod_logs(
    k8s: framework.utils.singletons.k8s.K8sHelper, name: str, namespace: str
) -> str:
    try:
        return k8s.logs(name=name, namespace=namespace)
    except Exception as exc:
        return f"<failed to read logs: {exc}>"


def create_or_replace_k8s_resource(
    k8s: framework.utils.singletons.k8s.K8sHelper,
    resource_type: str,
    resource_name: str,
    resource: object,
    namespace: str,
) -> None:
    try:
        fn = getattr(k8s.v1api, f"create_namespaced_{resource_type}")
        fn(namespace=namespace, body=resource)
    except kubernetes.client.rest.ApiException as exc:
        if exc.status != 409:
            raise
        fn = getattr(k8s.v1api, f"replace_namespaced_{resource_type}")
        fn(name=resource_name, namespace=namespace, body=resource)


def ensure_k8s_resource_deleted(
    k8s: framework.utils.singletons.k8s.K8sHelper,
    resource_type: str,
    resource_name: str,
    namespace: str,
) -> None:
    try:
        fn = getattr(k8s.v1api, f"delete_namespaced_{resource_type}")
        fn(name=resource_name, namespace=namespace)
    except kubernetes.client.rest.ApiException as exc:
        if exc.status != 404:
            raise
