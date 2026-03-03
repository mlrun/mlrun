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

import mlrun.common.runtimes.constants


def validate_sidecar_probes(sidecars: list[dict]) -> None:
    """Validate probe configurations in sidecars against Kubernetes V1Probe schema.

    Validates that each probe configuration has exactly one of the following:
    httpGet, exec, tcpSocket, or grpc.

    :param sidecars: List of sidecar dicts, each potentially containing probe configs.
    :raises mlrun.errors.MLRunInvalidArgumentError: If a probe has zero or more than one
        health check configuration key.
    """
    for sidecar in sidecars:
        for probe_type in mlrun.common.runtimes.constants.ProbeType.all():
            probe_config = sidecar.get(probe_type)
            if probe_config is None:
                continue

            # Count health check configuration keys
            present_keys = [
                key
                for key in mlrun.common.runtimes.constants.HEALTH_CHECK_KEYS
                if key in probe_config
            ]

            if len(present_keys) != 1:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Sidecar {probe_type} must have exactly one of "
                    f"the following configuration sections: "
                    f"{', '.join(mlrun.common.runtimes.constants.HEALTH_CHECK_KEYS)}"
                )
