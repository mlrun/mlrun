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

import typing

import pydantic

from .._shared.function import SecurityContextEnrichmentModes


class ImagePullSecret(pydantic.BaseModel):
    default: str | None = None


class Pipelines(pydantic.BaseModel):
    kfp_pod_user_unix_id: int | None = None


class SecurityContext(pydantic.BaseModel):
    default: str | None = None
    enrichment_mode: SecurityContextEnrichmentModes | None = None
    enrichment_group_id: int | None = None
    pipelines: Pipelines | None = None


class ServiceAccount(pydantic.BaseModel):
    default: str | None = None


class StateThresholds(pydantic.BaseModel):
    default: dict[str, str] | None = None


class Backoff(pydantic.BaseModel):
    default_base_delay: str | None = None
    min_base_delay: str | None = None


class RetrySpec(pydantic.BaseModel):
    backoff: Backoff


class FunctionSpec(pydantic.BaseModel):
    image_pull_secret: ImagePullSecret | None = None
    security_context: SecurityContext | None = None
    service_account: ServiceAccount | None = None
    state_thresholds: StateThresholds | None = None
    retry: RetrySpec | None = None

    model_config = pydantic.ConfigDict(extra="allow")


class Function(pydantic.BaseModel):
    spec: FunctionSpec | None = None
    application: dict[str, typing.Any] | None = None

    model_config = pydantic.ConfigDict(extra="allow")


class BatchingSpec(pydantic.BaseModel):
    # Set to True to enable batching
    enabled: bool
    # Maximal events to batch together. Default size is 10.
    batch_size: int | None = None
    # The maximum amount of time to wait before processing the batch. Default timeout is 1s.
    # Once this time passes, the batch is processed even if it hasn’t reached the full batch size.
    timeout: str | None = None

    def get_nuclio_batch_config(self):
        if not self.enabled:
            return None

        config = {"mode": "enable"}

        if self.batch_size:
            config["batchSize"] = self.batch_size

        if self.timeout:
            config["timeout"] = self.timeout

        return config
