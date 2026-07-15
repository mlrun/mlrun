# Copyright 2023 Iguazio
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

import pydantic.v1

from .._shared.function import SecurityContextEnrichmentModes


class ImagePullSecret(pydantic.v1.BaseModel):
    default: str | None


class Pipelines(pydantic.v1.BaseModel):
    kfp_pod_user_unix_id: int | None


class SecurityContext(pydantic.v1.BaseModel):
    default: str | None
    enrichment_mode: SecurityContextEnrichmentModes | None
    enrichment_group_id: int | None
    pipelines: Pipelines | None


class ServiceAccount(pydantic.v1.BaseModel):
    default: str | None


class StateThresholds(pydantic.v1.BaseModel):
    default: dict[str, str] | None


class Backoff(pydantic.v1.BaseModel):
    default_base_delay: str | None
    min_base_delay: str | None


class RetrySpec(pydantic.v1.BaseModel):
    backoff: Backoff


class FunctionSpec(pydantic.v1.BaseModel):
    image_pull_secret: ImagePullSecret | None
    security_context: SecurityContext | None
    service_account: ServiceAccount | None
    state_thresholds: StateThresholds | None
    retry: RetrySpec | None

    class Config:
        extra = pydantic.v1.Extra.allow


class Function(pydantic.v1.BaseModel):
    spec: FunctionSpec | None
    application: dict[str, typing.Any] | None

    class Config:
        extra = pydantic.v1.Extra.allow


class BatchingSpec(pydantic.v1.BaseModel):
    # Set to True to enable batching
    enabled: bool
    # Maximal events to batch together. Default size is 10.
    batch_size: int | None
    # The maximum amount of time to wait before processing the batch. Default timeout is 1s.
    # Once this time passes, the batch is processed even if it hasn’t reached the full batch size.
    timeout: str | None

    def get_nuclio_batch_config(self):
        if not self.enabled:
            return None

        config = {"mode": "enable"}

        if self.batch_size:
            config["batchSize"] = self.batch_size

        if self.timeout:
            config["timeout"] = self.timeout

        return config
