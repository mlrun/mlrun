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
#
from enum import Enum

import mlrun.common.schemas
from mlrun.common.types import StrEnum

internal_abort_task_id = "internal-abort"


class LogSources(Enum):
    AUTO = "auto"
    PERSISTENCY = "persistency"
    K8S = "k8s"


class MaskOperations(StrEnum):
    CONCEAL = "conceal"
    REDACT = "redact"


# These are alert templates that come built-in with the system and pre-populated on system start
# If we define new system templates, those should be added here

pre_defined_templates = [
    mlrun.common.schemas.AlertTemplate(
        template_name="JobFailed",
        template_description="Generic template for job failure alerts",
        system_generated=True,
        summary="A job has failed",
        severity=mlrun.common.schemas.alert.AlertSeverity.MEDIUM,
        trigger={"events": [mlrun.common.schemas.alert.EventKind.FAILED]},
        reset_policy=mlrun.common.schemas.alert.ResetPolicy.AUTO,
    ),
    mlrun.common.schemas.AlertTemplate(
        template_name="DataDriftDetected",
        template_description="Generic template for data drift detected alerts",
        system_generated=True,
        summary="Model data drift has been detected",
        severity=mlrun.common.schemas.alert.AlertSeverity.HIGH,
        trigger={"events": [mlrun.common.schemas.alert.EventKind.DATA_DRIFT_DETECTED]},
        reset_policy=mlrun.common.schemas.alert.ResetPolicy.AUTO,
    ),
    mlrun.common.schemas.AlertTemplate(
        template_name="DataDriftSuspected",
        template_description="Generic template for data drift suspected alerts",
        system_generated=True,
        summary="Model data drift is suspected",
        severity=mlrun.common.schemas.alert.AlertSeverity.MEDIUM,
        trigger={"events": [mlrun.common.schemas.alert.EventKind.DATA_DRIFT_SUSPECTED]},
        reset_policy=mlrun.common.schemas.alert.ResetPolicy.AUTO,
    ),
    mlrun.common.schemas.AlertTemplate(
        template_name="ModelMonitoringApplicationFailed",
        template_description="Generic template for model monitoring application failure alerts",
        system_generated=True,
        summary="An invalid event has been detected in the model monitoring application",
        severity=mlrun.common.schemas.alert.AlertSeverity.MEDIUM,
        trigger={
            "events": [mlrun.common.schemas.alert.EventKind.MM_APP_FAILED]
        },
        reset_policy=mlrun.common.schemas.alert.ResetPolicy.AUTO,
    ),
]
