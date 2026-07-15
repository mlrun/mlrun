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
"""Version-agnostic schema definitions shared by both Pydantic faces.

Holds enums, constants, regex patterns and plain (non-``BaseModel``) types that
are identical regardless of the installed Pydantic major. Both ``_v1`` and
``_v2`` import from here, so a shared definition has a single source and cannot
drift between faces.

This ``__init__`` re-exports the subset of shared names that the package
dispatcher exposes at the top level; per-topic modules (``_shared.<topic>``)
hold additional shared names imported directly by the faces.
"""

from .alert import (
    AlertActiveState,
)
from .api_gateway import (
    APIGatewayAuthenticationMode,
    APIGatewayState,
)
from .artifact import (
    ArtifactCategories,
)
from .auth import (
    AuthInfoKind,
    AuthorizationAction,
    AuthorizationResourceNamespace,
    AuthorizationResourceTypes,
    ProjectsRole,
)
from .background_task import (
    BackgroundTaskState,
)
from .clusterization_spec import (
    WaitForChiefToReachOnlineStateFeatureFlag,
)
from .constants import (
    APIStates,
    ArtifactPartitionByField,
    AuthorizationHeaderPrefixes,
    ClusterizationRole,
    CookieNames,
    DeletionStrategy,
    FeatureStorePartitionByField,
    HeaderNames,
    LogsCollectorMode,
    OrderType,
    PatchMode,
    RunPartitionByField,
    SortField,
)
from .events import (
    AuthSecretEventActions,
    DBConnectionEventActions,
    EventClientKinds,
    EventsModes,
    LogCollectorEventActions,
    MigrationEventActions,
    ProjectLifecycleEventActions,
    SecretEventActions,
)
from .frontend_spec import (
    NuclioStreamsFeatureFlag,
    PreemptionNodesFeatureFlag,
    ProjectMembershipFeatureFlag,
)
from .function import (
    FunctionState,
    PreemptionModes,
    SecurityContextEnrichmentModes,
)
from .http import (
    HTTPSessionRetryMode,
)
from .hub import (
    last_source_index,
)
from .k8s import (
    NodeSelectorOperator,
)
from .model_monitoring.constants import (
    DriftStatus,
    EndpointMode,
    EndpointType,
    EndpointUID,
    EventFieldType,
    EventKeyMetrics,
    FeatureSetFeatures,
    FileTargetKind,
    ModelEndpointCreationStrategy,
    ModelEndpointSchema,
    ModelMonitoringInfraLabel,
    ModelMonitoringMode,
    MonitoringFunctionNames,
    TSDBTarget,
    V3IOTSDBTables,
)
from .notification import (
    NotificationKind,
    NotificationSeverity,
    NotificationStatus,
)
from .object import (
    ObjectKind,
)
from .partition_interval import (
    PartitionInterval,
)
from .pipeline import (
    PipelinesPagination,
)
from .project import (
    ProjectDesiredState,
    ProjectState,
)
from .regex import (
    RegexMatchModes,
)
from .runtime_resource import (
    ListRuntimeResourcesGroupByField,
)
from .schedule import (
    ScheduleKinds,
)
from .secret import (
    SecretProviderName,
)
from .serving import (
    APIHandlerAction,
    ModelRunnerStepData,
    ModelsData,
    MonitoringData,
)

__all__ = [
    "APIGatewayAuthenticationMode",
    "APIGatewayState",
    "APIHandlerAction",
    "APIStates",
    "AlertActiveState",
    "ArtifactCategories",
    "ArtifactPartitionByField",
    "AuthInfoKind",
    "AuthSecretEventActions",
    "AuthorizationAction",
    "AuthorizationHeaderPrefixes",
    "AuthorizationResourceNamespace",
    "AuthorizationResourceTypes",
    "BackgroundTaskState",
    "ClusterizationRole",
    "CookieNames",
    "DBConnectionEventActions",
    "DeletionStrategy",
    "DriftStatus",
    "EndpointMode",
    "EndpointType",
    "EndpointUID",
    "EventClientKinds",
    "EventFieldType",
    "EventKeyMetrics",
    "EventsModes",
    "FeatureSetFeatures",
    "FeatureStorePartitionByField",
    "FileTargetKind",
    "FunctionState",
    "HTTPSessionRetryMode",
    "HeaderNames",
    "ListRuntimeResourcesGroupByField",
    "LogCollectorEventActions",
    "LogsCollectorMode",
    "MigrationEventActions",
    "ModelEndpointCreationStrategy",
    "ModelEndpointSchema",
    "ModelMonitoringInfraLabel",
    "ModelMonitoringMode",
    "ModelRunnerStepData",
    "ModelsData",
    "MonitoringData",
    "MonitoringFunctionNames",
    "NodeSelectorOperator",
    "NotificationKind",
    "NotificationSeverity",
    "NotificationStatus",
    "NuclioStreamsFeatureFlag",
    "ObjectKind",
    "OrderType",
    "PartitionInterval",
    "PatchMode",
    "PipelinesPagination",
    "PreemptionModes",
    "PreemptionNodesFeatureFlag",
    "ProjectDesiredState",
    "ProjectLifecycleEventActions",
    "ProjectMembershipFeatureFlag",
    "ProjectState",
    "ProjectsRole",
    "RegexMatchModes",
    "RunPartitionByField",
    "ScheduleKinds",
    "SecretEventActions",
    "SecretProviderName",
    "SecurityContextEnrichmentModes",
    "SortField",
    "TSDBTarget",
    "V3IOTSDBTables",
    "WaitForChiefToReachOnlineStateFeatureFlag",
    "last_source_index",
]
