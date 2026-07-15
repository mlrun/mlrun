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
"""Pydantic 1 schema face.

Today's schema models, relocated here unchanged (defined on the ``pydantic.v1``
namespace). Bound by the package dispatcher under Pydantic 1 — what the
client/SDK loads. Imports version-agnostic definitions from ``_shared``; imports
peer models from sibling ``_v1.<topic>`` modules.
"""

from .alert import (
    AlertActivation,
    AlertActivations,
    AlertConfig,
    AlertCriteria,
    AlertNotification,
    AlertTemplate,
    Event,
)
from .api_gateway import (
    APIGateway,
    APIGatewayBasicAuth,
    APIGatewayMetadata,
    APIGatewaysOutput,
    APIGatewaySpec,
    APIGatewayStatus,
    APIGatewayUpstream,
)
from .artifact import (
    Artifact,
    ArtifactIdentifier,
    ArtifactMetadata,
    ArtifactSpec,
)
from .auth import (
    AuthInfo,
    AuthorizationVerificationInput,
    Credentials,
)
from .background_task import (
    BackgroundTask,
    BackgroundTaskList,
    BackgroundTaskMetadata,
    BackgroundTaskSpec,
    BackgroundTaskStatus,
)
from .client_spec import (
    ClientSpec,
)
from .clusterization_spec import (
    ClusterizationSpec,
)
from .common import (
    ImageBuilder,
)
from .datastore_profile import (
    DatastoreProfile,
)
from .feature_store import (
    EntitiesOutput,
    EntitiesOutputV2,
    Entity,
    EntityListOutput,
    EntityRecord,
    Feature,
    FeatureListOutput,
    FeatureRecord,
    FeatureSet,
    FeatureSetDigestOutput,
    FeatureSetDigestOutputV2,
    FeatureSetDigestSpec,
    FeatureSetDigestSpecV2,
    FeatureSetIngestInput,
    FeatureSetIngestOutput,
    FeatureSetRecord,
    FeatureSetsOutput,
    FeatureSetSpec,
    FeatureSetsTagsOutput,
    FeaturesOutput,
    FeaturesOutputV2,
    FeatureVector,
    FeatureVectorRecord,
    FeatureVectorsOutput,
    FeatureVectorsTagsOutput,
)
from .frontend_spec import (
    ArtifactLimits,
    FeatureFlags,
    FrontendSpec,
)
from .function import (
    BatchingSpec,
)
from .hub import (
    HubCatalog,
    HubItem,
    HubObjectMetadata,
    HubSource,
    HubSourceSpec,
    IndexedHubSource,
)
from .k8s import (
    Resources,
    ResourceSpec,
)
from .memory_reports import (
    MostCommonObjectTypesReport,
    ObjectTypeReport,
)
from .model_monitoring.grafana import (
    GrafanaColumn,
    GrafanaNumberColumn,
    GrafanaStringColumn,
    GrafanaTable,
)
from .model_monitoring.model_endpoints import (
    Features,
    FeatureValues,
    ModelEndpoint,
    ModelEndpointDriftValues,
    ModelEndpointList,
    ModelEndpointMetadata,
    ModelEndpointSpec,
    ModelEndpointStatus,
)
from .notification import (
    Notification,
    NotificationState,
    NotificationSummary,
    SetNotificationRequest,
)
from .object import (
    ObjectMetadata,
    ObjectSpec,
    ObjectStatus,
)
from .pagination import (
    PaginationInfo,
)
from .pipeline import (
    PipelinesOutput,
)
from .project import (
    IguazioProject,
    Project,
    ProjectMetadata,
    ProjectMonitoringSpec,
    ProjectOut,
    ProjectOutput,
    ProjectOwner,
    ProjectsOutput,
    ProjectSpec,
    ProjectSpecOut,
    ProjectStatus,
    ProjectSummariesOutput,
    ProjectSummary,
)
from .runs import (
    RunIdentifier,
)
from .runtime_resource import (
    GroupedByJobRuntimeResourcesOutput,
    GroupedByProjectRuntimeResourcesOutput,
    KindRuntimeResources,
    RuntimeResource,
    RuntimeResources,
    RuntimeResourcesOutput,
)
from .schedule import (
    ScheduleCronTrigger,
    ScheduleIdentifier,
    ScheduleInput,
    ScheduleOutput,
    ScheduleRecord,
    SchedulesOutput,
    ScheduleUpdate,
)
from .secret import (
    AuthSecretData,
    DeleteSecretTokenResponse,
    DeleteSecretTokensResponse,
    ListSecretTokensResponse,
    SecretKeysData,
    SecretsData,
    SecretToken,
    SecretTokenInfo,
    StoreSecretTokensResponse,
)
from .tag import (
    Tag,
    TagObjects,
)
from .workflow import (
    GetWorkflowResponse,
    RerunWorkflowRequest,
    WorkflowRequest,
    WorkflowResponse,
    WorkflowSpec,
)

__all__ = [
    "APIGateway",
    "APIGatewayBasicAuth",
    "APIGatewayMetadata",
    "APIGatewaySpec",
    "APIGatewayStatus",
    "APIGatewayUpstream",
    "APIGatewaysOutput",
    "AlertActivation",
    "AlertActivations",
    "AlertConfig",
    "AlertCriteria",
    "AlertNotification",
    "AlertTemplate",
    "Artifact",
    "ArtifactIdentifier",
    "ArtifactLimits",
    "ArtifactMetadata",
    "ArtifactSpec",
    "AuthInfo",
    "AuthSecretData",
    "AuthorizationVerificationInput",
    "BackgroundTask",
    "BackgroundTaskList",
    "BackgroundTaskMetadata",
    "BackgroundTaskSpec",
    "BackgroundTaskStatus",
    "BatchingSpec",
    "ClientSpec",
    "ClusterizationSpec",
    "Credentials",
    "DatastoreProfile",
    "DeleteSecretTokenResponse",
    "DeleteSecretTokensResponse",
    "EntitiesOutput",
    "EntitiesOutputV2",
    "Entity",
    "EntityListOutput",
    "EntityRecord",
    "Event",
    "Feature",
    "FeatureFlags",
    "FeatureListOutput",
    "FeatureRecord",
    "FeatureSet",
    "FeatureSetDigestOutput",
    "FeatureSetDigestOutputV2",
    "FeatureSetDigestSpec",
    "FeatureSetDigestSpecV2",
    "FeatureSetIngestInput",
    "FeatureSetIngestOutput",
    "FeatureSetRecord",
    "FeatureSetSpec",
    "FeatureSetsOutput",
    "FeatureSetsTagsOutput",
    "FeatureValues",
    "FeatureVector",
    "FeatureVectorRecord",
    "FeatureVectorsOutput",
    "FeatureVectorsTagsOutput",
    "Features",
    "FeaturesOutput",
    "FeaturesOutputV2",
    "FrontendSpec",
    "GetWorkflowResponse",
    "GrafanaColumn",
    "GrafanaNumberColumn",
    "GrafanaStringColumn",
    "GrafanaTable",
    "GroupedByJobRuntimeResourcesOutput",
    "GroupedByProjectRuntimeResourcesOutput",
    "HubCatalog",
    "HubItem",
    "HubObjectMetadata",
    "HubSource",
    "HubSourceSpec",
    "IguazioProject",
    "ImageBuilder",
    "IndexedHubSource",
    "KindRuntimeResources",
    "ListSecretTokensResponse",
    "ModelEndpoint",
    "ModelEndpointDriftValues",
    "ModelEndpointList",
    "ModelEndpointMetadata",
    "ModelEndpointSpec",
    "ModelEndpointStatus",
    "MostCommonObjectTypesReport",
    "Notification",
    "NotificationState",
    "NotificationSummary",
    "ObjectMetadata",
    "ObjectSpec",
    "ObjectStatus",
    "ObjectTypeReport",
    "PaginationInfo",
    "PipelinesOutput",
    "Project",
    "ProjectMetadata",
    "ProjectMonitoringSpec",
    "ProjectOut",
    "ProjectOutput",
    "ProjectOwner",
    "ProjectSpec",
    "ProjectSpecOut",
    "ProjectStatus",
    "ProjectSummariesOutput",
    "ProjectSummary",
    "ProjectsOutput",
    "RerunWorkflowRequest",
    "ResourceSpec",
    "Resources",
    "RunIdentifier",
    "RuntimeResource",
    "RuntimeResources",
    "RuntimeResourcesOutput",
    "ScheduleCronTrigger",
    "ScheduleIdentifier",
    "ScheduleInput",
    "ScheduleOutput",
    "ScheduleRecord",
    "ScheduleUpdate",
    "SchedulesOutput",
    "SecretKeysData",
    "SecretToken",
    "SecretTokenInfo",
    "SecretsData",
    "SetNotificationRequest",
    "StoreSecretTokensResponse",
    "Tag",
    "TagObjects",
    "WorkflowRequest",
    "WorkflowResponse",
    "WorkflowSpec",
]
