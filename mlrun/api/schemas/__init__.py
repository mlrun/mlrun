# Copyright 2018 Iguazio
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
# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx

from .artifact import ArtifactCategories, ArtifactIdentifier, ArtifactsFormat
from .auth import (
    AuthInfo,
    AuthorizationAction,
    AuthorizationResourceTypes,
    AuthorizationVerificationInput,
    Credentials,
    ProjectsRole,
)
from .background_task import (
    BackgroundTask,
    BackgroundTaskMetadata,
    BackgroundTaskSpec,
    BackgroundTaskState,
    BackgroundTaskStatus,
)
from .client_spec import ClientSpec
from .clusterization_spec import (
    ClusterizationSpec,
    WaitForChiefToReachOnlineStateFeatureFlag,
)
from .constants import (
    APIStates,
    ClusterizationRole,
    DeletionStrategy,
    FeatureStorePartitionByField,
    HeaderNames,
    LogsCollectorMode,
    OrderType,
    PatchMode,
    RunPartitionByField,
    SortField,
)
from .feature_store import (
    EntitiesOutput,
    Entity,
    EntityListOutput,
    EntityRecord,
    Feature,
    FeatureListOutput,
    FeatureRecord,
    FeatureSet,
    FeatureSetDigestOutput,
    FeatureSetDigestSpec,
    FeatureSetIngestInput,
    FeatureSetIngestOutput,
    FeatureSetRecord,
    FeatureSetsOutput,
    FeatureSetSpec,
    FeatureSetsTagsOutput,
    FeaturesOutput,
    FeatureVector,
    FeatureVectorRecord,
    FeatureVectorsOutput,
    FeatureVectorsTagsOutput,
)
from .frontend_spec import (
    AuthenticationFeatureFlag,
    FeatureFlags,
    FrontendSpec,
    NuclioStreamsFeatureFlag,
    PreemptionNodesFeatureFlag,
    ProjectMembershipFeatureFlag,
)
from .function import FunctionState, PreemptionModes, SecurityContextEnrichmentModes
from .http import HTTPSessionRetryMode
from .k8s import NodeSelectorOperator, Resources, ResourceSpec
from .marketplace import (
    IndexedMarketplaceSource,
    MarketplaceCatalog,
    MarketplaceItem,
    MarketplaceObjectMetadata,
    MarketplaceSource,
    MarketplaceSourceSpec,
    last_source_index,
)
from .memory_reports import MostCommonObjectTypesReport, ObjectTypeReport
from .model_endpoints import (
    Features,
    FeatureValues,
    GrafanaColumn,
    GrafanaDataPoint,
    GrafanaNumberColumn,
    GrafanaStringColumn,
    GrafanaTable,
    GrafanaTimeSeriesTarget,
    Metric,
    ModelEndpoint,
    ModelEndpointList,
    ModelEndpointMetadata,
    ModelEndpointSpec,
    ModelEndpointStatus,
    ModelMonitoringMode,
    ModelMonitoringStoreKinds,
)
from .object import ObjectKind, ObjectMetadata, ObjectSpec, ObjectStatus
from .pipeline import PipelinesFormat, PipelinesOutput, PipelinesPagination
from .project import (
    IguazioProject,
    Project,
    ProjectDesiredState,
    ProjectMetadata,
    ProjectOwner,
    ProjectsFormat,
    ProjectsOutput,
    ProjectSpec,
    ProjectState,
    ProjectStatus,
    ProjectSummariesOutput,
    ProjectSummary,
)
from .runtime_resource import (
    GroupedByJobRuntimeResourcesOutput,
    GroupedByProjectRuntimeResourcesOutput,
    KindRuntimeResources,
    ListRuntimeResourcesGroupByField,
    RuntimeResource,
    RuntimeResources,
    RuntimeResourcesOutput,
)
from .schedule import (
    ScheduleCronTrigger,
    ScheduleInput,
    ScheduleKinds,
    ScheduleOutput,
    ScheduleRecord,
    SchedulesOutput,
    ScheduleUpdate,
)
from .secret import (
    AuthSecretData,
    SecretKeysData,
    SecretProviderName,
    SecretsData,
    UserSecretCreationRequest,
)
from .tag import Tag, TagObjects
