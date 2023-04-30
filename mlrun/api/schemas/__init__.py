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


import mlrun.common.schemas
from mlrun.utils.helpers import DeprecationHelper

# OldClsName = DeprecationHelper(NewClsName)
ArtifactCategories = DeprecationHelper(mlrun.common.schemas.ArtifactCategories)
ArtifactIdentifier = DeprecationHelper(mlrun.common.schemas.ArtifactIdentifier)
ArtifactsFormat = DeprecationHelper(mlrun.common.schemas.ArtifactsFormat)
AuthInfo = DeprecationHelper(mlrun.common.schemas.AuthInfo)
AuthorizationAction = DeprecationHelper(mlrun.common.schemas.AuthorizationAction)
AuthorizationResourceTypes = DeprecationHelper(
    mlrun.common.schemas.AuthorizationResourceTypes
)
AuthorizationVerificationInput = DeprecationHelper(
    mlrun.common.schemas.AuthorizationVerificationInput
)
Credentials = DeprecationHelper(mlrun.common.schemas.Credentials)
ProjectsRole = DeprecationHelper(mlrun.common.schemas.ProjectsRole)

BackgroundTask = DeprecationHelper(mlrun.common.schemas.BackgroundTask)
BackgroundTaskMetadata = DeprecationHelper(mlrun.common.schemas.BackgroundTaskMetadata)
BackgroundTaskSpec = DeprecationHelper(mlrun.common.schemas.BackgroundTaskSpec)
BackgroundTaskState = DeprecationHelper(mlrun.common.schemas.BackgroundTaskState)
BackgroundTaskStatus = DeprecationHelper(mlrun.common.schemas.BackgroundTaskStatus)
ClientSpe = DeprecationHelper(mlrun.common.schemas.ClientSpec)
ClusterizationSpec = DeprecationHelper(mlrun.common.schemas.ClusterizationSpec)
WaitForChiefToReachOnlineStateFeatureFlag = DeprecationHelper(
    mlrun.common.schemas.WaitForChiefToReachOnlineStateFeatureFlag
)
APIStates = DeprecationHelper(mlrun.common.schemas.APIStates)
ClusterizationRole = DeprecationHelper(mlrun.common.schemas.ClusterizationRole)
DeletionStrategy = DeprecationHelper(mlrun.common.schemas.DeletionStrategy)
FeatureStorePartitionByField = DeprecationHelper(
    mlrun.common.schemas.FeatureStorePartitionByField
)
HeaderNames = DeprecationHelper(mlrun.common.schemas.HeaderNames)
LogsCollectorMode = DeprecationHelper(mlrun.common.schemas.LogsCollectorMode)
OrderType = DeprecationHelper(mlrun.common.schemas.OrderType)
PatchMode = DeprecationHelper(mlrun.common.schemas.PatchMode)
RunPartitionByField = DeprecationHelper(mlrun.common.schemas.RunPartitionByField)
SortField = DeprecationHelper(mlrun.common.schemas.SortField)
EntitiesOutput = DeprecationHelper(mlrun.common.schemas.EntitiesOutput)
Entity = DeprecationHelper(mlrun.common.schemas.Entity)
EntityListOutput = DeprecationHelper(mlrun.common.schemas.EntityListOutput)
EntityRecord = DeprecationHelper(mlrun.common.schemas.EntityRecord)
Feature = DeprecationHelper(mlrun.common.schemas.Feature)
FeatureListOutput = DeprecationHelper(mlrun.common.schemas.FeatureListOutput)
FeatureRecord = DeprecationHelper(mlrun.common.schemas.FeatureRecord)
FeatureSet = DeprecationHelper(mlrun.common.schemas.FeatureSet)
FeatureSetDigestOutput = DeprecationHelper(mlrun.common.schemas.FeatureSetDigestOutput)
FeatureSetDigestSpec = DeprecationHelper(mlrun.common.schemas.FeatureSetDigestSpec)
FeatureSetIngestInput = DeprecationHelper(mlrun.common.schemas.FeatureSetIngestInput)
FeatureSetIngestOutput = DeprecationHelper(mlrun.common.schemas.FeatureSetIngestOutput)
FeatureSetRecord = DeprecationHelper(mlrun.common.schemas.FeatureSetRecord)
FeatureSetsOutput = DeprecationHelper(mlrun.common.schemas.FeatureSetsOutput)
FeatureSetSpec = DeprecationHelper(mlrun.common.schemas.FeatureSetSpec)
FeatureSetsTagsOutput = DeprecationHelper(mlrun.common.schemas.FeatureSetsTagsOutput)
FeaturesOutput = DeprecationHelper(mlrun.common.schemas.FeaturesOutput)
FeatureVector = DeprecationHelper(mlrun.common.schemas.FeatureVector)
FeatureVectorRecord = DeprecationHelper(mlrun.common.schemas.FeatureVectorRecord)
FeatureVectorsOutput = DeprecationHelper(mlrun.common.schemas.FeatureVectorsOutput)
FeatureVectorsTagsOutput = DeprecationHelper(
    mlrun.common.schemas.FeatureVectorsTagsOutput
)
AuthenticationFeatureFlag = DeprecationHelper(
    mlrun.common.schemas.AuthenticationFeatureFlag
)
FeatureFlags = DeprecationHelper(mlrun.common.schemas.FeatureFlags)
FrontendSpec = DeprecationHelper(mlrun.common.schemas.FrontendSpec)
NuclioStreamsFeatureFlag = DeprecationHelper(
    mlrun.common.schemas.NuclioStreamsFeatureFlag
)
PreemptionNodesFeatureFlag = DeprecationHelper(
    mlrun.common.schemas.PreemptionNodesFeatureFlag
)
ProjectMembershipFeatureFlag = DeprecationHelper(
    mlrun.common.schemas.ProjectMembershipFeatureFlag
)
FunctionState = DeprecationHelper(mlrun.common.schemas.FunctionState)
PreemptionModes = DeprecationHelper(mlrun.common.schemas.PreemptionModes)
SecurityContextEnrichmentModes = DeprecationHelper(
    mlrun.common.schemas.SecurityContextEnrichmentModes
)
HTTPSessionRetryMode = DeprecationHelper(mlrun.common.schemas.HTTPSessionRetryMode)
NodeSelectorOperator = DeprecationHelper(mlrun.common.schemas.NodeSelectorOperator)
Resources = DeprecationHelper(mlrun.common.schemas.Resources)
ResourceSpec = DeprecationHelper(mlrun.common.schemas.ResourceSpec)
IndexedMarketplaceSource = DeprecationHelper(
    mlrun.common.schemas.IndexedMarketplaceSource
)
MarketplaceCatalog = DeprecationHelper(mlrun.common.schemas.MarketplaceCatalog)
MarketplaceItem = DeprecationHelper(mlrun.common.schemas.MarketplaceItem)
MarketplaceObjectMetadata = DeprecationHelper(
    mlrun.common.schemas.MarketplaceObjectMetadata
)
MarketplaceSource = DeprecationHelper(mlrun.common.schemas.MarketplaceSource)
MarketplaceSourceSpec = DeprecationHelper(mlrun.common.schemas.MarketplaceSourceSpec)
last_source_index = DeprecationHelper(mlrun.common.schemas.last_source_index)
MostCommonObjectTypesReport = DeprecationHelper(
    mlrun.common.schemas.MostCommonObjectTypesReport
)
ObjectTypeReport = DeprecationHelper(mlrun.common.schemas.ObjectTypeReport)
Features = DeprecationHelper(mlrun.common.schemas.Features)
FeatureValues = DeprecationHelper(mlrun.common.schemas.FeatureValues)
GrafanaColumn = DeprecationHelper(mlrun.common.schemas.GrafanaColumn)
GrafanaDataPoint = DeprecationHelper(mlrun.common.schemas.GrafanaDataPoint)
GrafanaNumberColumn = DeprecationHelper(mlrun.common.schemas.GrafanaNumberColumn)
GrafanaStringColumn = DeprecationHelper(mlrun.common.schemas.GrafanaStringColumn)
GrafanaTable = DeprecationHelper(mlrun.common.schemas.GrafanaTable)
GrafanaTimeSeriesTarget = DeprecationHelper(
    mlrun.common.schemas.GrafanaTimeSeriesTarget
)
ModelEndpoint = DeprecationHelper(mlrun.common.schemas.ModelEndpoint)
ModelEndpointList = DeprecationHelper(mlrun.common.schemas.ModelEndpointList)
ModelEndpointMetadata = DeprecationHelper(mlrun.common.schemas.ModelEndpointMetadata)
ModelEndpointSpec = DeprecationHelper(mlrun.common.schemas.ModelEndpointSpec)
ModelEndpointStatus = DeprecationHelper(mlrun.common.schemas.ModelEndpointStatus)
ModelMonitoringStoreKinds = DeprecationHelper(
    mlrun.common.schemas.ModelMonitoringStoreKinds
)
NotificationSeverity = DeprecationHelper(mlrun.common.schemas.NotificationSeverity)
NotificationStatus = DeprecationHelper(mlrun.common.schemas.NotificationStatus)
ObjectKind = DeprecationHelper(mlrun.common.schemas.ObjectKind)
ObjectMetadata = DeprecationHelper(mlrun.common.schemas.ObjectMetadata)
ObjectSpec = DeprecationHelper(mlrun.common.schemas.ObjectSpec)
ObjectStatus = DeprecationHelper(mlrun.common.schemas.ObjectStatus)
PipelinesFormat = DeprecationHelper(mlrun.common.schemas.PipelinesFormat)
PipelinesOutput = DeprecationHelper(mlrun.common.schemas.PipelinesOutput)
PipelinesPagination = DeprecationHelper(mlrun.common.schemas.PipelinesPagination)
IguazioProject = DeprecationHelper(mlrun.common.schemas.IguazioProject)
Project = DeprecationHelper(mlrun.common.schemas.Project)
ProjectDesiredState = DeprecationHelper(mlrun.common.schemas.ProjectDesiredState)
ProjectMetadata = DeprecationHelper(mlrun.common.schemas.ProjectMetadata)
ProjectOwner = DeprecationHelper(mlrun.common.schemas.ProjectOwner)
ProjectsFormat = DeprecationHelper(mlrun.common.schemas.ProjectsFormat)
ProjectsOutput = DeprecationHelper(mlrun.common.schemas.ProjectsOutput)
ProjectSpec = DeprecationHelper(mlrun.common.schemas.ProjectSpec)
ProjectState = DeprecationHelper(mlrun.common.schemas.ProjectState)
ProjectStatus = DeprecationHelper(mlrun.common.schemas.ProjectStatus)
ProjectSummariesOutput = DeprecationHelper(mlrun.common.schemas.ProjectSummariesOutput)
ProjectSummary = DeprecationHelper(mlrun.common.schemas.ProjectSummary)
GroupedByJobRuntimeResourcesOutput = DeprecationHelper(
    mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput
)
GroupedByProjectRuntimeResourcesOutput = DeprecationHelper(
    mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput
)
KindRuntimeResources = DeprecationHelper(mlrun.common.schemas.KindRuntimeResources)
ListRuntimeResourcesGroupByField = DeprecationHelper(
    mlrun.common.schemas.ListRuntimeResourcesGroupByField
)
RuntimeResource = DeprecationHelper(mlrun.common.schemas.RuntimeResource)
RuntimeResources = DeprecationHelper(mlrun.common.schemas.RuntimeResources)
RuntimeResourcesOutput = DeprecationHelper(mlrun.common.schemas.RuntimeResourcesOutput)
ScheduleCronTrigger = DeprecationHelper(mlrun.common.schemas.ScheduleCronTrigger)
ScheduleInput = DeprecationHelper(mlrun.common.schemas.ScheduleInput)
ScheduleKinds = DeprecationHelper(mlrun.common.schemas.ScheduleKinds)
ScheduleOutput = DeprecationHelper(mlrun.common.schemas.ScheduleOutput)
ScheduleRecord = DeprecationHelper(mlrun.common.schemas.ScheduleRecord)
SchedulesOutput = DeprecationHelper(mlrun.common.schemas.SchedulesOutput)
ScheduleUpdate = DeprecationHelper(mlrun.common.schemas.ScheduleUpdate)
AuthSecretData = DeprecationHelper(mlrun.common.schemas.AuthSecretData)
SecretKeysData = DeprecationHelper(mlrun.common.schemas.SecretKeysData)
SecretProviderName = DeprecationHelper(mlrun.common.schemas.SecretProviderName)
SecretsData = DeprecationHelper(mlrun.common.schemas.SecretsData)
UserSecretCreationRequest = DeprecationHelper(
    mlrun.common.schemas.UserSecretCreationRequest
)
Tag = DeprecationHelper(mlrun.common.schemas.Tag)
TagObjects = DeprecationHelper(mlrun.common.schemas.TagObjects)
