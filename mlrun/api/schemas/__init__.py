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

"""
Schemas were moved to mlrun.common.schemas.
For backwards compatibility reasons, we have left this folder untouched so that older version of mlrun would
   be able to migrate to newer version without having to upgrade into an intermediate version.

The DeprecationHelper class is used to print a deprecation warning when the old import is used, and return the new
schema.
"""

import sys
import warnings

import mlrun.common.formatters
import mlrun.common.schemas
import mlrun.common.schemas.artifact as old_artifact
import mlrun.common.schemas.auth as old_auth
import mlrun.common.schemas.background_task as old_background_task
import mlrun.common.schemas.client_spec as old_client_spec
import mlrun.common.schemas.clusterization_spec as old_clusterization_spec
import mlrun.common.schemas.constants as old_constants
import mlrun.common.schemas.feature_store as old_feature_store
import mlrun.common.schemas.frontend_spec as old_frontend_spec
import mlrun.common.schemas.function as old_function
import mlrun.common.schemas.http as old_http
import mlrun.common.schemas.k8s as old_k8s
import mlrun.common.schemas.memory_reports as old_memory_reports
import mlrun.common.schemas.model_monitoring.grafana
import mlrun.common.schemas.object as old_object
import mlrun.common.schemas.pipeline as old_pipeline
import mlrun.common.schemas.project as old_project
import mlrun.common.schemas.runtime_resource as old_runtime_resource
import mlrun.common.schemas.schedule as old_schedule
import mlrun.common.schemas.secret as old_secret
import mlrun.common.schemas.tag as old_tag


class DeprecationHelper:
    """A helper class to deprecate old schemas"""

    def __init__(self, new_target, version="1.4.0"):
        self._new_target = new_target
        self._version = version

    def _warn(self):
        warnings.warn(
            f"mlrun.api.schemas.{self._new_target.__name__} is deprecated since version {self._version}, "
            f"Use mlrun.common.schemas.{self._new_target.__name__} instead.",
            FutureWarning,
        )

    def __call__(self, *args, **kwargs):
        self._warn()
        return self._new_target(*args, **kwargs)

    def __getattr__(self, attr):
        self._warn()
        return getattr(self._new_target, attr)


# for backwards compatibility, we need to inject the old import path to `sys.modules`
sys.modules["mlrun.api.schemas.artifact"] = old_artifact
sys.modules["mlrun.api.schemas.auth"] = old_auth
sys.modules["mlrun.api.schemas.background_task"] = old_background_task
sys.modules["mlrun.api.schemas.client_spec"] = old_client_spec
sys.modules["mlrun.api.schemas.clusterization_spec"] = old_clusterization_spec
sys.modules["mlrun.api.schemas.constants"] = old_constants
sys.modules["mlrun.api.schemas.feature_store"] = old_feature_store
sys.modules["mlrun.api.schemas.frontend_spec"] = old_frontend_spec
sys.modules["mlrun.api.schemas.function"] = old_function
sys.modules["mlrun.api.schemas.http"] = old_http
sys.modules["mlrun.api.schemas.k8s"] = old_k8s
sys.modules["mlrun.api.schemas.memory_reports"] = old_memory_reports
sys.modules["mlrun.api.schemas.object"] = old_object
sys.modules["mlrun.api.schemas.pipeline"] = old_pipeline
sys.modules["mlrun.api.schemas.project"] = old_project
sys.modules["mlrun.api.schemas.runtime_resource"] = old_runtime_resource
sys.modules["mlrun.api.schemas.schedule"] = old_schedule
sys.modules["mlrun.api.schemas.secret"] = old_secret
sys.modules["mlrun.api.schemas.tag"] = old_tag

# The DeprecationHelper class is used to print a deprecation warning when the old import is used,
# and return the new schema. This is done for backwards compatibility with mlrun.api.schemas.
ArtifactCategories = DeprecationHelper(mlrun.common.schemas.ArtifactCategories)
ArtifactIdentifier = DeprecationHelper(mlrun.common.schemas.ArtifactIdentifier)
ArtifactsFormat = DeprecationHelper(mlrun.common.formatters.ArtifactFormat)
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
IndexedHubSource = DeprecationHelper(mlrun.common.schemas.IndexedHubSource)
HubCatalog = DeprecationHelper(mlrun.common.schemas.HubCatalog)
HubItem = DeprecationHelper(mlrun.common.schemas.HubItem)
HubObjectMetadata = DeprecationHelper(mlrun.common.schemas.HubObjectMetadata)
HubSource = DeprecationHelper(mlrun.common.schemas.HubSource)
HubSourceSpec = DeprecationHelper(mlrun.common.schemas.HubSourceSpec)
last_source_index = DeprecationHelper(mlrun.common.schemas.last_source_index)
MostCommonObjectTypesReport = DeprecationHelper(
    mlrun.common.schemas.MostCommonObjectTypesReport
)
ObjectTypeReport = DeprecationHelper(mlrun.common.schemas.ObjectTypeReport)
Features = DeprecationHelper(mlrun.common.schemas.Features)
FeatureValues = DeprecationHelper(mlrun.common.schemas.FeatureValues)
GrafanaColumn = DeprecationHelper(
    mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn
)

GrafanaNumberColumn = DeprecationHelper(
    mlrun.common.schemas.model_monitoring.grafana.GrafanaNumberColumn
)
GrafanaStringColumn = DeprecationHelper(
    mlrun.common.schemas.model_monitoring.grafana.GrafanaStringColumn
)
GrafanaTable = DeprecationHelper(
    mlrun.common.schemas.model_monitoring.grafana.GrafanaTable
)
ModelEndpoint = DeprecationHelper(mlrun.common.schemas.ModelEndpoint)
ModelEndpointList = DeprecationHelper(mlrun.common.schemas.ModelEndpointList)
ModelEndpointMetadata = DeprecationHelper(mlrun.common.schemas.ModelEndpointMetadata)
ModelEndpointSpec = DeprecationHelper(mlrun.common.schemas.ModelEndpointSpec)
ModelEndpointStatus = DeprecationHelper(mlrun.common.schemas.ModelEndpointStatus)
NotificationSeverity = DeprecationHelper(mlrun.common.schemas.NotificationSeverity)
NotificationStatus = DeprecationHelper(mlrun.common.schemas.NotificationStatus)
ObjectKind = DeprecationHelper(mlrun.common.schemas.ObjectKind)
ObjectMetadata = DeprecationHelper(mlrun.common.schemas.ObjectMetadata)
ObjectSpec = DeprecationHelper(mlrun.common.schemas.ObjectSpec)
ObjectStatus = DeprecationHelper(mlrun.common.schemas.ObjectStatus)
PipelinesFormat = DeprecationHelper(mlrun.common.formatters.PipelineFormat)
PipelinesOutput = DeprecationHelper(mlrun.common.schemas.PipelinesOutput)
PipelinesPagination = DeprecationHelper(mlrun.common.schemas.PipelinesPagination)
IguazioProject = DeprecationHelper(mlrun.common.schemas.IguazioProject)
Project = DeprecationHelper(mlrun.common.schemas.Project)
ProjectDesiredState = DeprecationHelper(mlrun.common.schemas.ProjectDesiredState)
ProjectMetadata = DeprecationHelper(mlrun.common.schemas.ProjectMetadata)
ProjectOwner = DeprecationHelper(mlrun.common.schemas.ProjectOwner)
ProjectsFormat = DeprecationHelper(mlrun.common.formatters.ProjectFormat)
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
