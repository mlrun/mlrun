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
import enum
import typing

import mlrun.common.constants as mlrun_constants
import mlrun_pipelines.common.models


class PodPhases:
    """
    https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-phase
    """

    succeeded = "Succeeded"
    failed = "Failed"
    pending = "Pending"
    running = "Running"
    unknown = "Unknown"

    @staticmethod
    def terminal_phases():
        return [
            PodPhases.succeeded,
            PodPhases.failed,
        ]

    @staticmethod
    def all():
        return [
            PodPhases.succeeded,
            PodPhases.failed,
            PodPhases.pending,
            PodPhases.running,
            PodPhases.unknown,
        ]

    @staticmethod
    def pod_phase_to_run_state(pod_phase):
        if pod_phase not in PodPhases.all():
            raise ValueError(f"Invalid pod phase: {pod_phase}")
        return {
            PodPhases.succeeded: RunStates.completed,
            PodPhases.failed: RunStates.error,
            PodPhases.pending: RunStates.pending,
            PodPhases.running: RunStates.running,
            PodPhases.unknown: RunStates.unknown,
        }[pod_phase]


class ThresholdStates:
    # A pod can be in pending and scheduled state when the pod's container images are not already available on the node
    # where it is scheduled, or initialization tasks specified in the pod's configuration are not yet completed.
    pending_scheduled = "pending_scheduled"
    pending_not_scheduled = "pending_not_scheduled"
    executing = "executing"
    image_pull_backoff = "image_pull_backoff"

    @staticmethod
    def all():
        return [
            ThresholdStates.pending_scheduled,
            ThresholdStates.pending_not_scheduled,
            ThresholdStates.executing,
            ThresholdStates.image_pull_backoff,
        ]

    @staticmethod
    def from_pod_phase(pod_phase: str, pod: dict) -> typing.Optional[str]:
        if pod_phase == PodPhases.pending:
            if ThresholdStates.is_pod_in_image_pull_backoff(pod):
                return ThresholdStates.image_pull_backoff
            elif ThresholdStates.is_pod_scheduled(pod):
                return ThresholdStates.pending_scheduled
            else:
                return ThresholdStates.pending_not_scheduled

        elif pod_phase == PodPhases.running:
            return ThresholdStates.executing

        return None

    @staticmethod
    def is_pod_scheduled(pod: dict):
        conditions = pod["status"].get("conditions", []) or []
        for condition in conditions:
            if condition["type"] == "PodScheduled" and condition["status"] == "True":
                return True
        return False

    @staticmethod
    def is_pod_in_image_pull_backoff(pod: dict):
        container_statuses = pod.get("status").get("container_statuses", []) or []
        for container_status in container_statuses:
            state_waiting = container_status.get("state", {}).get("waiting", {}) or {}
            if state_waiting.get("reason", "") == "ImagePullBackOff":
                return True
        return False


class MPIJobCRDVersions:
    v1 = "v1"
    v1alpha1 = "v1alpha1"

    @staticmethod
    def all():
        return [MPIJobCRDVersions.v1, MPIJobCRDVersions.v1alpha1]

    @staticmethod
    def default():
        return MPIJobCRDVersions.v1alpha1

    @staticmethod
    def role_label_by_version(version):
        return {
            MPIJobCRDVersions.v1alpha1: mlrun_constants.MLRunInternalLabels.mpi_role_type,
            MPIJobCRDVersions.v1: mlrun_constants.MLRunInternalLabels.mpi_job_role,
        }[version]


class RunStates:
    completed = "completed"
    error = "error"
    running = "running"
    created = "created"
    pending = "pending"
    unknown = "unknown"
    aborted = "aborted"
    aborting = "aborting"
    skipped = "skipped"

    @staticmethod
    def all():
        return [
            RunStates.completed,
            RunStates.error,
            RunStates.running,
            RunStates.created,
            RunStates.pending,
            RunStates.unknown,
            RunStates.aborted,
            RunStates.aborting,
            RunStates.skipped,
        ]

    @staticmethod
    def terminal_states():
        return [
            RunStates.completed,
            RunStates.error,
            RunStates.aborted,
            RunStates.skipped,
        ]

    @staticmethod
    def error_states():
        return [
            RunStates.error,
            RunStates.aborted,
        ]

    @staticmethod
    def abortion_states():
        return [
            RunStates.aborted,
            RunStates.aborting,
        ]

    @staticmethod
    def error_and_abortion_states():
        return list(set(RunStates.error_states()) | set(RunStates.abortion_states()))

    @staticmethod
    def non_terminal_states():
        return list(set(RunStates.all()) - set(RunStates.terminal_states()))

    @staticmethod
    def not_allowed_for_deletion_states():
        return [
            RunStates.running,
            RunStates.pending,
            # TODO: add aborting state once we have it
        ]

    @staticmethod
    def notification_states():
        return RunStates.terminal_states() + [RunStates.running]

    @staticmethod
    def run_state_to_pipeline_run_status(run_state: str):
        if not run_state:
            return mlrun_pipelines.common.models.RunStatuses.runtime_state_unspecified

        if run_state not in RunStates.all():
            raise ValueError(f"Invalid run state: {run_state}")

        return {
            RunStates.completed: mlrun_pipelines.common.models.RunStatuses.succeeded,
            RunStates.error: mlrun_pipelines.common.models.RunStatuses.failed,
            RunStates.running: mlrun_pipelines.common.models.RunStatuses.running,
            RunStates.created: mlrun_pipelines.common.models.RunStatuses.pending,
            RunStates.pending: mlrun_pipelines.common.models.RunStatuses.pending,
            RunStates.unknown: mlrun_pipelines.common.models.RunStatuses.runtime_state_unspecified,
            RunStates.aborted: mlrun_pipelines.common.models.RunStatuses.canceled,
            RunStates.aborting: mlrun_pipelines.common.models.RunStatuses.canceling,
            RunStates.skipped: mlrun_pipelines.common.models.RunStatuses.skipped,
        }[run_state]

    @staticmethod
    def pipeline_run_status_to_run_state(pipeline_run_status):
        if pipeline_run_status not in mlrun_pipelines.common.models.RunStatuses.all():
            raise ValueError(f"Invalid pipeline run status: {pipeline_run_status}")
        return {
            mlrun_pipelines.common.models.RunStatuses.succeeded: RunStates.completed,
            mlrun_pipelines.common.models.RunStatuses.failed: RunStates.error,
            mlrun_pipelines.common.models.RunStatuses.running: RunStates.running,
            mlrun_pipelines.common.models.RunStatuses.pending: RunStates.pending,
            mlrun_pipelines.common.models.RunStatuses.canceled: RunStates.aborted,
            mlrun_pipelines.common.models.RunStatuses.canceling: RunStates.aborting,
            mlrun_pipelines.common.models.RunStatuses.skipped: RunStates.skipped,
            mlrun_pipelines.common.models.RunStatuses.runtime_state_unspecified: RunStates.unknown,
            mlrun_pipelines.common.models.RunStatuses.error: RunStates.error,
            mlrun_pipelines.common.models.RunStatuses.paused: RunStates.unknown,
            mlrun_pipelines.common.models.RunStatuses.unknown: RunStates.unknown,
        }[pipeline_run_status]


# TODO: remove this class in 1.10.0 - use only MlrunInternalLabels
class RunLabels(enum.Enum):
    owner = mlrun_constants.MLRunInternalLabels.owner
    v3io_user = mlrun_constants.MLRunInternalLabels.v3io_user

    @staticmethod
    def all():
        return [
            RunLabels.owner,
            RunLabels.v3io_user,
        ]


class SparkApplicationStates:
    """
    https://github.com/GoogleCloudPlatform/spark-on-k8s-operator/blob/master/pkg/apis/sparkoperator.k8s.io/v1beta2/types.go#L321
    """

    completed = "COMPLETED"
    failed = "FAILED"
    submitted = "SUBMITTED"
    running = "RUNNING"
    submission_failed = "SUBMISSION_FAILED"
    pending_rerun = "PENDING_RERUN"
    pending_submission = "PENDING_SUBMISSION"
    invalidating = "INVALIDATING"
    succeeding = "SUCCEEDING"
    failing = "FAILING"
    unknown = "UNKNOWN"

    @staticmethod
    def terminal_states():
        return [
            SparkApplicationStates.completed,
            SparkApplicationStates.failed,
        ]

    @staticmethod
    def all():
        return [
            SparkApplicationStates.completed,
            SparkApplicationStates.failed,
            SparkApplicationStates.submitted,
            SparkApplicationStates.running,
            SparkApplicationStates.submission_failed,
            SparkApplicationStates.pending_rerun,
            SparkApplicationStates.pending_submission,
            SparkApplicationStates.invalidating,
            SparkApplicationStates.succeeding,
            SparkApplicationStates.failing,
            SparkApplicationStates.unknown,
        ]

    @staticmethod
    def spark_application_state_to_run_state(spark_application_state):
        if spark_application_state not in SparkApplicationStates.all():
            raise ValueError(
                f"Invalid spark application state: {spark_application_state}"
            )
        return {
            SparkApplicationStates.completed: RunStates.completed,
            SparkApplicationStates.failed: RunStates.error,
            SparkApplicationStates.submitted: RunStates.running,
            SparkApplicationStates.running: RunStates.running,
            SparkApplicationStates.submission_failed: RunStates.running,
            SparkApplicationStates.pending_rerun: RunStates.running,
            SparkApplicationStates.pending_submission: RunStates.running,
            SparkApplicationStates.invalidating: RunStates.running,
            SparkApplicationStates.succeeding: RunStates.running,
            SparkApplicationStates.failing: RunStates.running,
            SparkApplicationStates.unknown: RunStates.unknown,
        }[spark_application_state]


class MPIJobV1Alpha1States:
    """
    https://github.com/kubeflow/mpi-operator/blob/master/pkg/apis/kubeflow/v1alpha1/types.go#L105
    """

    succeeded = "Succeeded"
    failed = "Failed"
    active = "Active"

    @staticmethod
    def terminal_states():
        return [
            MPIJobV1Alpha1States.succeeded,
            MPIJobV1Alpha1States.failed,
        ]

    @staticmethod
    def all():
        return [
            MPIJobV1Alpha1States.succeeded,
            MPIJobV1Alpha1States.failed,
            MPIJobV1Alpha1States.active,
        ]

    @staticmethod
    def mpijob_state_to_run_state(mpijob_state):
        if mpijob_state not in MPIJobV1Alpha1States.all():
            raise ValueError(f"Invalid MPI job state: {mpijob_state}")
        return {
            MPIJobV1Alpha1States.succeeded: RunStates.completed,
            SparkApplicationStates.failed: RunStates.error,
            MPIJobV1Alpha1States.active: RunStates.running,
        }[mpijob_state]


class MPIJobV1CleanPodPolicies:
    """
    https://github.com/kubeflow/common/blob/master/pkg/apis/common/v1/types.go#L137
    """

    all = "All"
    running = "Running"
    none = "None"
    undefined = ""

    @staticmethod
    def default():
        return MPIJobV1CleanPodPolicies.all


class NuclioIngressAddTemplatedIngressModes:
    always = "always"
    never = "never"
    on_cluster_ip = "onClusterIP"


class FunctionEnvironmentVariables:
    _env_prefix = "MLRUN_"
    auth_session = f"{_env_prefix}AUTH_SESSION"
