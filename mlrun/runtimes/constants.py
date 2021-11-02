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


class MPIJobCRDVersions(object):
    v1 = "v1"
    v1alpha1 = "v1alpha1"

    @staticmethod
    def all():
        return [MPIJobCRDVersions.v1, MPIJobCRDVersions.v1alpha1]

    @staticmethod
    def default():
        return MPIJobCRDVersions.v1alpha1


class RunStates(object):
    completed = "completed"
    error = "error"
    running = "running"
    created = "created"
    pending = "pending"
    unknown = "unknown"
    aborted = "aborted"

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
        ]

    @staticmethod
    def terminal_states():
        return [
            RunStates.completed,
            RunStates.error,
            RunStates.aborted,
        ]

    @staticmethod
    def non_terminal_states():
        return list(set(RunStates.all()) - set(RunStates.terminal_states()))


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
