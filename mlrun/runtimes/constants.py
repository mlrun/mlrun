class PodPhases:
    succeeded = "Succeeded"
    failed = "Failed"

    @staticmethod
    def stable_phases():
        return [
            PodPhases.succeeded,
            PodPhases.failed,
        ]

    @staticmethod
    def all():
        return [PodPhases.succeeded, PodPhases.failed]

    @staticmethod
    def pod_phase_to_run_state(pod_phase):
        if pod_phase not in PodPhases.all():
            raise ValueError(f"Invalid pod phase: {pod_phase}")
        return {
            PodPhases.succeeded: RunStates.completed,
            PodPhases.failed: RunStates.error,
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

    @staticmethod
    def all():
        return [RunStates.completed, RunStates.error]

    @staticmethod
    def stable_states():
        return [
            RunStates.completed,
            RunStates.error,
        ]
