class PodPhases:
    succeeded = 'Succeeded'
    failed = 'Failed'

    @staticmethod
    def stable_phases():
        return [
            PodPhases.succeeded,
            PodPhases.failed,
        ]


class MPIJobCRDVersions(object):
    v1 = 'v1'
    v1alpha1 = 'v1alpha1'

    @staticmethod
    def all():
        return [MPIJobCRDVersions.v1, MPIJobCRDVersions.v1alpha1]

    @staticmethod
    def default():
        return MPIJobCRDVersions.v1alpha1


class FunctionStates(object):
    completed = 'completed'
    error = 'error'

    @staticmethod
    def stable_phases():
        return [
            FunctionStates.completed,
            FunctionStates.error,
        ]
