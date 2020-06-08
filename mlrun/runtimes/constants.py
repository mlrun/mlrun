class PodPhases:
    succeeded = 'Succeeded'
    failed = 'Failed'

    @staticmethod
    def stable_phases():
        return [
            PodPhases.succeeded,
            PodPhases.failed,
        ]