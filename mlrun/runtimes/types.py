class MPIJobCRDVersions(object):
    v1 = 'v1'
    v1alpha1 = 'v1alpha1'

    @staticmethod
    def all():
        return [MPIJobCRDVersions.v1,
                MPIJobCRDVersions.v1alpha1]

    @staticmethod
    def default():
        return MPIJobCRDVersions.v1alpha1
