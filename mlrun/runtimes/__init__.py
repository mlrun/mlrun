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

from mlrun.config import config
from .base import RunError, BaseRuntime  # noqa
from .local import HandlerRuntime, LocalRuntime  # noqa
from .function import RemoteRuntime, new_model_server  # noqa
from .mpijob import MpiRuntimeV1Alpha1  # noqa
from .mpijob import MpiRuntimeV1  # noqa
from .daskjob import DaskCluster, get_dask_resource  # noqa
from .kubejob import KubejobRuntime  # noqa
from .sparkjob import SparkRuntime  # noqa
from .nuclio import nuclio_init_hook
from .serving import MLModelServer
from mlrun.k8s_utils import get_k8s_helper


class RuntimeKinds(object):
    remote = 'remote'
    nuclio = 'nuclio'
    dask = 'dask'
    job = 'job'
    spark = 'spark'
    mpijob = 'mpijob'

    @staticmethod
    def all():
        return [RuntimeKinds.remote,
                RuntimeKinds.nuclio,
                RuntimeKinds.dask,
                RuntimeKinds.job,
                RuntimeKinds.spark,
                RuntimeKinds.mpijob]


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


runtime_resources_map = {
    RuntimeKinds.dask: get_dask_resource()
}


def get_runtime_class(kind: str):
    if kind == RuntimeKinds.mpijob:
        return _resolve_mpi_runtime()

    kind_runtime_map = {
        RuntimeKinds.remote: RemoteRuntime,
        RuntimeKinds.nuclio: RemoteRuntime,
        RuntimeKinds.dask: DaskCluster,
        RuntimeKinds.job: KubejobRuntime,
        RuntimeKinds.spark: SparkRuntime
    }

    return kind_runtime_map[kind]


# resolve mpijob runtime according to the mpi-operator's supported crd-version
# if specified on mlrun config set it likewise,
# if not specified, try resolving it according to the mpi-operator, otherwise set to default
def _resolve_mpi_runtime():

    # try getting mpijob crd version from config
    mpijob_crd_version = config.mpijob_crd_version

    if not mpijob_crd_version:
        k8s_helper = get_k8s_helper()
        namespace = k8s_helper.ns()

        # set default mpijob crd version
        mpijob_crd_version = MPIJobCRDVersions.default()

        # try resolving according to mpi-operator that's running
        res = k8s_helper.list_pods(namespace=namespace, selector='component=mpi-operator')
        if len(res) > 0:
            mpi_operator_pod = res[0]
            mpijob_crd_version = mpi_operator_pod.metadata.labels.get('crd-version', mpijob_crd_version)

    if mpijob_crd_version not in MPIJobCRDVersions.all():
        raise ValueError('unsupported mpijob crd version: {}. supported versions: {}'.format(mpijob_crd_version,
                                                                                             MPIJobCRDVersions.all()))

    crd_version_to_runtime = {
        MPIJobCRDVersions.v1alpha1: MpiRuntimeV1Alpha1,
        MPIJobCRDVersions.v1: MpiRuntimeV1
    }
    return crd_version_to_runtime[mpijob_crd_version]
