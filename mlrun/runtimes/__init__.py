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

runtime_resources_map = {
    'dask': get_dask_resource()
}

supported_runtimes = ['remote', 'nuclio', 'dask', 'job', 'spark', 'mpijob']


def get_runtime(kind: str):
    if kind == 'mpijob':
        return infer_mpi_runtime()

    runtime_dict = {'remote': RemoteRuntime,
                    'nuclio': RemoteRuntime,
                    'dask': DaskCluster,
                    'job': KubejobRuntime,
                    'spark': SparkRuntime}


    return runtime_dict[kind]


# infer mpijob runtime according to the mpi-operator's supported crd-version
def infer_mpi_runtime():
    k8s_helper = get_k8s_helper()
    namespace = k8s_helper.ns()

    # default to v1alpha1 for backwards compatibility
    mpi_job_crd_version = 'v1alpha1'

    # get mpi-operator pod
    res = k8s_helper.list_pods(namespace=namespace, selector='release=mpi-operator')
    if len(res) > 0:
        mpi_operator_pod = res[0]
        mpi_job_crd_version = mpi_operator_pod.metadata.labels.get('crd-version', mpi_job_crd_version)

    crd_version_to_runtime = {
        'v1alpha1': MpiRuntimeV1Alpha1,
        'v1': MpiRuntimeV1
    }
    return crd_version_to_runtime[mpi_job_crd_version]
