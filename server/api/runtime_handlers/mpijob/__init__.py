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
import mlrun.common.constants as mlrun_constants
import mlrun.k8s_utils
import mlrun.utils.helpers
import server.api.utils.singletons.k8s
from mlrun.common.runtimes.constants import MPIJobCRDVersions
from mlrun.config import config
from server.api.runtime_handlers.mpijob.v1 import MpiV1RuntimeHandler

cached_mpijob_crd_version = None


def resolve_mpijob_crd_version():
    """
    Resolve mpijob runtime according to the mpi-operator's supported crd-version.
    If specified on mlrun config set it likewise.
    If not specified, try resolving it according to the mpi-operator, otherwise set to default.
    Since this is a heavy operation (sending requests to k8s/API), and it's unlikely that the crd version
    will change in any context - cache it.
    :return: mpi operator's crd-version
    """
    global cached_mpijob_crd_version
    if not cached_mpijob_crd_version:
        # try to resolve the crd version with K8s API
        # backoff to use default if needed
        mpijob_crd_version = (
            _resolve_mpijob_crd_version_best_effort() or MPIJobCRDVersions.default()
        )

        if mpijob_crd_version not in MPIJobCRDVersions.all():
            raise ValueError(
                f"Unsupported mpijob crd version: {mpijob_crd_version}. "
                f"Supported versions: {MPIJobCRDVersions.all()}"
            )
        cached_mpijob_crd_version = mpijob_crd_version

    return cached_mpijob_crd_version


def _resolve_mpijob_crd_version_best_effort():
    # config overrides everything
    if config.mpijob_crd_version:
        return config.mpijob_crd_version

    if not mlrun.k8s_utils.is_running_inside_kubernetes_cluster():
        return None

    k8s_helper = server.api.utils.singletons.k8s.get_k8s_helper()
    namespace = k8s_helper.resolve_namespace()

    # try resolving according to mpi-operator that's running
    res = k8s_helper.list_pods(
        namespace=namespace,
        selector=f"{mlrun_constants.MLRunInternalLabels.component}=mpi-operator",
    )

    if len(res) == 0:
        return None

    mpi_operator_pod = res[0]
    return mpi_operator_pod.metadata.labels.get("crd-version")
