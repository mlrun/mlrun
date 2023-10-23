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
from deprecated import deprecated

from mlrun.runtimes.constants import MPIJobCRDVersions
from mlrun.runtimes.mpijob.abstract import AbstractMPIJobRuntime


# TODO: Remove in 1.7.0
@deprecated(
    version="1.5.0",
    reason="v1alpha1 mpi will be removed in 1.7.0, use v1 instead",
    category=FutureWarning,
)
class MpiRuntimeV1Alpha1(AbstractMPIJobRuntime):
    crd_group = "kubeflow.org"
    crd_version = MPIJobCRDVersions.v1alpha1
    crd_plural = "mpijobs"
