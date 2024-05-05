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
from mlrun.common.runtimes.constants import MPIJobCRDVersions, MPIJobV1CleanPodPolicies
from mlrun.runtimes.mpijob.abstract import AbstractMPIJobRuntime, MPIResourceSpec


class MPIV1ResourceSpec(MPIResourceSpec):
    _dict_fields = MPIResourceSpec._dict_fields + ["clean_pod_policy"]

    def __init__(
        self,
        command=None,
        args=None,
        image=None,
        mode=None,
        volumes=None,
        volume_mounts=None,
        env=None,
        resources=None,
        default_handler=None,
        entry_points=None,
        description=None,
        workdir=None,
        replicas=None,
        image_pull_policy=None,
        service_account=None,
        build=None,
        image_pull_secret=None,
        mpi_args=None,
        clean_pod_policy=None,
        node_name=None,
        node_selector=None,
        affinity=None,
        priority_class_name=None,
        disable_auto_mount=False,
        pythonpath=None,
        tolerations=None,
        preemption_mode=None,
        security_context=None,
        clone_target_dir=None,
        state_thresholds=None,
    ):
        super().__init__(
            command=command,
            image=image,
            mode=mode,
            build=build,
            entry_points=entry_points,
            description=description,
            workdir=workdir,
            default_handler=default_handler,
            volumes=volumes,
            volume_mounts=volume_mounts,
            env=env,
            resources=resources,
            replicas=replicas,
            image_pull_policy=image_pull_policy,
            service_account=service_account,
            image_pull_secret=image_pull_secret,
            args=args,
            mpi_args=mpi_args,
            node_name=node_name,
            node_selector=node_selector,
            affinity=affinity,
            priority_class_name=priority_class_name,
            disable_auto_mount=disable_auto_mount,
            pythonpath=pythonpath,
            tolerations=tolerations,
            preemption_mode=preemption_mode,
            security_context=security_context,
            clone_target_dir=clone_target_dir,
            state_thresholds=state_thresholds,
        )
        self.clean_pod_policy = clean_pod_policy or MPIJobV1CleanPodPolicies.default()


class MpiRuntimeV1(AbstractMPIJobRuntime):
    crd_group = "kubeflow.org"
    crd_version = MPIJobCRDVersions.v1
    crd_plural = "mpijobs"

    @property
    def spec(self) -> MPIV1ResourceSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", MPIV1ResourceSpec)
