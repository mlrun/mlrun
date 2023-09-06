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
import typing
from copy import deepcopy

from deprecated import deprecated
from kubernetes import client

import mlrun.runtimes.pod
from mlrun.config import config as mlconf
from mlrun.execution import MLClientCtx
from mlrun.model import RunObject
from mlrun.runtimes.constants import MPIJobCRDVersions
from mlrun.runtimes.mpijob.abstract import AbstractMPIJobRuntime
from mlrun.utils import get_in, update_in


# TODO: Remove in 1.7.0
@deprecated(
    version="1.5.0",
    reason="v1alpha1 mpi will be removed in 1.7.0, use v1 instead",
    category=FutureWarning,
)
class MpiRuntimeV1Alpha1(AbstractMPIJobRuntime):
    _mpijob_template = {
        "apiVersion": "kubeflow.org/v1alpha1",
        "kind": "MPIJob",
        "metadata": {"name": "", "namespace": "default-tenant"},
        "spec": {
            "replicas": 1,
            "template": {
                "metadata": {},
                "spec": {
                    "containers": [
                        {
                            "image": "mlrun/mlrun",
                            "name": "base",
                            "command": [],
                            "env": [],
                            "volumeMounts": [],
                            "securityContext": {"capabilities": {"add": ["IPC_LOCK"]}},
                            "resources": {"limits": {}},
                        }
                    ],
                    "volumes": [],
                },
            },
        },
    }

    crd_group = "kubeflow.org"
    crd_version = MPIJobCRDVersions.v1alpha1
    crd_plural = "mpijobs"

    def _update_container(self, struct, key, value):
        struct["spec"]["template"]["spec"]["containers"][0][key] = value

    def _generate_mpi_job(
        self, runobj: RunObject, execution: MLClientCtx, meta: client.V1ObjectMeta
    ) -> typing.Dict:
        job = deepcopy(self._mpijob_template)

        pod_labels = deepcopy(meta.labels)
        pod_labels["mlrun/job"] = meta.name
        update_in(job, "metadata", meta.to_dict())
        update_in(job, "spec.template.metadata.labels", pod_labels)
        update_in(job, "spec.replicas", self.spec.replicas or 1)
        if self.spec.image:
            self._update_container(
                job,
                "image",
                self.full_image_path(
                    client_version=runobj.metadata.labels.get("mlrun/client_version"),
                    client_python_version=runobj.metadata.labels.get(
                        "mlrun/client_python_version"
                    ),
                ),
            )
        update_in(job, "spec.template.spec.volumes", self.spec.volumes)
        self._update_container(job, "volumeMounts", self.spec.volume_mounts)
        update_in(job, "spec.template.spec.nodeName", self.spec.node_name)
        update_in(job, "spec.template.spec.nodeSelector", self.spec.node_selector)
        update_in(
            job,
            "spec.template.spec.affinity",
            mlrun.runtimes.pod.get_sanitized_attribute(self.spec, "affinity"),
        )
        update_in(
            job,
            "spec.template.spec.tolerations",
            mlrun.runtimes.pod.get_sanitized_attribute(self.spec, "tolerations"),
        )
        update_in(
            job,
            "spec.template.spec.securityContext",
            mlrun.runtimes.pod.get_sanitized_attribute(self.spec, "security_context"),
        )
        if self.spec.priority_class_name and len(
            mlconf.get_valid_function_priority_class_names()
        ):
            update_in(
                job,
                "spec.template.spec.priorityClassName",
                self.spec.priority_class_name,
            )

        extra_env = self.generate_runtime_k8s_env(runobj)
        self._update_container(job, "env", extra_env + self.spec.env)
        if self.spec.image_pull_policy:
            self._update_container(job, "imagePullPolicy", self.spec.image_pull_policy)
        if self.spec.resources:
            self._update_container(job, "resources", self.spec.resources)
        if self.spec.workdir:
            self._update_container(job, "workingDir", self.spec.workdir)

        if self.spec.image_pull_secret:
            update_in(
                job,
                "spec.template.spec.imagePullSecrets",
                [{"name": self.spec.image_pull_secret}],
            )

        if self.spec.command:
            self._update_container(
                job, "command", ["mpirun", "python", self.spec.command] + self.spec.args
            )

        return job

    def _get_job_launcher_status(self, resp: typing.List) -> str:
        return get_in(resp, "status.launcherStatus")

    @staticmethod
    def _generate_pods_selector(name: str, launcher: bool) -> str:
        selector = "mlrun/class=mpijob"
        if name:
            selector += f",mpi_job_name={name}"
        if launcher:
            selector += ",mpi_role_type=launcher"

        return selector

    @staticmethod
    def _get_crd_info() -> typing.Tuple[str, str, str]:
        return (
            MpiRuntimeV1Alpha1.crd_group,
            MpiRuntimeV1Alpha1.crd_version,
            MpiRuntimeV1Alpha1.crd_plural,
        )
