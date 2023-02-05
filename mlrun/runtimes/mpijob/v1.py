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
import typing
from copy import deepcopy
from datetime import datetime

from kubernetes import client
from sqlalchemy.orm import Session

import mlrun.runtimes.pod
from mlrun.api.db.base import DBInterface
from mlrun.config import config as mlconf
from mlrun.execution import MLClientCtx
from mlrun.model import RunObject
from mlrun.runtimes.base import BaseRuntimeHandler, RunStates, RuntimeClassMode
from mlrun.runtimes.constants import MPIJobCRDVersions, MPIJobV1CleanPodPolicies
from mlrun.runtimes.mpijob.abstract import AbstractMPIJobRuntime, MPIResourceSpec
from mlrun.utils import get_in, update_in


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
        )
        self.clean_pod_policy = clean_pod_policy or MPIJobV1CleanPodPolicies.default()


class MpiRuntimeV1(AbstractMPIJobRuntime):
    _mpijob_pod_template = {
        "spec": {
            "containers": [
                {
                    "image": "mlrun/ml-models",
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
        "metadata": {},
    }

    crd_group = "kubeflow.org"
    crd_version = MPIJobCRDVersions.v1
    crd_plural = "mpijobs"

    @property
    def spec(self) -> MPIV1ResourceSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", MPIV1ResourceSpec)

    def _generate_mpi_job_template(self, launcher_pod_template, worker_pod_template):
        # https://github.com/kubeflow/mpi-operator/blob/master/pkg/apis/kubeflow/v1/types.go#L25
        # MPI job consists of Launcher and Worker which both are of type ReplicaSet
        # https://github.com/kubeflow/common/blob/master/pkg/apis/common/v1/types.go#L74
        return {
            "apiVersion": "kubeflow.org/v1",
            "kind": "MPIJob",
            "metadata": {"name": "", "namespace": "default-tenant"},
            "spec": {
                "slotsPerWorker": 1,
                "cleanPodPolicy": "All",
                "mpiReplicaSpecs": {
                    "Launcher": {"template": launcher_pod_template},
                    "Worker": {"replicas": 1, "template": worker_pod_template},
                },
            },
        }

    def _update_container(self, struct, key, value):
        struct["spec"]["containers"][0][key] = value

    def _enrich_launcher_configurations(self, launcher_pod_template, args):
        quoted_args = args or []
        quoted_mpi_args = []
        for arg in self.spec.mpi_args:
            quoted_mpi_args.append(arg)
        self._update_container(
            launcher_pod_template,
            "command",
            ["mpirun", *quoted_mpi_args, *quoted_args],
        )
        self._update_container(
            launcher_pod_template,
            "resources",
            mlconf.get_default_function_pod_resources(),
        )

    def _enrich_worker_configurations(self, worker_pod_template):
        if self.spec.resources:
            self._update_container(
                worker_pod_template, "resources", self.spec.resources
            )

    def _generate_mpi_job(
        self,
        runobj: RunObject,
        execution: MLClientCtx,
        meta: client.V1ObjectMeta,
    ) -> dict:
        pod_labels = deepcopy(meta.labels)
        pod_labels["mlrun/job"] = meta.name

        # Populate mpijob object

        # start by populating pod templates
        launcher_pod_template = deepcopy(self._mpijob_pod_template)
        worker_pod_template = deepcopy(self._mpijob_pod_template)
        command, args, extra_env = self._get_cmd_args(runobj)

        # configuration for both launcher and workers
        for pod_template in [launcher_pod_template, worker_pod_template]:
            if self.spec.image:
                self._update_container(
                    pod_template,
                    "image",
                    self.full_image_path(
                        client_version=runobj.metadata.labels.get(
                            "mlrun/client_version"
                        ),
                        client_python_version=runobj.metadata.labels.get(
                            "mlrun/client_python_version"
                        ),
                    ),
                )
            self._update_container(
                pod_template, "volumeMounts", self.spec.volume_mounts
            )
            self._update_container(pod_template, "env", extra_env + self.spec.env)
            if self.spec.image_pull_policy:
                self._update_container(
                    pod_template,
                    "imagePullPolicy",
                    self.spec.image_pull_policy,
                )
            if self.spec.workdir:
                self._update_container(pod_template, "workingDir", self.spec.workdir)
            if self.spec.image_pull_secret:
                update_in(
                    pod_template,
                    "spec.imagePullSecrets",
                    [{"name": self.spec.image_pull_secret}],
                )
            if self.spec.security_context:
                update_in(
                    pod_template,
                    "spec.securityContext",
                    mlrun.runtimes.pod.get_sanitized_attribute(
                        self.spec, "security_context"
                    ),
                )
            update_in(pod_template, "metadata.labels", pod_labels)
            update_in(pod_template, "spec.volumes", self.spec.volumes)
            update_in(pod_template, "spec.nodeName", self.spec.node_name)
            update_in(pod_template, "spec.nodeSelector", self.spec.node_selector)
            update_in(
                pod_template,
                "spec.affinity",
                mlrun.runtimes.pod.get_sanitized_attribute(self.spec, "affinity"),
            )
            update_in(
                pod_template,
                "spec.tolerations",
                mlrun.runtimes.pod.get_sanitized_attribute(self.spec, "tolerations"),
            )
            if self.spec.priority_class_name and len(
                mlconf.get_valid_function_priority_class_names()
            ):
                update_in(
                    pod_template,
                    "spec.priorityClassName",
                    self.spec.priority_class_name,
                )
            if self.spec.service_account:
                update_in(
                    pod_template, "spec.serviceAccountName", self.spec.service_account
                )

        # configuration for workers only
        # update resources only for workers because the launcher
        # doesn't require special resources (like GPUs, Memory, etc..)
        self._enrich_worker_configurations(worker_pod_template)

        # configuration for launcher only
        self._enrich_launcher_configurations(launcher_pod_template, [command] + args)

        # generate mpi job using both pod templates
        job = self._generate_mpi_job_template(
            launcher_pod_template, worker_pod_template
        )

        # update the replicas only for workers
        update_in(
            job,
            "spec.mpiReplicaSpecs.Worker.replicas",
            self.spec.replicas or 1,
        )

        update_in(
            job,
            "spec.cleanPodPolicy",
            self.spec.clean_pod_policy,
        )

        if execution.get_param("slots_per_worker"):
            update_in(
                job,
                "spec.slotsPerWorker",
                execution.get_param("slots_per_worker"),
            )

        update_in(job, "metadata", meta.to_dict())

        return job

    def _get_job_launcher_status(self, resp: typing.List) -> str:
        launcher_status = get_in(resp, "status.replicaStatuses.Launcher")
        if launcher_status is None:
            return ""

        for status in ["active", "failed", "succeeded"]:
            if launcher_status.get(status, 0) == 1:
                return status

        return ""

    @staticmethod
    def _generate_pods_selector(name: str, launcher: bool) -> str:
        selector = f"mpi-job-name={name}"
        if launcher:
            selector += ",mpi-job-role=launcher"

        return selector

    @staticmethod
    def _get_crd_info() -> typing.Tuple[str, str, str]:
        return (
            MpiRuntimeV1.crd_group,
            MpiRuntimeV1.crd_version,
            MpiRuntimeV1.crd_plural,
        )


class MpiV1RuntimeHandler(BaseRuntimeHandler):
    kind = "mpijob"
    class_modes = {
        RuntimeClassMode.run: "mpijob",
    }

    def _resolve_crd_object_status_info(
        self, db: DBInterface, db_session: Session, crd_object
    ) -> typing.Tuple[bool, typing.Optional[datetime], typing.Optional[str]]:
        """
        https://github.com/kubeflow/mpi-operator/blob/master/pkg/apis/kubeflow/v1/types.go#L29
        https://github.com/kubeflow/common/blob/master/pkg/apis/common/v1/types.go#L55
        """
        launcher_status = (
            crd_object.get("status", {}).get("replicaStatuses", {}).get("Launcher", {})
        )
        # the launcher status also has running property, but it's empty for
        # short period after the creation, so we're
        # checking terminal state by the completion time existence
        in_terminal_state = (
            crd_object.get("status", {}).get("completionTime", None) is not None
        )
        desired_run_state = RunStates.running
        completion_time = None
        if in_terminal_state:
            completion_time = datetime.fromisoformat(
                crd_object.get("status", {})
                .get("completionTime")
                .replace("Z", "+00:00")
            )
            desired_run_state = (
                RunStates.error
                if launcher_status.get("failed", 0) > 0
                else RunStates.completed
            )
        return in_terminal_state, completion_time, desired_run_state

    @staticmethod
    def _are_resources_coupled_to_run_object() -> bool:
        return True

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f"mlrun/uid={object_id}"

    @staticmethod
    def _get_run_completion_updates(run: dict) -> dict:

        # TODO: add a 'workers' section in run objects state, each worker will update its state while
        #  the run state will be resolved by the server.
        # update the run object state if empty so that it won't default to 'created' state
        update_in(run, "status.state", "running", append=False, replace=False)
        return {}

    @staticmethod
    def _get_crd_info() -> typing.Tuple[str, str, str]:
        return (
            MpiRuntimeV1.crd_group,
            MpiRuntimeV1.crd_version,
            MpiRuntimeV1.crd_plural,
        )
