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
from typing import Dict, Optional, Tuple

from kubernetes import client
from kubernetes.client.rest import ApiException
from sqlalchemy.orm import Session

from mlrun.api.db.base import DBInterface
from mlrun.config import config
from mlrun.db import get_run_db
from mlrun.runtimes.base import BaseRuntimeHandler
from mlrun.runtimes.constants import RunStates, SparkApplicationStates

from ..execution import MLClientCtx
from ..model import RunObject
from ..platforms.iguazio import mount_v3io_extended, mount_v3iod
from ..utils import get_in, logger, update_in
from .base import RunError
from .kubejob import KubejobRuntime
from .pod import KubeResourceSpec
from .utils import generate_resources

igz_deps = {
    "jars": [
        "/spark/v3io-libs/v3io-hcfs_2.11.jar",
        "/spark/v3io-libs/v3io-spark2-streaming_2.11.jar",
        "/spark/v3io-libs/v3io-spark2-object-dataframe_2.11.jar",
        "/igz/java/libs/scala-library-2.11.12.jar",
    ],
    "files": ["/igz/java/libs/v3io-pyspark.zip"],
}

allowed_types = ["Python", "Scala", "Java", "R"]

_sparkjob_template = {
    "apiVersion": "sparkoperator.k8s.io/v1beta2",
    "kind": "SparkApplication",
    "metadata": {"name": "", "namespace": "default-tenant"},
    "spec": {
        "mode": "cluster",
        "image": "",
        "imagePullPolicy": "IfNotPresent",
        "mainApplicationFile": "",
        "sparkVersion": "2.4.5",
        "restartPolicy": {
            "type": "OnFailure",
            "onFailureRetries": 3,
            "onFailureRetryInterval": 10,
            "onSubmissionFailureRetries": 5,
            "onSubmissionFailureRetryInterval": 20,
        },
        "deps": {},
        "volumes": [],
        "serviceAccount": "sparkapp",
        "driver": {
            "cores": 1,
            "coreLimit": "1200m",
            "memory": "512m",
            "labels": {},
            "volumeMounts": [],
            "env": [],
        },
        "executor": {
            "cores": 0,
            "instances": 0,
            "memory": "",
            "labels": {},
            "volumeMounts": [],
            "env": [],
        },
        "nodeSelector": {},
    },
}


class SparkJobSpec(KubeResourceSpec):
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
        replicas=None,
        image_pull_policy=None,
        service_account=None,
        image_pull_secret=None,
        driver_resources=None,
        executor_resources=None,
        job_type=None,
        python_version=None,
        spark_version=None,
        restart_policy=None,
        deps=None,
        main_class=None,
        default_handler=None,
        entry_points=None,
        description=None,
        workdir=None,
        build=None,
        spark_conf=None,
        hadoop_conf=None,
        node_selector=None,
    ):

        super().__init__(
            command=command,
            args=args,
            image=image,
            mode=mode,
            volumes=volumes,
            volume_mounts=volume_mounts,
            env=env,
            resources=resources,
            replicas=replicas,
            image_pull_policy=image_pull_policy,
            service_account=service_account,
            image_pull_secret=image_pull_secret,
            default_handler=default_handler,
            entry_points=entry_points,
            description=description,
            workdir=workdir,
            build=build,
            node_selector=node_selector,
        )

        self.driver_resources = driver_resources or {}
        self.executor_resources = executor_resources or {}
        self.spark_conf = spark_conf or {}
        self.hadoop_conf = hadoop_conf or {}
        self.job_type = job_type
        self.python_version = python_version
        self.spark_version = spark_version
        self.restart_policy = restart_policy or {}
        self.deps = deps
        self.main_class = main_class


class SparkRuntime(KubejobRuntime):
    group = "sparkoperator.k8s.io"
    version = "v1beta2"
    apiVersion = group + "/" + version
    kind = "spark"
    plural = "sparkapplications"

    @property
    def _default_image(self):
        _, driver_gpu = self._get_gpu_type_and_quantity(
            resources=self.spec.driver_resources["requests"]
        )
        _, executor_gpu = self._get_gpu_type_and_quantity(
            resources=self.spec.executor_resources["requests"]
        )
        gpu_bound = bool(driver_gpu or executor_gpu)
        if config.spark_app_image_tag and config.spark_app_image:
            return (
                config.spark_app_image
                + ("-cuda" if gpu_bound else "")
                + ":"
                + config.spark_app_image_tag
            )
        return None

    def deploy(
        self,
        watch=True,
        with_mlrun=True,
        skip_deployed=False,
        is_kfp=False,
        mlrun_version_specifier=None,
    ):
        """deploy function, build container with dependencies"""
        # connect will populate the config from the server config
        get_run_db()
        if not self.spec.build.base_image:
            self.spec.build.base_image = self._default_image
        return super().deploy(
            watch=watch,
            with_mlrun=with_mlrun,
            skip_deployed=skip_deployed,
            is_kfp=is_kfp,
            mlrun_version_specifier=mlrun_version_specifier,
        )

    @staticmethod
    def _get_gpu_type_and_quantity(resources):
        gpu_type = [
            resource_type
            for resource_type in resources.keys()
            if resource_type not in ["cpu", "memory"]
        ]
        if len(gpu_type) > 1:
            raise ValueError("Sparkjob supports only a single gpu type")
        gpu_quantity = resources[gpu_type[0]] if gpu_type else 0
        return gpu_type[0] if gpu_type else None, gpu_quantity

    def _run(self, runobj: RunObject, execution: MLClientCtx):
        if runobj.metadata.iteration:
            self.store_run(runobj)
        job = deepcopy(_sparkjob_template)
        meta = self._get_meta(runobj, True)
        pod_labels = deepcopy(meta.labels)
        pod_labels["mlrun/job"] = meta.name
        job_type = self.spec.job_type or "Python"
        update_in(job, "spec.type", job_type)
        if self.spec.job_type == "Python":
            update_in(job, "spec.pythonVersion", self.spec.python_version or "3")
        if self.spec.main_class:
            update_in(job, "spec.mainClass", self.spec.main_class)
        if self.spec.spark_version:
            update_in(job, "spec.sparkVersion", self.spec.spark_version)

        if self.spec.restart_policy:
            update_in(job, "spec.restartPolicy.type", self.spec.restart_policy["type"])
            update_in(
                job,
                "spec.restartPolicy.onFailureRetries",
                self.spec.restart_policy["retries"],
            )
            update_in(
                job,
                "spec.restartPolicy.onFailureRetryInterval",
                self.spec.restart_policy["retry_interval"],
            )
            update_in(
                job,
                "spec.restartPolicy.onSubmissionFailureRetries",
                self.spec.restart_policy["submission_retries"],
            )
            update_in(
                job,
                "spec.restartPolicy.onSubmissionFailureRetryInterval",
                self.spec.restart_policy["submission_retry_interval"],
            )

        update_in(job, "metadata", meta.to_dict())
        update_in(job, "spec.driver.labels", pod_labels)
        update_in(job, "spec.executor.labels", pod_labels)
        update_in(job, "spec.executor.instances", self.spec.replicas or 1)
        update_in(job, "spec.nodeSelector", self.spec.node_selector or {})

        if (not self.spec.image) and self._default_image:
            self.spec.image = self._default_image
        update_in(job, "spec.image", self.full_image_path())

        update_in(job, "spec.volumes", self.spec.volumes)

        extra_env = self._generate_runtime_env(runobj)
        extra_env = [{"name": k, "value": v} for k, v in extra_env.items()]

        update_in(job, "spec.driver.env", extra_env + self.spec.env)
        update_in(job, "spec.executor.env", extra_env + self.spec.env)
        update_in(job, "spec.driver.volumeMounts", self.spec.volume_mounts)
        update_in(job, "spec.executor.volumeMounts", self.spec.volume_mounts)
        update_in(job, "spec.deps", self.spec.deps)

        if self.spec.spark_conf:
            job["spec"]["sparkConf"] = {}
            for k, v in self.spec.spark_conf.items():
                job["spec"]["sparkConf"][f'"{k}"'] = f'"{v}"'

        if self.spec.hadoop_conf:
            job["spec"]["hadoopConf"] = {}
            for k, v in self.spec.hadoop_conf.items():
                job["spec"]["hadoopConf"][f'"{k}"'] = f'"{v}"'

        if "limits" in self.spec.executor_resources:
            if "cpu" in self.spec.executor_resources["limits"]:
                update_in(
                    job,
                    "spec.executor.coreLimit",
                    self.spec.executor_resources["limits"]["cpu"],
                )
        if "requests" in self.spec.executor_resources:
            if "cpu" in self.spec.executor_resources["requests"]:
                update_in(
                    job,
                    "spec.executor.cores",
                    self.spec.executor_resources["requests"]["cpu"],
                )
            if "memory" in self.spec.executor_resources["requests"]:
                update_in(
                    job,
                    "spec.executor.memory",
                    self.spec.executor_resources["requests"]["memory"],
                )
            gpu_type, gpu_quantity = self._get_gpu_type_and_quantity(
                resources=self.spec.executor_resources["requests"]
            )
            if gpu_type:
                update_in(job, "spec.executor.gpu.name", gpu_type)
                update_in(job, "spec.executor.gpu.quantity", gpu_quantity)
        if "limits" in self.spec.driver_resources:
            if "cpu" in self.spec.driver_resources["limits"]:
                update_in(
                    job,
                    "spec.driver.coreLimit",
                    self.spec.driver_resources["limits"]["cpu"],
                )
        if "requests" in self.spec.driver_resources:
            if "cpu" in self.spec.driver_resources["requests"]:
                update_in(
                    job,
                    "spec.driver.cores",
                    self.spec.driver_resources["requests"]["cpu"],
                )
            if "memory" in self.spec.driver_resources["requests"]:
                update_in(
                    job,
                    "spec.driver.memory",
                    self.spec.driver_resources["requests"]["memory"],
                )
            gpu_type, gpu_quantity = self._get_gpu_type_and_quantity(
                resources=self.spec.driver_resources["requests"]
            )
            if gpu_type:
                update_in(job, "spec.driver.gpu.name", gpu_type)
                update_in(job, "spec.driver.gpu.quantity", gpu_quantity)

        if self.spec.command:
            if "://" not in self.spec.command:
                self.spec.command = "local://" + self.spec.command
            update_in(job, "spec.mainApplicationFile", self.spec.command)
        update_in(job, "spec.arguments", self.spec.args or [])
        self._submit_job(job, meta.namespace)

        return None

    def _submit_job(self, job, namespace=None):
        k8s = self._get_k8s()
        namespace = k8s.resolve_namespace(namespace)
        try:
            resp = k8s.crdapi.create_namespaced_custom_object(
                SparkRuntime.group,
                SparkRuntime.version,
                namespace=namespace,
                plural=SparkRuntime.plural,
                body=job,
            )
            name = get_in(resp, "metadata.name", "unknown")
            logger.info(f"SparkJob {name} created")
            return resp
        except ApiException as exc:
            crd = f"{SparkRuntime.group}/{SparkRuntime.version}/{SparkRuntime.plural}"
            logger.error(f"Exception when creating SparkJob ({crd}): {exc}")
            raise RunError(f"Exception when creating SparkJob: {exc}")

    def get_job(self, name, namespace=None):
        k8s = self._get_k8s()
        namespace = k8s.resolve_namespace(namespace)
        try:
            resp = k8s.crdapi.get_namespaced_custom_object(
                SparkRuntime.group,
                SparkRuntime.version,
                namespace,
                SparkRuntime.plural,
                name,
            )
        except ApiException as exc:
            print(f"Exception when reading SparkJob: {exc}")
        return resp

    def _update_igz_jars(self, deps=igz_deps):
        if not self.spec.deps:
            self.spec.deps = {}
        if "jars" in deps:
            if "jars" not in self.spec.deps:
                self.spec.deps["jars"] = []
            self.spec.deps["jars"] += deps["jars"]
        if "files" in deps:
            if "files" not in self.spec.deps:
                self.spec.deps["files"] = []
            self.spec.deps["files"] += deps["files"]

    def with_igz_spark(self):
        self._update_igz_jars()
        self.apply(mount_v3io_extended())
        self.apply(
            mount_v3iod(
                namespace=config.namespace,
                v3io_config_configmap="spark-operator-v3io-config",
            )
        )

    def with_limits(self, mem=None, cpu=None, gpus=None, gpu_type="nvidia.com/gpu"):
        raise NotImplementedError(
            "In spark runtimes, please use with_driver_limits & with_executor_limits"
        )

    def with_requests(self, mem=None, cpu=None):
        raise NotImplementedError(
            "In spark runtimes, please use with_driver_requests & with_executor_requests"
        )

    def gpus(self, gpus, gpu_type="nvidia.com/gpu"):
        raise NotImplementedError(
            "In spark runtimes, please use with_driver_requests & with_executor_requests"
        )

    def with_node_selection(
        self,
        node_name: typing.Optional[str] = None,
        node_selector: typing.Optional[typing.Dict[str, str]] = None,
        affinity: typing.Optional[client.V1Affinity] = None,
    ):
        if node_name:
            raise NotImplementedError(
                "Setting node name is not supported for spark runtime"
            )
        if affinity:
            raise NotImplementedError(
                "Setting affinity is not supported for spark runtime"
            )
        super().with_node_selection(node_name, node_selector, affinity)

    def with_executor_requests(
        self, mem=None, cpu=None, gpus=None, gpu_type="nvidia.com/gpu"
    ):
        """set executor pod required cpu/memory/gpu resources"""
        update_in(
            self.spec.executor_resources,
            "requests",
            generate_resources(mem=mem, cpu=cpu, gpus=gpus, gpu_type=gpu_type),
        )

    def with_executor_limits(self, cpu=None):
        """set executor pod cpu limits"""
        update_in(
            self.spec.executor_resources, "limits", generate_resources(cpu=str(cpu))
        )

    def with_driver_requests(
        self, mem=None, cpu=None, gpus=None, gpu_type="nvidia.com/gpu"
    ):
        """set driver pod required cpu/memory/gpu resources"""
        update_in(
            self.spec.driver_resources,
            "requests",
            generate_resources(mem=mem, cpu=cpu, gpus=gpus, gpu_type=gpu_type),
        )

    def with_driver_limits(self, cpu=None):
        """set driver pod cpu limits"""
        update_in(
            self.spec.driver_resources, "limits", generate_resources(cpu=str(cpu))
        )

    def with_restart_policy(
        self,
        restart_type="OnFailure",
        retries=3,
        retry_interval=10,
        submission_retries=5,
        submission_retry_interval=20,
    ):
        """set restart policy
           restart_type=OnFailure/Never/Always"""
        update_in(self.spec.restart_policy, "type", restart_type)
        update_in(self.spec.restart_policy, "retries", retries)
        update_in(self.spec.restart_policy, "retry_interval", retry_interval)
        update_in(self.spec.restart_policy, "submission_retries", submission_retries)
        update_in(
            self.spec.restart_policy,
            "submission_retry_interval",
            submission_retry_interval,
        )

    def get_pods(self, name=None, namespace=None, driver=False):
        k8s = self._get_k8s()
        namespace = k8s.resolve_namespace(namespace)
        selector = "mlrun/class=spark"
        if name:
            selector += f",sparkoperator.k8s.io/app-name={name}"
        if driver:
            selector += ",spark-role=driver"
        pods = k8s.list_pods(selector=selector, namespace=namespace)
        if pods:
            return {p.metadata.name: p.status.phase for p in pods}

    def _get_driver(self, name, namespace=None):
        pods = self.get_pods(name, namespace, driver=True)
        if not pods:
            logger.error("no pod matches that job name")
            return
        _ = self._get_k8s()
        return list(pods.items())[0]

    @property
    def is_deployed(self):
        if (
            not self.spec.build.source
            and not self.spec.build.commands
            and not self.spec.build.extra
        ):
            return True
        return super().is_deployed

    @property
    def spec(self) -> SparkJobSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", SparkJobSpec)


class SparkRuntimeHandler(BaseRuntimeHandler):
    def _resolve_crd_object_status_info(
        self, db: DBInterface, db_session: Session, crd_object
    ) -> Tuple[bool, Optional[datetime], Optional[str]]:
        state = crd_object.get("status", {}).get("applicationState", {}).get("state")
        in_terminal_state = state in SparkApplicationStates.terminal_states()
        desired_run_state = SparkApplicationStates.spark_application_state_to_run_state(
            state
        )
        completion_time = None
        if in_terminal_state:
            completion_time = datetime.fromisoformat(
                crd_object.get("status", {})
                .get("terminationTime")
                .replace("Z", "+00:00")
            )
        return in_terminal_state, completion_time, desired_run_state

    def _update_ui_url(
        self,
        db: DBInterface,
        db_session: Session,
        project: str,
        uid: str,
        crd_object,
        run: Dict = None,
        leader_session: Optional[str] = None,
    ):
        app_state = (
            crd_object.get("status", {}).get("applicationState", {}).get("state")
        )
        state = SparkApplicationStates.spark_application_state_to_run_state(app_state)
        ui_url = None
        if state == RunStates.running:
            ui_url = (
                crd_object.get("status", {})
                .get("driverInfo", {})
                .get("webUIIngressAddress")
            )
        db_ui_url = run.get("status", {}).get("ui_url")
        if db_ui_url == ui_url:
            return
        run.setdefault("status", {})["ui_url"] = ui_url
        db.store_run(db_session, run, uid, project, leader_session=leader_session)

    @staticmethod
    def _are_resources_coupled_to_run_object() -> bool:
        return True

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f"mlrun/uid={object_id}"

    @staticmethod
    def _get_default_label_selector() -> str:
        return "mlrun/class=spark"

    @staticmethod
    def _get_crd_info() -> Tuple[str, str, str]:
        return SparkRuntime.group, SparkRuntime.version, SparkRuntime.plural
