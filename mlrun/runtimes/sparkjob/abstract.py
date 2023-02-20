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

import mlrun.errors
import mlrun.utils.regex
from mlrun.api.db.base import DBInterface
from mlrun.config import config
from mlrun.db import get_run_db
from mlrun.errors import err_to_str
from mlrun.runtimes.base import BaseRuntimeHandler
from mlrun.runtimes.constants import RunStates, SparkApplicationStates

from ...execution import MLClientCtx
from ...k8s_utils import get_k8s_helper
from ...model import RunObject
from ...platforms.iguazio import mount_v3io, mount_v3iod
from ...utils import (
    get_in,
    logger,
    update_in,
    verify_and_update_in,
    verify_field_regex,
    verify_list_and_update_in,
)
from ..base import RunError, RuntimeClassMode
from ..kubejob import KubejobRuntime
from ..pod import KubeResourceSpec
from ..utils import get_item_name

_service_account = "sparkapp"
_sparkjob_template = {
    "apiVersion": "sparkoperator.k8s.io/v1beta2",
    "kind": "SparkApplication",
    "metadata": {"name": "", "namespace": "default-tenant"},
    "spec": {
        "mode": "cluster",
        "image": "",
        "mainApplicationFile": "",
        "sparkVersion": "3.1.2",
        "restartPolicy": {
            "type": "OnFailure",
            "onFailureRetries": 0,
            "onFailureRetryInterval": 10,
            "onSubmissionFailureRetries": 3,
            "onSubmissionFailureRetryInterval": 20,
        },
        "deps": {},
        "volumes": [],
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
    },
}


allowed_types = ["Python", "Scala", "Java", "R"]


class AbstractSparkJobSpec(KubeResourceSpec):
    _dict_fields = KubeResourceSpec._dict_fields + [
        "driver_resources",
        "executor_resources",
        "job_type",
        "python_version",
        "spark_version",
        "restart_policy",
        "deps",
        "main_class",
        "spark_conf",
        "hadoop_conf",
        "use_default_image",
    ]

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
        use_default_image=False,
        priority_class_name=None,
        disable_auto_mount=False,
        pythonpath=None,
        node_name=None,
        affinity=None,
        tolerations=None,
        preemption_mode=None,
        security_context=None,
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
            priority_class_name=priority_class_name,
            disable_auto_mount=disable_auto_mount,
            pythonpath=pythonpath,
            node_name=node_name,
            affinity=affinity,
            tolerations=tolerations,
            preemption_mode=preemption_mode,
            security_context=security_context,
        )

        self._driver_resources = self.enrich_resources_with_default_pod_resources(
            "driver_resources", driver_resources
        )
        self._executor_resources = self.enrich_resources_with_default_pod_resources(
            "executor_resources", executor_resources
        )
        self.spark_conf = spark_conf or {}
        self.hadoop_conf = hadoop_conf or {}
        self.job_type = job_type
        self.python_version = python_version
        self.spark_version = spark_version
        self.restart_policy = restart_policy or {}
        self.deps = deps or {}
        self.main_class = main_class
        self.use_default_image = use_default_image

    @property
    def driver_resources(self) -> dict:
        return self._driver_resources

    @driver_resources.setter
    def driver_resources(self, resources):
        self._driver_resources = self.enrich_resources_with_default_pod_resources(
            "driver_resources", resources
        )

    @property
    def executor_resources(self) -> dict:
        return self._executor_resources

    @executor_resources.setter
    def executor_resources(self, resources):
        self._executor_resources = self.enrich_resources_with_default_pod_resources(
            "executor_resources", resources
        )


class AbstractSparkRuntime(KubejobRuntime):
    group = "sparkoperator.k8s.io"
    version = "v1beta2"
    apiVersion = group + "/" + version
    kind = "spark"
    plural = "sparkapplications"

    # the dot will make the api prefix the configured registry to the image name
    default_mlrun_image = ".spark-job-default-image"
    gpu_suffix = "-cuda"
    code_script = "spark-function-code.py"
    code_path = "/etc/config/mlrun"

    @classmethod
    def _get_default_deployed_mlrun_image_name(cls, with_gpu=False):
        suffix = cls.gpu_suffix if with_gpu else ""
        return cls.default_mlrun_image + suffix

    @classmethod
    def deploy_default_image(cls, with_gpu=False):
        from mlrun.run import new_function

        sj = new_function(kind=cls.kind, name="spark-default-image-deploy-temp")
        sj.spec.build.image = cls._get_default_deployed_mlrun_image_name(with_gpu)

        # setting required resources
        sj.with_executor_requests(cpu=1, mem="512m")
        sj.with_driver_requests(cpu=1, mem="512m")

        sj.deploy()
        get_run_db().delete_function(name=sj.metadata.name)

    def _is_using_gpu(self):
        driver_limits = self.spec.driver_resources.get("limits")
        driver_gpu = None
        if driver_limits:
            _, driver_gpu = self._get_gpu_type_and_quantity(resources=driver_limits)

        executor_limits = self.spec.executor_resources.get("limits")
        executor_gpu = None
        if executor_limits:
            _, executor_gpu = self._get_gpu_type_and_quantity(resources=executor_limits)

        return bool(driver_gpu or executor_gpu)

    @property
    def _default_image(self):
        if config.spark_app_image_tag and config.spark_app_image:
            return (
                config.spark_app_image
                + (self.gpu_suffix if self._is_using_gpu() else "")
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
        builder_env: dict = None,
        show_on_failure: bool = False,
    ):
        """deploy function, build container with dependencies

        :param watch:      wait for the deployment to complete (and print build logs)
        :param with_mlrun: add the current mlrun package to the container build
        :param skip_deployed: skip the build if we already have an image for the function
        :param is_kfp:  whether running inside KFP step (no need to use,
                        being enriched automatically if running inside one)
        :param mlrun_version_specifier:  which mlrun package version to include (if not current)
        :param builder_env:   Kaniko builder pod env vars dict (for config/credentials)
                              e.g. builder_env={"GIT_TOKEN": token}
        :param show_on_failure:  show logs only in case of build failure

        :return True if the function is ready (deployed)
        """
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
            builder_env=builder_env,
            show_on_failure=show_on_failure,
        )

    @staticmethod
    def _get_gpu_type_and_quantity(resources):
        gpu_type = [
            resource_type
            for resource_type in resources.keys()
            if resource_type not in ["cpu", "memory"]
        ]
        if len(gpu_type) > 1:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Sparkjob supports only a single gpu type"
            )
        gpu_quantity = resources[gpu_type[0]] if gpu_type else 0
        return gpu_type[0] if gpu_type else None, gpu_quantity

    def _validate(self, runobj: RunObject):
        # validating correctness of sparkjob's function name
        try:
            verify_field_regex(
                "run.metadata.name",
                runobj.metadata.name,
                mlrun.utils.regex.sparkjob_name,
            )

        except mlrun.errors.MLRunInvalidArgumentError as err:
            pattern_error = str(err).split(" ")[-1]
            if pattern_error == mlrun.utils.regex.sprakjob_length:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Job name '{runobj.metadata.name}' is not valid."
                    f" The job name must be not longer than 29 characters"
                )
            elif pattern_error in mlrun.utils.regex.label_value:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "a valid label must be an empty string or consist of alphanumeric characters,"
                    " '-', '_' or '.', and must start and end with an alphanumeric character"
                )
            elif pattern_error in mlrun.utils.regex.sparkjob_service_name:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "a valid label must consist of lower case alphanumeric characters or '-', start with "
                    "an alphabetic character, and end with an alphanumeric character"
                )
            else:
                raise err

        # validating existence of required fields
        if "requests" not in self.spec.executor_resources:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Sparkjob must contain executor requests"
            )
        if "requests" not in self.spec.driver_resources:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Sparkjob must contain driver requests"
            )

    def _get_spark_version(self):
        raise NotImplementedError()

    def _get_igz_deps(self):
        raise NotImplementedError()

    def _run(self, runobj: RunObject, execution: MLClientCtx):
        self._validate(runobj)

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
        update_in(
            job,
            "spec.sparkVersion",
            self.spec.spark_version or self._get_spark_version(),
        )

        if self.spec.image_pull_policy:
            verify_and_update_in(
                job, "spec.imagePullPolicy", self.spec.image_pull_policy, str
            )

        if self.spec.restart_policy:
            verify_and_update_in(
                job, "spec.restartPolicy.type", self.spec.restart_policy["type"], str
            )
            verify_and_update_in(
                job,
                "spec.restartPolicy.onFailureRetries",
                self.spec.restart_policy["retries"],
                int,
            )
            verify_and_update_in(
                job,
                "spec.restartPolicy.onFailureRetryInterval",
                self.spec.restart_policy["retry_interval"],
                int,
            )
            verify_and_update_in(
                job,
                "spec.restartPolicy.onSubmissionFailureRetries",
                self.spec.restart_policy["submission_retries"],
                int,
            )
            verify_and_update_in(
                job,
                "spec.restartPolicy.onSubmissionFailureRetryInterval",
                self.spec.restart_policy["submission_retry_interval"],
                int,
            )

        update_in(job, "metadata", meta.to_dict())
        update_in(job, "spec.driver.labels", pod_labels)
        update_in(job, "spec.executor.labels", pod_labels)
        verify_and_update_in(
            job,
            "spec.executor.instances",
            self.spec.replicas or 1,
            int,
        )
        if self.spec.image_pull_secret:
            update_in(job, "spec.imagePullSecrets", [self.spec.image_pull_secret])

        if self.spec.node_selector:
            update_in(job, "spec.nodeSelector", self.spec.node_selector)

        if not self.spec.image:
            if self.spec.use_default_image:
                self.spec.image = self._get_default_deployed_mlrun_image_name(
                    self._is_using_gpu()
                )
            elif self._default_image:
                self.spec.image = self._default_image

        update_in(
            job,
            "spec.image",
            self.full_image_path(
                client_version=runobj.metadata.labels.get("mlrun/client_version"),
                client_python_version=runobj.metadata.labels.get(
                    "mlrun/client_python_version"
                ),
            ),
        )

        update_in(job, "spec.volumes", self.spec.volumes)

        self._add_secrets_to_spec_before_running(runobj)

        command, args, extra_env = self._get_cmd_args(runobj)
        code = None
        if "MLRUN_EXEC_CODE" in [e.get("name") for e in extra_env]:
            code = f"""
import mlrun.__main__ as ml
ctx = ml.main.make_context('main', {args})
with ctx:
    result = ml.main.invoke(ctx)
"""

        update_in(job, "spec.driver.env", extra_env + self.spec.env)
        update_in(job, "spec.executor.env", extra_env + self.spec.env)
        update_in(job, "spec.driver.volumeMounts", self.spec.volume_mounts)
        update_in(job, "spec.executor.volumeMounts", self.spec.volume_mounts)
        update_in(job, "spec.deps", self.spec.deps)

        if self.spec.spark_conf:
            job["spec"]["sparkConf"] = {}
            for k, v in self.spec.spark_conf.items():
                job["spec"]["sparkConf"][f"{k}"] = f"{v}"

        if self.spec.hadoop_conf:
            job["spec"]["hadoopConf"] = {}
            for k, v in self.spec.hadoop_conf.items():
                job["spec"]["hadoopConf"][f"{k}"] = f"{v}"

        if "limits" in self.spec.executor_resources:
            if "cpu" in self.spec.executor_resources["limits"]:
                verify_and_update_in(
                    job,
                    "spec.executor.coreLimit",
                    self.spec.executor_resources["limits"]["cpu"],
                    str,
                )
        if "requests" in self.spec.executor_resources:
            verify_and_update_in(
                job,
                "spec.executor.cores",
                1,  # Must be set due to CRD validations. Will be overridden by coreRequest
                int,
            )
            if "cpu" in self.spec.executor_resources["requests"]:
                verify_and_update_in(
                    job,
                    "spec.executor.coreRequest",
                    str(
                        self.spec.executor_resources["requests"]["cpu"]
                    ),  # Backwards compatibility
                    str,
                )
            if "memory" in self.spec.executor_resources["requests"]:
                verify_and_update_in(
                    job,
                    "spec.executor.memory",
                    self.spec.executor_resources["requests"]["memory"],
                    str,
                )
            gpu_type, gpu_quantity = self._get_gpu_type_and_quantity(
                resources=self.spec.executor_resources["limits"]
            )
            if gpu_type:
                update_in(job, "spec.executor.gpu.name", gpu_type)
                if gpu_quantity:
                    verify_and_update_in(
                        job,
                        "spec.executor.gpu.quantity",
                        gpu_quantity,
                        int,
                    )
        if "limits" in self.spec.driver_resources:
            if "cpu" in self.spec.driver_resources["limits"]:
                verify_and_update_in(
                    job,
                    "spec.driver.coreLimit",
                    self.spec.driver_resources["limits"]["cpu"],
                    str,
                )
        if "requests" in self.spec.driver_resources:
            # CPU Requests for driver moved to child classes as spark3 supports string (i.e: "100m") and not only int
            if "memory" in self.spec.driver_resources["requests"]:
                verify_and_update_in(
                    job,
                    "spec.driver.memory",
                    self.spec.driver_resources["requests"]["memory"],
                    str,
                )
            gpu_type, gpu_quantity = self._get_gpu_type_and_quantity(
                resources=self.spec.driver_resources["limits"]
            )
            if gpu_type:
                update_in(job, "spec.driver.gpu.name", gpu_type)
                if gpu_quantity:
                    verify_and_update_in(
                        job,
                        "spec.driver.gpu.quantity",
                        gpu_quantity,
                        int,
                    )

        self._enrich_job(job)

        if self.spec.command:
            if "://" not in self.spec.command:
                self.spec.command = "local://" + self.spec.command
            update_in(job, "spec.mainApplicationFile", self.spec.command)

        verify_list_and_update_in(job, "spec.arguments", self.spec.args or [], str)
        self._submit_spark_job(job, meta, code)

        return None

    def _enrich_job(self, job):
        raise NotImplementedError()

    def _submit_spark_job(
        self,
        job,
        meta,
        code=None,
    ):
        namespace = meta.namespace
        k8s = self._get_k8s()
        namespace = k8s.resolve_namespace(namespace)
        if code:
            k8s_config_map = client.V1ConfigMap()
            k8s_config_map.metadata = meta
            k8s_config_map.metadata.name += "-script"
            k8s_config_map.data = {self.code_script: code}
            config_map = k8s.v1api.create_namespaced_config_map(
                namespace, k8s_config_map
            )
            config_map_name = config_map.metadata.name

            vol_src = client.V1ConfigMapVolumeSource(name=config_map_name)
            volume_name = "script"
            vol = client.V1Volume(name=volume_name, config_map=vol_src)
            vol_mount = client.V1VolumeMount(
                mount_path=self.code_path, name=volume_name
            )
            update_in(job, "spec.volumes", [vol], append=True)
            update_in(job, "spec.driver.volumeMounts", [vol_mount], append=True)
            update_in(job, "spec.executor.volumeMounts", [vol_mount], append=True)
            update_in(
                job,
                "spec.mainApplicationFile",
                f"local://{self.code_path}/{self.code_script}",
            )

        try:
            resp = k8s.crdapi.create_namespaced_custom_object(
                AbstractSparkRuntime.group,
                AbstractSparkRuntime.version,
                namespace=namespace,
                plural=AbstractSparkRuntime.plural,
                body=job,
            )
            name = get_in(resp, "metadata.name", "unknown")
            logger.info(f"SparkJob {name} created")
            return resp
        except ApiException as exc:
            crd = f"{AbstractSparkRuntime.group}/{AbstractSparkRuntime.version}/{AbstractSparkRuntime.plural}"
            logger.error(f"Exception when creating SparkJob ({crd}): {err_to_str(exc)}")
            raise RunError("Exception when creating SparkJob") from exc

    def get_job(self, name, namespace=None):
        k8s = self._get_k8s()
        namespace = k8s.resolve_namespace(namespace)
        try:
            resp = k8s.crdapi.get_namespaced_custom_object(
                AbstractSparkRuntime.group,
                AbstractSparkRuntime.version,
                namespace,
                AbstractSparkRuntime.plural,
                name,
            )
        except ApiException as exc:
            print(f"Exception when reading SparkJob: {err_to_str(exc)}")
        return resp

    def _update_igz_jars(self, deps):
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

    def with_igz_spark(self, mount_v3io_to_executor=True):
        self._update_igz_jars(deps=self._get_igz_deps())
        self.apply(mount_v3io(name="v3io"))

        # if we only want to mount v3io on the driver, move v3io
        # mounts from common volume mounts to driver volume mounts
        if not mount_v3io_to_executor:
            v3io_mounts = []
            non_v3io_mounts = []
            for mount in self.spec.volume_mounts:
                if get_item_name(mount) == "v3io":
                    v3io_mounts.append(mount)
                else:
                    non_v3io_mounts.append(mount)
            self.spec.volume_mounts = non_v3io_mounts
            self.spec.driver_volume_mounts += v3io_mounts

        self.apply(
            mount_v3iod(
                namespace=config.namespace,
                v3io_config_configmap="spark-operator-v3io-config",
            )
        )
        if config.spark_history_server_path:
            self.spec.spark_conf["spark.eventLog.enabled"] = "true"
            self.spec.spark_conf["spark.eventLog.dir"] = (
                "file://" + config.spark_history_server_path
            )

    def with_limits(
        self,
        mem=None,
        cpu=None,
        gpus=None,
        gpu_type="nvidia.com/gpu",
        patch: bool = False,
    ):
        raise NotImplementedError(
            "In spark runtimes, use 'with_driver_limits' & 'with_executor_limits'"
        )

    def with_requests(self, mem=None, cpu=None, patch: bool = False):
        raise NotImplementedError(
            "In spark runtimes, use 'with_driver_requests' & 'with_executor_requests'"
        )

    def gpus(self, gpus, gpu_type="nvidia.com/gpu"):
        raise NotImplementedError(
            "In spark runtimes, use 'with_driver_limits' & 'with_executor_limits'"
        )

    def with_node_selection(
        self,
        node_name: typing.Optional[str] = None,
        node_selector: typing.Optional[typing.Dict[str, str]] = None,
        affinity: typing.Optional[client.V1Affinity] = None,
        tolerations: typing.Optional[typing.List[client.V1Toleration]] = None,
    ):
        if node_name:
            raise NotImplementedError(
                "Setting node name is not supported for spark runtime"
            )
        # TODO add affinity support
        # https://github.com/GoogleCloudPlatform/spark-on-k8s-operator/blob/master/pkg/apis/sparkoperator.k8s.io/v1beta2/types.go#L491
        if affinity:
            raise NotImplementedError(
                "Setting affinity is not supported for spark runtime"
            )
        super().with_node_selection(node_name, node_selector, affinity, tolerations)

    def with_executor_requests(
        self, mem: str = None, cpu: str = None, patch: bool = False
    ):
        """
        set executor pod required cpu/memory/gpu resources
        by default it overrides the whole requests section, if you wish to patch specific resources use `patch=True`.
        """
        self.spec._verify_and_set_requests("executor_resources", mem, cpu, patch=patch)

    def with_executor_limits(
        self,
        cpu: str = None,
        gpus: int = None,
        gpu_type: str = "nvidia.com/gpu",
        patch: bool = False,
    ):
        """
        set executor pod limits
        by default it overrides the whole limits section, if you wish to patch specific resources use `patch=True`.
        """
        # in spark operator there is only use of mem passed through requests,
        # limits is set to the same value so passing mem=None
        self.spec._verify_and_set_limits(
            "executor_resources", None, cpu, gpus, gpu_type, patch=patch
        )

    def with_driver_requests(
        self, mem: str = None, cpu: str = None, patch: bool = False
    ):
        """
        set driver pod required cpu/memory/gpu resources
        by default it overrides the whole requests section, if you wish to patch specific resources use `patch=True`.
        """
        self.spec._verify_and_set_requests("driver_resources", mem, cpu, patch=patch)

    def with_driver_limits(
        self,
        cpu: str = None,
        gpus: int = None,
        gpu_type: str = "nvidia.com/gpu",
        patch: bool = False,
    ):
        """
        set driver pod cpu limits
        by default it overrides the whole limits section, if you wish to patch specific resources use `patch=True`.
        """
        # in spark operator there is only use of mem passed through requests,
        # limits is set to the same value so passing mem=None
        self.spec._verify_and_set_limits(
            "driver_resources", None, cpu, gpus, gpu_type, patch=patch
        )

    def with_restart_policy(
        self,
        restart_type="OnFailure",
        retries=0,
        retry_interval=10,
        submission_retries=3,
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

    def is_deployed(self):
        if (
            not self.spec.build.source
            and not self.spec.build.commands
            and not self.spec.build.extra
        ):
            return True
        return super().is_deployed()

    @property
    def spec(self) -> AbstractSparkJobSpec:
        raise NotImplementedError()

    @spec.setter
    def spec(self, spec):
        raise NotImplementedError()


class SparkRuntimeHandler(BaseRuntimeHandler):
    kind = "spark"
    class_modes = {
        RuntimeClassMode.run: "spark",
    }

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
            if crd_object.get("status", {}).get("terminationTime"):
                completion_time = datetime.fromisoformat(
                    crd_object.get("status", {})
                    .get("terminationTime")
                    .replace("Z", "+00:00")
                )
            else:
                last_submission_attempt_time = crd_object.get("status", {}).get(
                    "lastSubmissionAttemptTime"
                )
                if last_submission_attempt_time:
                    last_submission_attempt_time = last_submission_attempt_time.replace(
                        "Z", "+00:00"
                    )
                    completion_time = datetime.fromisoformat(
                        last_submission_attempt_time
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
        db.store_run(db_session, run, uid, project)

    @staticmethod
    def _are_resources_coupled_to_run_object() -> bool:
        return True

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f"mlrun/uid={object_id}"

    @staticmethod
    def _get_crd_info() -> Tuple[str, str, str]:
        return (
            AbstractSparkRuntime.group,
            AbstractSparkRuntime.version,
            AbstractSparkRuntime.plural,
        )

    def _delete_resources(
        self,
        db: DBInterface,
        db_session: Session,
        namespace: str,
        deleted_resources: typing.List[Dict],
        label_selector: str = None,
        force: bool = False,
        grace_period: int = None,
    ):
        """
        Handling config maps deletion
        """
        if grace_period is None:
            grace_period = config.runtime_resources_deletion_grace_period
        uids = []
        for crd_dict in deleted_resources:
            uid = crd_dict["metadata"].get("labels", {}).get("mlrun/uid", None)
            uids.append(uid)

        k8s_helper = get_k8s_helper()
        config_maps = k8s_helper.v1api.list_namespaced_config_map(
            namespace, label_selector=label_selector
        )
        for config_map in config_maps.items:
            try:
                uid = config_map.metadata.labels.get("mlrun/uid", None)
                if force or uid in uids:
                    k8s_helper.v1api.delete_namespaced_config_map(
                        config_map.metadata.name, namespace
                    )
                    logger.info(f"Deleted config map: {config_map.metadata.name}")
            except ApiException as exc:
                # ignore error if config map is already removed
                if exc.status != 404:
                    raise
