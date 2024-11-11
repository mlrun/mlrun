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
import datetime
import inspect
import socket
import time
from os import environ
from typing import Callable, Optional, Union

import mlrun.common.schemas
import mlrun.errors
import mlrun.k8s_utils
import mlrun.utils
import mlrun.utils.regex
from mlrun.errors import err_to_str

from ..config import config
from ..execution import MLClientCtx
from ..model import RunObject
from ..render import ipython_display
from ..utils import logger
from .base import FunctionStatus
from .kubejob import KubejobRuntime
from .local import exec_from_params, load_module
from .pod import KubeResourceSpec
from .utils import RunError, get_func_selector, log_std


class DaskSpec(KubeResourceSpec):
    _dict_fields = KubeResourceSpec._dict_fields + [
        "extra_pip",
        "remote",
        "service_type",
        "nthreads",
        "kfp_image",
        "node_port",
        "min_replicas",
        "max_replicas",
        "scheduler_timeout",
        "scheduler_resources",
        "worker_resources",
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
        build=None,
        default_handler=None,
        entry_points=None,
        description=None,
        replicas=None,
        image_pull_policy=None,
        service_account=None,
        image_pull_secret=None,
        extra_pip=None,
        remote=None,
        service_type=None,
        nthreads=None,
        kfp_image=None,
        node_port=None,
        min_replicas=None,
        max_replicas=None,
        scheduler_timeout=None,
        node_name=None,
        node_selector=None,
        affinity=None,
        scheduler_resources=None,
        worker_resources=None,
        priority_class_name=None,
        disable_auto_mount=False,
        pythonpath=None,
        workdir=None,
        tolerations=None,
        preemption_mode=None,
        security_context=None,
        clone_target_dir=None,
        state_thresholds=None,
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
            build=build,
            default_handler=default_handler,
            entry_points=entry_points,
            description=description,
            image_pull_secret=image_pull_secret,
            node_name=node_name,
            node_selector=node_selector,
            affinity=affinity,
            priority_class_name=priority_class_name,
            disable_auto_mount=disable_auto_mount,
            pythonpath=pythonpath,
            workdir=workdir,
            tolerations=tolerations,
            preemption_mode=preemption_mode,
            security_context=security_context,
            clone_target_dir=clone_target_dir,
            state_thresholds=state_thresholds,
        )
        self.args = args

        self.extra_pip = extra_pip
        self.remote = True if remote is None else remote  # make remote the default

        self.service_type = service_type
        self.kfp_image = kfp_image
        self.node_port = node_port
        self.min_replicas = min_replicas or 0
        self.max_replicas = max_replicas or 16
        # supported format according to https://github.com/dask/dask/blob/master/dask/utils.py#L1402
        self.scheduler_timeout = scheduler_timeout or "60 minutes"
        self.nthreads = nthreads or 1
        self._scheduler_resources = self.enrich_resources_with_default_pod_resources(
            "scheduler_resources", scheduler_resources
        )
        self._worker_resources = self.enrich_resources_with_default_pod_resources(
            "worker_resources", worker_resources
        )

        self.state_thresholds = None  # not supported in dask

    @property
    def scheduler_resources(self) -> dict:
        return self._scheduler_resources

    @scheduler_resources.setter
    def scheduler_resources(self, resources):
        self._scheduler_resources = self.enrich_resources_with_default_pod_resources(
            "scheduler_resources", resources
        )

    @property
    def worker_resources(self) -> dict:
        return self._worker_resources

    @worker_resources.setter
    def worker_resources(self, resources):
        self._worker_resources = self.enrich_resources_with_default_pod_resources(
            "worker_resources", resources
        )


class DaskStatus(FunctionStatus):
    def __init__(
        self,
        state=None,
        build_pod=None,
        scheduler_address=None,
        cluster_name=None,
        node_ports=None,
    ):
        super().__init__(state, build_pod)

        self.scheduler_address = scheduler_address
        self.cluster_name = cluster_name
        self.node_ports = node_ports


class DaskCluster(KubejobRuntime):
    kind = "dask"
    _is_nested = False
    _is_remote = False

    def __init__(self, spec=None, metadata=None):
        super().__init__(spec, metadata)
        self._cluster = None
        self.use_remote = not mlrun.k8s_utils.is_running_inside_kubernetes_cluster()
        self.spec.build.base_image = self.spec.build.base_image or "daskdev/dask:latest"

    @property
    def spec(self) -> DaskSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", DaskSpec)

    @property
    def status(self) -> DaskStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", DaskStatus)

    def is_deployed(self):
        if not self.spec.remote:
            return True
        return super().is_deployed()

    @property
    def initialized(self):
        return bool(self._cluster)

    def _load_db_status(self):
        meta = self.metadata
        if self._is_remote_api():
            db = self._get_db()
            db_func = None
            try:
                db_func = db.get_function(meta.name, meta.project, meta.tag)
            except Exception:
                pass

            if db_func and "status" in db_func:
                self.status = db_func["status"]
                if self.kfp:
                    logger.info(f"Dask status: {db_func['status']}")
                return "scheduler_address" in db_func["status"]

        return False

    def _start(self, watch=True):
        db = self._get_db()
        if not self._is_remote_api():
            self._cluster = db.start_function(function=self)
            return

        self.try_auto_mount_based_on_config()
        self._fill_credentials()
        if not self.is_deployed():
            raise RunError(
                "Function image is not built/ready, use .deploy()"
                " method first, or set base dask image (daskdev/dask:latest)"
            )

        self.save(versioned=False)
        background_task = db.start_function(func_url=self._function_uri())
        if watch:
            now = datetime.datetime.utcnow()
            timeout = now + datetime.timedelta(minutes=10)
            while now < timeout:
                background_task = db.get_project_background_task(
                    background_task.metadata.project, background_task.metadata.name
                )
                if (
                    background_task.status.state
                    in mlrun.common.schemas.BackgroundTaskState.terminal_states()
                ):
                    if (
                        background_task.status.state
                        == mlrun.common.schemas.BackgroundTaskState.failed
                    ):
                        raise mlrun.errors.MLRunRuntimeError(
                            "Failed bringing up dask cluster"
                        )
                    else:
                        function = db.get_function(
                            self.metadata.name,
                            self.metadata.project,
                            self.metadata.tag,
                        )
                        if function and function.get("status"):
                            self.status = function.get("status")
                        return
                time.sleep(5)
                now = datetime.datetime.utcnow()

    def close(self, running=True):
        from dask.distributed import default_client

        try:
            client = default_client()
            # shutdown the cluster first, then close the client
            client.shutdown()
            client.close()
        except ValueError:
            pass

    def get_status(self):
        meta = self.metadata
        selector = get_func_selector(meta.project, meta.name, meta.tag)
        db = self._get_db()
        return db.function_status(meta.project, meta.name, self.kind, selector)

    def cluster(self):
        return self._cluster

    def _remote_addresses(self):
        addr = self.status.scheduler_address
        dash = ""
        if config.remote_host:
            if self.spec.service_type == "NodePort" and self.use_remote:
                addr = f"{config.remote_host}:{self.status.node_ports.get('scheduler')}"

            if self.spec.service_type == "NodePort":
                dash = f"{config.remote_host}:{self.status.node_ports.get('dashboard')}"
            else:
                logger.info("To get a dashboard link, use NodePort service_type")

        return addr, dash

    @property
    def client(self):
        from dask.distributed import Client, default_client

        if self.spec.remote and not self.status.scheduler_address:
            if not self._load_db_status():
                self._start()

        if self.status.scheduler_address:
            addr, dash = self._remote_addresses()
            logger.info(f"Trying dask client at: {addr}")
            try:
                client = Client(addr)
            except OSError as exc:
                logger.warning(
                    f"Remote scheduler at {addr} not ready, will try to restart {err_to_str(exc)}"
                )

                status = self.get_status()
                if status != "running":
                    self._start()
                addr, dash = self._remote_addresses()
                client = Client(addr)

            logger.info(
                f"Using remote dask scheduler ({self.status.cluster_name}) at: {addr}"
            )
            if dash:
                ipython_display(
                    f'<a href="http://{dash}/status" target="_blank" >dashboard link: {dash}</a>',
                    alt_text=f"remote dashboard: {dash}",
                )

            return client
        try:
            return default_client()
        except ValueError:
            return Client()

    def deploy(
        self,
        watch=True,
        with_mlrun=None,
        skip_deployed=False,
        is_kfp=False,
        mlrun_version_specifier=None,
        builder_env: Optional[dict] = None,
        show_on_failure: bool = False,
        force_build: bool = False,
    ):
        """deploy function, build container with dependencies

        :param watch:                   wait for the deploy to complete (and print build logs)
        :param with_mlrun:              add the current mlrun package to the container build
        :param skip_deployed:           skip the build if we already have an image for the function
        :param is_kfp:                  deploy as part of a kfp pipeline
        :param mlrun_version_specifier: which mlrun package version to include (if not current)
        :param builder_env:             Kaniko builder pod env vars dict (for config/credentials)
                                        e.g. builder_env={"GIT_TOKEN": token}
        :param show_on_failure:         show logs only in case of build failure
        :param force_build:             force building the image, even when no changes were made

        :return:                        True if the function is ready (deployed)
        """
        return super().deploy(
            watch,
            with_mlrun,
            skip_deployed,
            is_kfp=is_kfp,
            mlrun_version_specifier=mlrun_version_specifier,
            builder_env=builder_env,
            show_on_failure=show_on_failure,
            force_build=force_build,
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
            "Use with_scheduler_limits/with_worker_limits to set resource limits",
        )

    def with_scheduler_limits(
        self,
        mem: Optional[str] = None,
        cpu: Optional[str] = None,
        gpus: Optional[int] = None,
        gpu_type: str = "nvidia.com/gpu",
        patch: bool = False,
    ):
        """
        set scheduler pod resources limits
        by default it overrides the whole limits section, if you wish to patch specific resources use `patch=True`.
        """
        self.spec._verify_and_set_limits(
            "scheduler_resources", mem, cpu, gpus, gpu_type, patch=patch
        )

    def with_worker_limits(
        self,
        mem: Optional[str] = None,
        cpu: Optional[str] = None,
        gpus: Optional[int] = None,
        gpu_type: str = "nvidia.com/gpu",
        patch: bool = False,
    ):
        """
        set worker pod resources limits
        by default it overrides the whole limits section, if you wish to patch specific resources use `patch=True`.
        """
        self.spec._verify_and_set_limits(
            "worker_resources", mem, cpu, gpus, gpu_type, patch=patch
        )

    def with_requests(self, mem=None, cpu=None, patch: bool = False):
        raise NotImplementedError(
            "Use with_scheduler_requests/with_worker_requests to set resource requests",
        )

    def with_scheduler_requests(
        self, mem: Optional[str] = None, cpu: Optional[str] = None, patch: bool = False
    ):
        """
        set scheduler pod resources requests
        by default it overrides the whole requests section, if you wish to patch specific resources use `patch=True`.
        """
        self.spec._verify_and_set_requests("scheduler_resources", mem, cpu, patch=patch)

    def with_worker_requests(
        self, mem: Optional[str] = None, cpu: Optional[str] = None, patch: bool = False
    ):
        """
        set worker pod resources requests
        by default it overrides the whole requests section, if you wish to patch specific resources use `patch=True`.
        """
        self.spec._verify_and_set_requests("worker_resources", mem, cpu, patch=patch)

    def set_state_thresholds(
        self,
        state_thresholds: dict[str, str],
        patch: bool = True,
    ):
        raise NotImplementedError(
            "State thresholds is not supported for Dask runtime yet, use spec.scheduler_timeout instead.",
        )

    def run(
        self,
        runspec: Optional[
            Union["mlrun.run.RunTemplate", "mlrun.run.RunObject", dict]
        ] = None,
        handler: Optional[Union[str, Callable]] = None,
        name: Optional[str] = "",
        project: Optional[str] = "",
        params: Optional[dict] = None,
        inputs: Optional[dict[str, str]] = None,
        out_path: Optional[str] = "",
        workdir: Optional[str] = "",
        artifact_path: Optional[str] = "",
        watch: Optional[bool] = True,
        schedule: Optional[Union[str, mlrun.common.schemas.ScheduleCronTrigger]] = None,
        hyperparams: Optional[dict[str, list]] = None,
        hyper_param_options: Optional[mlrun.model.HyperParamOptions] = None,
        verbose: Optional[bool] = None,
        scrape_metrics: Optional[bool] = None,
        local: Optional[bool] = False,
        local_code_path: Optional[str] = None,
        auto_build: Optional[bool] = None,
        param_file_secrets: Optional[dict[str, str]] = None,
        notifications: Optional[list[mlrun.model.Notification]] = None,
        returns: Optional[list[Union[str, dict[str, str]]]] = None,
        state_thresholds: Optional[dict[str, int]] = None,
        reset_on_run: Optional[bool] = None,
        **launcher_kwargs,
    ) -> RunObject:
        if state_thresholds:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "State thresholds is not supported for Dask runtime yet, use spec.scheduler_timeout instead."
            )
        return super().run(
            runspec=runspec,
            handler=handler,
            name=name,
            project=project,
            params=params,
            inputs=inputs,
            out_path=out_path,
            workdir=workdir,
            artifact_path=artifact_path,
            watch=watch,
            schedule=schedule,
            hyperparams=hyperparams,
            hyper_param_options=hyper_param_options,
            verbose=verbose,
            scrape_metrics=scrape_metrics,
            local=local,
            local_code_path=local_code_path,
            auto_build=auto_build,
            param_file_secrets=param_file_secrets,
            notifications=notifications,
            returns=returns,
            state_thresholds=state_thresholds,
            **launcher_kwargs,
        )

    def _run(self, runobj: RunObject, execution):
        handler = runobj.spec.handler
        self._force_handler(handler)

        # TODO: investigate if the following instructions could overwrite the environment on any MLRun API Pod
        # Such action could result on race conditions against other runtimes and MLRun itself
        extra_env = self._generate_runtime_env(runobj)
        environ.update(extra_env)

        context = MLClientCtx.from_dict(
            runobj.to_dict(),
            rundb=self.spec.rundb,
            autocommit=False,
            host=socket.gethostname(),
        )
        if not inspect.isfunction(handler):
            if not self.spec.command:
                raise ValueError(
                    "specified handler (string) without command "
                    "(py file path), specify command or use handler pointer"
                )
            # Do not embed the module in system as it is not persistent with the dask cluster
            handler = load_module(
                self.spec.command,
                handler,
                context=context,
                embed_in_sys=False,
            )
        client = self.client
        setattr(context, "dask_client", client)
        sout, serr = exec_from_params(handler, runobj, context)
        log_std(self._db_conn, runobj, sout, serr, skip=self.is_child, show=False)
        return context.to_dict()
