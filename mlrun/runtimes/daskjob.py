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
import datetime
import inspect
import socket
import time
import warnings
from os import environ
from typing import Dict, List, Optional, Union

from kubernetes.client.rest import ApiException
from sqlalchemy.orm import Session

import mlrun.api.schemas
import mlrun.errors
import mlrun.utils
import mlrun.utils.regex
from mlrun.api.db.base import DBInterface
from mlrun.runtimes.base import BaseRuntimeHandler

from ..config import config
from ..execution import MLClientCtx
from ..k8s_utils import get_k8s_helper
from ..model import RunObject
from ..render import ipython_display
from ..utils import logger, normalize_name, update_in
from .base import FunctionStatus
from .kubejob import KubejobRuntime
from .local import exec_from_params, load_module
from .pod import KubeResourceSpec, kube_resource_spec_to_pod_spec
from .utils import RunError, get_func_selector, get_resource_labels, log_std


def get_dask_resource():
    return {
        "scope": "function",
        "start": deploy_function,
        "status": get_obj_status,
    }


class DaskSpec(KubeResourceSpec):
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
        self.scheduler_resources = scheduler_resources or {}
        self.worker_resources = worker_resources or {}


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
        self.use_remote = not get_k8s_helper(
            silent=True
        ).is_running_inside_kubernetes_cluster()
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

    @property
    def is_deployed(self):
        if not self.spec.remote:
            return True
        return super().is_deployed

    @property
    def initialized(self):
        return True if self._cluster else False

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
                    logger.info(f"dask status: {db_func['status']}")
                return "scheduler_address" in db_func["status"]

        return False

    def _start(self, watch=True):
        if self._is_remote_api():
            self.try_auto_mount_based_on_config()
            self.fill_credentials()
            db = self._get_db()
            if not self.is_deployed:
                raise RunError(
                    "function image is not built/ready, use .deploy()"
                    " method first, or set base dask image (daskdev/dask:latest)"
                )

            self.save(versioned=False)
            background_task = db.remote_start(self._function_uri())
            if watch:
                now = datetime.datetime.utcnow()
                timeout = now + datetime.timedelta(minutes=10)
                while now < timeout:
                    background_task = db.get_project_background_task(
                        background_task.metadata.project, background_task.metadata.name
                    )
                    if (
                        background_task.status.state
                        in mlrun.api.schemas.BackgroundTaskState.terminal_states()
                    ):
                        if (
                            background_task.status.state
                            == mlrun.api.schemas.BackgroundTaskState.failed
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
        else:
            self._cluster = deploy_function(self)
            self.save(versioned=False)

    def close(self, running=True):
        from dask.distributed import default_client

        try:
            client = default_client()
            client.close()
        except ValueError:
            pass

        # meta = self.metadata
        # s = get_func_selector(meta.project, meta.name, meta.tag)
        # clean_objects(s, running)

    def get_status(self):
        meta = self.metadata
        s = get_func_selector(meta.project, meta.name, meta.tag)
        if self._is_remote_api():
            db = self._get_db()
            return db.remote_status(meta.project, meta.name, self.kind, s)

        status = get_obj_status(s)
        print(status)
        return status

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
                logger.info("to get a dashboard link, use NodePort service_type")

        return addr, dash

    @property
    def client(self):
        from dask.distributed import Client, default_client

        if self.spec.remote and not self.status.scheduler_address:
            if not self._load_db_status():
                self._start()

        if self.status.scheduler_address:
            addr, dash = self._remote_addresses()
            logger.info(f"trying dask client at: {addr}")
            try:
                client = Client(addr)
            except OSError as exc:
                logger.warning(
                    f"remote scheduler at {addr} not ready, will try to restart {exc}"
                )

                # todo: figure out if test is needed
                # if self._is_remote_api():
                #     raise Exception('no access to Kubernetes API')

                status = self.get_status()
                if status != "running":
                    self._start()
                addr, dash = self._remote_addresses()
                client = Client(addr)

            logger.info(
                f"using remote dask scheduler ({self.status.cluster_name}) at: {addr}"
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
        with_mlrun=True,
        skip_deployed=False,
        is_kfp=False,
        mlrun_version_specifier=None,
    ):
        """deploy function, build container with dependencies"""
        return super().deploy(
            watch,
            with_mlrun,
            skip_deployed,
            is_kfp=is_kfp,
            mlrun_version_specifier=mlrun_version_specifier,
        )

    def gpus(self, gpus, gpu_type="nvidia.com/gpu"):
        warnings.warn(
            "Dask's gpus will be deprecated in 0.8.0, and will be removed in 0.10.0, use "
            "with_scheduler_limits/with_worker_limits instead",
            # TODO: In 0.8.0 deprecate and replace gpus to with_worker/scheduler_limits in examples & demos (or maybe
            #  just change behavior ?)
            PendingDeprecationWarning,
        )
        # the scheduler/worker specific functions was introduced after the general one, to keep backwards compatibility
        # this function just sets the gpus for both of them
        update_in(self.spec.scheduler_resources, ["limits", gpu_type], gpus)
        update_in(self.spec.worker_resources, ["limits", gpu_type], gpus)

    def with_limits(self, mem=None, cpu=None, gpus=None, gpu_type="nvidia.com/gpu"):
        warnings.warn(
            "Dask's with_limits will be deprecated in 0.8.0, and will be removed in 0.10.0, use "
            "with_scheduler_limits/with_worker_limits instead",
            # TODO: In 0.8.0 deprecate and replace with_limits to with_worker/scheduler_limits in examples & demos (or
            #  maybe just change behavior ?)
            PendingDeprecationWarning,
        )
        # the scheduler/worker specific function was introduced after the general one, to keep backwards compatibility
        # this function just sets the limits for both of them
        self.with_scheduler_limits(mem, cpu, gpus, gpu_type)
        self.with_worker_limits(mem, cpu, gpus, gpu_type)

    def with_scheduler_limits(
        self, mem=None, cpu=None, gpus=None, gpu_type="nvidia.com/gpu"
    ):
        """set scheduler pod resources limits"""
        self._verify_and_set_limits("scheduler_resources", mem, cpu, gpus, gpu_type)

    def with_worker_limits(
        self, mem=None, cpu=None, gpus=None, gpu_type="nvidia.com/gpu"
    ):
        """set worker pod resources limits"""
        self._verify_and_set_limits("worker_resources", mem, cpu, gpus, gpu_type)

    def with_requests(self, mem=None, cpu=None):
        warnings.warn(
            "Dask's with_requests will be deprecated in 0.8.0, and will be removed in 0.10.0, use "
            "with_scheduler_requests/with_worker_requests instead",
            # TODO: In 0.8.0 deprecate and replace with_requests to with_worker/scheduler_requests in examples & demos
            #  (or maybe just change behavior ?)
            PendingDeprecationWarning,
        )
        # the scheduler/worker specific function was introduced after the general one, to keep backwards compatibility
        # this function just sets the requests for both of them
        self.with_scheduler_requests(mem, cpu)
        self.with_worker_requests(mem, cpu)

    def with_scheduler_requests(self, mem=None, cpu=None):
        """set scheduler pod resources requests"""
        self._verify_and_set_requests("scheduler_resources", mem, cpu)

    def with_worker_requests(self, mem=None, cpu=None):
        """set worker pod resources requests"""
        self._verify_and_set_requests("worker_resources", mem, cpu)

    def _run(self, runobj: RunObject, execution):

        handler = runobj.spec.handler
        self._force_handler(handler)

        extra_env = self._generate_runtime_env(runobj)
        environ.update(extra_env)

        if not inspect.isfunction(handler):
            if not self.spec.command:
                raise ValueError(
                    "specified handler (string) without command "
                    "(py file path), specify command or use handler pointer"
                )
            mod, handler = load_module(self.spec.command, handler)
        context = MLClientCtx.from_dict(
            runobj.to_dict(),
            rundb=self.spec.rundb,
            autocommit=False,
            host=socket.gethostname(),
        )
        client = self.client
        setattr(context, "dask_client", client)
        sout, serr = exec_from_params(handler, runobj, context)
        log_std(self._db_conn, runobj, sout, serr, skip=self.is_child, show=False)
        return context.to_dict()


def deploy_function(function: DaskCluster, secrets=None):

    # TODO: why is this here :|
    try:
        import dask
        from dask.distributed import Client, default_client  # noqa: F401
        from dask_kubernetes import KubeCluster, make_pod_spec  # noqa: F401
        from kubernetes_asyncio import client
    except ImportError as exc:
        print(
            "missing dask or dask_kubernetes, please run "
            '"pip install dask distributed dask_kubernetes", %s',
            exc,
        )
        raise exc

    spec = function.spec
    meta = function.metadata
    spec.remote = True

    image = function.full_image_path() or "daskdev/dask:latest"
    env = spec.env
    namespace = meta.namespace or config.namespace
    if spec.extra_pip:
        env.append(spec.extra_pip)

    pod_labels = get_resource_labels(function, scrape_metrics=config.scrape_metrics)
    worker_args = ["dask-worker", "--nthreads", str(spec.nthreads)]
    memory_limit = spec.resources.get("limits", {}).get("memory")
    if memory_limit:
        worker_args.extend(["--memory-limit", str(memory_limit)])
    if spec.args:
        worker_args.extend(spec.args)
    scheduler_args = ["dask-scheduler"]

    container_kwargs = {
        "name": "base",
        "image": image,
        "env": env,
        "image_pull_policy": spec.image_pull_policy,
        "volume_mounts": spec.volume_mounts,
    }
    scheduler_container = client.V1Container(
        resources=spec.scheduler_resources, args=scheduler_args, **container_kwargs
    )
    worker_container = client.V1Container(
        resources=spec.worker_resources, args=worker_args, **container_kwargs
    )

    scheduler_pod_spec = kube_resource_spec_to_pod_spec(spec, scheduler_container)
    worker_pod_spec = kube_resource_spec_to_pod_spec(spec, worker_container)
    for pod_spec in [scheduler_pod_spec, worker_pod_spec]:
        if spec.image_pull_secret:
            pod_spec.image_pull_secrets = [
                client.V1LocalObjectReference(name=spec.image_pull_secret)
            ]

    scheduler_pod = client.V1Pod(
        metadata=client.V1ObjectMeta(namespace=namespace, labels=pod_labels),
        # annotations=meta.annotation),
        spec=scheduler_pod_spec,
    )
    worker_pod = client.V1Pod(
        metadata=client.V1ObjectMeta(namespace=namespace, labels=pod_labels),
        # annotations=meta.annotation),
        spec=worker_pod_spec,
    )

    svc_temp = dask.config.get("kubernetes.scheduler-service-template")
    if spec.service_type or spec.node_port:
        if spec.node_port:
            spec.service_type = "NodePort"
            svc_temp["spec"]["ports"][1]["nodePort"] = spec.node_port
        update_in(svc_temp, "spec.type", spec.service_type)

    norm_name = normalize_name(meta.name)
    dask.config.set(
        {
            "kubernetes.scheduler-service-template": svc_temp,
            "kubernetes.name": "mlrun-" + norm_name + "-{uuid}",
        }
    )

    cluster = KubeCluster(
        worker_pod,
        scheduler_pod_template=scheduler_pod,
        deploy_mode="remote",
        namespace=namespace,
        idle_timeout=spec.scheduler_timeout,
    )

    logger.info(f"cluster {cluster.name} started at {cluster.scheduler_address}")

    function.status.scheduler_address = cluster.scheduler_address
    function.status.cluster_name = cluster.name
    if spec.service_type == "NodePort":
        ports = cluster.scheduler.service.spec.ports
        function.status.node_ports = {
            "scheduler": ports[0].node_port,
            "dashboard": ports[1].node_port,
        }

    if spec.replicas:
        cluster.scale(spec.replicas)
    else:
        cluster.adapt(minimum=spec.min_replicas, maximum=spec.max_replicas)

    return cluster


def get_obj_status(selector=[], namespace=None):
    k8s = get_k8s_helper()
    namespace = namespace or config.namespace
    selector = ",".join(["dask.org/component=scheduler"] + selector)
    pods = k8s.list_pods(namespace, selector=selector)
    status = ""
    for pod in pods:
        status = pod.status.phase.lower()
        print(pod)
        if status == "running":
            cluster = pod.metadata.labels.get("dask.org/cluster-name")
            logger.info(
                f"found running dask function {pod.metadata.name}, cluster={cluster}"
            )
            return status
        logger.info(
            f"found dask function {pod.metadata.name} in non ready state ({status})"
        )
    return status


class DaskRuntimeHandler(BaseRuntimeHandler):

    # Dask runtime resources are per function (and not per run).
    # It means that monitoring runtime resources state doesn't say anything about the run state.
    # Therefore dask run monitoring is done completely by the SDK, so overriding the monitoring method with no logic
    def monitor_runs(
        self, db: DBInterface, db_session: Session, leader_session: Optional[str] = None
    ):
        return

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f"mlrun/function={object_id}"

    @staticmethod
    def _get_possible_mlrun_class_label_values() -> List[str]:
        return ["dask"]

    def _enrich_list_resources_response(
        self,
        response: Union[
            mlrun.api.schemas.RuntimeResources,
            mlrun.api.schemas.GroupedByJobRuntimeResourcesOutput,
            mlrun.api.schemas.GroupedByProjectRuntimeResourcesOutput,
        ],
        namespace: str,
        label_selector: str = None,
        group_by: Optional[mlrun.api.schemas.ListRuntimeResourcesGroupByField] = None,
    ) -> Union[
        mlrun.api.schemas.RuntimeResources,
        mlrun.api.schemas.GroupedByJobRuntimeResourcesOutput,
        mlrun.api.schemas.GroupedByProjectRuntimeResourcesOutput,
    ]:
        """
        Handling listing service resources
        """
        enrich_needed = self._validate_if_enrich_is_needed_by_group_by(group_by)
        if not enrich_needed:
            return response
        k8s_helper = get_k8s_helper()
        services = k8s_helper.v1api.list_namespaced_service(
            namespace, label_selector=label_selector
        )
        service_resources = []
        for service in services.items:
            service_resources.append(
                mlrun.api.schemas.RuntimeResource(
                    name=service.metadata.name, labels=service.metadata.labels
                )
            )
        return self._enrich_service_resources_in_response(
            response, service_resources, group_by
        )

    def _build_output_from_runtime_resources(
        self,
        response: Union[
            mlrun.api.schemas.RuntimeResources,
            mlrun.api.schemas.GroupedByJobRuntimeResourcesOutput,
            mlrun.api.schemas.GroupedByProjectRuntimeResourcesOutput,
        ],
        runtime_resources_list: List[mlrun.api.schemas.RuntimeResources],
        group_by: Optional[mlrun.api.schemas.ListRuntimeResourcesGroupByField] = None,
    ):
        enrich_needed = self._validate_if_enrich_is_needed_by_group_by(group_by)
        if not enrich_needed:
            return response
        service_resources = []
        for runtime_resources in runtime_resources_list:
            service_resources += runtime_resources.service_resources
        return self._enrich_service_resources_in_response(
            response, service_resources, group_by
        )

    def _validate_if_enrich_is_needed_by_group_by(
        self,
        group_by: Optional[mlrun.api.schemas.ListRuntimeResourcesGroupByField] = None,
    ) -> bool:
        # Dask runtime resources are per function (and not per job) therefore, when grouping by job we're simply
        # omitting the dask runtime resources
        if group_by == mlrun.api.schemas.ListRuntimeResourcesGroupByField.job:
            return False
        elif group_by == mlrun.api.schemas.ListRuntimeResourcesGroupByField.project:
            return True
        elif group_by is not None:
            raise NotImplementedError(
                f"Provided group by field is not supported. group_by={group_by}"
            )
        return True

    def _enrich_service_resources_in_response(
        self,
        response: Union[
            mlrun.api.schemas.RuntimeResources,
            mlrun.api.schemas.GroupedByJobRuntimeResourcesOutput,
            mlrun.api.schemas.GroupedByProjectRuntimeResourcesOutput,
        ],
        service_resources: List[mlrun.api.schemas.RuntimeResource],
        group_by: Optional[mlrun.api.schemas.ListRuntimeResourcesGroupByField] = None,
    ):
        if group_by == mlrun.api.schemas.ListRuntimeResourcesGroupByField.project:
            for service_resource in service_resources:
                self._add_resource_to_grouped_by_project_resources_response(
                    response, "service_resources", service_resource
                )
        else:
            response.service_resources = service_resources
        return response

    def _delete_resources(
        self,
        db: DBInterface,
        db_session: Session,
        namespace: str,
        deleted_resources: List[Dict],
        label_selector: str = None,
        force: bool = False,
        grace_period: int = None,
    ):
        """
        Handling services deletion
        """
        if grace_period is None:
            grace_period = config.runtime_resources_deletion_grace_period
        service_names = []
        for pod_dict in deleted_resources:
            dask_component = (
                pod_dict["metadata"].get("labels", {}).get("dask.org/component")
            )
            cluster_name = (
                pod_dict["metadata"].get("labels", {}).get("dask.org/cluster-name")
            )
            if dask_component == "scheduler" and cluster_name:
                service_names.append(cluster_name)

        k8s_helper = get_k8s_helper()
        services = k8s_helper.v1api.list_namespaced_service(
            namespace, label_selector=label_selector
        )
        for service in services.items:
            try:
                if force or service.metadata.name in service_names:
                    k8s_helper.v1api.delete_namespaced_service(
                        service.metadata.name, namespace
                    )
                    logger.info(f"Deleted service: {service.metadata.name}")
            except ApiException as exc:
                # ignore error if service is already removed
                if exc.status != 404:
                    raise
