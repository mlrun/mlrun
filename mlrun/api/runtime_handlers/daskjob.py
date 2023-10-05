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
#
import typing
from typing import Dict, List, Optional, Union

from kubernetes.client.rest import ApiException
from sqlalchemy.orm import Session

import mlrun.api.utils.singletons.k8s
import mlrun.common.schemas
import mlrun.errors
import mlrun.k8s_utils
import mlrun.runtimes
import mlrun.runtimes.pod
import mlrun.utils
import mlrun.utils.regex
from mlrun.api.db.base import DBInterface
from mlrun.api.runtime_handlers.base import BaseRuntimeHandler
from mlrun.config import config
from mlrun.runtimes.base import RuntimeClassMode
from mlrun.runtimes.utils import get_k8s, get_resource_labels
from mlrun.utils import logger


def get_dask_resource():
    return {
        "scope": "function",
        "start": deploy_function,
        "status": get_obj_status,
    }


class DaskRuntimeHandler(BaseRuntimeHandler):
    kind = "dask"
    class_modes = {RuntimeClassMode.run: "dask"}

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
    def resolve_object_id(
        run: dict,
    ) -> typing.Optional[str]:
        """
        Resolves the object ID from the run object.
        In dask runtime, the object ID is the function name.
        :param run: run object
        :return: function name
        """

        function = run.get("spec", {}).get("function", None)
        if function:

            # a dask run's function field is in the format <project-name>/<function-name>@<run-uid>
            # we only want the function name
            project_and_function = function.split("@")[0]
            return project_and_function.split("/")[-1]

        return None

    def _enrich_list_resources_response(
        self,
        response: Union[
            mlrun.common.schemas.RuntimeResources,
            mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
            mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
        ],
        namespace: str,
        label_selector: str = None,
        group_by: Optional[
            mlrun.common.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ) -> Union[
        mlrun.common.schemas.RuntimeResources,
        mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
        mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
    ]:
        """
        Handling listing service resources
        """
        enrich_needed = self._validate_if_enrich_is_needed_by_group_by(group_by)
        if not enrich_needed:
            return response
        services = get_k8s().v1api.list_namespaced_service(
            namespace, label_selector=label_selector
        )
        service_resources = []
        for service in services.items:
            service_resources.append(
                mlrun.common.schemas.RuntimeResource(
                    name=service.metadata.name, labels=service.metadata.labels
                )
            )
        return self._enrich_service_resources_in_response(
            response, service_resources, group_by
        )

    def _build_output_from_runtime_resources(
        self,
        response: Union[
            mlrun.common.schemas.RuntimeResources,
            mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
            mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
        ],
        runtime_resources_list: List[mlrun.common.schemas.RuntimeResources],
        group_by: Optional[
            mlrun.common.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ):
        enrich_needed = self._validate_if_enrich_is_needed_by_group_by(group_by)
        if not enrich_needed:
            return response
        service_resources = []
        for runtime_resources in runtime_resources_list:
            if runtime_resources.service_resources:
                service_resources += runtime_resources.service_resources
        return self._enrich_service_resources_in_response(
            response, service_resources, group_by
        )

    def _validate_if_enrich_is_needed_by_group_by(
        self,
        group_by: Optional[
            mlrun.common.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ) -> bool:
        # Dask runtime resources are per function (and not per job) therefore, when grouping by job we're simply
        # omitting the dask runtime resources
        if group_by == mlrun.common.schemas.ListRuntimeResourcesGroupByField.job:
            return False
        elif group_by == mlrun.common.schemas.ListRuntimeResourcesGroupByField.project:
            return True
        elif group_by is not None:
            raise NotImplementedError(
                f"Provided group by field is not supported. group_by={group_by}"
            )
        return True

    def _enrich_service_resources_in_response(
        self,
        response: Union[
            mlrun.common.schemas.RuntimeResources,
            mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
            mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
        ],
        service_resources: List[mlrun.common.schemas.RuntimeResource],
        group_by: Optional[
            mlrun.common.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ):
        if group_by == mlrun.common.schemas.ListRuntimeResourcesGroupByField.project:
            for service_resource in service_resources:
                self._add_resource_to_grouped_by_project_resources_response(
                    response, "service_resources", service_resource
                )
        else:
            response.service_resources = service_resources
        return response

    def _delete_extra_resources(
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

        services = get_k8s().v1api.list_namespaced_service(
            namespace, label_selector=label_selector
        )
        for service in services.items:
            try:
                if force or service.metadata.name in service_names:
                    get_k8s().v1api.delete_namespaced_service(
                        service.metadata.name, namespace
                    )
                    logger.info(f"Deleted service: {service.metadata.name}")
            except ApiException as exc:
                # ignore error if service is already removed
                if exc.status != 404:
                    raise


def deploy_function(
    function: mlrun.runtimes.DaskCluster,
    secrets=None,
    client_version: str = None,
    client_python_version: str = None,
):
    _validate_dask_related_libraries_installed()

    scheduler_pod, worker_pod, function, namespace = enrich_dask_cluster(
        function, secrets, client_version, client_python_version
    )
    return initialize_dask_cluster(scheduler_pod, worker_pod, function, namespace)


def initialize_dask_cluster(scheduler_pod, worker_pod, function, namespace):
    import dask
    import dask_kubernetes

    spec, meta = function.spec, function.metadata

    svc_temp = dask.config.get("kubernetes.scheduler-service-template")
    if spec.service_type or spec.node_port:
        if spec.node_port:
            spec.service_type = "NodePort"
            svc_temp["spec"]["ports"][1]["nodePort"] = spec.node_port
        mlrun.utils.update_in(svc_temp, "spec.type", spec.service_type)

    dask.config.set(
        {
            "kubernetes.scheduler-service-template": svc_temp,
            "kubernetes.name": f"mlrun-{mlrun.utils.normalize_name(meta.name)}-{{uuid}}",
            # 5 minutes, to resiliently handle delicate/slow k8s clusters
            "kubernetes.scheduler-service-wait-timeout": 60 * 5,
            "distributed.comm.timeouts.connect": "300s",
        }
    )

    cluster = dask_kubernetes.KubeCluster(
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


def enrich_dask_cluster(
    function, secrets, client_version: str = None, client_python_version: str = None
):
    from dask.distributed import Client, default_client  # noqa: F401
    from dask_kubernetes import KubeCluster, make_pod_spec  # noqa: F401
    from kubernetes import client

    # Is it possible that the function will not have a project at this point?
    if function.metadata.project:
        function._add_secrets_to_spec_before_running(project=function.metadata.project)

    spec = function.spec
    meta = function.metadata
    spec.remote = True

    image = (
        function.full_image_path(
            client_version=client_version, client_python_version=client_python_version
        )
        # TODO: we might never enter here, since running a function requires defining an image
        or "daskdev/dask:latest"
    )
    env = spec.env
    env.extend(function.generate_runtime_k8s_env())
    namespace = meta.namespace or config.namespace
    if spec.extra_pip:
        env.append(spec.extra_pip)

    pod_labels = get_resource_labels(function, scrape_metrics=config.scrape_metrics)
    # TODO: 'dask-worker' is deprecated, new dask CLI was introduced in 2022.10.0.
    #  Upgrade when we drop python 3.7 support and use 'dask worker' instead
    worker_args = ["dask-worker", "--nthreads", str(spec.nthreads)]
    memory_limit = spec.worker_resources.get("limits", {}).get("memory")
    if memory_limit:
        worker_args.extend(["--memory-limit", str(memory_limit)])
    if spec.args:
        worker_args.extend(spec.args)
    # TODO: 'dask-scheduler' is deprecated, new dask CLI was introduced in 2022.10.0.
    #  Upgrade when we drop python 3.7 support and use 'dask scheduler' instead
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

    scheduler_pod_spec = mlrun.runtimes.pod.kube_resource_spec_to_pod_spec(
        spec, scheduler_container
    )
    worker_pod_spec = mlrun.runtimes.pod.kube_resource_spec_to_pod_spec(
        spec, worker_container
    )
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
    return scheduler_pod, worker_pod, function, namespace


def _validate_dask_related_libraries_installed():
    try:
        import dask  # noqa: F401
        from dask.distributed import Client, default_client  # noqa: F401
        from dask_kubernetes import KubeCluster, make_pod_spec  # noqa: F401
        from kubernetes import client  # noqa: F401
    except ImportError as exc:
        print(
            "missing dask or dask_kubernetes, please run "
            '"pip install dask distributed dask_kubernetes", %s',
            exc,
        )
        raise exc


def get_obj_status(selector=None, namespace=None):
    if selector is None:
        selector = []

    k8s = mlrun.api.utils.singletons.k8s.get_k8s_helper()
    namespace = namespace or config.namespace
    selector = ",".join(["dask.org/component=scheduler"] + selector)
    pods = k8s.list_pods(namespace, selector=selector)
    status = ""
    for pod in pods:
        status = pod.status.phase.lower()
        if status == "running":
            cluster = pod.metadata.labels.get("dask.org/cluster-name")
            logger.info(
                "Found running dask function",
                pod_name=pod.metadata.name,
                cluster=cluster,
            )
            return status
        logger.info(
            "Found dask function in non ready state",
            pod_name=pod.metadata.name,
            status=status,
        )
    return status
