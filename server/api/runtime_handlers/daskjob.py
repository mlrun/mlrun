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
from typing import Optional, Union

import semver
from kubernetes.client.rest import ApiException
from sqlalchemy.orm import Session

import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import mlrun.errors
import mlrun.k8s_utils
import mlrun.runtimes
import mlrun.runtimes.pod
import mlrun.utils
import mlrun.utils.regex
import server.api.utils.singletons.k8s
from mlrun.config import config
from mlrun.runtimes.base import RuntimeClassMode
from mlrun.utils import logger
from server.api.common.runtime_handlers import get_resource_labels
from server.api.db.base import DBInterface
from server.api.runtime_handlers import BaseRuntimeHandler


def get_dask_resource():
    return {
        "scope": "function",
        "start": deploy_function,
        "status": get_obj_status,
    }


class DaskRuntimeHandler(BaseRuntimeHandler):
    kind = "dask"
    class_modes = {RuntimeClassMode.run: "dask"}

    def run(
        self,
        runtime: mlrun.runtimes.BaseRuntime,
        run: mlrun.run.RunObject,
        execution: mlrun.execution.MLClientCtx,
    ):
        raise NotImplementedError(
            "Execution of dask jobs is done locally by the dask client"
        )

    # Dask runtime resources are per function (and not per run).
    # It means that monitoring runtime resources state doesn't say anything about the run state.
    # Therefore, dask run monitoring is done completely by the SDK, so overriding the monitoring method with no logic
    def monitor_runs(
        self, db: DBInterface, db_session: Session, leader_session: Optional[str] = None
    ) -> list[dict]:
        return []

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
        services = server.api.utils.singletons.k8s.get_k8s_helper().v1api.list_namespaced_service(
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
        runtime_resources_list: list[mlrun.common.schemas.RuntimeResources],
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
        service_resources: list[mlrun.common.schemas.RuntimeResource],
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
        deleted_resources: list[dict],
        label_selector: str = None,
        force: bool = False,
        grace_period: int = None,
        resource_deletion_grace_period: typing.Optional[int] = None,
    ):
        """
        Handling services deletion
        """
        service_names = []
        for pod_dict in deleted_resources:
            dask_component = (
                pod_dict["metadata"]
                .get("labels", {})
                .get(mlrun_constants.MLRunInternalLabels.dask_component)
            )
            cluster_name = (
                pod_dict["metadata"]
                .get("labels", {})
                .get(mlrun_constants.MLRunInternalLabels.dask_cluster_name)
            )
            if dask_component == "scheduler" and cluster_name:
                service_names.append(cluster_name)

        services = server.api.utils.singletons.k8s.get_k8s_helper().v1api.list_namespaced_service(
            namespace, label_selector=label_selector
        )
        for service in services.items:
            try:
                if force or service.metadata.name in service_names:
                    server.api.utils.singletons.k8s.get_k8s_helper().v1api.delete_namespaced_service(
                        service.metadata.name,
                        namespace,
                        grace_period_seconds=resource_deletion_grace_period,
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
            "distributed.comm.retry.count": 10,
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

    runtime_handler: server.api.runtime_handlers.daskjob.DaskRuntimeHandler = (
        server.api.runtime_handlers.DaskRuntimeHandler()
    )
    runtime_handler.add_secrets_to_spec_before_running(
        runtime=function, project_name=function.metadata.project
    )

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
    env = function.generate_runtime_k8s_env()

    # filter any spec.env that already exists in env
    # in other words, dont let spec.env override env (or not even duplicate it)
    # we dont want to override env to ensure k8s runtime envs are enforced and correct
    # leaving no room for human mistakes
    def get_env_name(env_: Union[client.V1EnvVar, dict]) -> str:
        if isinstance(env_, client.V1EnvVar):
            return env_.name
        return env_.get("name", "")

    env.extend(
        filter(
            lambda spec_env: not any(
                [
                    True
                    for _env in env
                    # spec_env might be V1EnvVar or a dict
                    # _env is just a dict
                    if get_env_name(spec_env) == get_env_name(_env)
                ]
            ),
            spec.env,
        )
    )

    namespace = meta.namespace or config.namespace
    if spec.extra_pip:
        env.append(spec.extra_pip)

    # remove duplicates by name
    env = list({get_env_name(spec_env): spec_env for spec_env in env}.values())

    pod_labels = get_resource_labels(function, scrape_metrics=config.scrape_metrics)

    worker_args = ["dask", "worker"]
    scheduler_args = ["dask", "scheduler"]
    # before mlrun 1.6.0, mlrun required a dask version that was not compatible with the new dask CLI
    # this assumes that the dask client version matches the dask cluster version
    is_legacy_dask = False
    try:
        is_legacy_dask = client_version and semver.VersionInfo.parse(
            client_version
        ) < semver.VersionInfo.parse("1.6.0-X")
    except ValueError:
        pass

    if is_legacy_dask:
        worker_args = ["dask-worker"]
        scheduler_args = ["dask-scheduler"]

    worker_args.extend(["--nthreads", str(spec.nthreads)])
    memory_limit = spec.worker_resources.get("limits", {}).get("memory")
    if memory_limit:
        worker_args.extend(["--memory-limit", str(memory_limit)])
    if spec.args:
        worker_args.extend(spec.args)

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

    # We query the project to enrich the worker and scheduler pod spec with the project's default node selector.
    # Since the dask runtime is a local run, and does not run in a dedicated k8s pod, node selectors for that run
    # are irrelevant, so we do not enrich the run object with the project node selector.
    # However, the node selector is still relevant for the Dask cluster's workers and scheduler, which do run
    # remotely on k8s. This ensures that the cluster pods follow the project's specified node selection.
    project = function._get_db().get_project(function.metadata.project)
    logger.debug(
        "Enriching Dask Cluster node selector from project and mlrun config",
        project_name=function.metadata.project,
        project_node_selector=project.spec.default_function_node_selector,
        mlconf_node_selector=mlrun.mlconf.get_default_function_node_selector(),
    )
    node_selector = mlrun.utils.helpers.merge_dicts_with_precedence(
        mlrun.mlconf.get_default_function_node_selector(),
        project.spec.default_function_node_selector,
        function.spec.node_selector,
    )
    scheduler_pod_spec = server.api.utils.singletons.k8s.kube_resource_spec_to_pod_spec(
        spec, scheduler_container, node_selector=node_selector
    )
    worker_pod_spec = server.api.utils.singletons.k8s.kube_resource_spec_to_pod_spec(
        spec, worker_container, node_selector=node_selector
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

    k8s = server.api.utils.singletons.k8s.get_k8s_helper()
    namespace = namespace or config.namespace
    selector = ",".join(
        [f"{mlrun_constants.MLRunInternalLabels.dask_component}=scheduler"] + selector
    )
    pods = k8s.list_pods(namespace, selector=selector)
    status = ""
    for pod in pods:
        status = pod.status.phase.lower()
        if status == "running":
            cluster = pod.metadata.labels.get(
                mlrun_constants.MLRunInternalLabels.dask_cluster_name
            )
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
