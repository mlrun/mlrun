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
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union

from kubernetes.client.rest import ApiException
from sqlalchemy.orm import Session

import mlrun.common.schemas
import mlrun.errors
import mlrun.launcher.factory
import mlrun.utils.helpers
import mlrun.utils.notifications
import mlrun.utils.regex
from mlrun.api.constants import LogSources
from mlrun.api.db.base import DBInterface
from mlrun.config import config
from mlrun.errors import err_to_str
from mlrun.runtimes import RuntimeClassMode
from mlrun.runtimes.constants import PodPhases, RunStates
from mlrun.runtimes.utils import get_k8s
from mlrun.utils import logger, now_date


class BaseRuntimeHandler(ABC):
    # setting here to allow tests to override
    kind = "base"
    class_modes: Dict[RuntimeClassMode, str] = {}
    wait_for_deletion_interval = 10
    pod_grace_period_seconds = 0

    @staticmethod
    @abstractmethod
    def _get_object_label_selector(object_id: str) -> str:
        """
        Should return the label selector to get only resources of a specific object (with id object_id)
        """
        pass

    def _should_collect_logs(self) -> bool:
        """
        There are some runtimes which we don't collect logs for using the log collector
        :return: whether it should collect log for it
        """
        return True

    def _get_possible_mlrun_class_label_values(
        self, class_mode: Union[RuntimeClassMode, str] = None
    ) -> List[str]:
        """
        Should return the possible values of the mlrun/class label for runtime resources that are of this runtime
        handler kind
        """
        if not class_mode:
            return list(self.class_modes.values())
        class_mode = self.class_modes.get(class_mode, None)
        return [class_mode] if class_mode else []

    def list_resources(
        self,
        project: str,
        object_id: Optional[str] = None,
        label_selector: str = None,
        group_by: Optional[
            mlrun.common.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ) -> Union[
        mlrun.common.schemas.RuntimeResources,
        mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
        mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
    ]:
        # We currently don't support removing runtime resources in non k8s env
        if not get_k8s().is_running_inside_kubernetes_cluster():
            return {}
        namespace = get_k8s().resolve_namespace()
        label_selector = self.resolve_label_selector(project, object_id, label_selector)
        pods = self._list_pods(namespace, label_selector)
        pod_resources = self._build_pod_resources(pods)
        crd_objects = self._list_crd_objects(namespace, label_selector)
        crd_resources = self._build_crd_resources(crd_objects)
        response = self._build_list_resources_response(
            pod_resources, crd_resources, group_by
        )
        response = self._enrich_list_resources_response(
            response, namespace, label_selector, group_by
        )
        return response

    def build_output_from_runtime_resources(
        self,
        runtime_resources_list: List[mlrun.common.schemas.RuntimeResources],
        group_by: Optional[
            mlrun.common.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ):
        pod_resources = []
        crd_resources = []
        for runtime_resources in runtime_resources_list:
            pod_resources += runtime_resources.pod_resources
            crd_resources += runtime_resources.crd_resources
        response = self._build_list_resources_response(
            pod_resources, crd_resources, group_by
        )
        response = self._build_output_from_runtime_resources(
            response, runtime_resources_list, group_by
        )
        return response

    def delete_resources(
        self,
        db: DBInterface,
        db_session: Session,
        label_selector: str = None,
        force: bool = False,
        grace_period: int = None,
    ):
        if grace_period is None:
            grace_period = config.runtime_resources_deletion_grace_period
        # We currently don't support removing runtime resources in non k8s env
        if not get_k8s().is_running_inside_kubernetes_cluster():
            return
        namespace = get_k8s().resolve_namespace()
        label_selector = self.resolve_label_selector("*", label_selector=label_selector)
        crd_group, crd_version, crd_plural = self._get_crd_info()
        if crd_group and crd_version and crd_plural:
            deleted_resources = self._delete_crd_resources(
                db,
                db_session,
                namespace,
                label_selector,
                force,
                grace_period,
            )
        else:
            deleted_resources = self._delete_pod_resources(
                db,
                db_session,
                namespace,
                label_selector,
                force,
                grace_period,
            )
        self._delete_extra_resources(
            db,
            db_session,
            namespace,
            deleted_resources,
            label_selector,
            force,
            grace_period,
        )

    def delete_runtime_object_resources(
        self,
        db: DBInterface,
        db_session: Session,
        object_id: str,
        label_selector: str = None,
        force: bool = False,
        grace_period: int = None,
    ):
        if grace_period is None:
            grace_period = config.runtime_resources_deletion_grace_period
        label_selector = self._add_object_label_selector_if_needed(
            object_id, label_selector
        )
        self.delete_resources(db, db_session, label_selector, force, grace_period)

    def monitor_runs(self, db: DBInterface, db_session: Session):
        namespace = get_k8s().resolve_namespace()
        label_selector = self._get_default_label_selector()
        crd_group, crd_version, crd_plural = self._get_crd_info()
        runtime_resource_is_crd = False
        if crd_group and crd_version and crd_plural:
            runtime_resource_is_crd = True
            runtime_resources = self._list_crd_objects(namespace, label_selector)
        else:
            runtime_resources = self._list_pods(namespace, label_selector)
        project_run_uid_map = self._list_runs_for_monitoring(db, db_session)
        # project -> uid -> {"name": <runtime-resource-name>}
        run_runtime_resources_map = {}
        for runtime_resource in runtime_resources:
            project, uid, name = self._resolve_runtime_resource_run(runtime_resource)
            run_runtime_resources_map.setdefault(project, {})
            run_runtime_resources_map.get(project).update({uid: {"name": name}})
            try:
                self._monitor_runtime_resource(
                    db,
                    db_session,
                    project_run_uid_map,
                    runtime_resource,
                    runtime_resource_is_crd,
                    namespace,
                    project,
                    uid,
                    name,
                )
            except Exception as exc:
                logger.warning(
                    "Failed monitoring runtime resource. Continuing",
                    runtime_resource_name=runtime_resource["metadata"]["name"],
                    project_name=project,
                    namespace=namespace,
                    exc=err_to_str(exc),
                    traceback=traceback.format_exc(),
                )
        for project, runs in project_run_uid_map.items():
            if runs:
                for run_uid, run in runs.items():
                    try:
                        if not run:
                            run = db.read_run(db_session, run_uid, project)
                        if self.kind == run.get("metadata", {}).get("labels", {}).get(
                            "kind", ""
                        ):
                            self._ensure_run_not_stuck_on_non_terminal_state(
                                db,
                                db_session,
                                project,
                                run_uid,
                                run,
                                run_runtime_resources_map,
                            )
                    except Exception as exc:
                        logger.warning(
                            "Failed ensuring run not stuck. Continuing",
                            run_uid=run_uid,
                            run=run,
                            project=project,
                            exc=err_to_str(exc),
                            traceback=traceback.format_exc(),
                        )

    def _ensure_run_not_stuck_on_non_terminal_state(
        self,
        db: DBInterface,
        db_session: Session,
        project: str,
        run_uid: str,
        run: dict = None,
        run_runtime_resources_map: dict = None,
    ):
        """
        Ensuring that a run does not become trapped in a non-terminal state as a result of not finding
        corresponding k8s resource.
        This can occur when a node is evicted or preempted, causing the resources to be removed from the resource
        listing when the final state recorded in the database is non-terminal.
        This will have a significant impact on scheduled jobs, since they will not be created until the
        previous run reaches a terminal state (because of concurrency limit)
        """
        now = now_date()
        db_run_state = run.get("status", {}).get("state")
        if not db_run_state:
            # we are setting the run state to a terminal state to avoid log spamming, this is mainly sanity as we are
            # setting state to runs when storing new runs.
            logger.info(
                "Runs monitoring found a run without state, updating to a terminal state",
                project=project,
                uid=run_uid,
                db_run_state=db_run_state,
                now=now,
            )
            run.setdefault("status", {})["state"] = RunStates.error
            run.setdefault("status", {})["last_update"] = now.isoformat()
            db.store_run(db_session, run, run_uid, project)
            return
        if db_run_state in RunStates.non_terminal_states():
            if run_runtime_resources_map and run_uid in run_runtime_resources_map.get(
                project, {}
            ):
                # if found resource there is no need to continue
                return
            last_update_str = run.get("status", {}).get("last_update")
            debounce_period = (
                config.resolve_runs_monitoring_missing_runtime_resources_debouncing_interval()
            )
            if last_update_str is None:
                logger.info(
                    "Runs monitoring found run in non-terminal state without last update time set, "
                    "updating last update time to now, to be able to evaluate next time if something changed",
                    project=project,
                    uid=run_uid,
                    db_run_state=db_run_state,
                    now=now,
                    debounce_period=debounce_period,
                )
                run.setdefault("status", {})["last_update"] = now.isoformat()
                db.store_run(db_session, run, run_uid, project)
                return

            if datetime.fromisoformat(last_update_str) > now - timedelta(
                seconds=debounce_period
            ):
                # we are setting non-terminal states to runs before the run is actually applied to k8s, meaning there is
                # a timeframe where the run exists and no runtime resources exist and it's ok, therefore we're applying
                # a debounce period before setting the state to error
                logger.warning(
                    "Monitoring did not discover a runtime resource that corresponded to a run in a "
                    "non-terminal state. but record has recently updated. Debouncing",
                    project=project,
                    uid=run_uid,
                    db_run_state=db_run_state,
                    last_update=datetime.fromisoformat(last_update_str),
                    now=now,
                    debounce_period=debounce_period,
                )
            else:
                logger.info(
                    "Updating run state", run_uid=run_uid, run_state=RunStates.error
                )
                run.setdefault("status", {})["state"] = RunStates.error
                run.setdefault("status", {})[
                    "reason"
                ] = "A runtime resource related to this run could not be found"
                run.setdefault("status", {})["last_update"] = now.isoformat()
                db.store_run(db_session, run, run_uid, project)

    def _add_object_label_selector_if_needed(
        self,
        object_id: Optional[str] = None,
        label_selector: Optional[str] = None,
    ):
        if object_id:
            object_label_selector = self._get_object_label_selector(object_id)
            if label_selector:
                label_selector = ",".join([object_label_selector, label_selector])
            else:
                label_selector = object_label_selector
        return label_selector

    @staticmethod
    def _get_main_runtime_resource_label_selector() -> str:
        """
        There are some runtimes which might have multiple k8s resources attached to a one runtime, in this case
        we don't want to pull logs from all but rather only for the "driver"/"launcher" etc
        :return: the label selector
        """
        return ""

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
        Override this to list resources other then pods or CRDs (which are handled by the base class)
        """
        return response

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
        """
        Override this to add runtime resources other than pods or CRDs (which are handled by the base class) to the
        output
        """
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
        Override this to handle deletion of resources other than pods or CRDs (which are handled by the base class)
        Note that this is happening after the deletion of the CRDs or the pods
        Note to add this at the beginning:
        if grace_period is None:
            grace_period = config.runtime_resources_deletion_grace_period
        """
        pass

    def _resolve_crd_object_status_info(
        self, db: DBInterface, db_session: Session, crd_object
    ) -> Tuple[bool, Optional[datetime], Optional[str]]:
        """
        Override this if the runtime has CRD resources.
        :return: Tuple with:
        1. bool determining whether the crd object is in terminal state
        2. datetime of when the crd object got into terminal state (only when the crd object in terminal state)
        3. the desired run state matching the crd object state
        """
        return False, None, None

    def _update_ui_url(
        self,
        db: DBInterface,
        db_session: Session,
        project: str,
        uid: str,
        crd_object,
        run: Dict,
    ):
        """
        Update the UI URL for relevant jobs.
        """
        pass

    def _resolve_pod_status_info(
        self, db: DBInterface, db_session: Session, pod: Dict
    ) -> Tuple[bool, Optional[datetime], Optional[str]]:
        """
        :return: Tuple with:
        1. bool determining whether the pod is in terminal state
        2. datetime of when the pod got into terminal state (only when the pod in terminal state)
        3. the run state matching the pod state
        """
        in_terminal_state = pod["status"]["phase"] in PodPhases.terminal_phases()
        run_state = PodPhases.pod_phase_to_run_state(pod["status"]["phase"])
        last_container_completion_time = None
        if in_terminal_state:
            for container_status in pod["status"].get("container_statuses", []):
                if container_status.get("state", {}).get("terminated"):
                    container_completion_time = container_status["state"][
                        "terminated"
                    ].get("finished_at")

                    # take latest completion time
                    if (
                        not last_container_completion_time
                        or last_container_completion_time < container_completion_time
                    ):
                        last_container_completion_time = container_completion_time

        return in_terminal_state, last_container_completion_time, run_state

    def _get_default_label_selector(
        self, class_mode: Union[RuntimeClassMode, str] = None
    ) -> str:
        """
        Override this to add a default label selector
        """
        class_values = self._get_possible_mlrun_class_label_values(class_mode)
        if not class_values:
            return ""
        if len(class_values) == 1:
            return f"mlrun/class={class_values[0]}"
        return f"mlrun/class in ({', '.join(class_values)})"

    @staticmethod
    def _get_crd_info() -> Tuple[str, str, str]:
        """
        Override this if the runtime has CRD resources. this should return the CRD info:
        crd group, crd version, crd plural
        """
        return "", "", ""

    @staticmethod
    def _are_resources_coupled_to_run_object() -> bool:
        """
        Some resources are tightly coupled to mlrun Run object, for example, for each Run of a Function of the job kind
        a kubernetes job is being generated, on the opposite a Function of the daskjob kind generates a dask cluster,
        and every Run is being executed using this cluster, i.e. no resources are created for the Run.
        This function should return true for runtimes in which Run are coupled to the underlying resources and therefore
        aspects of the Run (like its state) should be taken into consideration on resources deletion
        """
        return False

    @staticmethod
    def _expect_pods_without_uid() -> bool:
        return False

    def _list_pods(self, namespace: str, label_selector: str = None) -> List:
        pods = get_k8s().list_pods(namespace, selector=label_selector)
        # when we work with custom objects (list_namespaced_custom_object) it's always a dict, to be able to generalize
        # code working on runtime resource (either a custom object or a pod) we're transforming to dicts
        pods = [pod.to_dict() for pod in pods]
        return pods

    def _list_crd_objects(self, namespace: str, label_selector: str = None) -> List:
        crd_group, crd_version, crd_plural = self._get_crd_info()
        crd_objects = []
        if crd_group and crd_version and crd_plural:
            try:
                crd_objects = get_k8s().crdapi.list_namespaced_custom_object(
                    crd_group,
                    crd_version,
                    namespace,
                    crd_plural,
                    label_selector=label_selector,
                )
            except ApiException as exc:
                # ignore error if crd is not defined
                if exc.status != 404:
                    raise
            else:
                crd_objects = crd_objects["items"]
        return crd_objects

    def resolve_label_selector(
        self,
        project: str,
        object_id: Optional[str] = None,
        label_selector: Optional[str] = None,
        class_mode: Union[RuntimeClassMode, str] = None,
        with_main_runtime_resource_label_selector: bool = False,
    ) -> str:
        default_label_selector = self._get_default_label_selector(class_mode=class_mode)

        if label_selector:
            label_selector = ",".join([default_label_selector, label_selector])
        else:
            label_selector = default_label_selector

        if project and project != "*":
            label_selector = ",".join([label_selector, f"mlrun/project={project}"])

        label_selector = self._add_object_label_selector_if_needed(
            object_id, label_selector
        )

        if with_main_runtime_resource_label_selector:
            main_runtime_resource_label_selector = (
                self._get_main_runtime_resource_label_selector()
            )
            if main_runtime_resource_label_selector:
                label_selector = ",".join(
                    [label_selector, main_runtime_resource_label_selector]
                )

        return label_selector

    @staticmethod
    def resolve_object_id(
        run: dict,
    ) -> Optional[str]:
        """
        Get the object id from the run object
        Override this if the object id is not the run uid
        :param run: run object
        :return: object id
        """
        return run.get("metadata", {}).get("uid", None)

    def _wait_for_pods_deletion(
        self,
        namespace: str,
        deleted_pods: List[Dict],
        label_selector: str = None,
    ):
        deleted_pod_names = [pod_dict["metadata"]["name"] for pod_dict in deleted_pods]

        def _verify_pods_removed():
            pods = get_k8s().v1api.list_namespaced_pod(
                namespace, label_selector=label_selector
            )
            existing_pod_names = [pod.metadata.name for pod in pods.items]
            still_in_deletion_pods = set(existing_pod_names).intersection(
                deleted_pod_names
            )
            if still_in_deletion_pods:
                raise RuntimeError(
                    f"Pods are still in deletion process: {still_in_deletion_pods}"
                )

        if deleted_pod_names:
            timeout = 180
            logger.debug(
                "Waiting for pods deletion",
                timeout=timeout,
                interval=self.wait_for_deletion_interval,
            )
            mlrun.utils.retry_until_successful(
                self.wait_for_deletion_interval,
                timeout,
                logger,
                True,
                _verify_pods_removed,
            )

    def _wait_for_crds_underlying_pods_deletion(
        self,
        deleted_crds: List[Dict],
        label_selector: str = None,
    ):
        # we're using here the run identifier as the common ground to identify which pods are relevant to which CRD, so
        # if they are not coupled we are not able to wait - simply return
        # NOTE - there are surely smarter ways to do this, without depending on the run object, but as of writing this
        # none of the runtimes using CRDs are like that, so not handling it now
        if not self._are_resources_coupled_to_run_object():
            return

        def _verify_crds_underlying_pods_removed():
            project_uid_crd_map = {}
            for crd in deleted_crds:
                project, uid, _ = self._resolve_runtime_resource_run(crd)
                if not uid or not project:
                    logger.warning(
                        "Could not resolve run uid from crd. Skipping waiting for pods deletion",
                        crd=crd,
                    )
                    continue
                project_uid_crd_map.setdefault(project, {})[uid] = crd["metadata"][
                    "name"
                ]
            still_in_deletion_crds_to_pod_names = {}
            jobs_runtime_resources: mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput = self.list_resources(
                "*",
                label_selector=label_selector,
                group_by=mlrun.common.schemas.ListRuntimeResourcesGroupByField.job,
            )
            for project, project_jobs in jobs_runtime_resources.items():
                if project not in project_uid_crd_map:
                    continue
                for job_uid, job_runtime_resources in jobs_runtime_resources[
                    project
                ].items():
                    if job_uid not in project_uid_crd_map[project]:
                        continue
                    if job_runtime_resources.pod_resources:
                        still_in_deletion_crds_to_pod_names[
                            project_uid_crd_map[project][job_uid]
                        ] = [
                            pod_resource.name
                            for pod_resource in job_runtime_resources.pod_resources
                        ]
            if still_in_deletion_crds_to_pod_names:
                raise RuntimeError(
                    f"CRD underlying pods are still in deletion process: {still_in_deletion_crds_to_pod_names}"
                )

        if deleted_crds:
            timeout = 180
            logger.debug(
                "Waiting for CRDs underlying pods deletion",
                timeout=timeout,
                interval=self.wait_for_deletion_interval,
            )
            mlrun.utils.retry_until_successful(
                self.wait_for_deletion_interval,
                timeout,
                logger,
                True,
                _verify_crds_underlying_pods_removed,
            )

    def _delete_pod_resources(
        self,
        db: DBInterface,
        db_session: Session,
        namespace: str,
        label_selector: str = None,
        force: bool = False,
        grace_period: int = None,
    ) -> List[Dict]:
        if grace_period is None:
            grace_period = config.runtime_resources_deletion_grace_period
        pods = get_k8s().v1api.list_namespaced_pod(
            namespace, label_selector=label_selector
        )
        deleted_pods = []
        for pod in pods.items:
            pod_dict = pod.to_dict()

            # best effort - don't let one failure in pod deletion to cut the whole operation
            try:
                (
                    in_terminal_state,
                    last_update,
                    run_state,
                ) = self._resolve_pod_status_info(db, db_session, pod_dict)
                if not force:
                    if not in_terminal_state:
                        continue

                    # give some grace period if we have last update time
                    now = datetime.now(timezone.utc)
                    if (
                        last_update is not None
                        and last_update + timedelta(seconds=float(grace_period)) > now
                    ):
                        continue

                # if resources are tightly coupled to the run object - we want to perform some actions on the run object
                # before deleting them
                if self._are_resources_coupled_to_run_object():
                    try:
                        self._pre_deletion_runtime_resource_run_actions(
                            db, db_session, pod_dict, run_state
                        )
                    except Exception as exc:
                        # Don't prevent the deletion for failure in the pre deletion run actions
                        logger.warning(
                            "Failure in pod run pre-deletion actions. Continuing",
                            exc=repr(exc),
                            pod_name=pod.metadata.name,
                        )

                get_k8s().delete_pod(
                    pod.metadata.name, namespace, self.pod_grace_period_seconds
                )
                deleted_pods.append(pod_dict)
            except Exception as exc:
                logger.warning(
                    f"Cleanup failed processing pod {pod.metadata.name}: {repr(exc)}. Continuing"
                )
        # TODO: don't wait for pods to be deleted, client should poll the deletion status
        self._wait_for_pods_deletion(namespace, deleted_pods, label_selector)
        return deleted_pods

    def _delete_crd_resources(
        self,
        db: DBInterface,
        db_session: Session,
        namespace: str,
        label_selector: str = None,
        force: bool = False,
        grace_period: int = None,
    ) -> List[Dict]:
        if grace_period is None:
            grace_period = config.runtime_resources_deletion_grace_period
        crd_group, crd_version, crd_plural = self._get_crd_info()
        deleted_crds = []
        try:
            crd_objects = get_k8s().crdapi.list_namespaced_custom_object(
                crd_group,
                crd_version,
                namespace,
                crd_plural,
                label_selector=label_selector,
            )
        except ApiException as exc:
            # ignore error if crd is not defined
            if exc.status != 404:
                raise
        else:
            for crd_object in crd_objects["items"]:
                # best effort - don't let one failure in pod deletion to cut the whole operation
                try:
                    (
                        in_terminal_state,
                        last_update,
                        desired_run_state,
                    ) = self._resolve_crd_object_status_info(db, db_session, crd_object)
                    if not force:
                        if not in_terminal_state:
                            continue

                        # give some grace period if we have last update time
                        now = datetime.now(timezone.utc)
                        if (
                            last_update is not None
                            and last_update + timedelta(seconds=float(grace_period))
                            > now
                        ):
                            continue

                    # if resources are tightly coupled to the run object - we want to perform some actions on the run
                    # object before deleting them
                    if self._are_resources_coupled_to_run_object():
                        try:
                            self._pre_deletion_runtime_resource_run_actions(
                                db,
                                db_session,
                                crd_object,
                                desired_run_state,
                            )
                        except Exception as exc:
                            # Don't prevent the deletion for failure in the pre deletion run actions
                            logger.warning(
                                "Failure in crd object run pre-deletion actions. Continuing",
                                exc=err_to_str(exc),
                                crd_object_name=crd_object["metadata"]["name"],
                            )

                    get_k8s().delete_crd(
                        crd_object["metadata"]["name"],
                        crd_group,
                        crd_version,
                        crd_plural,
                        namespace,
                    )
                    deleted_crds.append(crd_object)
                except Exception:
                    exc = traceback.format_exc()
                    crd_object_name = crd_object["metadata"]["name"]
                    logger.warning(
                        f"Cleanup failed processing CRD object {crd_object_name}: {err_to_str(exc)}. Continuing"
                    )
        self._wait_for_crds_underlying_pods_deletion(deleted_crds, label_selector)
        return deleted_crds

    def _pre_deletion_runtime_resource_run_actions(
        self,
        db: DBInterface,
        db_session: Session,
        runtime_resource: Dict,
        run_state: str,
    ):
        project, uid, name = self._resolve_runtime_resource_run(runtime_resource)

        # if cannot resolve related run nothing to do
        if not uid:
            if not self._expect_pods_without_uid():
                logger.warning(
                    "Could not resolve run uid from runtime resource. Skipping pre-deletion actions",
                    runtime_resource=runtime_resource,
                )
                raise ValueError("Could not resolve run uid from runtime resource")
            else:
                return

        logger.info(
            "Performing pre-deletion actions before cleaning up runtime resources",
            project=project,
            uid=uid,
        )

        _, _, run = self._ensure_run_state(
            db, db_session, project, uid, name, run_state
        )
        self._ensure_run_logs_collected(db, db_session, project, uid, run=run)

    def _is_runtime_resource_run_in_terminal_state(
        self,
        db: DBInterface,
        db_session: Session,
        runtime_resource: Dict,
    ) -> Tuple[bool, Optional[datetime]]:
        """
        A runtime can have different underlying resources (like pods or CRDs) - to generalize we call it runtime
        resource. This function will verify whether the Run object related to this runtime resource is in transient
        state. This is useful in order to determine whether an object can be removed. for example, a kubejob's pod
        might be in completed state, but we would like to verify that the run is completed as well to verify the logs
        were collected before we're removing the pod.

        :returns: bool determining whether the run in terminal state, and the last update time if it exists
        """
        project, uid, _ = self._resolve_runtime_resource_run(runtime_resource)

        # if no uid, assume in terminal state
        if not uid:
            return True, None

        run = db.read_run(db_session, uid, project)
        last_update = None
        last_update_str = run.get("status", {}).get("last_update")
        if last_update_str is not None:
            last_update = datetime.fromisoformat(last_update_str)

        if run.get("status", {}).get("state") not in RunStates.terminal_states():
            return False, last_update

        return True, last_update

    def _list_runs_for_monitoring(
        self, db: DBInterface, db_session: Session, states: list = None
    ):
        last_update_time_from = None
        if config.monitoring.runs.list_runs_time_period_in_days:
            last_update_time_from = (
                datetime.now()
                - timedelta(
                    days=int(config.monitoring.runs.list_runs_time_period_in_days)
                )
            ).isoformat()

        runs = db.list_runs(
            db_session,
            project="*",
            states=states,
            last_update_time_from=last_update_time_from,
        )
        project_run_uid_map = {}
        run_with_missing_data = []
        duplicated_runs = []
        for run in runs:
            project = run.get("metadata", {}).get("project")
            uid = run.get("metadata", {}).get("uid")
            if not uid or not project:
                run_with_missing_data.append(run.get("metadata", {}))
                continue
            current_run = project_run_uid_map.setdefault(project, {}).get(uid)

            # sanity
            if current_run:
                duplicated_runs = {
                    "monitored_run": current_run.get(["metadata"]),
                    "duplicated_run": run.get(["metadata"]),
                }
                continue

            project_run_uid_map[project][uid] = run

        # If there are duplications or runs with missing data it probably won't be fixed
        # Monitoring is running periodically and we don't want to log on every problem we found which will spam the log
        # so we're aggregating the problems and logging only once per aggregation
        if duplicated_runs:
            logger.warning(
                "Found duplicated runs (same uid). Heuristically monitoring the first one found",
                duplicated_runs=duplicated_runs,
            )

        if run_with_missing_data:
            logger.warning(
                "Found runs with missing data. They will not be monitored",
                run_with_missing_data=run_with_missing_data,
            )

        return project_run_uid_map

    def _monitor_runtime_resource(
        self,
        db: DBInterface,
        db_session: Session,
        project_run_uid_map: Dict,
        runtime_resource: Dict,
        runtime_resource_is_crd: bool,
        namespace: str,
        project: str = None,
        uid: str = None,
        name: str = None,
    ):
        if not project and not uid and not name:
            project, uid, name = self._resolve_runtime_resource_run(runtime_resource)
        if not project or not uid:
            # Currently any build pod won't have UID and therefore will cause this log message to be printed which
            # spams the log
            # TODO: uncomment the log message when builder become a kind / starts having a UID
            # logger.warning(
            #     "Could not resolve run project or uid from runtime resource, can not monitor run. Continuing",
            #     project=project,
            #     uid=uid,
            #     runtime_resource_name=runtime_resource["metadata"]["name"],
            #     namespace=namespace,
            # )
            return
        run = project_run_uid_map.get(project, {}).get(uid)
        if runtime_resource_is_crd:
            (
                _,
                _,
                run_state,
            ) = self._resolve_crd_object_status_info(db, db_session, runtime_resource)
        else:
            (
                _,
                _,
                run_state,
            ) = self._resolve_pod_status_info(db, db_session, runtime_resource)

        _, updated_run_state, run = self._ensure_run_state(
            db,
            db_session,
            project,
            uid,
            name,
            run_state,
            run,
            search_run=True,
        )

        # Update the UI URL after ensured run state because it also ensures that a run exists
        # (A runtime resource might exist before the run is created)
        self._update_ui_url(db, db_session, project, uid, runtime_resource, run)

        if updated_run_state in RunStates.terminal_states():
            self._ensure_run_logs_collected(db, db_session, project, uid, run=run)

    def _build_list_resources_response(
        self,
        pod_resources: List[mlrun.common.schemas.RuntimeResource] = None,
        crd_resources: List[mlrun.common.schemas.RuntimeResource] = None,
        group_by: Optional[
            mlrun.common.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ) -> Union[
        mlrun.common.schemas.RuntimeResources,
        mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
        mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
    ]:
        if crd_resources is None:
            crd_resources = []
        if pod_resources is None:
            pod_resources = []

        if group_by is None:
            return mlrun.common.schemas.RuntimeResources(
                crd_resources=crd_resources, pod_resources=pod_resources
            )
        else:
            if group_by == mlrun.common.schemas.ListRuntimeResourcesGroupByField.job:
                return self._build_grouped_by_job_list_resources_response(
                    pod_resources, crd_resources
                )
            elif (
                group_by
                == mlrun.common.schemas.ListRuntimeResourcesGroupByField.project
            ):
                return self._build_grouped_by_project_list_resources_response(
                    pod_resources, crd_resources
                )
            else:
                raise NotImplementedError(
                    f"Provided group by field is not supported. group_by={group_by}"
                )

    def _build_grouped_by_project_list_resources_response(
        self,
        pod_resources: List[mlrun.common.schemas.RuntimeResource] = None,
        crd_resources: List[mlrun.common.schemas.RuntimeResource] = None,
    ) -> mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput:
        resources = {}
        for pod_resource in pod_resources:
            self._add_resource_to_grouped_by_project_resources_response(
                resources, "pod_resources", pod_resource
            )
        for crd_resource in crd_resources:
            self._add_resource_to_grouped_by_project_resources_response(
                resources, "crd_resources", crd_resource
            )
        return resources

    def _build_grouped_by_job_list_resources_response(
        self,
        pod_resources: List[mlrun.common.schemas.RuntimeResource] = None,
        crd_resources: List[mlrun.common.schemas.RuntimeResource] = None,
    ) -> mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput:
        resources = {}
        for pod_resource in pod_resources:
            self._add_resource_to_grouped_by_job_resources_response(
                resources, "pod_resources", pod_resource
            )
        for crd_resource in crd_resources:
            self._add_resource_to_grouped_by_job_resources_response(
                resources, "crd_resources", crd_resource
            )
        return resources

    def _add_resource_to_grouped_by_project_resources_response(
        self,
        resources: mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
        resource_field_name: str,
        resource: mlrun.common.schemas.RuntimeResource,
    ):
        if "mlrun/class" in resource.labels:
            project = resource.labels.get("mlrun/project", "")
            mlrun_class = resource.labels["mlrun/class"]
            kind = self._resolve_kind_from_class(mlrun_class)
            self._add_resource_to_grouped_by_field_resources_response(
                project, kind, resources, resource_field_name, resource
            )

    def _add_resource_to_grouped_by_job_resources_response(
        self,
        resources: mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
        resource_field_name: str,
        resource: mlrun.common.schemas.RuntimeResource,
    ):
        if "mlrun/uid" in resource.labels:
            project = resource.labels.get("mlrun/project", config.default_project)
            uid = resource.labels["mlrun/uid"]
            self._add_resource_to_grouped_by_field_resources_response(
                project, uid, resources, resource_field_name, resource
            )

    @staticmethod
    def _add_resource_to_grouped_by_field_resources_response(
        first_field_value: str,
        second_field_value: str,
        resources: mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
        resource_field_name: str,
        resource: mlrun.common.schemas.RuntimeResource,
    ):
        if first_field_value not in resources:
            resources[first_field_value] = {}
        if second_field_value not in resources[first_field_value]:
            resources[first_field_value][
                second_field_value
            ] = mlrun.common.schemas.RuntimeResources(
                pod_resources=[], crd_resources=[]
            )
        if not getattr(
            resources[first_field_value][second_field_value], resource_field_name
        ):
            setattr(
                resources[first_field_value][second_field_value],
                resource_field_name,
                [],
            )
        getattr(
            resources[first_field_value][second_field_value], resource_field_name
        ).append(resource)

    @staticmethod
    def _resolve_kind_from_class(mlrun_class: str) -> str:
        class_to_kind_map = {}
        for kind in mlrun.runtimes.RuntimeKinds.runtime_with_handlers():
            runtime_handler = mlrun.api.runtime_handlers.get_runtime_handler(kind)
            class_values = runtime_handler._get_possible_mlrun_class_label_values()
            for value in class_values:
                class_to_kind_map[value] = kind
        return class_to_kind_map[mlrun_class]

    @staticmethod
    def _get_run_label_selector(project: str, run_uid: str):
        return f"mlrun/project={project},mlrun/uid={run_uid}"

    @staticmethod
    def _ensure_run_logs_collected(
        db: DBInterface, db_session: Session, project: str, uid: str, run: Dict = None
    ):
        # import here to avoid circular imports
        import mlrun.api.crud as crud

        log_file_exists, _ = crud.Logs().log_file_exists_for_run_uid(project, uid)
        if not log_file_exists:
            # this stays for now for backwards compatibility in case we would not use the log collector but rather
            # the legacy method to pull logs
            logs_from_k8s = crud.Logs()._get_logs_legacy_method(
                db_session, project, uid, source=LogSources.K8S, run=run
            )
            if logs_from_k8s:
                logger.info("Storing run logs", project=project, uid=uid)
                crud.Logs().store_log(logs_from_k8s, project, uid, append=False)

    @staticmethod
    def _ensure_run_state(
        db: DBInterface,
        db_session: Session,
        project: str,
        uid: str,
        name: str,
        run_state: str,
        run: Dict = None,
        search_run: bool = True,
    ) -> Tuple[bool, str, dict]:
        if run is None:
            run = {}
        if not run and search_run:
            try:
                run = db.read_run(db_session, uid, project)
            except mlrun.errors.MLRunNotFoundError:
                run = {}
        if not run:
            logger.warning(
                "Run not found. A new run will be created",
                project=project,
                uid=uid,
                desired_run_state=run_state,
                search_run=search_run,
            )
            run = {"metadata": {"project": project, "name": name, "uid": uid}}
        db_run_state = run.get("status", {}).get("state")
        if db_run_state:

            if db_run_state == run_state:
                return False, run_state, run

            # if the current run state is terminal and different from the desired - log
            if db_run_state in RunStates.terminal_states():

                # This can happen when the SDK running in the user's Run updates the Run's state to terminal, but
                # before it exits, when the runtime resource is still running, the API monitoring (here) is executed
                if run_state not in RunStates.terminal_states():
                    now = datetime.now(timezone.utc)
                    last_update_str = run.get("status", {}).get("last_update")
                    if last_update_str is not None:
                        last_update = datetime.fromisoformat(last_update_str)
                        debounce_period = config.runs_monitoring_interval
                        if last_update > now - timedelta(
                            seconds=float(debounce_period)
                        ):
                            logger.warning(
                                "Monitoring found non-terminal state on runtime resource but record has recently "
                                "updated to terminal state. Debouncing",
                                project=project,
                                uid=uid,
                                db_run_state=db_run_state,
                                run_state=run_state,
                                last_update=last_update,
                                now=now,
                                debounce_period=debounce_period,
                            )
                            return False, run_state, run

                logger.warning(
                    "Run record has terminal state but monitoring found different state on runtime resource. Changing",
                    project=project,
                    uid=uid,
                    db_run_state=db_run_state,
                    run_state=run_state,
                )

        logger.info("Updating run state", run_state=run_state)
        run.setdefault("status", {})["state"] = run_state
        run.setdefault("status", {})["last_update"] = now_date().isoformat()
        db.store_run(db_session, run, uid, project)

        return True, run_state, run

    @staticmethod
    def _resolve_runtime_resource_run(runtime_resource: Dict) -> Tuple[str, str, str]:
        project = (
            runtime_resource.get("metadata", {}).get("labels", {}).get("mlrun/project")
        )
        if not project:
            project = config.default_project
        uid = runtime_resource.get("metadata", {}).get("labels", {}).get("mlrun/uid")
        name = (
            runtime_resource.get("metadata", {})
            .get("labels", {})
            .get("mlrun/name", "no-name")
        )
        return project, uid, name

    @staticmethod
    def _build_pod_resources(pods) -> List[mlrun.common.schemas.RuntimeResource]:
        pod_resources = []
        for pod in pods:
            pod_resources.append(
                mlrun.common.schemas.RuntimeResource(
                    name=pod["metadata"]["name"],
                    labels=pod["metadata"]["labels"],
                    status=pod["status"],
                )
            )
        return pod_resources

    @staticmethod
    def _build_crd_resources(
        custom_objects,
    ) -> List[mlrun.common.schemas.RuntimeResource]:
        crd_resources = []
        for custom_object in custom_objects:
            crd_resources.append(
                mlrun.common.schemas.RuntimeResource(
                    name=custom_object["metadata"]["name"],
                    labels=custom_object["metadata"]["labels"],
                    status=custom_object.get("status", {}),
                )
            )
        return crd_resources
