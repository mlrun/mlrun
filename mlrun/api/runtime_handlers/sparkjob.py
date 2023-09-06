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
from datetime import datetime
from typing import Dict, Optional, Tuple

from kubernetes.client.rest import ApiException
from sqlalchemy.orm import Session

from mlrun.api.db.base import DBInterface
from mlrun.api.runtime_handlers.base import BaseRuntimeHandler
from mlrun.runtimes.base import RuntimeClassMode
from mlrun.runtimes.constants import RunStates, SparkApplicationStates
from mlrun.runtimes.sparkjob.abstract import AbstractSparkRuntime
from mlrun.runtimes.utils import get_k8s
from mlrun.utils import logger


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
        run: Dict,
    ):
        if not run:
            logger.warning(
                "Run object was not provided, cannot update the UI URL",
                project=project,
                uid=uid,
                run=run,
            )
            return

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
    def _get_main_runtime_resource_label_selector() -> str:
        """
        There are some runtimes which might have multiple k8s resources attached to a one runtime, in this case
        we don't want to pull logs from all but rather only for the "driver"/"launcher" etc
        :return: the label selector
        """
        return "spark-role=driver"

    @staticmethod
    def _get_crd_info() -> Tuple[str, str, str]:
        return (
            AbstractSparkRuntime.group,
            AbstractSparkRuntime.version,
            AbstractSparkRuntime.plural,
        )

    def _delete_extra_resources(
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
        uids = []
        for crd_dict in deleted_resources:
            uid = crd_dict["metadata"].get("labels", {}).get("mlrun/uid", None)
            uids.append(uid)

        config_maps = get_k8s().v1api.list_namespaced_config_map(
            namespace, label_selector=label_selector
        )
        for config_map in config_maps.items:
            try:
                uid = config_map.metadata.labels.get("mlrun/uid", None)
                if force or uid in uids:
                    get_k8s().v1api.delete_namespaced_config_map(
                        config_map.metadata.name, namespace
                    )
                    logger.info(f"Deleted config map: {config_map.metadata.name}")
            except ApiException as exc:
                # ignore error if config map is already removed
                if exc.status != 404:
                    raise
