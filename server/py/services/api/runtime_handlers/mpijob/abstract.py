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
import abc
from typing import Optional

from kubernetes import client
from sqlalchemy.orm import Session

import mlrun.common.constants as mlrun_constants
import mlrun.k8s_utils
import mlrun.utils.helpers
from mlrun.runtimes.base import RuntimeClassMode
from mlrun.runtimes.mpijob import AbstractMPIJobRuntime
from mlrun.utils import logger

import framework.utils.singletons.k8s
from framework.db.base import DBInterface
from services.api.runtime_handlers import KubeRuntimeHandler


class AbstractMPIJobRuntimeHandler(KubeRuntimeHandler, abc.ABC):
    kind = "mpijob"
    class_modes = {
        RuntimeClassMode.run: "mpijob",
    }

    def run(
        self,
        runtime: AbstractMPIJobRuntime,
        run: mlrun.run.RunObject,
        execution: mlrun.execution.MLClientCtx,
    ):
        if run.metadata.iteration:
            runtime.store_run(run)

        meta = self._get_meta(runtime, run, True)

        self.add_secrets_to_spec_before_running(
            runtime, project_name=run.metadata.project
        )

        job = self._generate_mpi_job(runtime, run, execution, meta)

        self._submit_mpijob(job, meta.namespace)

        # fetch the launcher pod status
        resp = self.get_job(meta.name, meta.namespace)
        status = self._get_job_launcher_status(resp)

        if status:
            # map the CRD state to run state and set hostname if launcher started
            state = self._crd_state_to_run_state(status)
            launcher, _ = self._get_launcher(meta.name, meta.namespace)
            execution.set_hostname(launcher)
            status_text = (
                f"MpiJob {meta.name} launcher pod {launcher} is in state {state}"
            )
            logger.info(
                "MpiJob launcher pod state update",
                name=meta.name,
                launcher=launcher,
                state=state,
            )
        else:
            # no state yet, assume pending
            state = mlrun.run.RunStatuses.pending
            status_text = f"MpiJob {meta.name} pending - awaiting launcher pod startup"
            logger.info(
                "Waiting for MpiJob launcher pod to start", name=meta.name, state=state
            )

        # update execution state and run status
        execution.set_state(state)
        run.status.status_text = status_text

    def get_pods(self, name=None, namespace=None, launcher=False):
        namespace = framework.utils.singletons.k8s.get_k8s_helper().resolve_namespace(
            namespace
        )

        selector = self._generate_pods_selector(name, launcher)

        pods = framework.utils.singletons.k8s.get_k8s_helper().list_pods(
            selector=selector, namespace=namespace
        )
        if pods:
            return {p.metadata.name: p.status.phase for p in pods}

    def get_job(self, name, namespace=None):
        mpi_group, mpi_version, mpi_plural = self._get_crd_info()
        namespace = framework.utils.singletons.k8s.get_k8s_helper().resolve_namespace(
            namespace
        )
        try:
            resp = framework.utils.singletons.k8s.get_k8s_helper().crdapi.get_namespaced_custom_object(
                mpi_group, mpi_version, namespace, mpi_plural, name
            )
        except client.exceptions.ApiException as exc:
            logger.warning(
                "Exception when reading MPIJob", error=mlrun.errors.err_to_str(exc)
            )
            return None
        return resp

    @abc.abstractmethod
    def _generate_mpi_job(
        self,
        runtime: AbstractMPIJobRuntime,
        run: mlrun.run.RunObject,
        execution: mlrun.execution.MLClientCtx,
        meta: client.V1ObjectMeta,
    ) -> dict:
        pass

    @abc.abstractmethod
    def _get_job_launcher_status(self, resp: list) -> str:
        pass

    @staticmethod
    @abc.abstractmethod
    def _generate_pods_selector(name: str, launcher: bool) -> str:
        pass

    # should return the mpijob CRD information -> (group, version, plural)
    @staticmethod
    @abc.abstractmethod
    def _get_crd_info() -> tuple[str, str, str]:
        pass

    def _get_launcher(self, name, namespace=None):
        pods = self.get_pods(name, namespace, launcher=True)
        if not pods:
            logger.error("no pod matches that job name")
            return
        return list(pods.items())[0]

    def _submit_mpijob(self, job, namespace=None):
        mpi_group, mpi_version, mpi_plural = self._get_crd_info()

        namespace = framework.utils.singletons.k8s.get_k8s_helper().resolve_namespace(
            namespace
        )
        try:
            resp = framework.utils.singletons.k8s.get_k8s_helper().crdapi.create_namespaced_custom_object(
                mpi_group,
                mpi_version,
                namespace=namespace,
                plural=mpi_plural,
                body=job,
            )
            name = mlrun.utils.helpers.get_in(resp, "metadata.name", "unknown")
            logger.info(f"MpiJob {name} created")
            return resp
        except client.rest.ApiException as exc:
            logger.error(
                f"Exception when creating MPIJob: {mlrun.errors.err_to_str(exc)}"
            )
            raise mlrun.runtimes.utils.RunError(
                "Exception when creating MPIJob"
            ) from exc

    @staticmethod
    def _crd_state_to_run_state(state: str) -> str:
        state = state.lower()
        mapping = {
            "active": mlrun.common.runtimes.constants.RunStates.running,
            "failed": mlrun.common.runtimes.constants.RunStates.error,
        }
        return mapping.get(state, state)

    def _ensure_run_state(
        self,
        db: DBInterface,
        db_session: Session,
        project: str,
        uid: str,
        name: str,
        run_state: str,
        run: Optional[dict] = None,
        search_run: bool = True,
        runtime_resource: Optional[dict] = None,
    ) -> tuple[bool, str, dict]:
        _, run_state, run = super()._ensure_run_state(
            db,
            db_session,
            project,
            uid,
            name,
            run_state,
            run,
            search_run,
            runtime_resource,
        )

        execution = mlrun.execution.MLClientCtx.from_dict(run, store_run=False)
        pod_name = (
            run["metadata"]
            .get("labels", {})
            .get(mlrun_constants.MLRunInternalLabels.job, "")
        )

        # ensure hostname is set if not already assigned
        if pod_name and not execution.host:
            namespace = mlrun.mlconf.namespace
            launcher, _ = self._get_launcher(pod_name, namespace)
            execution.set_hostname(launcher)

            # persist the hostname change in the DB
            updates = {"status.host": launcher}
            run = db.update_run(
                db_session,
                updates=updates,
                uid=uid,
                project=project,
            )
        return True, run_state, run
