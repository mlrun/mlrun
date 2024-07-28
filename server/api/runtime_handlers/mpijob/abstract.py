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
import time

from kubernetes import client

import mlrun.k8s_utils
import mlrun.utils.helpers
import server.api.utils.singletons.k8s
from mlrun.config import config
from mlrun.runtimes.base import RuntimeClassMode
from mlrun.runtimes.mpijob import AbstractMPIJobRuntime
from mlrun.utils import logger
from server.api.runtime_handlers import KubeRuntimeHandler


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

        resp = self._submit_mpijob(job, meta.namespace)

        state = None
        timeout = int(config.submit_timeout) or 120
        for _ in range(timeout):
            resp = self.get_job(meta.name, meta.namespace)
            state = self._get_job_launcher_status(resp)
            if resp and state:
                break
            time.sleep(1)

        if resp:
            logger.info(f"MpiJob {meta.name} state={state or 'unknown'}")
            if state:
                state = self._crd_state_to_run_state(state)
                launcher, _ = self._get_launcher(meta.name, meta.namespace)
                execution.set_hostname(launcher)
                execution.set_state(state)
                txt = f"MpiJob {meta.name} launcher pod {launcher} state {state}"
                logger.info(txt)
                run.status.status_text = txt

            else:
                pods_phases = self.get_pods(meta.name, meta.namespace)
                txt = f"MpiJob status unknown or failed, check pods: {pods_phases}"
                logger.warning(txt)
                run.status.status_text = txt

    def get_pods(self, name=None, namespace=None, launcher=False):
        namespace = server.api.utils.singletons.k8s.get_k8s_helper().resolve_namespace(
            namespace
        )

        selector = self._generate_pods_selector(name, launcher)

        pods = server.api.utils.singletons.k8s.get_k8s_helper().list_pods(
            selector=selector, namespace=namespace
        )
        if pods:
            return {p.metadata.name: p.status.phase for p in pods}

    def get_job(self, name, namespace=None):
        mpi_group, mpi_version, mpi_plural = self._get_crd_info()
        namespace = server.api.utils.singletons.k8s.get_k8s_helper().resolve_namespace(
            namespace
        )
        try:
            resp = server.api.utils.singletons.k8s.get_k8s_helper().crdapi.get_namespaced_custom_object(
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

        namespace = server.api.utils.singletons.k8s.get_k8s_helper().resolve_namespace(
            namespace
        )
        try:
            resp = server.api.utils.singletons.k8s.get_k8s_helper().crdapi.create_namespaced_custom_object(
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
