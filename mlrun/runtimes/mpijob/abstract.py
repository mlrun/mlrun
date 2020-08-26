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
import abc
import time
import typing
import os

from kubernetes import client

from mlrun.config import config
from mlrun.execution import MLClientCtx
from mlrun.model import RunObject
from mlrun.runtimes.kubejob import KubejobRuntime
from mlrun.runtimes.utils import AsyncLogWriter, RunError
from mlrun.utils import logger, get_in


class AbstractMPIJobRuntime(KubejobRuntime, abc.ABC):
    kind = "mpijob"
    _is_nested = False

    @abc.abstractmethod
    def _generate_mpi_job(
        self, runobj: RunObject, execution: MLClientCtx, meta: client.V1ObjectMeta
    ) -> typing.Dict:
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
    def _get_crd_info() -> typing.Tuple[str, str, str]:
        pass

    def _pretty_print_jobs(self, items: typing.List):
        print("{:10} {:20} {:21} {}".format("status", "name", "start", "end"))
        for i in items:
            print(
                "{:10} {:20} {:21} {}".format(
                    self._get_job_launcher_status(i),
                    get_in(i, "metadata.name", ""),
                    get_in(i, "status.startTime", ""),
                    get_in(i, "status.completionTime", ""),
                )
            )

    def _run(self, runobj: RunObject, execution: MLClientCtx):

        if runobj.metadata.iteration:
            self.store_run(runobj)

        meta = self._get_meta(runobj, True)

        job = self._generate_mpi_job(runobj, execution, meta)

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
            logger.info("MpiJob {} state={}".format(meta.name, state or "unknown"))
            if state:
                state = state.lower()
                launcher, _ = self._get_launcher(meta.name, meta.namespace)
                execution.set_hostname(launcher)
                execution.set_state("running" if state == "active" else state)
                if self.kfp:
                    writer = AsyncLogWriter(self._db_conn, runobj)
                    status = self._get_k8s().watch(
                        launcher, meta.namespace, writer=writer
                    )
                    logger.info(
                        "MpiJob {} finished with state {}".format(meta.name, status)
                    )
                    if status == "succeeded":
                        execution.set_state("completed")
                    else:
                        execution.set_state(
                            "error",
                            "MpiJob {} finished with state {}".format(
                                meta.name, status
                            ),
                        )
                else:
                    txt = "MpiJob {} launcher pod {} state {}".format(
                        meta.name, launcher, state
                    )
                    logger.info(txt)
                    runobj.status.status_text = txt
            else:
                txt = "MpiJob status unknown or failed, check pods: {}".format(
                    self.get_pods(meta.name, meta.namespace)
                )
                logger.warning(txt)
                runobj.status.status_text = txt
                if self.kfp:
                    execution.set_state("error", txt)

        return None

    def _submit_mpijob(self, job, namespace=None):
        mpi_group, mpi_version, mpi_plural = self._get_crd_info()

        k8s = self._get_k8s()
        namespace = k8s.resolve_namespace(namespace)
        try:
            resp = k8s.crdapi.create_namespaced_custom_object(
                mpi_group, mpi_version, namespace=namespace, plural=mpi_plural, body=job
            )
            name = get_in(resp, "metadata.name", "unknown")
            logger.info("MpiJob {} created".format(name))
            return resp
        except client.rest.ApiException as e:
            logger.error("Exception when creating MPIJob: %s" % e)
            raise RunError("Exception when creating MPIJob: %s" % e)

    def delete_job(self, name, namespace=None):
        mpi_group, mpi_version, mpi_plural = self._get_crd_info()
        k8s = self._get_k8s()
        namespace = k8s.resolve_namespace(namespace)
        try:
            # delete the mpi job
            body = client.V1DeleteOptions()
            resp = k8s.crdapi.delete_namespaced_custom_object(
                mpi_group, mpi_version, namespace, mpi_plural, name, body
            )
            logger.info("del status: {}".format(get_in(resp, "status", "unknown")))
        except client.rest.ApiException as e:
            print("Exception when deleting MPIJob: %s" % e)

    def list_jobs(self, namespace=None, selector="", show=True):
        mpi_group, mpi_version, mpi_plural = self._get_crd_info()
        k8s = self._get_k8s()
        namespace = k8s.resolve_namespace(namespace)
        try:
            resp = k8s.crdapi.list_namespaced_custom_object(
                mpi_group,
                mpi_version,
                namespace,
                mpi_plural,
                watch=False,
                label_selector=selector,
            )
        except client.rest.ApiException as e:
            print("Exception when reading MPIJob: %s" % e)

        items = []
        if resp:
            items = resp.get("items", [])
            if show and items:
                self._pretty_print_jobs(items)
        return items

    def get_job(self, name, namespace=None):
        mpi_group, mpi_version, mpi_plural = self._get_crd_info()
        k8s = self._get_k8s()
        namespace = k8s.resolve_namespace(namespace)
        try:
            resp = k8s.crdapi.get_namespaced_custom_object(
                mpi_group, mpi_version, namespace, mpi_plural, name
            )
        except client.rest.ApiException as e:
            print("Exception when reading MPIJob: %s" % e)
        return resp

    def get_pods(self, name=None, namespace=None, launcher=False):
        k8s = self._get_k8s()
        namespace = k8s.resolve_namespace(namespace)

        selector = self._generate_pods_selector(name, launcher)

        pods = k8s.list_pods(selector=selector, namespace=namespace)
        if pods:
            return {p.metadata.name: p.status.phase for p in pods}

    def _get_launcher(self, name, namespace=None):
        pods = self.get_pods(name, namespace, launcher=True)
        if not pods:
            logger.error("no pod matches that job name")
            return
        # TODO: Why was this here?
        # k8s = self._get_k8s()
        return list(pods.items())[0]

    def with_tracing(
        self, log_file_path: str = None, enable_cycle_markers: bool = False
    ):
        """Add Horovod Timeline activity tracking to the job to analyse
        its performence.

        The data will be saved as JSON to {log_file_path}. It can then be viewed via
        a trace viewer like chrome or edge's `edge://tracing`.

        More information can be found in the official documentation:
        https://horovod.readthedocs.io/en/latest/timeline_include.html

        Args:
            log_file_path (str, optional):         filepath for the json log file.
                                                   Defaults to <artifacts_path>/hvd_logs/trace.log.
            enable_cycle_markers (bool, optional): Add cycle markers to the log for
                                                   Tensor Fusion aid. Could make the trace very crowded.
                                                   Defaults to False.
        """

        log_path = (
            os.path.join(config.artifact_path, "hvd_logs", "trace.log")
            if log_file_path is None
            else log_file_path
        )
        horovod_timeline_settings = {
            "HOROVOD_TIMELINE": log_path,
            "HOROVOD_TIMELINE_MARK_CYCLES": int(enable_cycle_markers),
        }
        self.set_envs(horovod_timeline_settings)

    def with_autotune(
        self,
        log_file_path: str = None,
        warmup_samples: int = None,
        steps_per_sample: int = None,
        bayes_opt_max_samples: int = None,
        gaussian_process_noise: float = None,
    ):
        """Adds an Autotuner to help optimize Horovod's Parameters for better performence.

        The autotuner will collect metrics and tune horovod's parameters while running using
        Bayesian optimiation. This may affect the performence of the run initially but after
        arriving to the best parameters should increase performence.

        Since autotuning imposes a tradeoff between early performence for better performence
        later on, It's advised to enable it when both:
        - Training should take a long timeout
        - Scaling efficiency was found lacking with the default settings

        More information can be found in the official documentation:
        https://horovod.readthedocs.io/en/latest/autotune_include.html

        Args:
            log_file_path (str, optional):            filepath for the csv log file.
                                                      Defaults to <artifacts_path>/hvd_logs/autotune.csv
            warmup_samples (int, optional):           number of discarded samples at the beginning of the training
                                                      process. Defaults to None.
            steps_per_sample (int, optional):         steps per sample. Defaults to None.
            bayes_opt_max_samples (int, optional):    maximum number of samples. Defaults to None.
            gaussian_process_noise (float, optional): Bayes optimizer's Alpha (noise regularization), to
                                                      account for network and resources variance.
                                                      Defaults to None.
        """

        log_path = (
            os.path.join(config.artifact_path, "hvd_logs", "autotune.csv")
            if log_file_path is None
            else log_file_path
        )
        horovod_autotune_settings = {
            "HOROVOD_AUTOTUNE": "1",
            "HOROVOD_AUTOTUNE_LOG": log_path,
        }
        if warmup_samples is not None:
            horovod_autotune_settings["autotune-warmup-samples"] = warmup_samples
        if steps_per_sample is not None:
            horovod_autotune_settings["autotune-steps-per-sample"] = steps_per_sample
        if bayes_opt_max_samples is not None:
            horovod_autotune_settings[
                "autotune-bayes-opt-max-samples"
            ] = bayes_opt_max_samples
        if gaussian_process_noise is not None:
            horovod_autotune_settings[
                "autotune-gaussian-process-noise"
            ] = gaussian_process_noise

        self.set_envs(horovod_autotune_settings)
