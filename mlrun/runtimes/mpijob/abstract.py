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
import os
from typing import Optional

from mlrun.config import config
from mlrun.runtimes.kubejob import KubejobRuntime
from mlrun.runtimes.pod import KubeResourceSpec
from mlrun.utils import update_in


class MPIResourceSpec(KubeResourceSpec):
    _dict_fields = KubeResourceSpec._dict_fields + ["mpi_args"]

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
        default_handler=None,
        entry_points=None,
        description=None,
        workdir=None,
        replicas=None,
        image_pull_policy=None,
        service_account=None,
        build=None,
        image_pull_secret=None,
        mpi_args=None,
        node_name=None,
        node_selector=None,
        affinity=None,
        priority_class_name=None,
        disable_auto_mount=False,
        pythonpath=None,
        tolerations=None,
        preemption_mode=None,
        security_context=None,
        clone_target_dir=None,
        state_thresholds=None,
    ):
        super().__init__(
            command=command,
            image=image,
            mode=mode,
            build=build,
            entry_points=entry_points,
            description=description,
            workdir=workdir,
            default_handler=default_handler,
            volumes=volumes,
            volume_mounts=volume_mounts,
            env=env,
            resources=resources,
            replicas=replicas,
            image_pull_policy=image_pull_policy,
            service_account=service_account,
            image_pull_secret=image_pull_secret,
            args=args,
            node_name=node_name,
            node_selector=node_selector,
            affinity=affinity,
            priority_class_name=priority_class_name,
            disable_auto_mount=disable_auto_mount,
            pythonpath=pythonpath,
            tolerations=tolerations,
            preemption_mode=preemption_mode,
            security_context=security_context,
            clone_target_dir=clone_target_dir,
            state_thresholds=state_thresholds,
        )
        self.mpi_args = mpi_args or [
            "-x",
            "NCCL_SOCKET_NTHREADS=2",
            "-x",
            "NCCL_NSOCKS_PERTHREAD=8",
            "-x",
            "NCCL_MIN_NCHANNELS=4",
        ]


class AbstractMPIJobRuntime(KubejobRuntime, abc.ABC):
    kind = "mpijob"
    # nested i.e. hyper-param loop will use the same CRD/containers (vs CRD per iteration)
    _is_nested = True

    @property
    def spec(self) -> MPIResourceSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", MPIResourceSpec)

    @staticmethod
    def _get_run_completion_updates(run: dict) -> dict:
        # TODO: add a 'workers' section in run objects state, each worker will update its state while
        #  the run state will be resolved by the server.
        # update the run object state if empty so that it won't default to 'created' state
        update_in(run, "status.state", "running", append=False, replace=False)
        return {}

    def with_tracing(
        self, log_file_path: Optional[str] = None, enable_cycle_markers: bool = False
    ):
        """Add Horovod Timeline activity tracking to the job to analyse
        its performance.

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
        log_file_path: Optional[str] = None,
        warmup_samples: Optional[int] = None,
        steps_per_sample: Optional[int] = None,
        bayes_opt_max_samples: Optional[int] = None,
        gaussian_process_noise: Optional[float] = None,
    ):
        """Adds an Autotuner to help optimize Horovod's Parameters for better performance.

        The autotuner will collect metrics and tune horovod's parameters while running using
        Bayesian optimiation. This may affect the performance of the run initially but after
        arriving to the best parameters should increase performance.

        Since autotuning imposes a tradeoff between early performance for better performance
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
            horovod_autotune_settings["autotune-bayes-opt-max-samples"] = (
                bayes_opt_max_samples
            )
        if gaussian_process_noise is not None:
            horovod_autotune_settings["autotune-gaussian-process-noise"] = (
                gaussian_process_noise
            )

        self.set_envs(horovod_autotune_settings)

    def set_mpi_args(self, args: list[str]) -> None:
        """Sets the runtime's mpi arguments to args.

        Parameters
        ----------
        args : typing.List[str]
            Arguments to be used for the mpi-operator

        Raises
        ------
        ValueError
            args is of type `List[str]` and can only accept `str` parameters.

        Example
        -------
        ```
        # Define the wanted MPI arguments
        mpi_args = []
        mpi_args.append("-x")
        mpi_args.append("NCCL_DEBUG=INFO")
        mpi_args.append("-x")
        mpi_args.append("NCCL_SOCKET_NTHREADS=2")
        mpi_args.append("-x")
        mpi_args.append("NCCL_NSOCKS_PERTHREAD=8")
        mpi_args.append("-x")
        mpi_args.append("NCCL_MIN_NCHANNELS=4")

        # Set the MPI arguments in the function
        fn.set_mpi_args(mpi_args)
        ```

        Notes
        -----
        * This will replace existing args.

        """

        # Verify that we are given only strings
        if not all([isinstance(arg, str) for arg in args]):
            raise ValueError(
                "Args is of type `List[str]` and can only accept `str` type params."
            )

        self.spec.mpi_args = args
