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
import os
import typing

import kubernetes
import sqlalchemy.orm
from kubernetes import client
from kubernetes.client.rest import ApiException
from packaging.version import parse as parse_version

import mlrun
import mlrun.common.constants as mlrun_constants
import server.api.db.base as api_db_base
import server.api.utils.singletons.k8s
from mlrun.runtimes.base import RuntimeClassMode
from mlrun.utils import logger
from server.api.runtime_handlers import BaseRuntimeHandler


class KubeRuntimeHandler(BaseRuntimeHandler):
    kind = "job"
    class_modes = {RuntimeClassMode.run: "job", RuntimeClassMode.build: "build"}

    @staticmethod
    def _get_kubernetes_lifecycle_handler_class():
        try:
            if parse_version(kubernetes.__version__) > parse_version("22.6.0"):
                return client.V1LifecycleHandler
        except ImportError:
            return client.V1Handler
        return client.V1Handler

    def run(
        self,
        runtime: mlrun.runtimes.KubejobRuntime,
        run: mlrun.run.RunObject,
        execution: mlrun.execution.MLClientCtx,
    ):
        command, args, extra_env = self._get_cmd_args(runtime, run)
        run_node_selector = run.spec.node_selector

        if run.metadata.iteration:
            runtime.store_run(run)
        new_meta = self._get_meta(runtime, run)

        self.add_secrets_to_spec_before_running(
            runtime, project_name=run.metadata.project
        )
        workdir = self._resolve_workdir(runtime)

        pod_spec = func_to_pod(
            runtime.full_image_path(
                client_version=run.metadata.labels.get(
                    mlrun_constants.MLRunInternalLabels.client_version
                ),
                client_python_version=run.metadata.labels.get(
                    mlrun_constants.MLRunInternalLabels.client_python_version
                ),
            ),
            runtime,
            extra_env,
            command,
            args,
            workdir,
            self._get_lifecycle(),
            node_selector=run_node_selector,
        )
        pod = client.V1Pod(metadata=new_meta, spec=pod_spec)
        try:
            (
                pod_name,
                namespace,
            ) = server.api.utils.singletons.k8s.get_k8s_helper().create_pod(pod)
        except ApiException as exc:
            raise mlrun.runtimes.utils.RunError(mlrun.errors.err_to_str(exc)) from exc

        txt = "Job is running in the background"
        logger.info(txt, pod_name=pod_name)
        run.status.status_text = f"{txt}, pod: {pod_name}"

    def _get_cmd_args(
        self,
        runtime: mlrun.runtimes.KubejobRuntime,
        run: mlrun.run.RunObject,
    ):
        extra_env = runtime._generate_runtime_env(run)
        if runtime.spec.pythonpath:
            extra_env["PYTHONPATH"] = runtime.spec.pythonpath
        args = []
        command = runtime.spec.command
        code = (
            runtime.spec.build.functionSourceCode
            if hasattr(runtime.spec, "build")
            else None
        )

        if run.spec.handler and runtime.spec.mode == "pass":
            raise ValueError('cannot use "pass" mode with handler')

        if code:
            extra_env["MLRUN_EXEC_CODE"] = code

        load_archive = (
            runtime.spec.build.load_source_on_run and runtime.spec.build.source
        )
        need_mlrun = code or load_archive or runtime.spec.mode != "pass"

        if need_mlrun:
            args = ["run", "--name", run.metadata.name, "--from-env"]
            if run.spec.handler:
                args += ["--handler", run.spec.handler]
            if runtime.spec.mode:
                args += ["--mode", runtime.spec.mode]
            if runtime.spec.build.origin_filename:
                args += ["--origin-file", runtime.spec.build.origin_filename]

            if load_archive:
                if code:
                    raise ValueError("cannot specify both code and source archive")
                args += ["--source", runtime.spec.build.source]
                if runtime.spec.workdir:
                    # set the absolute/relative path to the cloned code
                    args += ["--workdir", runtime.spec.workdir]

            if command:
                args += [command]

            if runtime.spec.args:
                if not command:
                    # * is a placeholder for the url argument in the run CLI command,
                    # where the code is passed in the `MLRUN_EXEC_CODE` meaning there is no "actual" file to execute
                    # until the run command will create that file from the env param.
                    args += ["*"]
                args = args + runtime.spec.args

            command = "mlrun"
        else:
            command = command.format(**run.spec.parameters)
            if runtime.spec.args:
                args = [arg.format(**run.spec.parameters) for arg in runtime.spec.args]

        extra_env = [{"name": k, "value": v} for k, v in extra_env.items()]
        return command, args, extra_env

    @staticmethod
    def _resolve_workdir(runtime: mlrun.runtimes.KubejobRuntime):
        """
        The workdir is relative to the source root, if the source is not loaded on run then the workdir
        is relative to the clone target dir (where the source was copied to).
        Otherwise, if the source is loaded on run, the workdir is resolved on the run as well.
        If the workdir is absolute, keep it as is.
        """
        workdir = runtime.spec.workdir
        if runtime.spec.build.source and runtime.spec.build.load_source_on_run:
            # workdir will be set AFTER the clone which is done in the pre-run of local runtime
            return None

        if workdir and os.path.isabs(workdir):
            return workdir

        if runtime.spec.build.source_code_target_dir:
            workdir = workdir or ""
            workdir = workdir.removeprefix("./")

            return os.path.join(runtime.spec.build.source_code_target_dir, workdir)

        return workdir

    @staticmethod
    def _expect_pods_without_uid() -> bool:
        """
        builder pods are handled as part of this runtime handler - they are not coupled to run object, therefore they
        don't have the uid in their labels
        """
        return True

    @staticmethod
    def are_resources_coupled_to_run_object() -> bool:
        return True

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f"{mlrun_constants.MLRunInternalLabels.uid}={object_id}"

    @staticmethod
    def _get_lifecycle():
        return None


class DatabricksRuntimeHandler(KubeRuntimeHandler):
    kind = "databricks"
    class_modes = {RuntimeClassMode.run: "databricks"}

    @staticmethod
    def _get_lifecycle():
        script_path = "/mlrun/mlrun/runtimes/databricks_job/databricks_cancel_task.py"
        handler_class = (
            DatabricksRuntimeHandler._get_kubernetes_lifecycle_handler_class()
        )
        pre_stop_handler = handler_class(
            _exec=client.V1ExecAction(command=["python", script_path])
        )
        return client.V1Lifecycle(pre_stop=pre_stop_handler)

    def _delete_pod_resources(
        self,
        db: api_db_base.DBInterface,
        db_session: sqlalchemy.orm.Session,
        namespace: str,
        label_selector: str = None,
        force: bool = False,
        grace_period: int = None,
        resource_deletion_grace_period: typing.Optional[int] = None,
    ) -> list[dict]:
        # override the grace period for the deletion of the pods
        # because the databricks pods needs to signal the databricks cluster to stop the run
        return super()._delete_pod_resources(
            db,
            db_session,
            namespace,
            label_selector,
            force,
            grace_period,
            # coupled with "databricks_runtime.py:DatabricksSpec"
            resource_deletion_grace_period=60,
        )


def func_to_pod(
    image=None,
    runtime=None,
    extra_env=None,
    command=None,
    args=None,
    workdir=None,
    lifecycle=None,
    node_selector=None,
):
    container = client.V1Container(
        name="base",
        image=image,
        env=extra_env + runtime.spec.env,
        command=[command] if command else None,
        args=args,
        working_dir=workdir,
        image_pull_policy=runtime.spec.image_pull_policy,
        volume_mounts=runtime.spec.volume_mounts,
        resources=runtime.spec.resources,
        lifecycle=lifecycle,
    )

    pod_spec = server.api.utils.singletons.k8s.kube_resource_spec_to_pod_spec(
        runtime.spec, container, node_selector
    )

    if runtime.spec.image_pull_secret:
        pod_spec.image_pull_secrets = [
            client.V1LocalObjectReference(name=runtime.spec.image_pull_secret)
        ]

    return pod_spec
