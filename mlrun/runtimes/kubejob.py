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

import time
import typing

from kubernetes import client
from kubernetes.client.rest import ApiException

import mlrun.api.schemas
import mlrun.errors
from mlrun.runtimes.base import BaseRuntimeHandler

from ..builder import build_runtime
from ..db import RunDBError
from ..kfpops import build_op
from ..model import RunObject
from ..utils import get_in, logger
from .base import RunError
from .pod import KubeResource, kube_resource_spec_to_pod_spec
from .utils import AsyncLogWriter


class KubejobRuntime(KubeResource):
    kind = "job"
    _is_nested = True

    _is_remote = True

    @property
    def is_deployed(self):
        """check if the function is deployed (have a valid container)"""
        if self.spec.image:
            return True

        if self._is_remote_api():
            db = self._get_db()
            try:
                db.get_builder_status(self, logs=False)
            except Exception:
                pass

        if self.spec.image:
            return True
        if self.status.state and self.status.state == "ready":
            return True
        return False

    def with_source_archive(self, source, pythonpath=None, pull_at_runtime=True):
        """load the code from git/tar/zip archive at runtime or build

        :param source:     valid path to git, zip, or tar file, e.g.
                           git://github.com/mlrun/something.git
                           http://some/url/file.zip
        :param pythonpath: python search path relative to the archive root or absolute (e.g. './subdir')
        :param pull_at_runtime: load the archive into the container at job runtime vs on build/deploy
        """
        self.spec.build.load_source_on_run = pull_at_runtime
        self.spec.build.source = source
        if pythonpath:
            self.spec.pythonpath = pythonpath

    def build_config(
        self,
        image="",
        base_image=None,
        commands: list = None,
        secret=None,
        source=None,
        extra=None,
        load_source_on_run=None,
    ):
        """specify builder configuration for the deploy operation

        :param image:      target image name/path
        :param base_image: base image name/path
        :param commands:   list of docker build (RUN) commands e.g. ['pip install pandas']
        :param secret:     k8s secret for accessing the docker registry
        :param source:     source git/tar archive to load code from in to the context/workdir
                           e.g. git://github.com/mlrun/something.git#development
        :param extra:      extra Dockerfile lines
        :param load_source_on_run: load the archive code into the container at runtime vs at build time
        """
        if image:
            self.spec.build.image = image
        if commands:
            if not isinstance(commands, list):
                raise ValueError("commands must be a string list")
            self.spec.build.commands = self.spec.build.commands or []
            self.spec.build.commands += commands
        if extra:
            self.spec.build.extra = extra
        if secret:
            self.spec.build.secret = secret
        if base_image:
            self.spec.build.base_image = base_image
        if source:
            self.spec.build.source = source
        if load_source_on_run:
            self.spec.build.load_source_on_run = load_source_on_run

    def deploy(
        self,
        watch=True,
        with_mlrun=True,
        skip_deployed=False,
        is_kfp=False,
        mlrun_version_specifier=None,
        builder_env: dict = None,
    ):
        """deploy function, build container with dependencies

        :param watch:      wait for the deploy to complete (and print build logs)
        :param with_mlrun: add the current mlrun package to the container build
        :param skip_deployed: skip the build if we already have an image for the function
        :param mlrun_version_specifier:  which mlrun package version to include (if not current)
        :param builder_env:   Kaniko builder pod env vars dict (for config/credentials)
                              e.g. builder_env={"GIT_TOKEN": token}
        """

        build = self.spec.build

        if not build.source and not build.commands and not build.extra and with_mlrun:
            logger.info(
                "running build to add mlrun package, set "
                "with_mlrun=False to skip if its already in the image"
            )
        self.status.state = ""

        # When we're in pipelines context we must watch otherwise the pipelines pod will exit before the operation
        # is actually done. (when a pipelines pod exits, the pipeline step marked as done)
        if is_kfp:
            watch = True

        if self._is_remote_api():
            db = self._get_db()
            data = db.remote_builder(
                self,
                with_mlrun,
                mlrun_version_specifier,
                skip_deployed,
                builder_env=builder_env,
            )
            self.status = data["data"].get("status", None)
            self.spec.image = get_in(data, "data.spec.image")
            ready = data.get("ready", False)
            if not ready:
                logger.info(
                    f"Started building image: {data.get('data', {}).get('spec', {}).get('build', {}).get('image')}"
                )
            if watch and not ready:
                state = self._build_watch(watch)
                ready = state == "ready"
                self.status.state = state
        else:
            self.save(versioned=False)
            ready = build_runtime(
                mlrun.api.schemas.AuthInfo(),
                self,
                with_mlrun,
                mlrun_version_specifier,
                skip_deployed,
                watch,
            )
            self.save(versioned=False)

        if watch and not ready:
            raise mlrun.errors.MLRunRuntimeError("Deploy failed")
        return ready

    def _build_watch(self, watch=True, logs=True):
        db = self._get_db()
        offset = 0
        try:
            text, _ = db.get_builder_status(self, 0, logs=logs)
        except RunDBError:
            raise ValueError("function or build process not found")

        if text:
            print(text)
        if watch:
            while self.status.state in ["pending", "running"]:
                offset += len(text)
                time.sleep(2)
                text, _ = db.get_builder_status(self, offset, logs=logs)
                if text:
                    print(text, end="")

        return self.status.state

    def builder_status(self, watch=True, logs=True):
        if self._is_remote_api():
            return self._build_watch(watch, logs)

        else:
            pod = self.status.build_pod
            if not self.status.state == "ready" and pod:
                k8s = self._get_k8s()
                status = k8s.get_pod_status(pod)
                if logs:
                    if watch:
                        status = k8s.watch(pod)
                    else:
                        resp = k8s.logs(pod)
                        if resp:
                            print(resp.encode())

                if status == "succeeded":
                    self.status.build_pod = None
                    self.status.state = "ready"
                    logger.info("build completed successfully")
                    return "ready"
                if status in ["failed", "error"]:
                    self.status.state = status
                    logger.error(f" build {status}, watch the build pod logs: {pod}")
                    return status

                logger.info(f"builder status is: {status}, wait for it to complete")
            return None

    def deploy_step(
        self,
        image=None,
        base_image=None,
        commands: list = None,
        secret_name="",
        with_mlrun=True,
        skip_deployed=False,
    ):
        function_name = self.metadata.name or "function"
        name = f"deploy_{function_name}"
        # mark that the function/image is built as part of the pipeline so other places
        # which use the function will grab the updated image/status
        self._build_in_pipeline = True
        return build_op(
            name,
            self,
            image=image,
            base_image=base_image,
            commands=commands,
            secret_name=secret_name,
            with_mlrun=with_mlrun,
            skip_deployed=skip_deployed,
        )

    def _run(self, runobj: RunObject, execution):

        command, args, extra_env = self._get_cmd_args(runobj)

        if runobj.metadata.iteration:
            self.store_run(runobj)
        k8s = self._get_k8s()
        new_meta = self._get_meta(runobj)

        if self._secrets:
            if self._secrets.has_vault_source():
                self._add_vault_params_to_spec(runobj)
            if self._secrets.has_azure_vault_source():
                self._add_azure_vault_params_to_spec(
                    self._secrets.get_azure_vault_k8s_secret()
                )
            self._add_project_k8s_secrets_to_spec(
                self._secrets.get_k8s_secrets(), runobj
            )
        else:
            self._add_project_k8s_secrets_to_spec(None, runobj)

        pod_spec = func_to_pod(
            self.full_image_path(), self, extra_env, command, args, self.spec.workdir
        )
        pod = client.V1Pod(metadata=new_meta, spec=pod_spec)
        try:
            pod_name, namespace = k8s.create_pod(pod)
        except ApiException as exc:
            raise RunError(str(exc))

        if pod_name and self.kfp:
            writer = AsyncLogWriter(self._db_conn, runobj)
            status = k8s.watch(pod_name, namespace, writer=writer)

            if status in ["failed", "error"]:
                raise RunError(f"pod exited with {status}, check logs")
        else:
            txt = f"Job is running in the background, pod: {pod_name}"
            logger.info(txt)
            runobj.status.status_text = txt

        return None


def func_to_pod(image, runtime, extra_env, command, args, workdir):
    container = client.V1Container(
        name="base",
        image=image,
        env=extra_env + runtime.spec.env,
        command=[command],
        args=args,
        working_dir=workdir,
        image_pull_policy=runtime.spec.image_pull_policy,
        volume_mounts=runtime.spec.volume_mounts,
        resources=runtime.spec.resources,
    )

    pod_spec = kube_resource_spec_to_pod_spec(runtime.spec, container)

    if runtime.spec.image_pull_secret:
        pod_spec.image_pull_secrets = [
            client.V1LocalObjectReference(name=runtime.spec.image_pull_secret)
        ]

    return pod_spec


class KubeRuntimeHandler(BaseRuntimeHandler):
    @staticmethod
    def _expect_pods_without_uid() -> bool:
        """
        builder pods are handled as part of this runtime handler - they are not coupled to run object, therefore they
        don't have the uid in their labels
        """
        return True

    @staticmethod
    def _are_resources_coupled_to_run_object() -> bool:
        return True

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f"mlrun/uid={object_id}"

    @staticmethod
    def _get_possible_mlrun_class_label_values() -> typing.List[str]:
        return ["build", "job"]
