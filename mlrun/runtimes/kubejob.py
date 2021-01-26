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
from base64 import b64encode

from kubernetes import client
from kubernetes.client.rest import ApiException

from mlrun.runtimes.base import BaseRuntimeHandler
from .base import RunError
from .funcdoc import update_function_entry_points
from .pod import KubeResource
from .utils import AsyncLogWriter, generate_function_image_name
from ..builder import build_runtime
from ..db import RunDBError
from ..kfpops import build_op
from ..model import RunObject
from ..utils import logger, get_in


class KubejobRuntime(KubeResource):
    kind = "job"
    _is_nested = True

    _is_remote = True

    def with_code(self, from_file="", body=None, with_doc=True):
        """Update the function code
        This function eliminates the need to build container images every time we edit the code

        :param from_file:   blank for current notebook, or path to .py/.ipynb file
        :param body:        will use the body as the function code
        :param with_doc:    update the document of the function parameters

        :return: function object
        """
        if (not body and not from_file) or (from_file and from_file.endswith(".ipynb")):
            from nuclio import build_file

            _, _, body = build_file(from_file)

        if from_file:
            with open(from_file) as fp:
                body = fp.read()
        self.spec.build.functionSourceCode = b64encode(body.encode("utf-8")).decode(
            "utf-8"
        )
        if with_doc:
            update_function_entry_points(self, body)
        return self

    @property
    def is_deployed(self):
        if self.spec.image:
            return True

        if self._is_remote_api():
            db = self._get_db()
            try:
                db.get_builder_status(self, logs=False)
            except RunDBError:
                pass

        if self.spec.image:
            return True
        if self.status.state and self.status.state == "ready":
            return True
        return False

    def build_config(
        self, image="", base_image=None, commands: list = None, secret=None, source=None
    ):
        if image:
            self.spec.build.image = image
        if commands:
            if not isinstance(commands, list):
                raise ValueError("commands must be a string list")
            self.spec.build.commands = self.spec.build.commands or []
            self.spec.build.commands += commands
        if secret:
            self.spec.build.secret = secret
        if base_image:
            self.spec.build.base_image = base_image
        if source:
            self.spec.build.source = source

    def build(self, **kw):
        raise ValueError(".build() is deprecated, use .deploy() instead")

    def deploy(
        self,
        watch=True,
        with_mlrun=True,
        skip_deployed=False,
        is_kfp=False,
        mlrun_version_specifier=None,
    ):
        """deploy function, build container with dependencies"""

        if skip_deployed and self.is_deployed:
            self.status.state = "ready"
            self.save(versioned=False)
            return True

        build = self.spec.build
        if not build.source and not build.commands and not with_mlrun:
            if not self.spec.image:
                raise ValueError(
                    "noting to build and image is not specified, "
                    "please set the function image or build args"
                )
            self.status.state = "ready"
            self.save(versioned=False)
            return True

        if not build.source and not build.commands and with_mlrun:
            logger.info(
                "running build to add mlrun package, set "
                "with_mlrun=False to skip if its already in the image"
            )

        self.spec.build.image = self.spec.build.image or generate_function_image_name(
            self
        )
        self.status.state = ""

        if self._is_remote_api() and not is_kfp:
            db = self._get_db()
            logger.info(
                "starting remote build, image: {}".format(self.spec.build.image)
            )
            data = db.remote_builder(self, with_mlrun, mlrun_version_specifier)
            self.status = data["data"].get("status", None)
            self.spec.image = get_in(data, "data.spec.image")
            ready = data.get("ready", False)
            if watch:
                state = self._build_watch(watch)
                ready = state == "ready"
                self.status.state = state
        else:
            self.save(versioned=False)
            ready = build_runtime(
                self, with_mlrun, mlrun_version_specifier, watch or is_kfp
            )
            self.save(versioned=False)

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
                    logger.error(
                        " build {}, watch the build pod logs: {}".format(status, pod)
                    )
                    return status

                logger.info(
                    "builder status is: {}, wait for it to complete".format(status)
                )
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

        name = "deploy_{}".format(self.metadata.name or "function")
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

        with_mlrun = (not self.spec.mode) or (self.spec.mode != "pass")
        command, args, extra_env = self._get_cmd_args(runobj, with_mlrun)
        extra_env = [{"name": k, "value": v} for k, v in extra_env.items()]

        if runobj.metadata.iteration:
            self.store_run(runobj)
        k8s = self._get_k8s()
        new_meta = self._get_meta(runobj)

        pod_spec = func_to_pod(
            self.full_image_path(), self, extra_env, command, args, self.spec.workdir
        )
        pod = client.V1Pod(metadata=new_meta, spec=pod_spec)
        try:
            pod_name, namespace = k8s.create_pod(pod)
        except ApiException as e:
            raise RunError(str(e))

        if pod_name and self.kfp:
            writer = AsyncLogWriter(self._db_conn, runobj)
            status = k8s.watch(pod_name, namespace, writer=writer)

            if status in ["failed", "error"]:
                raise RunError(f"pod exited with {status}, check logs")
        else:
            txt = "Job is running in the background, pod: {}".format(pod_name)
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

    pod_spec = client.V1PodSpec(
        containers=[container],
        restart_policy="Never",
        volumes=runtime.spec.volumes,
        service_account=runtime.spec.service_account,
    )

    if runtime.spec.image_pull_secret:
        pod_spec.image_pull_secrets = [
            client.V1LocalObjectReference(name=runtime.spec.image_pull_secret)
        ]

    return pod_spec


class KubeRuntimeHandler(BaseRuntimeHandler):
    @staticmethod
    def _consider_run_on_resources_deletion() -> bool:
        return True

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f"mlrun/uid={object_id}"

    @staticmethod
    def _get_default_label_selector() -> str:
        return "mlrun/class in (build, job)"
