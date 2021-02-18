# Copyright 2021 Iguazio
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
import re

from .pod import KubeResourceSpec
from mlrun.runtimes import KubejobRuntime, KubeRuntimeHandler
from mlrun.config import config
from mlrun.db import get_run_db
from ..execution import MLClientCtx
from ..model import RunObject
from ..platforms.iguazio import mount_v3io_extended, mount_v3iod
from subprocess import run


class SparkClientSpec(KubeResourceSpec):
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
        igz_spark=None,
    ):
        super().__init__(
            command=command,
            args=args,
            image=image,
            mode=mode,
            build=build,
            entry_points=entry_points,
            description=description,
            workdir=workdir,
            default_handler=default_handler,
        )
        self.igz_spark = igz_spark


class SparkClientRuntime(KubejobRuntime):
    kind = "sparkclient"

    @property
    def spec(self) -> SparkClientSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", SparkClientSpec)

    def with_igz_spark(self, spark_service):
        self.spec.igz_spark = True
        self.spec.env.append({"name": "MLRUN_SPARK_CLIENT_IGZ_SPARK", "value": "true"})
        self.apply(mount_v3io_extended())
        self.apply(
            mount_v3iod(
                namespace="default-tenant",
                v3io_config_configmap=spark_service + "-submit",
            )
        )

    @property
    def is_deployed(self):
        if (
            not self.spec.build.source
            and not self.spec.build.commands
            and not self.spec.build.extra
            and self.spec.igz_spark
        ):
            return True
        return super().is_deployed

    @property
    def _default_image(self):
        if (
            self.spec.igz_spark
            and config.spark_app_image
            and config.spark_app_image_tag
        ):
            app_image = re.sub("spark-app", "shell", config.spark_app_image)
            # this is temporary until we get the image name from external config
            return app_image + ":" + config.spark_app_image_tag
        return None

    def deploy(self, watch=True, with_mlrun=True, skip_deployed=False, is_kfp=False):
        """deploy function, build container with dependencies"""
        # connect will populate the config from the server config
        get_run_db()
        if not self.spec.build.base_image:
            self.spec.build.base_image = self._default_image
        return super().deploy(
            watch=watch,
            with_mlrun=with_mlrun,
            skip_deployed=skip_deployed,
            is_kfp=is_kfp,
        )

    def _run(self, runobj: RunObject, execution: MLClientCtx):
        # if not self.spec.image:
        #    self.spec.image = self._default_image
        super()._run(runobj, execution)


class SparkClientRuntimeHandler(KubeRuntimeHandler):
    @staticmethod
    def _consider_run_on_resources_deletion() -> bool:
        return True

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f"mlrun/uid={object_id}"

    @staticmethod
    def _get_default_label_selector() -> str:
        return "mlrun/class=sparkclient"


def igz_spark_pre_hook():
    run(["/bin/bash", "/etc/config/v3io/spark-job-init.sh"])
