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
import typing
from subprocess import run

from mlrun.config import config

from ..model import RunObject
from ..platforms.iguazio import mount_v3io_extended, mount_v3iod
from .kubejob import KubejobRuntime, KubeRuntimeHandler
from .pod import KubeResourceSpec


class RemoteSparkSpec(KubeResourceSpec):
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
        provider=None,
        node_name=None,
        node_selector=None,
        affinity=None,
        priority_class_name=None,
        disable_auto_mount=False,
        pythonpath=None,
    ):
        super().__init__(
            command=command,
            args=args,
            image=image,
            mode=mode,
            volumes=volumes,
            volume_mounts=volume_mounts,
            env=env,
            resources=resources,
            default_handler=default_handler,
            entry_points=entry_points,
            description=description,
            workdir=workdir,
            replicas=replicas,
            image_pull_policy=image_pull_policy,
            service_account=service_account,
            build=build,
            image_pull_secret=image_pull_secret,
            node_name=node_name,
            node_selector=node_selector,
            affinity=affinity,
            priority_class_name=priority_class_name,
            disable_auto_mount=disable_auto_mount,
            pythonpath=pythonpath,
        )
        self.provider = provider


class RemoteSparkProviders(object):
    iguazio = "iguazio"


class RemoteSparkRuntime(KubejobRuntime):
    kind = "remote-spark"
    default_image = ".remote-spark-default-image"

    @classmethod
    def deploy_default_image(cls):
        from mlrun import get_run_db
        from mlrun.run import new_function

        sj = new_function(
            kind="remote-spark", name="remote-spark-default-image-deploy-temp"
        )
        sj.spec.build.image = cls.default_image
        sj.with_spark_service(spark_service="dummy-spark")
        sj.deploy()
        get_run_db().delete_function(name=sj.metadata.name)

    @property
    def is_deployed(self):
        if (
            not self.spec.build.source
            and not self.spec.build.commands
            and not self.spec.build.extra
        ):
            return True
        return super().is_deployed

    def _run(self, runobj: RunObject, execution):
        self.spec.image = self.spec.image or self.default_image
        super()._run(runobj=runobj, execution=execution)

    @property
    def spec(self) -> RemoteSparkSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", RemoteSparkSpec)

    def with_spark_service(self, spark_service, provider=RemoteSparkProviders.iguazio):
        """Attach spark service to function"""
        self.spec.provider = provider
        if provider == RemoteSparkProviders.iguazio:
            self.spec.env.append(
                {"name": "MLRUN_SPARK_CLIENT_IGZ_SPARK", "value": "true"}
            )
            self.apply(mount_v3io_extended())
            self.apply(
                mount_v3iod(
                    namespace=config.namespace,
                    v3io_config_configmap=spark_service + "-submit",
                )
            )

    @property
    def _resolve_default_base_image(self):
        if (
            self.spec.provider == RemoteSparkProviders.iguazio
            and config.spark_app_image
            and config.spark_app_image_tag
        ):
            app_image = re.sub("spark-app", "shell", config.spark_app_image)
            # this is temporary until we get the image name from external config
            return app_image + ":" + config.spark_app_image_tag
        return None

    def deploy(
        self,
        watch=True,
        with_mlrun=True,
        skip_deployed=False,
        is_kfp=False,
        mlrun_version_specifier=None,
    ):
        """deploy function, build container with dependencies"""
        # connect will populate the config from the server config
        if not self.spec.build.base_image:
            self.spec.build.base_image = self._resolve_default_base_image
        return super().deploy(
            watch=watch,
            with_mlrun=with_mlrun,
            skip_deployed=skip_deployed,
            is_kfp=is_kfp,
            mlrun_version_specifier=mlrun_version_specifier,
        )


class RemoteSparkRuntimeHandler(KubeRuntimeHandler):
    @staticmethod
    def _are_resources_coupled_to_run_object() -> bool:
        return True

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f"mlrun/uid={object_id}"

    @staticmethod
    def _get_possible_mlrun_class_label_values() -> typing.List[str]:
        return ["remote-spark"]


def igz_spark_pre_hook():
    run(["/bin/bash", "/etc/config/v3io/spark-job-init.sh"])
