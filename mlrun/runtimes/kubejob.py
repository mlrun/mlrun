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
import warnings

import mlrun.common.schemas
import mlrun.db
import mlrun.errors
from mlrun_pipelines.common.ops import build_op

from ..model import RunObject
from .pod import KubeResource


class KubejobRuntime(KubeResource):
    kind = "job"
    _is_nested = True

    _is_remote = True

    def is_deployed(self):
        """check if the function is deployed (has a valid container)"""
        if self.spec.image:
            return True

        db = self._get_db()
        try:
            # getting builder status enriches the runtime when it needs to be fetched from the API,
            # otherwise it's a no-op
            db.get_builder_status(self, logs=False)
        except Exception:
            pass

        if self.spec.image:
            return True
        if self.status.state and self.status.state == "ready":
            return True
        return False

    def with_source_archive(
        self, source, workdir=None, handler=None, pull_at_runtime=True, target_dir=None
    ):
        """load the code from git/tar/zip archive at runtime or build

        :param source:          valid absolute path or URL to git, zip, or tar file, e.g.
                                git://github.com/mlrun/something.git
                                http://some/url/file.zip
                                note path source must exist on the image or exist locally when run is local
                                (it is recommended to use 'workdir' when source is a filepath instead)
        :param handler:         default function handler
        :param workdir:         working dir relative to the archive root (e.g. './subdir') or absolute to the image root
        :param pull_at_runtime: load the archive into the container at job runtime vs on build/deploy
        :param target_dir:      target dir on runtime pod or repo clone / archive extraction
        """
        self._configure_mlrun_build_with_source(
            source=source,
            workdir=workdir,
            handler=handler,
            pull_at_runtime=pull_at_runtime,
            target_dir=target_dir,
        )

    def build_config(
        self,
        image="",
        base_image=None,
        commands: typing.Optional[list] = None,
        secret=None,
        source=None,
        extra=None,
        load_source_on_run=None,
        with_mlrun=None,
        auto_build=None,
        requirements=None,
        overwrite=False,
        prepare_image_for_deploy=True,
        requirements_file=None,
        builder_env=None,
        extra_args=None,
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
        :param with_mlrun: add the current mlrun package to the container build
        :param auto_build: when set to True and the function require build it will be built on the first
                           function run, use only if you dont plan on changing the build config between runs
        :param requirements: a list of packages to install
        :param requirements_file: requirements file to install
        :param overwrite:  overwrite existing build configuration (currently applies to requirements and commands)
           * False: the new params are merged with the existing
           * True: the existing params are replaced by the new ones
        :param prepare_image_for_deploy:    prepare the image/base_image spec for deployment
        :param extra_args:  A string containing additional builder arguments in the format of command-line options,
            e.g. extra_args="--skip-tls-verify --build-arg A=val"
        :param builder_env: Kaniko builder pod env vars dict (for config/credentials)
            e.g. builder_env={"GIT_TOKEN": token}
        """
        if not overwrite:
            # TODO: change overwrite default to True in 1.10.0
            warnings.warn(
                "The `overwrite` parameter default will change from 'False' to 'True' in 1.10.0.",
                mlrun.utils.OverwriteBuildParamsWarning,
            )
        image = mlrun.utils.helpers.remove_image_protocol_prefix(image)
        self.spec.build.build_config(
            image=image,
            base_image=base_image,
            commands=commands,
            secret=secret,
            source=source,
            extra=extra,
            load_source_on_run=load_source_on_run,
            with_mlrun=with_mlrun,
            auto_build=auto_build,
            requirements=requirements,
            requirements_file=requirements_file,
            overwrite=overwrite,
            builder_env=builder_env,
            extra_args=extra_args,
        )

        if prepare_image_for_deploy:
            self.prepare_image_for_deploy()

    def deploy(
        self,
        watch: bool = True,
        with_mlrun: typing.Optional[bool] = None,
        skip_deployed: bool = False,
        is_kfp: bool = False,
        mlrun_version_specifier: typing.Optional[bool] = None,
        builder_env: typing.Optional[dict] = None,
        show_on_failure: bool = False,
        force_build: bool = False,
    ) -> bool:
        """Deploy function, build container with dependencies

        :param watch:                   Wait for the deploy to complete (and print build logs)
        :param with_mlrun:              Add the current mlrun package to the container build
        :param skip_deployed:           Skip the build if we already have an image for the function
        :param is_kfp:                  Deploy as part of a kfp pipeline
        :param mlrun_version_specifier: Which mlrun package version to include (if not current)
        :param builder_env:             Kaniko builder pod env vars dict (for config/credentials)
                                        e.g. builder_env={"GIT_TOKEN": token}
        :param show_on_failure:         Show logs only in case of build failure
        :param force_build:             Set True for force building the image, even when no changes were made

        :return: True if the function is ready (deployed)
        """

        build = self.spec.build
        with_mlrun = self._resolve_build_with_mlrun(with_mlrun)

        self.status.state = ""
        if build.base_image:
            # clear the image so build will not be skipped
            self.spec.image = ""

        return self._build_image(
            builder_env=builder_env,
            force_build=force_build,
            mlrun_version_specifier=mlrun_version_specifier,
            show_on_failure=show_on_failure,
            skip_deployed=skip_deployed,
            watch=watch,
            is_kfp=is_kfp,
            with_mlrun=with_mlrun,
        )

    def deploy_step(
        self,
        image=None,
        base_image=None,
        commands: typing.Optional[list] = None,
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
        raise NotImplementedError(
            f"Running a {self.kind} function from the client is not supported. Use .run() to submit the job to the API."
        )
