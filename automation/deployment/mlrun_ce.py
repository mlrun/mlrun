# Copyright 2023 MLRun Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform
import subprocess
import sys
import typing

import click

import mlrun.utils


class Constants:
    helm_repo_name = "mlrun-ce"
    helm_release_name = "mlrun-ce"
    helm_chart_name = f"{helm_repo_name}/{helm_release_name}"
    helm_repo_url = "https://mlrun.github.io/ce"
    registry_credentials_secret_name = "registry-credentials"
    mlrun_image_values = ["mlrun.api", "mlrun.ui", "jupyterNotebook"]
    disableable_deployments = ["pipelines", "kube-prometheus-stack", "spark-operator"]


class CommunityEditionDeployer:
    def __init__(
        self,
        namespace: str,
        log_level: str = "info",
        log_file: str = None,
    ) -> None:
        self._debug = log_level == "debug"
        self._log_file_handler = None
        self._logger = mlrun.utils.create_logger(level=log_level, name="automation")
        if log_file:
            self._log_file_handler = open(log_file, "w")
            self._logger.set_handler(
                "file", self._log_file_handler, mlrun.utils.HumanReadableFormatter()
            )
        self._namespace = namespace

    def deploy(
        self,
        registry_url: str = None,
        registry_username: str = None,
        registry_password: str = None,
        chart_version: str = None,
        mlrun_version: str = None,
        override_mlrun_api_image: str = None,
        override_mlrun_ui_image: str = None,
        override_jupyter_image: str = None,
        disable_pipelines: bool = False,
        disable_prometheus_stack: bool = False,
        disable_spark_operator: bool = False,
        devel: bool = False,
        minikube: bool = False,
    ) -> None:
        self._prepare_prerequisites(registry_url, registry_username, registry_password)
        helm_arguments = self._generate_helm_install_arguments(
            registry_url,
            chart_version,
            mlrun_version,
            override_mlrun_api_image,
            override_mlrun_ui_image,
            override_jupyter_image,
            disable_pipelines,
            disable_prometheus_stack,
            disable_spark_operator,
            devel,
            minikube,
        )

        self._logger.info(
            "Installing helm chart with arguments", helm_arguments=helm_arguments
        )
        self._run_command("helm", helm_arguments)

        self._teardown()

    def delete(
        self,
        skip_uninstall: bool = False,
        cleanup_registry_secret: bool = True,
        cleanup_volumes: bool = False,
        cleanup_namespace: bool = False,
    ) -> None:
        if cleanup_namespace:
            self._logger.warning(
                "Cleaning up entire namespace", namespace=self._namespace
            )
            self._run_command("kubectl", ["delete", "namespace", self._namespace])
            return

        if not skip_uninstall:
            self._logger.info(
                "Cleaning up helm release", release=Constants.helm_release_name
            )
            self._run_command(
                "helm",
                [
                    "--namespace",
                    self._namespace,
                    "uninstall",
                    Constants.helm_release_name,
                ],
            )

        if cleanup_volumes:
            self._logger.warning("Cleaning up mlrun volumes")
            self._run_command(
                "kubectl",
                [
                    "--namespace",
                    self._namespace,
                    "delete",
                    "pvc",
                    "-l",
                    f"app.kubernetes.io/name={Constants.helm_release_name}",
                ],
            )

        if cleanup_registry_secret:
            self._logger.warning(
                "Cleaning up registry secret",
                secret_name=Constants.registry_credentials_secret_name,
            )
            self._run_command(
                "kubectl",
                [
                    "--namespace",
                    self._namespace,
                    "delete",
                    "secret",
                    Constants.registry_credentials_secret_name,
                ],
            )

        self._teardown()

    def patch_minikube_images(
        self,
        mlrun_api_image: str = None,
        mlrun_ui_image: str = None,
        jupyter_image: str = None,
    ) -> None:
        for image in [mlrun_api_image, mlrun_ui_image, jupyter_image]:
            if image:
                self._run_command("minikube", ["load", image])

        self._teardown()

    def _teardown(self):
        if self._log_file_handler:
            self._log_file_handler.close()

    def _prepare_prerequisites(
        self,
        registry_url: str = None,
        registry_username: str = None,
        registry_password: str = None,
    ) -> None:
        self._logger.info("Preparing prerequisites")
        self._logger.info("Creating namespace", namespace=self._namespace)
        self._run_command("kubectl", ["create", "namespace", self._namespace])

        self._logger.debug("Adding helm repo")
        self._run_command(
            "helm", ["repo", "add", Constants.helm_repo_name, Constants.helm_repo_url]
        )

        self._logger.debug("Updating helm repo")
        self._run_command("helm", ["repo", "update"])

        if registry_url and registry_username and registry_password:
            self._create_registry_credentials_secret(
                registry_url, registry_username, registry_password
            )
        else:
            self._logger.warning(
                "Registry credentials were not provided, skipping registry secret creation "
                "and assuming it already exists"
            )

    def _generate_helm_install_arguments(
        self,
        registry_url: str = None,
        chart_version: str = None,
        mlrun_version: str = None,
        override_mlrun_api_image: str = None,
        override_mlrun_ui_image: str = None,
        override_jupyter_image: str = None,
        disable_pipelines: bool = False,
        disable_prometheus_stack: bool = False,
        disable_spark_operator: bool = False,
        devel: bool = False,
        minikube: bool = False,
    ) -> typing.List[str]:
        helm_arguments = [
            "--namespace",
            self._namespace,
            "upgrade",
            "--install",
            Constants.helm_release_name,
            "--wait",
            "--timeout",
            "960s",
        ]

        if self._debug:
            helm_arguments.append("--debug")

        for helm_key, helm_value in self._generate_helm_values(
            registry_url,
            mlrun_version,
            override_mlrun_api_image,
            override_mlrun_ui_image,
            override_jupyter_image,
            disable_pipelines,
            disable_prometheus_stack,
            disable_spark_operator,
            minikube,
        ).items():
            helm_arguments.extend(
                [
                    "--set",
                    f"{helm_key}={helm_value}",
                ]
            )

        helm_arguments.append(Constants.helm_chart_name)

        if chart_version:
            self._logger.warning(
                "Installing specific chart version", chart_version=chart_version
            )
            helm_arguments.extend(
                [
                    "--version",
                    chart_version,
                ]
            )

        if devel:
            self._logger.warning("Installing development chart version")
            helm_arguments.append("--devel")

        return helm_arguments

    def _generate_helm_values(
        self,
        registry_url: str = None,
        mlrun_version: str = None,
        override_mlrun_api_image: str = None,
        override_mlrun_ui_image: str = None,
        override_jupyter_image: str = None,
        disable_pipelines: bool = False,
        disable_prometheus_stack: bool = False,
        disable_spark_operator: bool = False,
        minikube: bool = False,
    ) -> typing.Dict[str, str]:

        helm_values = {
            "global.registry.url": registry_url,
            "global.registry.secretName": Constants.registry_credentials_secret_name,
            "global.externalHostAddress": self._get_minikube_ip()
            if minikube
            else self._get_host_ip(),
        }

        if mlrun_version:
            self._set_mlrun_version_in_helm_values(helm_values, mlrun_version)

        for value, overriden_image in zip(
            Constants.mlrun_image_values,
            [
                override_mlrun_api_image,
                override_mlrun_ui_image,
                override_jupyter_image,
            ],
        ):
            if overriden_image:
                self._override_image_in_helm_values(helm_values, value, overriden_image)

        for deployment, disabled in zip(
            Constants.disableable_deployments,
            [
                disable_pipelines,
                disable_prometheus_stack,
                disable_spark_operator,
            ],
        ):
            if disabled:
                self._disable_deployment_in_helm_values(helm_values, deployment)

        # TODO: We need to fix the pipelines metadata grpc server to work on arm
        if self._check_platform_architecture() == "arm":
            self._logger.warning(
                "Kubeflow Pipelines is not supported on ARM architecture. Disabling KFP installation."
            )
            self._disable_deployment_in_helm_values(helm_values, "pipelines")

        self._logger.debug(
            "Generated helm values",
            helm_values=helm_values,
        )

        return helm_values

    def _create_registry_credentials_secret(
        self, registry_url: str, registry_username: str, registry_password: str
    ) -> None:
        self._logger.debug(
            "Creating registry credentials secret",
            secret_name=Constants.registry_credentials_secret_name,
        )
        self._run_command(
            "kubectl",
            [
                "--namespace",
                self._namespace,
                "create",
                "secret",
                "docker-registry",
                Constants.registry_credentials_secret_name,
                f"--docker-server={registry_url}",
                f"--docker-username={registry_username}",
                f"--docker-password={registry_password}",
            ],
        )

    def _check_platform_architecture(self) -> str:
        if platform.system() == "Darwin":
            translated, _, exit_status = self._run_command(
                "sysctl",
                ["-n", "sysctl.proc_translated"],
                live=False,
            )
            is_rosetta = translated.strip() == b"1" and exit_status == 0

            if is_rosetta:
                return "arm"

        return platform.processor()

    def _get_host_ip(self) -> str:
        if platform.system() == "Darwin":
            return self._run_command("ipconfig", ["getifaddr", "en0"], live=False)[
                0
            ].strip()
        elif platform.system() == "Linux":
            return (
                self._run_command("hostname", ["-I"], live=False)[0].split()[0].strip()
            )
        else:
            raise NotImplementedError(
                f"Platform {platform.system()} is not supported for this action"
            )

    def _get_minikube_ip(self) -> str:
        return self._run_command("minikube", ["ip"], live=False)[0].strip()

    def _set_mlrun_version_in_helm_values(
        self, helm_values: typing.Dict[str, str], mlrun_version: str
    ) -> None:
        self._logger.warning(
            "Installing specific mlrun version", mlrun_version=mlrun_version
        )
        for image in Constants.mlrun_image_values:
            helm_values[f"{image}.image.tag"] = mlrun_version

    def _override_image_in_helm_values(
        self,
        helm_values: typing.Dict[str, str],
        image_helm_value: str,
        overriden_image: str,
    ) -> None:
        (
            overriden_image_repo,
            overriden_image_tag,
        ) = overriden_image.split(":")
        self._logger.warning(
            "Overriding image", image=image_helm_value, overriden_image=overriden_image
        )
        helm_values[f"{image_helm_value}.image.repository"] = overriden_image_repo
        helm_values[f"{image_helm_value}.image.tag"] = overriden_image_tag

    def _disable_deployment_in_helm_values(
        self, helm_values: typing.Dict[str, str], deployment: str
    ) -> None:
        self._logger.warning("Disabling deployment", deployment=deployment)
        helm_values[f"{deployment}.enabled"] = "false"

    def _run_command(
        self,
        command: str,
        args: list = None,
        workdir: str = None,
        stdin: str = None,
        live: bool = True,
    ) -> (str, str, int):
        stdout, stderr, exit_status = "", "", 0
        if workdir:
            command = f"cd {workdir}; " + command
        if args:
            command += " " + " ".join(args)

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            shell=True,
        )

        if stdin:
            process.stdin.write(bytes(stdin, "ascii"))
            process.stdin.close()

        if live:
            for line in iter(process.stdout.readline, b""):
                stdout += str(line)
                sys.stdout.write(line.decode(sys.stdout.encoding))
                if self._log_file_handler:
                    self._log_file_handler.write(line.decode(sys.stdout.encoding))
        else:
            stdout = process.stdout.read()

        stderr = process.stderr.read()

        exit_status = process.wait()

        return stdout, stderr, exit_status


@click.group(help="MLRun Community Edition Deployment CLI Tool")
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
@click.option(
    "-f",
    "--log-file",
    help="Path to log file. If not specified, will log to stdout",
)
@click.pass_context
def cli(ctx, debug: bool = False, log_file: str = None):
    ctx.obj["DEBUG"] = debug
    ctx.obj["LOG_FILE"] = log_file


@cli.command()
@click.option(
    "-n",
    "--namespace",
    default="mlrun",
    help="Namespace to install the platform in. Defaults to 'mlrun'",
)
@click.option(
    "-mv",
    "--mlrun-version",
    help="Version of mlrun to install. If not specified, will install the latest version",
)
@click.option(
    "-cv",
    "--chart-version",
    help="Version of the mlrun chart to install. If not specified, will install the latest version",
)
@click.option(
    "--registry-url",
    help="URL of the container registry to use for storing images",
)
@click.option(
    "--registry-username",
    help="Username of the container registry to use for storing images",
)
@click.option(
    "--registry-password",
    help="Password of the container registry to use for storing images",
)
@click.option(
    "--override-mlrun-api-image",
    help="Override the mlrun-api image. Format: <repo>:<tag>",
)
@click.option(
    "--override-mlrun-ui-image",
    help="Override the mlrun-ui image. Format: <repo>:<tag>",
)
@click.option(
    "--override-jupyter-image",
    help="Override the jupyter image. Format: <repo>:<tag>",
)
@click.option(
    "--disable-pipelines",
    is_flag=True,
    help="Disable the installation of Kubeflow Pipelines",
)
@click.option(
    "--disable-prometheus-stack",
    is_flag=True,
    help="Disable the installation of the Prometheus stack",
)
@click.option(
    "--disable-spark-operator",
    is_flag=True,
    help="Disable the installation of the Spark operator",
)
@click.option(
    "--devel",
    is_flag=True,
    help="Get the latest RC version of the mlrun chart. (Only works if --chart-version is not specified)",
)
@click.option(
    "-m",
    "--minikube",
    is_flag=True,
    help="Install the mlrun chart in local minikube.",
)
@click.pass_context
def deploy(
    ctx,
    namespace: str = "mlrun",
    mlrun_version: str = None,
    chart_version: str = None,
    registry_url: str = None,
    registry_username: str = None,
    registry_password: str = None,
    override_mlrun_api_image: str = None,
    override_mlrun_ui_image: str = None,
    override_jupyter_image: str = None,
    disable_pipelines: bool = False,
    disable_prometheus_stack: bool = False,
    disable_spark_operator: bool = False,
    devel: bool = False,
    minikube: bool = False,
):
    deployer = CommunityEditionDeployer(
        namespace=namespace,
        log_level="debug" if ctx.obj["DEBUG"] else "info",
        log_file=ctx.obj["LOG_FILE"],
    )
    deployer.deploy(
        mlrun_version=mlrun_version,
        chart_version=chart_version,
        registry_url=registry_url,
        registry_username=registry_username,
        registry_password=registry_password,
        override_mlrun_api_image=override_mlrun_api_image,
        override_mlrun_ui_image=override_mlrun_ui_image,
        override_jupyter_image=override_jupyter_image,
        disable_pipelines=disable_pipelines,
        disable_prometheus_stack=disable_prometheus_stack,
        disable_spark_operator=disable_spark_operator,
        devel=devel,
        minikube=minikube,
    )


@cli.command()
@click.option(
    "-n",
    "--namespace",
    default="mlrun",
    help="Namespace to install the platform in. Defaults to 'mlrun'",
)
@click.option(
    "--skip-uninstall",
    is_flag=True,
    help="Skip uninstalling the Helm chart. Useful if already uninstalled and you want to perform cleanup only",
)
@click.option(
    "--skip-cleanup-registry-secret",
    is_flag=True,
    help="Skip deleting the registry secret created during installation",
)
@click.option(
    "--cleanup-volumes",
    is_flag=True,
    help="Delete the PVCs created during installation. WARNING: This will result in data loss!",
)
@click.option(
    "--cleanup-namespace",
    is_flag=True,
    help="Delete the namespace created during installation. This overrides the other cleanup options. "
    "WARNING: This will result in data loss!",
)
@click.pass_context
def delete(
    ctx,
    namespace: str = "mlrun",
    skip_uninstall: bool = False,
    skip_cleanup_registry_secret: bool = False,
    cleanup_volumes: bool = False,
    cleanup_namespace: bool = False,
):
    deployer = CommunityEditionDeployer(
        namespace=namespace,
        log_level="debug" if ctx.obj["DEBUG"] else "info",
        log_file=ctx.obj["LOG_FILE"],
    )
    deployer.delete(
        skip_uninstall=skip_uninstall,
        cleanup_registry_secret=not skip_cleanup_registry_secret,
        cleanup_volumes=cleanup_volumes,
        cleanup_namespace=cleanup_namespace,
    )


@cli.command()
@click.option(
    "--mlrun-api-image",
    help="Override the mlrun-api image. Format: <repo>:<tag>",
)
@click.option(
    "--mlrun-ui-image",
    help="Override the mlrun-ui image. Format: <repo>:<tag>",
)
@click.option(
    "--jupyter-image",
    help="Override the jupyter image. Format: <repo>:<tag>",
)
@click.pass_context
def patch_minikube_images(
    ctx,
    mlrun_api_image: str = None,
    mlrun_ui_image: str = None,
    jupyter_image: str = None,
):
    deployer = CommunityEditionDeployer(
        namespace="",
        log_level="debug" if ctx.obj["DEBUG"] else "info",
        log_file=ctx.obj["LOG_FILE"],
    )
    deployer.patch_minikube_images(
        mlrun_api_image=mlrun_api_image,
        mlrun_ui_image=mlrun_ui_image,
        jupyter_image=jupyter_image,
    )


if __name__ == "__main__":
    cli(obj={})
