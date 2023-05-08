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
import os.path
import platform
import subprocess
import sys
import typing

import requests

import mlrun.utils


class Constants:
    helm_repo_name = "mlrun-ce"
    helm_release_name = "mlrun-ce"
    helm_chart_name = f"{helm_repo_name}/{helm_release_name}"
    helm_repo_url = "https://mlrun.github.io/ce"
    default_registry_secret_name = "registry-credentials"
    mlrun_image_values = ["mlrun.api", "mlrun.ui", "jupyterNotebook"]
    disableable_deployments = ["pipelines", "kube-prometheus-stack", "spark-operator"]
    minikube_registry_port = 5000


class CommunityEditionDeployer:
    """
    Deployer for MLRun Community Edition (CE) stack.
    """

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
        registry_url: str,
        registry_username: str = None,
        registry_password: str = None,
        registry_secret_name: str = None,
        chart_version: str = None,
        mlrun_version: str = None,
        override_mlrun_api_image: str = None,
        override_mlrun_ui_image: str = None,
        override_jupyter_image: str = None,
        disable_pipelines: bool = False,
        disable_prometheus_stack: bool = False,
        disable_spark_operator: bool = False,
        skip_registry_validation: bool = False,
        devel: bool = False,
        minikube: bool = False,
        sqlite: str = None,
        upgrade: bool = False,
        custom_values: typing.List[str] = None,
    ) -> None:
        """
        Deploy MLRun CE stack.
        :param registry_url: URL of the container registry to use for storing images
        :param registry_username: Username for the container registry (not required if registry_secret_name is provided)
        :param registry_password: Password for the container registry (not required if registry_secret_name is provided)
        :param registry_secret_name: Name of the secret containing the credentials for the container registry
        :param chart_version: Version of the helm chart to deploy (defaults to the latest stable version)
        :param mlrun_version: Version of MLRun to deploy (defaults to the latest stable version)
        :param override_mlrun_api_image: Override the default MLRun API image
        :param override_mlrun_ui_image: Override the default MLRun UI image
        :param override_jupyter_image: Override the default Jupyter image
        :param disable_pipelines: Disable the deployment of the pipelines component
        :param disable_prometheus_stack: Disable the deployment of the Prometheus stack component
        :param disable_spark_operator: Disable the deployment of the Spark operator component
        :param skip_registry_validation: Skip the validation of the registry URL
        :param devel: Deploy the development version of the helm chart
        :param minikube: Deploy the helm chart with minikube configuration
        :param sqlite: Path to sqlite file to use as the mlrun database. If not supplied, will use MySQL deployment
        :param upgrade: Upgrade an existing MLRun CE deployment
        :param custom_values: List of custom values to pass to the helm chart
        """
        self._prepare_prerequisites(
            registry_url,
            registry_username,
            registry_password,
            registry_secret_name,
            skip_registry_validation,
            minikube,
        )
        helm_arguments = self._generate_helm_install_arguments(
            registry_url,
            registry_secret_name,
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
            sqlite,
            upgrade,
            custom_values,
        )

        self._logger.info(
            "Installing helm chart with arguments", helm_arguments=helm_arguments
        )
        stdout, stderr, exit_status = run_command("helm", helm_arguments)
        if exit_status != 0:
            self._logger.error(
                "Failed to install helm chart",
                stderr=stderr,
                exit_status=exit_status,
            )
            raise RuntimeError("Failed to install helm chart")

        self._teardown()

    def delete(
        self,
        skip_uninstall: bool = False,
        sqlite: str = None,
        cleanup_registry_secret: bool = True,
        cleanup_volumes: bool = False,
        cleanup_namespace: bool = False,
        registry_secret_name: str = Constants.default_registry_secret_name,
    ) -> None:
        """
        Delete MLRun CE stack.
        :param skip_uninstall: Skip the uninstallation of the helm chart
        :param sqlite: Path to sqlite file to delete (if needed).
        :param cleanup_registry_secret: Delete the registry secret
        :param cleanup_volumes: Delete the MLRun volumes
        :param cleanup_namespace: Delete the entire namespace
        :param registry_secret_name: Name of the registry secret to delete
        """
        if cleanup_namespace:
            self._logger.warning(
                "Cleaning up entire namespace", namespace=self._namespace
            )
            run_command("kubectl", ["delete", "namespace", self._namespace])
            return

        if not skip_uninstall:
            self._logger.info(
                "Cleaning up helm release", release=Constants.helm_release_name
            )
            run_command(
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
            run_command(
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
                secret_name=registry_secret_name,
            )
            run_command(
                "kubectl",
                [
                    "--namespace",
                    self._namespace,
                    "delete",
                    "secret",
                    registry_secret_name,
                ],
            )

        if sqlite:
            os.remove(sqlite)

        self._teardown()

    def patch_minikube_images(
        self,
        mlrun_api_image: str = None,
        mlrun_ui_image: str = None,
        jupyter_image: str = None,
    ) -> None:
        """
        Patch the MLRun CE stack images in minikube.
        :param mlrun_api_image: MLRun API image to use
        :param mlrun_ui_image: MLRun UI image to use
        :param jupyter_image: Jupyter image to use
        """
        for image in [mlrun_api_image, mlrun_ui_image, jupyter_image]:
            if image:
                run_command("minikube", ["load", image])

        self._teardown()

    def _teardown(self):
        """
        Teardown the CLI tool.
        Close the log file handler if exists.
        """
        if self._log_file_handler:
            self._log_file_handler.close()

    def _prepare_prerequisites(
        self,
        registry_url: str,
        registry_username: str = None,
        registry_password: str = None,
        registry_secret_name: str = None,
        skip_registry_validation: bool = False,
        minikube: bool = False,
    ) -> None:
        """
        Prepare the prerequisites for the MLRun CE stack deployment.
        Creates namespace, adds helm repository, creates registry secret if needed.
        :param registry_url: URL of the registry to use
        :param registry_username: Username of the registry to use (not required if registry_secret_name is provided)
        :param registry_password: Password of the registry to use (not required if registry_secret_name is provided)
        :param registry_secret_name: Name of the registry secret to use
        :param skip_registry_validation: Skip the validation of the registry URL
        :param minikube: Whether to deploy on minikube
        """
        self._logger.info("Preparing prerequisites")
        skip_registry_validation = skip_registry_validation or (
            registry_url is None and minikube
        )
        if not skip_registry_validation:
            self._validate_registry_url(registry_url)

        self._logger.info("Creating namespace", namespace=self._namespace)
        run_command("kubectl", ["create", "namespace", self._namespace])

        self._logger.debug("Adding helm repo")
        run_command(
            "helm", ["repo", "add", Constants.helm_repo_name, Constants.helm_repo_url]
        )

        self._logger.debug("Updating helm repo")
        run_command("helm", ["repo", "update"])

        if registry_username and registry_password:
            self._create_registry_credentials_secret(
                registry_url, registry_username, registry_password
            )
        elif registry_secret_name is not None:
            self._logger.warning(
                "Using existing registry secret", secret_name=registry_secret_name
            )
        else:
            raise ValueError(
                "Either registry credentials or registry secret name must be provided"
            )

    def _generate_helm_install_arguments(
        self,
        registry_url: str = None,
        registry_secret_name: str = None,
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
        sqlite: str = None,
        upgrade: bool = False,
        custom_values: typing.List[str] = None,
    ) -> typing.List[str]:
        """
        Generate the helm install arguments.
        :param registry_url: URL of the registry to use
        :param registry_secret_name: Name of the registry secret to use
        :param chart_version: Version of the chart to use
        :param mlrun_version: Version of MLRun to use
        :param override_mlrun_api_image: Override MLRun API image to use
        :param override_mlrun_ui_image: Override MLRun UI image to use
        :param override_jupyter_image: Override Jupyter image to use
        :param disable_pipelines: Disable pipelines
        :param disable_prometheus_stack: Disable Prometheus stack
        :param disable_spark_operator: Disable Spark operator
        :param devel: Use development chart
        :param minikube: Use minikube
        :param sqlite: Path to sqlite file to use as the mlrun database. If not supplied, will use MySQL deployment
        :param upgrade: Upgrade an existing MLRun CE deployment
        :param custom_values: List of custom values to use
        :return: List of helm install arguments
        """
        helm_arguments = [
            "--namespace",
            self._namespace,
            "upgrade",
            Constants.helm_release_name,
            Constants.helm_chart_name,
            "--install",
            "--wait",
            "--timeout",
            "960s",
        ]

        if self._debug:
            helm_arguments.append("--debug")

        if upgrade:
            helm_arguments.append("--reuse-values")

        for helm_key, helm_value in self._generate_helm_values(
            registry_url,
            registry_secret_name,
            mlrun_version,
            override_mlrun_api_image,
            override_mlrun_ui_image,
            override_jupyter_image,
            disable_pipelines,
            disable_prometheus_stack,
            disable_spark_operator,
            sqlite,
            minikube,
        ).items():
            helm_arguments.extend(
                [
                    "--set",
                    f"{helm_key}={helm_value}",
                ]
            )

        for value in custom_values:
            helm_arguments.extend(
                [
                    "--set",
                    value,
                ]
            )

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
        registry_url: str,
        registry_secret_name: str = None,
        mlrun_version: str = None,
        override_mlrun_api_image: str = None,
        override_mlrun_ui_image: str = None,
        override_jupyter_image: str = None,
        disable_pipelines: bool = False,
        disable_prometheus_stack: bool = False,
        disable_spark_operator: bool = False,
        sqlite: str = None,
        minikube: bool = False,
    ) -> typing.Dict[str, str]:
        """
        Generate the helm values.
        :param registry_url: URL of the registry to use
        :param registry_secret_name: Name of the registry secret to use
        :param mlrun_version: Version of MLRun to use
        :param override_mlrun_api_image: Override MLRun API image to use
        :param override_mlrun_ui_image: Override MLRun UI image to use
        :param override_jupyter_image: Override Jupyter image to use
        :param disable_pipelines: Disable pipelines
        :param disable_prometheus_stack: Disable Prometheus stack
        :param disable_spark_operator: Disable Spark operator
        :param sqlite: Path to sqlite file to use as the mlrun database. If not supplied, will use MySQL deployment
        :param minikube: Use minikube
        :return: Dictionary of helm values
        """
        host_ip = self._get_minikube_ip() if minikube else self._get_host_ip()
        if not registry_url and minikube:
            registry_url = f"{host_ip}:{Constants.minikube_registry_port}"

        helm_values = {
            "global.registry.url": registry_url,
            "global.registry.secretName": f'"{registry_secret_name}"'  # adding quotes in case of empty string
            if registry_secret_name is not None
            else Constants.default_registry_secret_name,
            "global.externalHostAddress": host_ip,
            "nuclio.dashboard.externalIPAddresses[0]": host_ip,
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

        if sqlite:
            dir_path = os.path.dirname(sqlite)
            helm_values.update(
                {
                    "mlrun.httpDB.dbType": "sqlite",
                    "mlrun.httpDB.dirPath": dir_path,
                    "mlrun.httpDB.dsn": f"sqlite:///{sqlite}?check_same_thread=false",
                    "mlrun.httpDB.oldDsn": '""',
                }
            )

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
        self,
        registry_url: str,
        registry_username: str,
        registry_password: str,
        registry_secret_name: str = None,
    ) -> None:
        """
        Create a registry credentials secret.
        :param registry_url: URL of the registry to use
        :param registry_username: Username of the registry to use
        :param registry_password: Password of the registry to use
        :param registry_secret_name: Name of the registry secret to use
        """
        registry_secret_name = (
            registry_secret_name
            if registry_secret_name is not None
            else Constants.default_registry_secret_name
        )
        self._logger.debug(
            "Creating registry credentials secret",
            secret_name=registry_secret_name,
        )
        run_command(
            "kubectl",
            [
                "--namespace",
                self._namespace,
                "create",
                "secret",
                "docker-registry",
                registry_secret_name,
                f"--docker-server={registry_url}",
                f"--docker-username={registry_username}",
                f"--docker-password={registry_password}",
            ],
        )

    @staticmethod
    def _check_platform_architecture() -> str:
        """
        Check the platform architecture. If running on macOS, check if Rosetta is enabled.
        Used for kubeflow pipelines which is not supported on ARM architecture (specifically the metadata grpc server).
        :return: Platform architecture
        """
        if platform.system() == "Darwin":
            translated, _, exit_status = run_command(
                "sysctl",
                ["-n", "sysctl.proc_translated"],
                live=False,
            )
            is_rosetta = translated.strip() == b"1" and exit_status == 0

            if is_rosetta:
                return "arm"

        return platform.processor()

    def _get_host_ip(self) -> str:
        """
        Get the host machine IP.
        :return: Host IP
        """
        if platform.system() == "Darwin":
            return (
                run_command("ipconfig", ["getifaddr", "en0"], live=False)[0]
                .strip()
                .decode("utf-8")
            )
        elif platform.system() == "Linux":
            return (
                run_command("hostname", ["-I"], live=False)[0]
                .split()[0]
                .strip()
                .decode("utf-8")
            )
        else:
            raise NotImplementedError(
                f"Platform {platform.system()} is not supported for this action"
            )

    @staticmethod
    def _get_minikube_ip() -> str:
        """
        Get the minikube IP.
        :return: Minikube IP
        """
        return run_command("minikube", ["ip"], live=False)[0].strip().decode("utf-8")

    def _validate_registry_url(self, registry_url):
        """
        Validate the registry url. Send simple GET request to the registry url.
        :param registry_url: URL of the registry to use
        """
        if not registry_url:
            raise ValueError("Registry url is required")
        try:
            response = requests.get(registry_url)
            response.raise_for_status()
        except Exception as exc:
            self._logger.error("Failed to validate registry url", exc=exc)
            raise exc

    def _set_mlrun_version_in_helm_values(
        self, helm_values: typing.Dict[str, str], mlrun_version: str
    ) -> None:
        """
        Set the mlrun version in all the image tags in the helm values.
        :param helm_values: Helm values to update
        :param mlrun_version: MLRun version to use
        """
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
        """
        Override an image in the helm values.
        :param helm_values: Helm values to update
        :param image_helm_value: Helm value of the image to override
        :param overriden_image: Image with which to override
        """
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
        """
        Disable a deployment in the helm values.
        :param helm_values: Helm values to update
        :param deployment: Deployment to disable
        """
        self._logger.warning("Disabling deployment", deployment=deployment)
        helm_values[f"{deployment}.enabled"] = "false"


def run_command(
    command: str,
    args: list = None,
    workdir: str = None,
    stdin: str = None,
    live: bool = True,
    log_file_handler: typing.IO[str] = None,
) -> (str, str, int):
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

    stdout = _handle_command_stdout(process.stdout, log_file_handler, live)
    stderr = process.stderr.read()
    exit_status = process.wait()

    return stdout, stderr, exit_status


def _handle_command_stdout(
    stdout_stream: typing.IO[bytes],
    log_file_handler: typing.IO[str] = None,
    live: bool = True,
) -> str:
    def _write_to_log_file(text: bytes):
        if log_file_handler:
            log_file_handler.write(text.decode(sys.stdout.encoding))

    stdout = ""
    if live:
        for line in iter(stdout_stream.readline, b""):
            stdout += str(line)
            sys.stdout.write(line.decode(sys.stdout.encoding))
            _write_to_log_file(line)
    else:
        stdout = stdout_stream.read()
        _write_to_log_file(stdout)

    return stdout
