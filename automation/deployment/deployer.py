# Copyright 2023 Iguazio
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
import logging
import os.path
import platform
import subprocess
import sys
import typing

import paramiko
import requests


class Constants:
    helm_repo_name = "mlrun-ce"
    helm_release_name = "mlrun-ce"
    default_helm_chart_name = f"{helm_repo_name}/{helm_release_name}"
    helm_repo_url = "https://mlrun.github.io/ce"
    default_registry_secret_name = "registry-credentials"
    mlrun_image_values = [
        "mlrun.api",
        "mlrun.ui",
        "jupyterNotebook",
        "mlrun.api.sidecars.logCollector",
    ]
    disableable_components = [
        "pipelines",
        "kube-prometheus-stack",
        "spark-operator",
        "mlrun.api.sidecars.logCollector",
    ]
    minikube_registry_port = 5000
    log_format = "> %(asctime)s [%(levelname)s] %(message)s"


class ExcecutionParams:
    def __init__(
        self,
        registry_url: str,
        registry_secret_name: typing.Optional[str] = None,
        chart_name: typing.Optional[str] = None,
        chart_version: typing.Optional[str] = None,
        mlrun_version: typing.Optional[str] = None,
        override_mlrun_api_image: typing.Optional[str] = None,
        override_mlrun_log_collector_image: typing.Optional[str] = None,
        override_mlrun_ui_image: typing.Optional[str] = None,
        override_jupyter_image: typing.Optional[str] = None,
        disable_pipelines: bool = False,
        force_enable_pipelines: bool = False,
        disable_prometheus_stack: bool = False,
        disable_spark_operator: bool = False,
        disable_log_collector: bool = False,
        devel: bool = False,
        minikube: bool = False,
        sqlite: typing.Optional[str] = None,
        upgrade: bool = False,
        custom_values: typing.Optional[list[str]] = None,
    ):
        self.registry_url = registry_url
        self.registry_secret_name = registry_secret_name
        self.chart_name = chart_name
        self.chart_version = chart_version
        self.mlrun_version = mlrun_version
        self.override_mlrun_api_image = override_mlrun_api_image
        self.override_mlrun_log_collector_image = override_mlrun_log_collector_image
        self.override_mlrun_ui_image = override_mlrun_ui_image
        self.override_jupyter_image = override_jupyter_image
        self.disable_pipelines = disable_pipelines
        self.force_enable_pipelines = force_enable_pipelines
        self.disable_prometheus_stack = disable_prometheus_stack
        self.disable_spark_operator = disable_spark_operator
        self.disable_log_collector = disable_log_collector
        self.devel = devel
        self.minikube = minikube
        self.sqlite = sqlite
        self.upgrade = upgrade
        self.custom_values = custom_values


class CommunityEditionDeployer:
    """
    Deployer for MLRun Community Edition (CE) stack.
    """

    def __init__(
        self,
        namespace: str,
        log_level: str = "info",
        log_file: typing.Optional[str] = None,
        remote: typing.Optional[str] = None,
        remote_ssh_username: typing.Optional[str] = None,
        remote_ssh_password: typing.Optional[str] = None,
        chart_name: typing.Optional[str] = None,
    ) -> None:
        self._debug = log_level == "debug"
        self._log_file_handler = None
        logging.basicConfig(format="> %(asctime)s [%(levelname)s] %(message)s")
        self._logger = logging.getLogger("automation")
        self._logger.setLevel(log_level.upper())

        if log_file:
            self._log_file_handler = open(log_file, "a")
            # using StreamHandler instead of FileHandler (which opens a file descriptor) so the same file descriptor
            # can be used for command stdout as well as the logs.
            handler = logging.StreamHandler(self._log_file_handler)
            handler.setFormatter(logging.Formatter(Constants.log_format))
            self._logger.addHandler(handler)

        self._namespace = namespace
        self._chart_name = chart_name or Constants.default_helm_chart_name
        self._remote = remote
        self._remote_ssh_username = remote_ssh_username or os.environ.get(
            "MLRUN_REMOTE_SSH_USERNAME"
        )
        self._remote_ssh_password = remote_ssh_password or os.environ.get(
            "MLRUN_REMOTE_SSH_PASSWORD"
        )
        self._ssh_client = None
        if self._remote:
            self.connect_to_remote()

    def connect_to_remote(self):
        self._log("info", "Connecting to remote machine", remote=self._remote)
        self._ssh_client = paramiko.SSHClient()
        self._ssh_client.set_missing_host_key_policy(paramiko.RejectPolicy)
        self._ssh_client.connect(
            self._remote,
            username=self._remote_ssh_username,
            password=self._remote_ssh_password,
        )

    def deploy(
        self,
        registry_url: str,
        registry_username: typing.Optional[str] = None,
        registry_password: typing.Optional[str] = None,
        registry_secret_name: typing.Optional[str] = None,
        chart_name: typing.Optional[str] = None,
        chart_version: typing.Optional[str] = None,
        mlrun_version: typing.Optional[str] = None,
        override_mlrun_api_image: typing.Optional[str] = None,
        override_mlrun_log_collector_image: typing.Optional[str] = None,
        override_mlrun_ui_image: typing.Optional[str] = None,
        override_jupyter_image: typing.Optional[str] = None,
        disable_pipelines: bool = False,
        force_enable_pipelines: bool = False,
        disable_prometheus_stack: bool = False,
        disable_spark_operator: bool = False,
        disable_log_collector: bool = False,
        skip_registry_validation: bool = False,
        devel: bool = False,
        minikube: bool = False,
        sqlite: typing.Optional[str] = None,
        upgrade: bool = False,
        custom_values: typing.Optional[list[str]] = None,
    ) -> None:
        """
        Deploy MLRun CE stack.
        :param registry_url:        URL of the container registry to use for storing images
        :param registry_username:   Username for the container registry (required unless providing registry_secret_name)
        :param registry_password:   Password for the container registry (required unless providing registry_secret_name)
        :param registry_secret_name:    Name of the secret containing the credentials for the container registry
        :param chart_name:          Name or local path of the helm chart to deploy (defaults to mlrun-ce/mlrun-ce)
        :param chart_version:       Version of the helm chart to deploy (defaults to the latest stable version)
        :param mlrun_version:       Version of MLRun to deploy (defaults to the latest stable version)
        :param override_mlrun_api_image:            Override the default MLRun API image
        :param override_mlrun_log_collector_image:  Override the default MLRun Log Collector image
        :param override_mlrun_ui_image:             Override the default MLRun UI image
        :param override_jupyter_image:              Override the default Jupyter image
        :param disable_pipelines:           Disable the deployment of the pipelines component
        :param force_enable_pipelines:      Force the pipelines component to be installed
        :param disable_prometheus_stack:    Disable the deployment of the Prometheus stack component
        :param disable_spark_operator:      Disable the deployment of the Spark operator component
        :param disable_log_collector:       Disable the mlrun API log collector sidecar and use legacy mode instead
        :param skip_registry_validation:    Skip the validation of the registry URL
        :param devel:       Deploy the development version of the helm chart
        :param minikube:    Deploy the helm chart with minikube configuration
        :param sqlite:      Path to sqlite file to use as the mlrun database. If not supplied, will use MySQL deployment
        :param upgrade:         Upgrade an existing MLRun CE deployment
        :param custom_values:   List of custom values to pass to the helm chart
        """
        self._prepare_prerequisites(
            registry_url,
            registry_username,
            registry_password,
            registry_secret_name,
            skip_registry_validation,
            minikube,
        )

        ep = ExcecutionParams(
            registry_url,
            registry_secret_name,
            chart_name,
            chart_version,
            mlrun_version,
            override_mlrun_api_image,
            override_mlrun_log_collector_image,
            override_mlrun_ui_image,
            override_jupyter_image,
            disable_pipelines,
            force_enable_pipelines,
            disable_prometheus_stack,
            disable_spark_operator,
            disable_log_collector,
            devel,
            minikube,
            sqlite,
            upgrade,
            custom_values,
        )

        helm_arguments = self._generate_helm_install_arguments(ep)

        self._log(
            "info",
            "Installing helm chart with arguments",
            helm_arguments=helm_arguments,
        )
        stdout, stderr, exit_status = self._run_command("helm", helm_arguments)
        if exit_status != 0:
            self._log(
                "error",
                "Failed to install helm chart",
                stderr=stderr.strip().decode("utf-8"),
                exit_status=exit_status,
            )
            raise RuntimeError("Failed to install helm chart")

        self._teardown()

    def delete(
        self,
        skip_uninstall: bool = False,
        sqlite: typing.Optional[str] = None,
        cleanup_registry_secret: bool = True,
        cleanup_volumes: bool = False,
        cleanup_namespace: bool = False,
        registry_secret_name: str = Constants.default_registry_secret_name,
    ) -> None:
        """
        Delete MLRun CE stack.
        :param skip_uninstall:  Skip the uninstallation of the helm chart
        :param sqlite:      Path to sqlite file to delete (if needed).
        :param cleanup_registry_secret: Delete the registry secret
        :param cleanup_volumes:         Delete the MLRun volumes
        :param cleanup_namespace:       Delete the entire namespace
        :param registry_secret_name:    Name of the registry secret to delete
        """
        if cleanup_namespace:
            self._log(
                "warning", "Cleaning up entire namespace", namespace=self._namespace
            )
            self._run_command("kubectl", ["delete", "namespace", self._namespace])
            return

        if not skip_uninstall:
            self._log(
                "info", "Cleaning up helm release", release=Constants.helm_release_name
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
            self._log("warning", "Cleaning up mlrun volumes")
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
            self._log(
                "warning",
                "Cleaning up registry secret",
                secret_name=registry_secret_name,
            )
            self._run_command(
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
        mlrun_api_image: typing.Optional[str] = None,
        mlrun_ui_image: typing.Optional[str] = None,
        jupyter_image: typing.Optional[str] = None,
    ) -> None:
        """
        Patch the MLRun CE stack images in minikube.
        :param mlrun_api_image: MLRun API image to use
        :param mlrun_ui_image:  MLRun UI image to use
        :param jupyter_image:   Jupyter image to use
        """
        for image in [mlrun_api_image, mlrun_ui_image, jupyter_image]:
            if image:
                self._run_command("minikube", ["load", image])

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
        registry_username: typing.Optional[str] = None,
        registry_password: typing.Optional[str] = None,
        registry_secret_name: typing.Optional[str] = None,
        skip_registry_validation: bool = False,
        minikube: bool = False,
    ) -> None:
        """
        Prepare the prerequisites for the MLRun CE stack deployment.
        Creates namespace, adds helm repository, creates registry secret if needed.
        :param registry_url:         URL of the registry to use
        :param registry_username:    Username of the registry to use (not required if registry_secret_name is provided)
        :param registry_password:    Password of the registry to use (not required if registry_secret_name is provided)
        :param registry_secret_name: Name of the registry secret to use
        :param skip_registry_validation: Skip the validation of the registry URL
        :param minikube:    Whether to deploy on minikube
        """
        self._log("info", "Preparing prerequisites")
        skip_registry_validation = skip_registry_validation or (
            registry_url is None and minikube
        )
        if not skip_registry_validation:
            self._validate_registry_url(registry_url)

        self._log("info", "Creating namespace", namespace=self._namespace)
        self._run_command("kubectl", ["create", "namespace", self._namespace])

        self._log("debug", "Adding helm repo")
        self._run_command(
            "helm", ["repo", "add", Constants.helm_repo_name, Constants.helm_repo_url]
        )

        self._log("debug", "Updating helm repo")
        self._run_command("helm", ["repo", "update"])

        if registry_username and registry_password:
            self._create_registry_credentials_secret(
                registry_url, registry_username, registry_password
            )
        elif registry_secret_name is not None:
            self._log(
                "warning",
                "Using existing registry secret",
                secret_name=registry_secret_name,
            )
        else:
            raise ValueError(
                "Either registry credentials or registry secret name must be provided"
            )

    def _generate_helm_install_arguments(
        self,
        ep: ExcecutionParams,
    ) -> list[str]:
        """
        Generate the helm install arguments.
        :param ep:  Execution parameters
        :return:    List of helm install arguments
        """
        helm_arguments = [
            "--namespace",
            self._namespace,
            "upgrade",
            Constants.helm_release_name,
            self._chart_name,
            "--install",
            "--wait",
            "--timeout",
            "960s",
        ]

        if self._debug:
            helm_arguments.append("--debug")

        if ep.upgrade:
            helm_arguments.append("--reuse-values")

        for helm_key, helm_value in self._generate_helm_values(ep).items():
            helm_arguments.extend(
                [
                    "--set",
                    f"{helm_key}={helm_value}",
                ]
            )

        for value in ep.custom_values:
            helm_arguments.extend(
                [
                    "--set",
                    value,
                ]
            )

        if ep.chart_version:
            self._log(
                "warning",
                "Installing specific chart version",
                chart_version=ep.chart_version,
            )
            helm_arguments.extend(
                [
                    "--version",
                    ep.chart_version,
                ]
            )

        if ep.devel:
            self._log("warning", "Installing development chart version")
            helm_arguments.append("--devel")

        return helm_arguments

    def _generate_helm_values(
        self,
        ep: ExcecutionParams,
    ) -> dict[str, str]:
        """
        Generate the helm values.
        :return: Dictionary of helm values
        """
        host_ip = self._get_minikube_ip() if ep.minikube else self._get_host_ip()
        if not ep.registry_url and ep.minikube:
            ep.registry_url = f"{host_ip}:{Constants.minikube_registry_port}"

        helm_values = {
            "global.registry.url": ep.registry_url,
            "global.registry.secretName": f'"{ep.registry_secret_name}"'  # adding quotes in case of empty string
            if ep.registry_secret_name is not None
            else Constants.default_registry_secret_name,
            "global.externalHostAddress": host_ip,
            "nuclio.dashboard.externalIPAddresses[0]": host_ip,
        }

        if ep.mlrun_version:
            self._set_mlrun_version_in_helm_values(helm_values, ep.mlrun_version)

        for value, overriden_image in zip(
            Constants.mlrun_image_values,
            [
                ep.override_mlrun_api_image,
                ep.override_mlrun_ui_image,
                ep.override_jupyter_image,
                ep.override_mlrun_log_collector_image,
            ],
        ):
            if overriden_image:
                self._override_image_in_helm_values(helm_values, value, overriden_image)

        for component, disabled in zip(
            Constants.disableable_components,
            [
                ep.disable_pipelines,
                ep.disable_prometheus_stack,
                ep.disable_spark_operator,
                ep.disable_log_collector,
            ],
        ):
            self._toggle_component_in_helm_values(helm_values, component, disabled)

        if ep.sqlite:
            dir_path = os.path.dirname(ep.sqlite)
            helm_values.update(
                {
                    "mlrun.httpDB.dbType": "sqlite",
                    "mlrun.httpDB.dirPath": dir_path,
                    "mlrun.httpDB.dsn": f"sqlite:///{ep.sqlite}?check_same_thread=false",
                    "mlrun.httpDB.oldDsn": '""',
                }
            )

        if ep.force_enable_pipelines and ep.disable_pipelines:
            error_message = "--force-enable-pipelines and --disable-pipelines may not be used together. Aborting"
            self._log(
                "error",
                error_message,
            )
            raise ValueError(error_message)

        # TODO: We need to fix the pipelines metadata grpc server to work on arm
        if self._check_platform_architecture() == "arm":
            self._log(
                "warning",
                "Kubeflow Pipelines is not supported on ARM architectures",
            )
            if not ep.force_enable_pipelines:
                self._log(
                    "warning",
                    "Kubeflow Pipelines won't be installed",
                )
                self._toggle_component_in_helm_values(helm_values, "pipelines", True)
            else:
                self._log(
                    "warning",
                    "Kubeflow Pipelines will be installed, but it may not work. Proceed at your own risk",
                )
                self._toggle_component_in_helm_values(helm_values, "pipelines", False)

        self._log(
            "debug",
            "Generated helm values",
            helm_values=helm_values,
        )

        return helm_values

    def _create_registry_credentials_secret(
        self,
        registry_url: str,
        registry_username: str,
        registry_password: str,
        registry_secret_name: typing.Optional[str] = None,
    ) -> None:
        """
        Create a registry credentials secret.
        :param registry_url:         URL of the registry to use
        :param registry_username:    Username of the registry to use
        :param registry_password:    Password of the registry to use
        :param registry_secret_name: Name of the registry secret to use
        """
        registry_secret_name = (
            registry_secret_name
            if registry_secret_name is not None
            else Constants.default_registry_secret_name
        )
        self._log(
            "debug",
            "Creating registry credentials secret",
            secret_name=registry_secret_name,
        )
        self._run_command(
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

    def _check_platform_architecture(self) -> str:
        """
        Check the platform architecture. If running on macOS, check if Rosetta is enabled.
        Used for kubeflow pipelines which is not supported on ARM architecture (specifically the metadata grpc server).
        :return: Platform architecture
        """
        if self._remote:
            self._log(
                "warning",
                "Cannot check platform architecture on remote machine, assuming x86",
            )
            return "x86"

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
        """
        Get the host machine IP.
        :return: Host IP
        """
        if platform.system() == "Darwin":
            return (
                self._run_command("ipconfig", ["getifaddr", "en0"], live=False)[0]
                .strip()
                .decode("utf-8")
            )
        elif platform.system() == "Linux":
            return (
                self._run_command("hostname", ["-I"], live=False)[0]
                .split()[0]
                .strip()
                .decode("utf-8")
            )
        else:
            raise NotImplementedError(
                f"Platform {platform.system()} is not supported for this action"
            )

    def _get_minikube_ip(self) -> str:
        """
        Get the minikube IP.
        :return: Minikube IP
        """
        return (
            self._run_command("minikube", ["ip"], live=False)[0].strip().decode("utf-8")
        )

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
            self._log("error", "Failed to validate registry url", exc=exc)
            raise exc

    def _set_mlrun_version_in_helm_values(
        self, helm_values: dict[str, str], mlrun_version: str
    ) -> None:
        """
        Set the mlrun version in all the image tags in the helm values.
        :param helm_values: Helm values to update
        :param mlrun_version: MLRun version to use
        """
        self._log(
            "warning", "Installing specific mlrun version", mlrun_version=mlrun_version
        )
        for image in Constants.mlrun_image_values:
            helm_values[f"{image}.image.tag"] = mlrun_version

    def _override_image_in_helm_values(
        self,
        helm_values: dict[str, str],
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
        self._log(
            "warning",
            "Overriding image",
            image=image_helm_value,
            overriden_image=overriden_image,
        )
        helm_values[f"{image_helm_value}.image.repository"] = overriden_image_repo
        helm_values[f"{image_helm_value}.image.tag"] = overriden_image_tag

    def _toggle_component_in_helm_values(
        self, helm_values: dict[str, str], component: str, disable: bool
    ) -> None:
        """
        Disable a deployment in the helm values.
        :param helm_values: Helm values to update
        :param component: Component to toggle
        :param disable: Whether to disable the deployment
        """
        self._log("debug", "Toggling component", component=component, disable=disable)
        value = "false" if disable else "true"
        helm_values[f"{component}.enabled"] = value

    def _run_command(
        self,
        command: str,
        args: typing.Optional[list] = None,
        workdir: typing.Optional[str] = None,
        stdin: typing.Optional[str] = None,
        live: bool = True,
    ) -> (str, str, int):
        if self._remote:
            return run_command_remotely(
                self._ssh_client,
                command=command,
                args=args,
                workdir=workdir,
                stdin=stdin,
                live=live,
                log_file_handler=self._log_file_handler,
            )
        else:
            return run_command(
                command=command,
                args=args,
                workdir=workdir,
                stdin=stdin,
                live=live,
                log_file_handler=self._log_file_handler,
            )

    def _log(self, level: str, message: str, **kwargs: typing.Any) -> None:
        more = f": {kwargs}" if kwargs else ""
        self._logger.log(logging.getLevelName(level.upper()), f"{message}{more}")


def run_command(
    command: str,
    args: typing.Optional[list] = None,
    workdir: typing.Optional[str] = None,
    stdin: typing.Optional[str] = None,
    live: bool = True,
    log_file_handler: typing.Optional[typing.IO[str]] = None,
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


def run_command_remotely(
    ssh_client: paramiko.SSHClient,
    command: str,
    args: typing.Optional[list] = None,
    workdir: typing.Optional[str] = None,
    stdin: typing.Optional[str] = None,
    live: bool = True,
    log_file_handler: typing.Optional[typing.IO[str]] = None,
) -> (str, str, int):
    if workdir:
        command = f"cd {workdir}; " + command
    if args:
        command += " " + " ".join(args)

    stdin_stream, stdout_stream, stderr_stream = ssh_client.exec_command(command)

    if stdin:
        stdin_stream.write(stdin)
        stdin_stream.close()

    stdout = _handle_command_stdout(stdout_stream, log_file_handler, live, remote=True)
    stderr = stderr_stream.read()
    exit_status = stdout_stream.channel.recv_exit_status()

    return stdout, stderr, exit_status


def _handle_command_stdout(
    stdout_stream: typing.Union[typing.IO[bytes], paramiko.channel.ChannelFile],
    log_file_handler: typing.Optional[typing.IO[str]] = None,
    live: bool = True,
    remote: bool = False,
) -> str:
    def _maybe_decode(text: typing.Union[str, bytes]) -> str:
        if isinstance(text, bytes):
            return text.decode(sys.stdout.encoding)
        return text

    def _write_to_log_file(text: bytes):
        if log_file_handler:
            log_file_handler.write(_maybe_decode(text))

    stdout = ""
    if live:
        for line in iter(stdout_stream.readline, b""):
            # remote stream never ends, so we need to break when there's no more data
            if remote and not line:
                break
            stdout += str(line)
            sys.stdout.write(_maybe_decode(line))
            _write_to_log_file(line)
    else:
        stdout = stdout_stream.read()
        _write_to_log_file(stdout)

    return stdout
