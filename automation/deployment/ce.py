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

import sys
from typing import Optional

import click
from deployer import CommunityEditionDeployer

common_options = [
    click.option(
        "-v",
        "--verbose",
        is_flag=True,
        help="Enable debug logging",
    ),
    click.option(
        "-f",
        "--log-file",
        help="Path to log file. If not specified, will log only to stdout",
    ),
    click.option(
        "--remote",
        help="Remote host to deploy to. If not specified, will deploy to the local host",
    ),
    click.option(
        "--remote-ssh-username",
        help="Username to use when connecting to the remote host via SSH. "
        "If not specified, will use MLRUN_REMOTE_SSH_USERNAME environment variable",
    ),
    click.option(
        "--remote-ssh-password",
        help="Password to use when connecting to the remote host via SSH. "
        "If not specified, will use MLRUN_REMOTE_SSH_PASSWORD environment variable",
    ),
]

common_deployment_options = [
    click.option(
        "-n",
        "--namespace",
        default="mlrun",
        help="Namespace to install the platform in. Defaults to 'mlrun'",
    ),
    click.option(
        "--registry-secret-name",
        help="Name of the secret containing the credentials for the container registry to use for storing images",
    ),
    click.option(
        "--sqlite",
        help="Path to sqlite file to use as the mlrun database. If not supplied, will use MySQL deployment",
    ),
]


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


def order_click_options(func):
    func.__click_params__ = list(
        reversed(sorted(func.__click_params__, key=lambda option: option.name))
    )
    return func


@click.group(help="MLRun Community Edition Deployment CLI Tool")
def cli():
    pass


@cli.command(help="Deploy (or upgrade) MLRun Community Edition")
@order_click_options
@click.option(
    "-mv",
    "--mlrun-version",
    help="Version of mlrun to install. If not specified, will install the latest version",
)
@click.option(
    "--chart",
    help="Specify a custom helm chart name or local path",
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
    "--override-mlrun-log-collector-image",
    help="Override the mlrun-log-collector image. Format: <repo>:<tag>",
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
    "--force-enable-pipelines",
    is_flag=True,
    help="Install Kubeflow Pipelines even when the available CPU architecture doesn't support it",
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
    "--disable-log-collector",
    is_flag=True,
    help="Disable the mlrun API log collector sidecar and use legacy mode instead",
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
    help="Install the mlrun chart in local minikube",
)
@click.option(
    "--set",
    "set_",
    help="Set custom values for the mlrun chart. Format: <key>=<value>",
    multiple=True,
)
@click.option(
    "--upgrade",
    is_flag=True,
    help="Upgrade the existing mlrun installation",
)
@click.option(
    "--skip-registry-validation",
    is_flag=True,
    help="Skip validation of the registry URL",
)
@add_options(common_options)
@add_options(common_deployment_options)
def deploy(
    verbose: bool = False,
    log_file: Optional[str] = None,
    namespace: str = "mlrun",
    remote: Optional[str] = None,
    remote_ssh_username: Optional[str] = None,
    remote_ssh_password: Optional[str] = None,
    mlrun_version: Optional[str] = None,
    chart: Optional[str] = None,
    chart_version: Optional[str] = None,
    registry_url: Optional[str] = None,
    registry_secret_name: Optional[str] = None,
    registry_username: Optional[str] = None,
    registry_password: Optional[str] = None,
    override_mlrun_api_image: Optional[str] = None,
    override_mlrun_log_collector_image: Optional[str] = None,
    override_mlrun_ui_image: Optional[str] = None,
    override_jupyter_image: Optional[str] = None,
    disable_pipelines: bool = False,
    force_enable_pipelines: bool = False,
    disable_prometheus_stack: bool = False,
    disable_spark_operator: bool = False,
    disable_log_collector: bool = False,
    skip_registry_validation: bool = False,
    sqlite: Optional[str] = None,
    devel: bool = False,
    minikube: bool = False,
    upgrade: bool = False,
    set_: Optional[list[str]] = None,
):
    deployer = CommunityEditionDeployer(
        namespace=namespace,
        log_level="debug" if verbose else "info",
        log_file=log_file,
        remote=remote,
        remote_ssh_username=remote_ssh_username,
        remote_ssh_password=remote_ssh_password,
        chart_name=chart,
    )
    deployer.deploy(
        registry_url=registry_url,
        registry_username=registry_username,
        registry_password=registry_password,
        registry_secret_name=registry_secret_name,
        mlrun_version=mlrun_version,
        chart_name=chart,
        chart_version=chart_version,
        override_mlrun_api_image=override_mlrun_api_image,
        override_mlrun_log_collector_image=override_mlrun_log_collector_image,
        override_mlrun_ui_image=override_mlrun_ui_image,
        override_jupyter_image=override_jupyter_image,
        disable_pipelines=disable_pipelines,
        force_enable_pipelines=force_enable_pipelines,
        disable_prometheus_stack=disable_prometheus_stack,
        disable_spark_operator=disable_spark_operator,
        disable_log_collector=disable_log_collector,
        skip_registry_validation=skip_registry_validation,
        devel=devel,
        minikube=minikube,
        sqlite=sqlite,
        upgrade=upgrade,
        custom_values=set_,
    )


@cli.command(help="Uninstall MLRun Community Edition Deployment")
@order_click_options
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
@add_options(common_options)
@add_options(common_deployment_options)
def delete(
    verbose: bool = False,
    log_file: Optional[str] = None,
    namespace: str = "mlrun",
    remote: Optional[str] = None,
    remote_ssh_username: Optional[str] = None,
    remote_ssh_password: Optional[str] = None,
    registry_secret_name: Optional[str] = None,
    skip_uninstall: bool = False,
    skip_cleanup_registry_secret: bool = False,
    cleanup_volumes: bool = False,
    cleanup_namespace: bool = False,
    sqlite: Optional[str] = None,
):
    deployer = CommunityEditionDeployer(
        namespace=namespace,
        log_level="debug" if verbose else "info",
        log_file=log_file,
        remote=remote,
        remote_ssh_username=remote_ssh_username,
        remote_ssh_password=remote_ssh_password,
    )
    deployer.delete(
        skip_uninstall=skip_uninstall,
        sqlite=sqlite,
        cleanup_registry_secret=not skip_cleanup_registry_secret,
        cleanup_volumes=cleanup_volumes,
        cleanup_namespace=cleanup_namespace,
        registry_secret_name=registry_secret_name,
    )


@cli.command(
    help="Patch MLRun Community Edition Deployment images to minikube. "
    "Useful if overriding images and running in minikube"
)
@order_click_options
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
@add_options(common_options)
def patch_minikube_images(
    remote: Optional[str] = None,
    remote_ssh_username: Optional[str] = None,
    remote_ssh_password: Optional[str] = None,
    verbose: bool = False,
    log_file: Optional[str] = None,
    mlrun_api_image: Optional[str] = None,
    mlrun_ui_image: Optional[str] = None,
    jupyter_image: Optional[str] = None,
):
    deployer = CommunityEditionDeployer(
        namespace="",
        log_level="debug" if verbose else "info",
        log_file=log_file,
        remote=remote,
        remote_ssh_username=remote_ssh_username,
        remote_ssh_password=remote_ssh_password,
    )
    deployer.patch_minikube_images(
        mlrun_api_image=mlrun_api_image,
        mlrun_ui_image=mlrun_ui_image,
        jupyter_image=jupyter_image,
    )


if __name__ == "__main__":
    try:
        cli()
    except Exception as exc:
        print("Unexpected error:", exc)
        sys.exit(1)
