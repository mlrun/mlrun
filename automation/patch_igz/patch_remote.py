#!/usr/bin/env python3
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
#
import datetime
import io
import json
import logging
import os
import shlex
import subprocess
import time
import typing

import click
import coloredlogs
import paramiko
import yaml

log_level = logging.INFO
fmt = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=log_level)
logger = logging.getLogger("mlrun-patch")
coloredlogs.install(level=log_level, logger=logger, fmt=fmt)


class Constants:
    mandatory_fields = {"DATA_NODES", "SSH_USER", "SSH_PASSWORD", "DOCKER_REGISTRY"}
    api_container = "mlrun-api"
    log_collector_container = "mlrun-log-collector"
    api = "api"
    mlrun = "mlrun"
    mlrun_kfp = "mlrun-kfp"
    log_collector = "log-collector"
    targets_to_image_name = {
        api: api_container,
        mlrun: mlrun,
        mlrun_kfp: mlrun_kfp,
        log_collector: log_collector,
    }


class MLRunPatcher:
    def __init__(
        self,
        conf_file: str,
        patch_file: str,
        reset_db: bool,
        image_tag: str,
        patch_log_collector_image: bool,
        patch_mlrun_image: bool,
        skip_patch_api: bool,
        patch_alerts: bool,
    ):
        self._config = yaml.safe_load(conf_file)
        patch_yaml_data = yaml.safe_load(patch_file)
        self._deploy_patch = patch_yaml_data
        self._reset_db = reset_db
        self._image_tag = image_tag
        self._patch_log_collector_image = bool(patch_log_collector_image)
        self._validate_config()
        self._patch_mlrun_image = patch_mlrun_image
        self._skip_patch_api = skip_patch_api
        self._patch_alerts = patch_alerts

        if self._skip_patch_api and self._patch_alerts:
            raise ValueError("Cannot skip api and patch alerts at the same time")

        cluster_data_nodes = self._config["DATA_NODES"]
        if not isinstance(cluster_data_nodes, list):
            cluster_data_nodes = [cluster_data_nodes]
        self._cluster_data_nodes = cluster_data_nodes
        self._deployments = [
            "mlrun-api-chief",
            "mlrun-api-worker",
        ]
        if self._patch_alerts:
            self._deployments.append("mlrun-alerts")

    def patch(self):
        image_tag = self._get_current_version()
        targets = []
        if not self._skip_patch_api:
            targets.append(Constants.api)
        if self._patch_mlrun_image:
            targets.extend([Constants.mlrun, Constants.mlrun_kfp])
        if self._patch_log_collector_image:
            targets.append(Constants.log_collector)
        if not targets:
            raise ValueError("No targets to patch")
        self._docker_login_if_configured()

        target_to_built_images = self._make_targets(
            targets=targets,
            image_tag=image_tag,
        )
        # Build and push Docker images
        built_images = self._tag_images_for_multi_node_registries(
            target_to_built_images.values()
        )
        self._push_docker_images(built_images)

        # Connect to the first node and start deployment patching process
        node = self._cluster_data_nodes[0]
        self._connect_to_node(node)
        if self._patch_log_collector_image:
            self._replace_deployment_images(
                Constants.log_collector_container,
                target_to_built_images[Constants.log_collector],
            )
        if not self._skip_patch_api:
            try:
                # Replace deployment policies and images
                self._patch_deployment_from_file()
                self._replace_deployment_images(
                    Constants.api_container, target_to_built_images[Constants.api]
                )

                # Reset or rollout deployment as necessary
                if self._reset_db:
                    self._reset_mlrun_db()
                else:
                    self._rollout_deployment()

                self._wait_deployment_ready()

            finally:
                # Check status of pods after deployment
                out = self._exec_remote(
                    ["kubectl", "-n", "default-tenant", "get", "pods"]
                )
                for line in out.splitlines():
                    if (
                        Constants.api_container in line
                        or Constants.log_collector_container in line
                    ):
                        logger.info(line)

                self._disconnect_from_node()

        logger.info(
            "Deployed branch successfully! (Note: This may not survive system restarts)"
        )

    def _docker_login_if_configured(self):
        registry_username = self._config.get("REGISTRY_USERNAME")
        registry_password = self._config.get("REGISTRY_PASSWORD")
        docker_registry = self._config.get("DOCKER_REGISTRY")
        if not registry_username:
            return
        command = [
            "docker",
            "login",
            docker_registry or "",
            "--username",
            registry_username,
            "--password-stdin",
        ]
        completed_process = subprocess.run(
            command, input=registry_password.encode() + b"\n", capture_output=True
        )
        if completed_process.returncode != 0:
            raise RuntimeError(
                f"Failed to login to docker registry. Error: {completed_process.stderr}"
            )

    def _validate_config(self):
        missing_fields = Constants.mandatory_fields - set(self._config.keys())
        if len(missing_fields) > 0:
            raise RuntimeError(f"Mandatory options not defined: {missing_fields}")

        registry_username = self._config.get("REGISTRY_USERNAME")
        registry_password = self._config.get("REGISTRY_PASSWORD")
        if registry_username is not None and registry_password is None:
            raise RuntimeError(
                "REGISTRY_USERNAME defined, yet REGISTRY_PASSWORD is not defined"
            )

        if self._reset_db and "DB_USER" not in self._config:
            raise RuntimeError("Must define DB_USER if requesting DB reset")

    def _get_current_version(self) -> str:
        if "unstable" in self._image_tag:
            return "unstable"
        return self._image_tag

    def _make_targets(
        self,
        targets: list[str],
        image_tag: str,
    ) -> dict[str, str]:
        for target in targets:
            logger.info(f"Building mlrun docker images: {target}:{image_tag}")

        mlrun_docker_registry = self._config["DOCKER_REGISTRY"].rstrip("/")
        mlrun_docker_repo = self._config.get("DOCKER_REPO")

        if mlrun_docker_repo:
            mlrun_docker_registry = (
                f"{mlrun_docker_registry}/{mlrun_docker_repo.rstrip('/')}"
            )

        env = {
            "MLRUN_VERSION": image_tag,
            "MLRUN_DOCKER_REPO": mlrun_docker_registry,
        }
        cmd = ["make"]
        cmd.extend(targets)
        self._exec_local(cmd, live=True, env=env)

        return {
            target: f"{mlrun_docker_registry}/{Constants.targets_to_image_name[target]}:{image_tag}"
            for target in targets
        }

    def _connect_to_node(self, node):
        logger.debug(f"Connecting to {node}")

        self._ssh_client = paramiko.SSHClient()
        self._ssh_client.set_missing_host_key_policy(paramiko.MissingHostKeyPolicy)
        self._ssh_client.connect(
            node,
            username=self._config["SSH_USER"],
            password=self._config["SSH_PASSWORD"],
            look_for_keys=False,
            allow_agent=False,
        )

    def _disconnect_from_node(self):
        self._ssh_client.close()

    def _tag_images_for_multi_node_registries(self, built_images):
        if self._config.get("SKIP_MULTI_NODE_PUSH") == "true":
            return

        resolve_built_images = []
        for built_image in built_images:
            for node in self._cluster_data_nodes:
                if node in built_image:
                    resolve_built_images.append(built_image)
                    for replacement_node in self._cluster_data_nodes:
                        if replacement_node != node:
                            replaced_built_image = built_image.replace(
                                node, replacement_node
                            )
                            self._exec_local(
                                [
                                    "docker",
                                    "tag",
                                    built_image,
                                    replaced_built_image,
                                ],
                                live=True,
                            )
                            resolve_built_images.append(replaced_built_image)

                    # Once we found the node configured in the built_image we can stop because it is only possible
                    # to specify one node when building the image
                    break

        return resolve_built_images or built_images

    def _push_docker_images(self, built_images):
        logger.info(f"Pushing mlrun docker images: {built_images}")
        for image in built_images:
            self._exec_local(
                cmd=[
                    "docker",
                    "push",
                    image,
                ],
                live=True,
            )

    def _patch_deployment_from_file(self):
        for deployment in self._deployments:
            logger.info(f"Patching deployment {deployment}")
            deployment_patch = (
                self._deploy_patch["mlrun_alerts"]
                if "-alerts" in deployment
                else self._deploy_patch["mlrun_api"]
            )
            self._exec_remote(
                [
                    "kubectl",
                    "-n",
                    "default-tenant",
                    "patch",
                    "deployment",
                    deployment,
                    "-p",
                    f"{json.dumps(deployment_patch)}",
                ]
            )

    def _replace_deployment_images(self, container, built_image):
        if self._config.get("OVERWRITE_IMAGE_REGISTRY"):
            docker_registry, overwrite_registry = self._resolve_overwrite_registry()
            built_image = built_image.replace(
                docker_registry,
                overwrite_registry,
            )

        for deployment in self._deployments:
            logger.info(f"Replace container {container} for {deployment}")
            self._exec_remote(
                [
                    "kubectl",
                    "-n",
                    "default-tenant",
                    "set",
                    "image",
                    f"deployment/{deployment}",
                    f"{container}={built_image}",
                ]
            )

    def _rollout_deployment(self):
        for deployment in self._deployments:
            logger.info(f"Restarting deployment {deployment}")
            self._exec_remote(
                [
                    "kubectl",
                    "-n",
                    "default-tenant",
                    "rollout",
                    "restart",
                    f"deployment/{deployment}",
                ]
            )

    def _wait_deployment_ready(self):
        for deployment in self._deployments:
            logger.info(f"Waiting for {deployment} to become ready")
            self._exec_remote(
                [
                    "kubectl",
                    "-n",
                    "default-tenant",
                    "rollout",
                    "status",
                    "deployment",
                    deployment,
                    "--timeout=120s",
                ],
                live=True,
            )

        self._wait_for_pods_readiness()

    def _wait_for_pods_readiness(self):
        """
        Waits for a pod to become ready.
        Since some deployments' strategy is RollingUpdate, using 'kubectl wait --for condition=Ready' sometimes times
        out because it waits for the terminating pod to be ready. To mitigate it, we use smaller timeouts and retries
        """

        logger.info("Waiting for mlrun pods to become ready")

        timeout = datetime.datetime.now() + datetime.timedelta(seconds=300)
        while datetime.datetime.now() < timeout:
            try:
                self._exec_remote(
                    [
                        "kubectl",
                        "-n",
                        "default-tenant",
                        "wait",
                        "pods",
                        "-l",
                        "app.kubernetes.io/name=mlrun",
                        "--for",
                        "condition=Ready",
                        "--timeout=20s",
                    ],
                    live=True,
                )
                break
            except RuntimeError:
                # Retry until timeout is reached
                time.sleep(5)

    def _reset_mlrun_db(self):
        mlrun_api_services_deployment_selector = (
            "app.kubernetes.io/component!=ui,"
            "app.kubernetes.io/component!=db,"
            "app.kubernetes.io/name=mlrun"
        )

        # in form of "deployment1:replicas1\ndeployment2:replicas2\n..."
        current_mlrun_api_services_output = (
            self._exec_remote(
                [
                    "kubectl",
                    "-n",
                    "default-tenant",
                    "get",
                    "deployments",
                    "--selector",
                    mlrun_api_services_deployment_selector,
                    '-o=jsonpath=\'{range .items[*]}{.metadata.name}{":"}{.spec.replicas}{"\\n"}{end}\'',
                ]
            )
            .strip()
            .strip("'")
            .strip()
        )
        # in form of {"deployment1": replicas1, "deployment2": replicas2, ...}
        current_non_mlrun_db_services = {
            deployment: int(replicas)
            for deployment, replicas in dict(
                output.split(":")
                for output in current_mlrun_api_services_output.split()
            ).items()
        }

        logger.info("Scaling down services")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "scale",
                "deploy",
                "--selector",
                mlrun_api_services_deployment_selector,
                "--replicas=0",
            ],
        )
        logger.info("Waiting for pods to go down")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "wait",
                "--for=delete",
                "--timeout=300s",
                "pod",
                "--selector",
                mlrun_api_services_deployment_selector,
            ],
            live=True,
        )

        logger.info("Reset DB")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "exec",
                "-it",
                "deployment/mlrun-db",
                "--container",
                "mlrun-db",
                "--",
                "mysql",
                "--user",
                self._config["DB_USER"],
                "--socket",
                "/var/run/mysqld/mysql.sock",
                "--execute",
                "DROP DATABASE mlrun; CREATE DATABASE mlrun",
            ],
            live=True,
        )

        for deployment, replicas in current_non_mlrun_db_services.items():
            logger.info(f"Scaling up {deployment} with {replicas}")
            self._exec_remote(
                [
                    "kubectl",
                    "-n",
                    "default-tenant",
                    "scale",
                    "deploy",
                    deployment,
                    f"--replicas={replicas}",
                ],
            )

    @staticmethod
    def _execute_local_proc_interactive(cmd, env=None):
        env = os.environ | (env or {})
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
        yield from proc.stdout
        proc.stdout.close()
        ret_code = proc.wait()
        if ret_code:
            raise subprocess.CalledProcessError(ret_code, cmd)

    def _exec_local(
        self, cmd: list[str], live: bool = False, env: typing.Optional[dict] = None
    ) -> str:
        logger.debug("Exec local: %s", " ".join(cmd))
        buf = io.StringIO()
        for line in self._execute_local_proc_interactive(cmd, env):
            buf.write(line)
            if live:
                print(line, end="")
        output = buf.getvalue()
        return output

    def _exec_remote(self, cmd: list[str], live=False) -> str:
        cmd_str = shlex.join(cmd)
        logger.debug("Exec remote: %s", cmd_str)
        stdin_stream, stdout_stream, stderr_stream = self._ssh_client.exec_command(
            cmd_str
        )

        stdout = ""
        if live:
            while True:
                line = stdout_stream.readline()
                stdout += line
                if not line:
                    break
                print(line, end="")
        else:
            stdout = stdout_stream.read().decode("utf8")

        stderr = stderr_stream.read().decode("utf8")

        exit_status = stdout_stream.channel.recv_exit_status()

        if exit_status:
            raise RuntimeError(
                f"Command '{cmd_str}' finished with failure ({exit_status})\n{stderr}"
            )

        return stdout

    def _resolve_overwrite_registry(self):
        docker_registry = self._config["DOCKER_REGISTRY"]
        overwrite_registry = self._config["OVERWRITE_IMAGE_REGISTRY"]
        if docker_registry.endswith("/"):
            docker_registry = docker_registry[:-1]
        if overwrite_registry.endswith("/"):
            overwrite_registry = overwrite_registry[:-1]

        return docker_registry, overwrite_registry


@click.command(help="mlrun-api deployer to remote system")
@click.option("-v", "--verbose", is_flag=True, help="Print what we are doing")
@click.option(
    "-c",
    "--config",
    help="Config file",
    default="automation/patch_igz/patch_env.yml",
    type=click.File(mode="r"),
    show_default=True,
)
@click.option(
    "-pf",
    "--patch-file",
    help="Kubernetes deployment patch file",
    default="automation/patch_igz/patch-api.yml",
    type=click.File(mode="r"),
    show_default=True,
)
@click.option(
    "-r", "--reset-db", is_flag=True, help="Reset mlrun DB after deploying api"
)
@click.option(
    "-t",
    "--tag",
    default="0.0.0+unstable",
    help="Tag to use for the API. Defaults to unstable (latest and greatest)",
)
@click.option(
    "-lc",
    "--log-collector",
    is_flag=True,
    help="Deploy the log collector",
)
@click.option(
    "-ml",
    "--mlrun",
    is_flag=True,
    help="Deploy the mlrun image",
)
@click.option(
    "-sa",
    "--skip-api",
    is_flag=True,
    help="Deploy the mlrun API image",
)
@click.option(
    "--alerts",
    is_flag=True,
    help="Deploy the the alerts service",
)
def main(
    verbose: bool,
    config: str,
    patch_file: str,
    reset_db: bool,
    tag: str,
    log_collector: bool,
    mlrun: bool,
    skip_api: bool,
    alerts: bool,
):
    if verbose:
        coloredlogs.set_level(logging.DEBUG)

    MLRunPatcher(
        conf_file=config,
        patch_file=patch_file,
        reset_db=reset_db,
        image_tag=tag,
        patch_log_collector_image=log_collector,
        patch_mlrun_image=mlrun,
        skip_patch_api=skip_api,
        patch_alerts=alerts,
    ).patch()


if __name__ == "__main__":
    main()
