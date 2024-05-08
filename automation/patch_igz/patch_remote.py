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

import io
import json
import logging
import os
import shlex
import subprocess
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


class MLRunPatcher:
    class Consts:
        mandatory_fields = {"DATA_NODES", "SSH_USER", "SSH_PASSWORD", "DOCKER_REGISTRY"}
        api_container = "mlrun-api"
        log_collector_container = "mlrun-log-collector"

    def __init__(self, conf_file, patch_file, reset_db, tag, log_collector):
        self._config = yaml.safe_load(conf_file)
        patch_yaml_data = yaml.safe_load(patch_file)
        self._deploy_patch = json.dumps(patch_yaml_data)
        self._reset_db = reset_db
        self._tag = tag
        self._patch_log_collector = bool(log_collector)
        self._validate_config()

    def patch_mlrun_api(self):
        vers = self._get_current_version()

        nodes = self._config["DATA_NODES"]
        if not isinstance(nodes, list):
            nodes = [nodes]

        image_tag = self._get_image_tag(vers)
        built_api_image = self._make_mlrun("api", image_tag, "mlrun-api")

        built_images = [built_api_image]
        built_log_collector_image = None
        if self._patch_log_collector:
            built_log_collector_image = self._make_mlrun(
                "log-collector", image_tag, "log-collector"
            )
            built_images.append(built_log_collector_image)

        self._docker_login_if_configured()

        self._push_docker_images(built_images)

        node = nodes[0]
        self._connect_to_node(node)
        try:
            self._replace_deploy_policy()
            self._replace_deployment_images(self.Consts.api_container, built_api_image)
            if self._patch_log_collector:
                # sanity
                if not built_log_collector_image:
                    raise RuntimeError("Log collector image not built")
                self._replace_deployment_images(
                    self.Consts.log_collector_container, built_log_collector_image
                )

            if self._reset_db:
                self._reset_mlrun_db()
                self._wait_deployment_ready()
            else:
                self._rollout_deployment()
                self._wait_deployment_ready()

        finally:
            out = self._exec_remote(
                [
                    "kubectl",
                    "-n",
                    "default-tenant",
                    "get",
                    "pods",
                ],
            )

            for line in out.splitlines():
                if self.Consts.api_container in line:
                    logger.info(line)
                elif self.Consts.log_collector_container in line:
                    logger.info(line)

            self._disconnect_from_node()

        logger.info(
            "Deployed branch successfully! Yay! (Note this may not survive system restarts)"
        )

    def _docker_login_if_configured(self):
        registry_username = self._config.get("REGISTRY_USERNAME")
        registry_password = self._config.get("REGISTRY_PASSWORD")
        if registry_username is not None:
            self._exec_local(
                [
                    "docker",
                    "login",
                    "--username",
                    registry_username,
                    "--password",
                    registry_password,
                ],
                live=True,
            )

    def _validate_config(self):
        missing_fields = self.Consts.mandatory_fields - set(self._config.keys())
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
        if "unstable" in self._tag:
            return "unstable"
        return self._tag

    def _make_mlrun(self, target, image_tag, image_name) -> str:
        logger.info(f"Building mlrun docker image: {target}:{image_tag}")
        env = {
            "MLRUN_VERSION": image_tag,
            "MLRUN_DOCKER_REPO": self._config["DOCKER_REGISTRY"],
        }
        cmd = ["make", target]
        self._exec_local(cmd, live=True, env=env)
        return f"{self._config['DOCKER_REGISTRY']}/{image_name}:{image_tag}"

    def _connect_to_node(self, node):
        logger.debug(f"Connecting to {node}")

        self._ssh_client = paramiko.SSHClient()
        self._ssh_client.set_missing_host_key_policy(paramiko.WarningPolicy)
        self._ssh_client.connect(
            node,
            username=self._config["SSH_USER"],
            password=self._config["SSH_PASSWORD"],
        )

    def _disconnect_from_node(self):
        self._ssh_client.close()

    def _push_docker_images(self, built_images):
        logger.info(f"Pushing mlrun docker images: {built_images}")

        for image in built_images:
            self._exec_local(
                [
                    "docker",
                    "push",
                    image,
                ],
                live=True,
            )

    def _replace_deploy_policy(self):
        logger.info("Patching mlrun-api-chief deployment")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "patch",
                "deployment",
                "mlrun-api-chief",
                "-p",
                f"{self._deploy_patch}",
            ]
        )

        logger.info("Patching mlrun-api-worker deployment")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "patch",
                "deployment",
                "mlrun-api-worker",
                "-p",
                f"{self._deploy_patch}",
            ]
        )

    def _replace_deployment_images(self, container, built_image):
        logger.info("Replace mlrun-api-chief")
        if self._config.get("OVERWRITE_IMAGE_REGISTRY"):
            built_image = built_image.replace(
                self._config["DOCKER_REGISTRY"],
                self._config["OVERWRITE_IMAGE_REGISTRY"],
            )

        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "set",
                "image",
                "deployment/mlrun-api-chief",
                f"{container}={built_image}",
            ]
        )

        logger.info("Replace mlrun-api-worker")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "set",
                "image",
                "deployment/mlrun-api-worker",
                f"{container}={built_image}",
            ]
        )

    def _rollout_deployment(self):
        logger.info("Restarting deployment")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "rollout",
                "restart",
                "deployment",
                "mlrun-api-chief",
                "mlrun-api-worker",
            ]
        )

    def _wait_deployment_ready(self):
        logger.info("Waiting for mlrun-api-chief to become ready")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "rollout",
                "status",
                "deployment",
                "mlrun-api-chief",
                "--timeout=120s",
            ],
            live=True,
        )
        logger.info("Waiting for mlrun-api-worker to become ready")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "rollout",
                "status",
                "deployment",
                "mlrun-api-worker",
                "--timeout=120s",
            ],
            live=True,
        )
        logger.info("Waiting for mlrun-api to become ready")
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
                "--timeout=240s",
            ],
            live=True,
        )

    def _reset_mlrun_db(self):
        curr_worker_replicas = (
            self._exec_remote(
                [
                    "kubectl",
                    "-n",
                    "default-tenant",
                    "get",
                    "deployment",
                    "mlrun-api-worker",
                    "-o=jsonpath='{.spec.replicas}'",
                ]
            )
            .strip()
            .strip("'")
        )
        logger.info("Detected current worker replicas: %s", curr_worker_replicas)

        logger.info("Scaling down mlrun-api-chief")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "scale",
                "deploy",
                "mlrun-api-chief",
                "--replicas=0",
            ],
        )
        logger.info("Scaling down mlrun-api-worker")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "scale",
                "deploy",
                "mlrun-api-worker",
                "--replicas=0",
            ],
        )

        logger.info("Waiting for mlrun-api-chief to go down")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "wait",
                "pods",
                "-l",
                "app.kubernetes.io/sub-component=chief",
                "--for=delete",
                "--timeout=60s",
            ],
            live=True,
        )

        logger.info("Waiting for mlrun-api-worker to go down")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "wait",
                "pods",
                "-l",
                "app.kubernetes.io/sub-component=worker",
                "--for=delete",
                "--timeout=60s",
            ],
            live=True,
        )

        mlrun_db_pod = self._get_db_pod()
        if mlrun_db_pod is None:
            raise RuntimeError("Unable to find DB pod")

        logger.info("Reset DB")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "exec",
                "-it",
                mlrun_db_pod,
                "-c",
                "mlrun-db",
                "--",
                "mysql",
                "-u",
                self._config["DB_USER"],
                "-S",
                "/var/run/mysqld/mysql.sock",
                "-e",
                "DROP DATABASE mlrun; CREATE DATABASE mlrun",
            ],
            live=True,
        )

        logger.info("Scaling up mlrun-api-chief")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "scale",
                "deploy",
                "mlrun-api-chief",
                "--replicas=1",
            ],
        )
        logger.info("Scaling up mlrun-api-worker")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "scale",
                "deploy",
                "mlrun-api-worker",
                f"--replicas={curr_worker_replicas}",
            ],
        )

    def _get_db_pod(self):
        cmd = ["kubectl", "-n", "default-tenant", "get", "pod"]

        for line in self._exec_remote(cmd).splitlines()[1:]:
            if "mlrun-db" in line:
                return line.split()[0]

    @staticmethod
    def _get_image_tag(tag) -> str:
        return f"{tag}"

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
    help="Deploy the log collector as well",
)
def main(verbose, config, patch_file, reset_db, tag, log_collector):
    if verbose:
        coloredlogs.set_level(logging.DEBUG)

    MLRunPatcher(config, patch_file, reset_db, tag, log_collector).patch_mlrun_api()


if __name__ == "__main__":
    main()
