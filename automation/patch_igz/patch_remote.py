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
import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from typing import List

import click
import coloredlogs
import paramiko
import yaml

log_level = logging.INFO
fmt = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=log_level)
logger = logging.getLogger("mlrun-patch")
coloredlogs.install(level=log_level, logger=logger, fmt=fmt)


class MLRunPatcher(object):
    class Consts(object):
        mandatory_fields = {"DATA_NODES", "SSH_USER", "SSH_PASSWORD", "DOCKER_REGISTRY"}

    def __init__(self, conf_file):
        self._config = yaml.safe_load(conf_file)
        self._validate_config()

    def patch_mlrun_api(self):
        vers = self._get_current_version()

        nodes = self._config["DATA_NODES"]
        if not isinstance(nodes, list):
            nodes = [nodes]

        image_tag = self._get_image_tag(vers)
        built_image = self._make_mlrun_api(image_tag)

        self._docker_login_if_configured()

        self._push_docker_image(built_image)

        node = nodes[0]
        self._connect_to_node(node)
        try:
            self._replace_deploy_policy()
            self._replace_deployment_images(built_image)
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
                if "mlrun-api" in line:
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

    @contextmanager
    def _add_mlrun_src_to_path(self):
        mlrun_src = os.path.dirname(os.path.abspath(__file__)) + "/.."
        sys.path.append(mlrun_src)
        yield sys.path
        sys.path.remove(mlrun_src)

    def _get_current_version(self) -> str:
        with self._add_mlrun_src_to_path():
            from version import version_file

            return str(version_file.read_unstable_version_prefix())

    def _make_mlrun_api(self, image_tag) -> str:
        logger.info("Building mlrun-api docker image")
        os.environ["MLRUN_VERSION"] = image_tag
        os.environ["MLRUN_DOCKER_REPO"] = self._config["DOCKER_REGISTRY"]
        cmd = ["make", "api"]
        self._exec_local(cmd, live=True)
        return f"{self._config['DOCKER_REGISTRY']}/mlrun-api:{image_tag}"

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

    def _push_docker_image(self, built_image):
        logger.info("Pushing mlrun-api docker image")

        self._exec_local(
            [
                "docker",
                "push",
                built_image,
            ],
            live=True,
        )

    def _replace_deploy_policy(self):
        policy = """'{"spec":{"template":{"spec":{"containers":[{"name":"mlrun-api","imagePullPolicy":"Always"}]}}}}'"""
        logger.info("Change mlrun-api-chief pull policy")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "patch",
                "deployment",
                "mlrun-api-chief",
                "-p",
                policy,
            ]
        )
        logger.info("Change mlrun-api-worker pull policy")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "patch",
                "deployment",
                "mlrun-api-worker",
                "-p",
                policy,
            ]
        )

    def _replace_deployment_images(self, built_image):
        logger.info("Replace mlrun-api-chief")
        self._exec_remote(
            [
                "kubectl",
                "-n",
                "default-tenant",
                "set",
                "image",
                "deployment/mlrun-api-chief",
                f"mlrun-api={built_image}",
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
                f"mlrun-api={built_image}",
            ]
        )

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

    @staticmethod
    def _get_image_tag(tag) -> str:
        return f"{tag}"

    @staticmethod
    def _execute_local_proc_interactive(cmd):
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            yield line
        proc.stdout.close()
        ret_code = proc.wait()
        if ret_code:
            raise subprocess.CalledProcessError(ret_code, cmd)

    def _exec_local(self, cmd: List[str], live=False) -> str:
        logger.debug("Exec local: %s", " ".join(cmd))
        buf = io.StringIO()
        for line in self._execute_local_proc_interactive(cmd):
            buf.write(line)
            if live:
                print(line, end="")
        output = buf.getvalue()
        return output

    def _exec_remote(self, cmd: List[str], live=False) -> str:
        cmd_str = " ".join(cmd)

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
@click.option("--verbose", is_flag=True, help="Print what we are doing")
@click.option(
    "-c",
    "--config",
    help="Config file",
    default="automation/patch_igz/patch_env.yml",
    type=click.File(mode="r"),
    show_default=True,
)
def main(verbose, config):
    if verbose:
        coloredlogs.set_level(logging.DEBUG)

    MLRunPatcher(config).patch_mlrun_api()


if __name__ == "__main__":
    main()
