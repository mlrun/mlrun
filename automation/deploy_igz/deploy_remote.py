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
import re
import subprocess
import sys
from typing import List

import click
import coloredlogs
import paramiko
import yaml

log_level = logging.INFO
fmt = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=log_level)
logger = logging.getLogger("mlrun-deploy")
coloredlogs.install(level=log_level, logger=logger, fmt=fmt)


class MlRunDeployer(object):
    class Consts(object):
        mandatory_fields = ["DATA_NODES", "USER", "PASSWORD", "DOCKER_REGISTRY"]

    def __init__(self, conf_file):
        self._config = yaml.safe_load(conf_file)
        for key in self.Consts.mandatory_fields:
            if self._config.get(key, None) is None:
                raise RuntimeError(f"Mandatory option {key} not defined")
        self._username = self._config.get("REGISTRY_USERNAME")
        self._password = self._config.get("REGISTRY_PASSWORD")
        if self._username is not None and self._password is None:
            raise RuntimeError(
                "REGISTRY_USERNAME defined, yet REGISTRY_PASSWORD is not defined"
            )

    @staticmethod
    def _get_image_tag(branch, tag):
        return f"v{tag}-{branch}"

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

    def _exec_remote(self, cmd: List[str], live=False):
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

    def _get_current_version(self) -> str:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

        from version import version_file

        return str(version_file.read_unstable_version_prefix())

    def _find_current_git_branch(self) -> str:
        cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        return self._exec_local(cmd).strip()

    def _make_mlrun_api(self, image_tag):
        logger.info("Building mlrun-api docker image")
        os.environ["MLRUN_VERSION"] = image_tag
        os.environ["MLRUN_DOCKER_REPO"] = self._config["DOCKER_REGISTRY"]
        cmd = ["make", "api"]
        return self._exec_local(cmd, live=True).strip()

    def _connect_to_node(self, node):
        logger.debug(f"Connecting to {node}")

        self._ssh_client = paramiko.SSHClient()
        self._ssh_client.set_missing_host_key_policy(paramiko.WarningPolicy)
        self._ssh_client.connect(
            node,
            username=self._config["USER"],
            password=self._config["PASSWORD"],
        )

    def _disconnect_from_node(self):
        self._ssh_client.close()

    def do_replacing(self):
        branch = re.sub(r"\\|@|/|:|#|~|\+", "_", self._find_current_git_branch())
        vers = self._get_current_version()

        nodes = self._config["DATA_NODES"]
        if not isinstance(nodes, list):
            nodes = [nodes]

        image_tag = self._get_image_tag(branch, vers)
        self._make_mlrun_api(image_tag)
        built_image = "{}/mlrun-api:{}".format(
            self._config["DOCKER_REGISTRY"], image_tag
        )

        if self._username is not None:
            self._exec_local(
                [
                    "docker",
                    "login",
                    "--username",
                    self._username,
                    "--password",
                    self._password,
                ],
                live=True,
            )

        logger.info("Pushing mlrun-api docker image")

        self._exec_local(
            [
                "docker",
                "push",
                built_image,
            ],
            live=True,
        )

        policy = """'{"spec":{"template":{"spec":{"containers":[{"name":"mlrun-api","imagePullPolicy":"Always"}]}}}}'"""
        node = nodes[0]
        self._connect_to_node(node)
        try:
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
                    "--timeout=300s",
                ],
                live=True,
            )
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


@click.command(help="mlrun-api deployer to remote system")
@click.option("--verbose", is_flag=True, help="Print what we are doing")
@click.option(
    "-c",
    "--config",
    help="Config file",
    default="automation/deploy_igz/deploy_env.yml",
    type=click.File(mode="r"),
    show_default=True,
)
def main(verbose, config):
    if verbose:
        coloredlogs.set_level(logging.DEBUG)

    MlRunDeployer(config).do_replacing()


if __name__ == "__main__":
    main()
