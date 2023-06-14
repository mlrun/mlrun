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

import argparse
import subprocess
from typing import List

import giturlparse
import paramiko
import yaml


class MlRunDeployer(object):
    class Consts(object):
        mandatory_fields = ["APP_NODE", "USER", "PASSWORD"]
        fetch_folder = "/tmp/mlrun_temp"

    def __init__(self, args):
        self._verbose = args.verbose
        self._config = yaml.safe_load(args.config)
        for key in self.Consts.mandatory_fields:
            if self._config.get(key, None) is None:
                raise RuntimeError(f"Mandatory {key} not defined")

    def _exec_local(self, cmd: List[str]) -> str:
        if self._verbose:
            print("Exec local: ", " ".join(cmd))
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    def _exec_remote(self, cmd: List[str], set_work_dir=False, live=False):
        if set_work_dir:
            cmd_str = f"cd {self.Consts.fetch_folder}; " + " ".join(cmd)
        else:
            cmd_str = " ".join(cmd)
        if self._verbose:
            print("Exec remote: ", cmd_str)

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
            stdout = stdout_stream.read()

        stderr = stderr_stream.read().decode("utf8")

        exit_status = stdout_stream.channel.recv_exit_status()

        if exit_status:
            raise RuntimeError(
                f"Command '{cmd_str}' exited with failure ({exit_status})\n{stderr}"
            )

    def _find_git_repo(self) -> str:
        cmd = ["git", "remote", "get-url", "origin"]

        return self._exec_local(cmd).strip()

    def _find_current_git_branch(self) -> str:
        cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        return self._exec_local(cmd).strip()

    def _connect_to_node(self):
        if self._verbose:
            print("Connecting to {}", self._config["APP_NODE"])

        self._ssh_client = paramiko.SSHClient()
        self._ssh_client.set_missing_host_key_policy(paramiko.WarningPolicy)
        self._ssh_client.connect(
            self._config["APP_NODE"],
            username=self._config["USER"],
            password=self._config["PASSWORD"],
        )

    def do_replacing(self):
        def get_config(self, var, func):
            val = self._config.get(var, None)
            if val is None:
                val = func()
                print(f"Using {val}, as {var} was not set")
            return val

        repo = get_config(self, "GIT_REPO", self._find_git_repo)
        branch = get_config(self, "GIT_BRANCH", self._find_current_git_branch)

        if repo.startswith("git"):
            p = giturlparse.parse(repo)
            repo = p.url2https

        self._connect_to_node()

        try:
            self._exec_remote(["rm", "-rf", self.Consts.fetch_folder])
            self._exec_remote(["mkdir", "-p", self.Consts.fetch_folder])
            print("Cloning repo on remote")
            self._exec_remote(
                ["git", "clone", "--progress", "--branch", branch, repo],
                set_work_dir=True,
                live=True,
            )
            replace_cmd = ["cd", "mlrun", ";", "./replace_mlrun.py"]
            if self._verbose:
                replace_cmd.append("--verbose")
            print("Running replace commands on remote")
            self._exec_remote(replace_cmd, set_work_dir=True, live=True)
            print("Deployed branch sucessfully! Yay!")
        finally:
            self._ssh_client.close()


def main():
    parser = argparse.ArgumentParser(description="Replace mlrun on an app node")

    parser.add_argument(
        "--verbose", help="Print what we are doing", action="store_true"
    )

    parser.add_argument(
        "-c",
        "--config",
        help="Config file (default: %(default)s)",
        default="deploy_env.yml",
        type=argparse.FileType("r"),
        metavar="FILE",
    )

    args = parser.parse_args()

    MlRunDeployer(args).do_replacing()


if __name__ == "__main__":
    main()
