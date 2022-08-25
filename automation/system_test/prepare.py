# Copyright 2018 Iguazio
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
import json
import logging
import pathlib
import subprocess
import sys
import time

import click
import paramiko
import requests
import semver
import yaml

import mlrun.utils

logger = mlrun.utils.create_logger(level="debug", name="automation")
logging.getLogger("paramiko").setLevel(logging.DEBUG)


class SystemTestPreparer:
    class Constants:
        ci_dir_name = "mlrun-automation"
        homedir = pathlib.Path("/home/iguazio/")
        workdir = homedir / ci_dir_name
        mlrun_code_path = workdir / "mlrun"
        system_tests_env_yaml = pathlib.Path("tests") / "system" / "env.yml"

        git_url = "https://github.com/mlrun/mlrun.git"
        provctl_releases = "https://api.github.com/repos/iguazio/provazio/releases"
        provctl_release_search_amount = 10
        provctl_binary_format = "provctl-{release_name}-linux-amd64"

    def __init__(
        self,
        mlrun_version: str = None,
        mlrun_commit: str = None,
        override_image_registry: str = None,
        override_image_repo: str = None,
        override_mlrun_images: str = None,
        data_cluster_ip: str = None,
        data_cluster_ssh_username: str = None,
        data_cluster_ssh_password: str = None,
        app_cluster_ssh_password: str = None,
        github_access_token: str = None,
        mlrun_dbpath: str = None,
        webapi_direct_http: str = None,
        framesd_url: str = None,
        username: str = None,
        access_key: str = None,
        iguazio_version: str = None,
        spark_service: str = None,
        password: str = None,
        debug: bool = False,
    ):
        self._logger = logger
        self._debug = debug
        self._mlrun_version = mlrun_version
        self._mlrun_commit = mlrun_commit
        self._override_image_registry = (
            override_image_registry.strip().strip("/") + "/"
            if override_image_registry is not None
            else override_image_registry
        )
        self._override_image_repo = override_image_repo
        self._override_mlrun_images = override_mlrun_images
        self._data_cluster_ip = data_cluster_ip
        self._data_cluster_ssh_username = data_cluster_ssh_username
        self._data_cluster_ssh_password = data_cluster_ssh_password
        self._app_cluster_ssh_password = app_cluster_ssh_password
        self._github_access_token = github_access_token
        self._iguazio_version = iguazio_version

        self._env_config = {
            "MLRUN_DBPATH": mlrun_dbpath,
            "V3IO_API": webapi_direct_http,
            "V3IO_FRAMESD": framesd_url,
            "V3IO_USERNAME": username,
            "V3IO_ACCESS_KEY": access_key,
            "MLRUN_SYSTEM_TESTS_DEFAULT_SPARK_SERVICE": spark_service,
        }
        if password:
            self._env_config["V3IO_PASSWORD"] = password

    def prepare_local_env(self):
        self._prepare_env_local()

    def connect_to_remote(self):
        self._logger.info(
            "Connecting to data-cluster", data_cluster_ip=self._data_cluster_ip
        )
        if not self._debug:
            self._ssh_client = paramiko.SSHClient()
            self._ssh_client.set_missing_host_key_policy(paramiko.WarningPolicy)
            self._ssh_client.connect(
                self._data_cluster_ip,
                username=self._data_cluster_ssh_username,
                password=self._data_cluster_ssh_password,
            )

    def run(self):

        self.connect_to_remote()

        # for sanity clean up before starting the run
        self.clean_up_remote_workdir(close_ssh_client=False)

        self._prepare_env_remote()

        self._override_mlrun_api_env()

        provctl_path = self._download_provctl()
        self._patch_mlrun(provctl_path)

    def clean_up_remote_workdir(self, close_ssh_client: bool = True):
        self._logger.info(
            "Cleaning up remote workdir", workdir=str(self.Constants.homedir)
        )
        self._run_command(
            f"rm -rf {self.Constants.workdir}", workdir=str(self.Constants.homedir)
        )

        if close_ssh_client and not self._debug:
            self._ssh_client.close()

    def _run_command(
        self,
        command: str,
        args: list = None,
        workdir: str = None,
        stdin: str = None,
        live: bool = True,
        suppress_errors: bool = False,
        local: bool = False,
        detach: bool = False,
        verbose: bool = True,
    ) -> str:
        workdir = workdir or str(self.Constants.workdir)
        stdout, stderr, exit_status = "", "", 0

        log_command_location = "locally" if local else "on data cluster"

        if verbose:
            self._logger.debug(
                f"Running command {log_command_location}",
                command=command,
                args=args,
                stdin=stdin,
                workdir=workdir,
            )
        if self._debug:
            return ""
        try:
            if local:
                stdout, stderr, exit_status = self._run_command_locally(
                    command, args, workdir, stdin, live
                )
            else:
                stdout, stderr, exit_status = self._run_command_remotely(
                    command,
                    args,
                    workdir,
                    stdin,
                    live,
                    detach,
                    verbose,
                )
            if exit_status != 0 and not suppress_errors:
                raise RuntimeError(f"Command failed with exit status: {exit_status}")
        except (paramiko.SSHException, RuntimeError) as exc:
            if verbose:
                self._logger.error(
                    f"Failed running command {log_command_location}",
                    command=command,
                    error=exc,
                    stdout=stdout,
                    stderr=stderr,
                    exit_status=exit_status,
                )
            raise
        else:
            if verbose:
                self._logger.debug(
                    f"Successfully ran command {log_command_location}",
                    command=command,
                    stdout=stdout,
                    stderr=stderr,
                    exit_status=exit_status,
                )
            return stdout

    def _run_command_remotely(
        self,
        command: str,
        args: list = None,
        workdir: str = None,
        stdin: str = None,
        live: bool = True,
        detach: bool = False,
        verbose: bool = True,
    ) -> (str, str, int):
        workdir = workdir or self.Constants.workdir
        stdout, stderr, exit_status = "", "", 0

        command = f"cd {workdir}; " + command
        if args:
            command += " " + " ".join(args)

        if detach:
            command = f"screen -d -m bash -c '{command}'"
            if verbose:
                self._logger.debug("running command in detached mode", command=command)

        stdin_stream, stdout_stream, stderr_stream = self._ssh_client.exec_command(
            command
        )

        if stdin:
            stdin_stream.write(stdin)
            stdin_stream.close()

        if live:
            while True:
                line = stdout_stream.readline()
                stdout += line
                if not line:
                    break
                print(line, end="")
        else:
            stdout = stdout_stream.read()

        stderr = stderr_stream.read()

        exit_status = stdout_stream.channel.recv_exit_status()

        return stdout, stderr, exit_status

    @staticmethod
    def _run_command_locally(
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
        else:
            stdout = process.stdout.read()

        stderr = process.stderr.read()

        exit_status = process.wait()

        return stdout, stderr, exit_status

    def _get_provctl_version_and_url(self):
        def extract_version_from_release(release):
            tag = release["tag_name"]
            version = tag
            # remove prefix v if exists
            version = version.replace("v", "")
            return semver.VersionInfo.parse(version)

        response = requests.get(
            self.Constants.provctl_releases,
            headers={"Authorization": f"token {self._github_access_token}"},
        )
        response.raise_for_status()
        provazio_releases = json.loads(response.content)
        stable_provazio_releases = list(
            filter(lambda release: release["tag_name"] != "unstable", provazio_releases)
        )
        latest_provazio_releases = stable_provazio_releases[
            : self.Constants.provctl_release_search_amount
        ]
        # This should protect us from taking a backport release (assuming there are never
        # {provctl_release_search_amount} backport releases in a row)
        latest_provazio_releases.sort(key=extract_version_from_release, reverse=True)
        for provazio_release in latest_provazio_releases:
            for asset in provazio_release["assets"]:
                if asset["name"] == self.Constants.provctl_binary_format.format(
                    release_name=provazio_release["name"]
                ):
                    self._logger.debug(
                        "Got provctl release url",
                        release=provazio_release["name"],
                        name=asset["name"],
                        url=asset["url"],
                    )
                    return asset["name"], asset["url"]

        raise RuntimeError(
            f"provctl binary not found in {self.Constants.provctl_release_search_amount} latest releases"
        )

    def _prepare_env_remote(self):
        self._run_command(
            "mkdir",
            args=["-p", str(self.Constants.workdir)],
        )

    def _prepare_env_local(self):
        contents = yaml.safe_dump(self._env_config)
        filepath = str(self.Constants.system_tests_env_yaml)
        self._logger.debug("Populating system tests env.yml", filepath=filepath)
        self._run_command(
            "cat > ",
            workdir=".",
            args=[filepath],
            stdin=contents,
            local=True,
        )

    def _override_mlrun_api_env(self):
        version_specifier = (
            f"mlrun[complete] @ git+https://github.com/mlrun/mlrun@{self._mlrun_commit}"
            if self._mlrun_commit
            else "mlrun[complete]"
        )
        data = {
            "MLRUN_HTTPDB__BUILDER__MLRUN_VERSION_SPECIFIER": version_specifier,
            # Disable the scheduler minimum allowed interval to allow fast tests (default minimum is 10 minutes, which
            # will make our tests really long)
            "MLRUN_HTTPDB__SCHEDULING__MIN_ALLOWED_INTERVAL": "0 Seconds",
            # to allow batch_function to have parquet files sooner
            "MLRUN_MODEL_ENDPOINT_MONITORING__PARQUET_BATCHING_MAX_EVENTS": "100",
        }
        if self._override_image_registry:
            data["MLRUN_IMAGES_REGISTRY"] = f"{self._override_image_registry}"
        override_mlrun_registry_manifest = {
            "apiVersion": "v1",
            "data": data,
            "kind": "ConfigMap",
            "metadata": {"name": "mlrun-override-env", "namespace": "default-tenant"},
        }
        manifest_file_name = "override_mlrun_registry.yml"
        self._run_command(
            "cat > ",
            args=[manifest_file_name],
            stdin=yaml.safe_dump(override_mlrun_registry_manifest),
        )

        self._run_command(
            "kubectl",
            args=["apply", "-f", manifest_file_name],
        )

    def _download_provctl(self):
        provctl, provctl_url = self._get_provctl_version_and_url()
        self._logger.debug("Downloading provctl to data node", provctl_url=provctl_url)
        self._run_command(
            "curl",
            args=[
                "--verbose",
                "--location",
                "--remote-header-name",
                "--remote-name",
                "--header",
                '"Accept: application/octet-stream"',
                "--header",
                f'"Authorization: token {self._github_access_token}"',
                provctl_url,
            ],
        )
        self._run_command("chmod", args=["+x", provctl])
        return provctl

    def _run_and_wait_until_successful(
        self,
        command: str,
        command_name: str,
        max_retries: int = 60,
        interval: int = 10,
    ):
        finished = False
        retries = 0
        start_time = datetime.datetime.now()
        while not finished and retries < max_retries:
            try:
                self._run_command(command, verbose=False)
                finished = True

            except Exception:
                self._logger.debug(
                    f"Command {command_name} didn't complete yet, trying again in {interval} seconds",
                    retry_number=retries,
                )
                retries += 1
                time.sleep(interval)

        if retries >= max_retries and not finished:
            self._logger.info(
                f"Command {command_name} timeout passed and not finished, failing..."
            )
            raise mlrun.errors.MLRunTimeoutError()
        total_seconds_took = (datetime.datetime.now() - start_time).total_seconds()
        self._logger.info(
            f"Command {command_name} took {total_seconds_took} seconds to finish"
        )

    def _patch_mlrun(self, provctl_path):
        time_string = time.strftime("%Y%m%d-%H%M%S")
        self._logger.debug(
            "Creating mlrun patch archive", mlrun_version=self._mlrun_version
        )
        mlrun_archive = f"./mlrun-{self._mlrun_version}.tar"

        override_image_arg = ""
        if self._override_mlrun_images:
            override_image_arg = f"--override-images {self._override_mlrun_images}"

        provctl_create_patch_log = (
            f"{str(self.Constants.workdir)}/provctl-create-patch-{time_string}.log"
        )
        self._run_command(
            f"./{provctl_path}",
            args=[
                f"--logger-file-path={provctl_create_patch_log}",
                "create-patch",
                "appservice",
                override_image_arg,
                "--gzip-flag=-1",
                "-v",
                f"--target-iguazio-version={str(self._iguazio_version)}",
                "mlrun",
                self._mlrun_version,
                mlrun_archive,
            ],
            detach=True,
        )
        self._run_and_wait_until_successful(
            command=f"grep 'Patch archive prepared' {provctl_create_patch_log}",
            command_name="provctl create patch",
            max_retries=25,
            interval=60,
        )
        # print provctl create patch log
        self._run_command(f"cat {provctl_create_patch_log}")

        self._logger.info("Patching MLRun version", mlrun_version=self._mlrun_version)
        provctl_patch_mlrun_log = (
            f"{str(self.Constants.workdir)}/provctl-patch-mlrun-{time_string}.log"
        )
        self._run_command(
            f"./{provctl_path}",
            args=[
                f"--logger-file-path={provctl_patch_mlrun_log}",
                "--app-cluster-password",
                self._app_cluster_ssh_password,
                "--data-cluster-password",
                self._data_cluster_ssh_password,
                "patch",
                "appservice",
                "mlrun",
                mlrun_archive,
            ],
            detach=True,
        )
        self._run_and_wait_until_successful(
            command=f"grep 'Finished patching appservice' {provctl_patch_mlrun_log}",
            command_name="provctl patch mlrun",
            max_retries=25,
            interval=60,
        )
        # print provctl patch mlrun log
        self._run_command(f"cat {provctl_patch_mlrun_log}")


@click.group()
def main():
    pass


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("mlrun-version", type=str, required=True)
@click.option(
    "--override-image-registry",
    "-oireg",
    default=None,
    help="Override default mlrun docker image registry.",
)
@click.option(
    "--override-image-repo",
    "-oirep",
    default=None,
    help="Override default mlrun docker image repository name.",
)
@click.option(
    "--override-mlrun-images",
    "-omi",
    default=None,
    help="Override default images (comma delimited list).",
)
@click.option(
    "--mlrun-commit",
    "-mc",
    default=None,
    help="The commit (in mlrun/mlrun) of the tested mlrun version.",
)
@click.argument("data-cluster-ip", type=str, required=True)
@click.argument("data-cluster-ssh-username", type=str, required=True)
@click.argument("data-cluster-ssh-password", type=str, required=True)
@click.argument("app-cluster-ssh-password", type=str, required=True)
@click.argument("github-access-token", type=str, required=True)
@click.argument("mlrun-dbpath", type=str, required=True)
@click.argument("webapi-direct-url", type=str, required=True)
@click.argument("framesd-url", type=str, required=True)
@click.argument("username", type=str, required=True)
@click.argument("access-key", type=str, required=True)
@click.argument("iguazio-version", type=str, default=None, required=True)
@click.argument("spark-service", type=str, required=True)
@click.argument("password", type=str, default=None, required=False)
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    help="Don't run the ci only show the commands that will be run",
)
def run(
    mlrun_version: str,
    mlrun_commit: str,
    override_image_registry: str,
    override_image_repo: str,
    override_mlrun_images: str,
    data_cluster_ip: str,
    data_cluster_ssh_username: str,
    data_cluster_ssh_password: str,
    app_cluster_ssh_password: str,
    github_access_token: str,
    mlrun_dbpath: str,
    webapi_direct_url: str,
    framesd_url: str,
    username: str,
    access_key: str,
    iguazio_version: str,
    spark_service: str,
    password: str,
    debug: bool,
):
    system_test_preparer = SystemTestPreparer(
        mlrun_version,
        mlrun_commit,
        override_image_registry,
        override_image_repo,
        override_mlrun_images,
        data_cluster_ip,
        data_cluster_ssh_username,
        data_cluster_ssh_password,
        app_cluster_ssh_password,
        github_access_token,
        mlrun_dbpath,
        webapi_direct_url,
        framesd_url,
        username,
        access_key,
        iguazio_version,
        spark_service,
        password,
        debug,
    )
    try:
        system_test_preparer.run()
    except Exception as exc:
        logger.error("Failed running system test automation", exc=exc)
        raise


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("mlrun-dbpath", type=str, required=True)
@click.argument("webapi-direct-url", type=str, required=True)
@click.argument("framesd-url", type=str, required=True)
@click.argument("username", type=str, required=True)
@click.argument("access-key", type=str, required=True)
@click.argument("spark-service", type=str, required=True)
@click.argument("password", type=str, default=None, required=False)
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    help="Don't run the ci only show the commands that will be run",
)
def env(
    mlrun_dbpath: str,
    webapi_direct_url: str,
    framesd_url: str,
    username: str,
    access_key: str,
    spark_service: str,
    password: str,
    debug: bool,
):
    system_test_preparer = SystemTestPreparer(
        mlrun_version=None,
        mlrun_commit=None,
        override_image_registry=None,
        override_image_repo=None,
        override_mlrun_images=None,
        data_cluster_ip=None,
        data_cluster_ssh_password=None,
        app_cluster_ssh_password=None,
        github_access_token=None,
        mlrun_dbpath=mlrun_dbpath,
        webapi_direct_http=webapi_direct_url,
        framesd_url=framesd_url,
        username=username,
        access_key=access_key,
        iguazio_version=None,
        spark_service=spark_service,
        password=password,
        debug=debug,
    )
    try:
        system_test_preparer.prepare_local_env()
    except Exception as exc:
        logger.error("Failed preparing local system test environment", exc=exc)
        raise


if __name__ == "__main__":
    main()
