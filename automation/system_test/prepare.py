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
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import typing
import urllib.parse

import boto3
import click
import paramiko
import yaml


class Logger:
    def __init__(self, name, **kwargs):
        self._logger = logging.getLogger(name)
        level = kwargs.get("level", logging.INFO)
        self._logger.setLevel(level)
        if not self._logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)

    def log(self, level: str, message: str, **kwargs: typing.Any) -> None:
        more = f": {kwargs}" if kwargs else ""
        self._logger.log(logging.getLevelName(level.upper()), f"{message}{more}")


project_dir = pathlib.Path(__file__).resolve().parent.parent.parent
logger = Logger(level=logging.DEBUG, name="automation")
logging.getLogger("paramiko").setLevel(logging.DEBUG)


class SystemTestPreparer:
    class Constants:
        ci_dir_name = "mlrun-automation"
        homedir = pathlib.Path("/home/iguazio/")
        workdir = homedir / ci_dir_name
        igz_version_file = homedir / "igz" / "version.txt"
        mlrun_code_path = workdir / "mlrun"
        provctl_path = workdir / "provctl"
        system_tests_env_yaml = (
            project_dir / pathlib.Path("tests") / "system" / "env.yml"
        )
        namespace = "default-tenant"

        git_url = "https://github.com/mlrun/mlrun.git"

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
        provctl_download_url: str = None,
        provctl_download_s3_access_key: str = None,
        provctl_download_s3_key_id: str = None,
        mlrun_dbpath: str = None,
        username: str = None,
        access_key: str = None,
        iguazio_version: str = None,
        slack_webhook_url: str = None,
        mysql_user: str = None,
        mysql_password: str = None,
        purge_db: bool = False,
        debug: bool = False,
        branch: str = None,
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
        self._provctl_download_url = provctl_download_url
        self._provctl_download_s3_access_key = provctl_download_s3_access_key
        self._provctl_download_s3_key_id = provctl_download_s3_key_id
        self._iguazio_version = iguazio_version
        self._mysql_user = mysql_user
        self._mysql_password = mysql_password
        self._purge_db = purge_db
        self._ssh_client = None

        self._env_config = {
            "MLRUN_DBPATH": mlrun_dbpath,
            "V3IO_USERNAME": username,
            "V3IO_ACCESS_KEY": access_key,
            "MLRUN_SYSTEM_TESTS_SLACK_WEBHOOK_URL": slack_webhook_url,
            "MLRUN_SYSTEM_TESTS_BRANCH": branch,
            # Setting to MLRUN_SYSTEM_TESTS_GIT_TOKEN instead of GIT_TOKEN, to not affect tests which doesn't need it
            # (e.g. tests which use public repos, therefor doesn't need that access token)
            "MLRUN_SYSTEM_TESTS_GIT_TOKEN": github_access_token,
        }

    def prepare_local_env(self, save_to_path: str = ""):
        self._prepare_env_local(save_to_path)

    def connect_to_remote(self):
        if not self._debug and self._data_cluster_ip:
            self._logger.log(
                "info",
                "Connecting to data-cluster",
                data_cluster_ip=self._data_cluster_ip,
            )
            self._ssh_client = paramiko.SSHClient()
            self._ssh_client.set_missing_host_key_policy(paramiko.WarningPolicy)
            self._ssh_client.connect(
                self._data_cluster_ip,
                username=self._data_cluster_ssh_username,
                password=self._data_cluster_ssh_password,
            )

    def run(self):
        self.connect_to_remote()

        try:
            logger.log("debug", "installing dev utilities")
            self._install_dev_utilities()
            logger.log("debug", "installing dev utilities - done")
        except Exception as exp:
            self._logger.log(
                "error", "error on install dev utilities", exception=str(exp)
            )

        # for sanity clean up before starting the run
        self.clean_up_remote_workdir()

        self._prepare_env_remote()

        self._resolve_iguazio_version()

        self._download_provctl()

        self._override_mlrun_api_env()

        # purge of the database needs to be executed before patching mlrun so that the mlrun migrations
        # that run as part of the patch would succeed even if we move from a newer version to an older one
        # e.g from development branch which is (1.4.0) and has a newer alembic revision than 1.3.x which is (1.3.1)
        if self._purge_db:
            self._purge_mlrun_db()

        self._patch_mlrun()

    def clean_up_remote_workdir(self):
        self._logger.log(
            "info", "Cleaning up remote workdir", workdir=str(self.Constants.workdir)
        )
        self._run_command(
            f"rm -rf {self.Constants.workdir}", workdir=str(self.Constants.homedir)
        )

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
    ) -> (bytes, bytes):
        workdir = workdir or str(self.Constants.workdir)
        stdout, stderr, exit_status = "", "", 0

        log_command_location = "locally" if local else "on data cluster"

        if verbose:
            self._logger.log(
                "debug",
                f"Running command {log_command_location}",
                command=command,
                args=args,
                stdin=stdin,
                workdir=workdir,
            )
        if self._debug:
            return b"", b""
        try:
            if local:
                stdout, stderr, exit_status = run_command(
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
            err_log_kwargs = {
                "error": str(exc),
                "stdout": stdout,
                "stderr": stderr,
                "exit_status": exit_status,
            }
            if verbose:
                err_log_kwargs["command"] = command

            self._logger.log(
                "error",
                f"Failed running command {log_command_location}",
                **err_log_kwargs,
            )
            raise
        else:
            if verbose:
                self._logger.log(
                    "debug",
                    f"Successfully ran command {log_command_location}",
                    command=command,
                    stdout=stdout,
                    stderr=stderr,
                    exit_status=exit_status,
                )
            return stdout, stderr

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
                self._logger.log(
                    "debug", "running command in detached mode", command=command
                )

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

    def _prepare_env_remote(self):
        self._run_command(
            "mkdir",
            args=["-p", str(self.Constants.workdir)],
            workdir=str(self.Constants.homedir),
        )

    def _prepare_env_local(self, save_to_path: str = ""):
        filepath = save_to_path or str(self.Constants.system_tests_env_yaml)
        backup_filepath = str(self.Constants.system_tests_env_yaml) + ".bak"
        self._logger.log("debug", "Populating system tests env.yml", filepath=filepath)

        # if filepath exists, backup the file first (to avoid overriding it)
        if os.path.isfile(filepath) and not os.path.isfile(backup_filepath):
            self._logger.log(
                "debug", "Backing up existing env.yml", destination=backup_filepath
            )
            shutil.copy(filepath, backup_filepath)

        # enrichment can be done only if ssh client is initialized
        if self._ssh_client:
            self._enrich_env()
        serialized_env_config = self._serialize_env_config()
        with open(filepath, "w") as f:
            f.write(serialized_env_config)

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
            "metadata": {
                "name": "mlrun-override-env",
                "namespace": self.Constants.namespace,
            },
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

    def _enrich_env(self):
        devutils_outputs = self._get_devutils_status()
        if "redis" in devutils_outputs:
            self._logger.log("debug", "Enriching env with redis info")
            # uncomment when url is accessible from outside the cluster
            # self._env_config["MLRUN_REDIS__URL"] = f"redis://{devutils_outputs['redis']['app_url']}"
            # self._env_config["REDIS_USER"] = devutils_outputs["redis"]["username"]
            # self._env_config["REDIS_PASSWORD"] = devutils_outputs["redis"]["password"]

        api_url_host = self._get_ingress_host("datanode-dashboard")
        framesd_host = self._get_ingress_host("framesd")
        v3io_api_host = self._get_ingress_host("webapi")
        spark_service_name = self._get_service_name("app=spark,component=spark-master")
        self._env_config["MLRUN_IGUAZIO_API_URL"] = f"https://{api_url_host}"
        self._env_config["V3IO_FRAMESD"] = f"https://{framesd_host}"
        self._env_config[
            "MLRUN_SYSTEM_TESTS_DEFAULT_SPARK_SERVICE"
        ] = spark_service_name
        self._env_config["V3IO_API"] = f"https://{v3io_api_host}"

    def _install_dev_utilities(self):
        list_uninstall = [
            "dev_utilities.py",
            "uninstall",
            "--redis",
            "--mysql",
            "--redisinsight",
            "--kafka",
        ]
        list_install = [
            "dev_utilities.py",
            "install",
            "--redis",
            "--mysql",
            "--redisinsight",
            "--kafka",
            "--ipadd",
            os.environ.get("IP_ADDR_PREFIX", "localhost"),
        ]
        self._run_command("rm", args=["-rf", "/home/iguazio/dev_utilities"])
        self._run_command("python3", args=list_uninstall, workdir="/home/iguazio/")
        self._run_command("python3", args=list_install, workdir="/home/iguazio/")

    def _download_provctl(self):
        # extract bucket name, object name from s3 file path
        # https://<bucket-name>.s3.amazonaws.com/<object-name>
        # s3://<bucket-name>/<object-name>
        parsed_url = urllib.parse.urlparse(self._provctl_download_url)
        if self._provctl_download_url.startswith("s3://"):
            object_name = parsed_url.path.lstrip("/")
            bucket_name = parsed_url.netloc
        else:
            object_name = parsed_url.path.lstrip("/")
            bucket_name = parsed_url.netloc.split(".")[0]
        # download provctl from s3
        with tempfile.NamedTemporaryFile() as local_provctl_path:
            self._logger.log(
                "debug",
                "Downloading provctl",
                bucket_name=bucket_name,
                object_name=object_name,
                local_path=local_provctl_path.name,
            )
            s3_client = boto3.client(
                "s3",
                aws_secret_access_key=self._provctl_download_s3_access_key,
                aws_access_key_id=self._provctl_download_s3_key_id,
            )
            s3_client.download_file(bucket_name, object_name, local_provctl_path.name)
            # upload provctl to data node
            self._logger.log(
                "debug",
                "Uploading provctl to datanode",
                remote_path=str(self.Constants.provctl_path),
                local_path=local_provctl_path.name,
            )
            sftp_client = self._ssh_client.open_sftp()
            sftp_client.put(local_provctl_path.name, str(self.Constants.provctl_path))
            sftp_client.close()
        # make provctl executable
        self._run_command("chmod", args=["+x", str(self.Constants.provctl_path)])

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
                self._logger.log(
                    "debug",
                    f"Command {command_name} didn't complete yet, trying again in {interval} seconds",
                    retry_number=retries,
                )
                retries += 1
                time.sleep(interval)

        if retries >= max_retries and not finished:
            self._logger.log(
                "info",
                f"Command {command_name} timeout passed and not finished, failing...",
            )
            raise RuntimeError("Command timeout passed and not finished")
        total_seconds_took = (datetime.datetime.now() - start_time).total_seconds()
        self._logger.log(
            "info",
            f"Command {command_name} took {total_seconds_took} seconds to finish",
        )

    def _patch_mlrun(self):
        time_string = time.strftime("%Y%m%d-%H%M%S")
        self._logger.log(
            "debug", "Creating mlrun patch archive", mlrun_version=self._mlrun_version
        )
        mlrun_archive = f"./mlrun-{self._mlrun_version}.tar"

        override_image_arg = ""
        if self._override_mlrun_images:
            override_image_arg = f"--override-images {self._override_mlrun_images}"

        provctl_create_patch_log = f"/tmp/provctl-create-patch-{time_string}.log"
        self._run_command(
            str(self.Constants.provctl_path),
            args=[
                "--verbose",
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

        self._logger.log(
            "info", "Patching MLRun version", mlrun_version=self._mlrun_version
        )
        provctl_patch_mlrun_log = f"/tmp/provctl-patch-mlrun-{time_string}.log"
        self._run_command(
            str(self.Constants.provctl_path),
            args=[
                "--verbose",
                f"--logger-file-path={provctl_patch_mlrun_log}",
                "--app-cluster-password",
                self._app_cluster_ssh_password,
                "--data-cluster-password",
                self._data_cluster_ssh_password,
                "patch",
                "appservice",
                # we force because by default provctl doesn't allow downgrading between version but due to system tests
                # running on multiple branches this might occur.
                "--force",
                "mlrun",
                mlrun_archive,
                # enable audit events - will be ignored by provctl if mlrun version does not support it
                # TODO: remove when setup is upgraded to iguazio version >= 3.5.4 since audit events
                #  are enabled by default
                "--feature-gates",
                "mlrun.auditevents=enabled",
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

    def _resolve_iguazio_version(self):
        # iguazio version is optional, if not provided, we will try to resolve it from the data node
        if not self._iguazio_version:
            self._logger.log("info", "Resolving iguazio version")
            self._iguazio_version, _ = self._run_command(
                f"cat {self.Constants.igz_version_file}",
                verbose=False,
                live=False,
            )
            self._iguazio_version = self._iguazio_version.strip().decode()
        self._logger.log(
            "info", "Resolved iguazio version", iguazio_version=self._iguazio_version
        )

    def _purge_mlrun_db(self):
        """
        Purge mlrun db - exec into mlrun-db pod, delete the database and scale down mlrun pods
        """
        self._delete_mlrun_db()
        self._scale_down_mlrun_deployments()

    def _delete_mlrun_db(self):
        self._logger.log("info", "Deleting mlrun db")

        mlrun_db_pod_name_cmd = self._get_pod_name_command(
            labels={
                "app.kubernetes.io/component": "db",
                "app.kubernetes.io/instance": "mlrun",
            },
        )
        if not mlrun_db_pod_name_cmd:
            self._logger.log("info", "No mlrun db pod found")
            return

        self._logger.log(
            "info", "Deleting mlrun db pod", mlrun_db_pod_name_cmd=mlrun_db_pod_name_cmd
        )

        password = ""
        if self._mysql_password:
            password = f"-p {self._mysql_password} "

        drop_db_cmd = f"mysql --socket=/run/mysqld/mysql.sock -u {self._mysql_user} {password}-e 'DROP DATABASE mlrun;'"
        self._run_kubectl_command(
            args=[
                "exec",
                "-n",
                self.Constants.namespace,
                "-it",
                mlrun_db_pod_name_cmd,
                "--",
                drop_db_cmd,
            ],
            verbose=False,
        )

    def _get_pod_name_command(self, labels):
        labels_selector = ",".join([f"{k}={v}" for k, v in labels.items()])
        pod_name, stderr = self._run_kubectl_command(
            args=[
                "get",
                "pods",
                "--namespace",
                self.Constants.namespace,
                "--selector",
                labels_selector,
                "|",
                "tail",
                "-n",
                "1",
                "|",
                "awk",
                "'{print $1}'",
            ],
        )
        if b"No resources found" in stderr or not pod_name:
            return None
        return pod_name.strip()

    def _scale_down_mlrun_deployments(self):
        # scaling down to avoid automatically deployments restarts and failures
        self._logger.log("info", "scaling down mlrun deployments")
        self._run_kubectl_command(
            args=[
                "scale",
                "deployment",
                "-n",
                self.Constants.namespace,
                "mlrun-api-chief",
                "mlrun-api-worker",
                "mlrun-db",
                "--replicas=0",
            ]
        )

    def _run_kubectl_command(self, args, verbose=True):
        return self._run_command(
            command="kubectl",
            args=args,
            verbose=verbose,
        )

    def _serialize_env_config(self, allow_none_values: bool = False):
        env_config = self._env_config.copy()

        # we sanitize None values from config to avoid "null" values in yaml
        if not allow_none_values:
            for key in list(env_config):
                if env_config[key] is None:
                    del env_config[key]

        return yaml.safe_dump(env_config)

    def _get_ingress_host(self, ingress_name: str):
        host, stderr = self._run_kubectl_command(
            args=[
                "get",
                "ingress",
                "--namespace",
                self.Constants.namespace,
                ingress_name,
                "--output",
                "jsonpath={'@.spec.rules[0].host'}",
            ],
        )
        if stderr:
            raise RuntimeError(
                f"Failed getting {ingress_name} ingress host. Error: {stderr}"
            )
        return host.strip()

    def _get_service_name(self, label_selector):
        service_name, stderr = self._run_kubectl_command(
            args=[
                "get",
                "deployment",
                "--namespace",
                self.Constants.namespace,
                "-l",
                label_selector,
                "--output",
                "jsonpath={'@.items[0].metadata.labels.release'}",
            ],
        )
        if stderr:
            raise RuntimeError(f"Failed getting service name. Error: {stderr}")
        return service_name.strip()

    def _get_devutils_status(self):
        out, err = "", ""
        try:
            out, err = self._run_command(
                "python3",
                [
                    "/home/iguazio/dev_utilities.py",
                    "status",
                    "--redis",
                    "--kafka",
                    "--mysql",
                    "--redisinsight",
                    "--output",
                    "json",
                ],
            )
        except Exception as exc:
            self._logger.log(
                "warning", "Failed to enrich env", exc=exc, err=err, out=out
            )

        return json.loads(out or "{}")


@click.group()
def main():
    pass


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.option("--mlrun-version")
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
@click.option("--data-cluster-ip", required=True)
@click.option("--data-cluster-ssh-username", required=True)
@click.option("--data-cluster-ssh-password", required=True)
@click.option("--app-cluster-ssh-password", required=True)
@click.option("--provctl-download-url", required=True)
@click.option("--provctl-download-s3-access-key", required=True)
@click.option("--provctl-download-s3-key-id", required=True)
@click.option("--username", required=True)
@click.option("--access-key", required=True)
@click.option("--iguazio-version", default=None)
@click.option("--mysql-user")
@click.option("--mysql-password")
@click.option("--purge-db", "-pdb", is_flag=True, help="Purge mlrun db")
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
    provctl_download_url: str,
    provctl_download_s3_access_key: str,
    provctl_download_s3_key_id: str,
    username: str,
    access_key: str,
    iguazio_version: str,
    mysql_user: str,
    mysql_password: str,
    purge_db: bool,
    debug: bool,
):
    system_test_preparer = SystemTestPreparer(
        mlrun_version=mlrun_version,
        mlrun_commit=mlrun_commit,
        override_image_registry=override_image_registry,
        override_image_repo=override_image_repo,
        override_mlrun_images=override_mlrun_images,
        data_cluster_ip=data_cluster_ip,
        data_cluster_ssh_username=data_cluster_ssh_username,
        data_cluster_ssh_password=data_cluster_ssh_password,
        app_cluster_ssh_password=app_cluster_ssh_password,
        provctl_download_url=provctl_download_url,
        provctl_download_s3_access_key=provctl_download_s3_access_key,
        provctl_download_s3_key_id=provctl_download_s3_key_id,
        username=username,
        access_key=access_key,
        iguazio_version=iguazio_version,
        mysql_user=mysql_user,
        mysql_password=mysql_password,
        purge_db=purge_db,
        debug=debug,
    )
    try:
        system_test_preparer.run()
    except Exception as exc:
        logger.log("error", "Failed running system test automation", exc=exc)
        raise


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.option("--mlrun-dbpath", help="The mlrun api address", required=True)
@click.option("--data-cluster-ip")
@click.option("--data-cluster-ssh-username")
@click.option("--data-cluster-ssh-password")
@click.option("--username", help="Iguazio running username")
@click.option("--access-key", help="Iguazio running user access key")
@click.option(
    "--slack-webhook-url", help="Slack webhook url to send tests notifications to"
)
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    help="Don't run the ci only show the commands that will be run",
)
@click.option("--branch", help="The mlrun branch to run the tests against")
@click.option(
    "--github-access-token",
    help="Github access token to use for fetching private functions",
)
@click.option(
    "--save-to-path",
    help="Path to save the compiled env file to",
)
def env(
    data_cluster_ip: str,
    data_cluster_ssh_username: str,
    data_cluster_ssh_password: str,
    mlrun_dbpath: str,
    username: str,
    access_key: str,
    slack_webhook_url: str,
    debug: bool,
    branch: str,
    github_access_token: str,
    save_to_path: str,
):
    system_test_preparer = SystemTestPreparer(
        data_cluster_ip=data_cluster_ip,
        data_cluster_ssh_password=data_cluster_ssh_password,
        data_cluster_ssh_username=data_cluster_ssh_username,
        mlrun_dbpath=mlrun_dbpath,
        username=username,
        access_key=access_key,
        debug=debug,
        slack_webhook_url=slack_webhook_url,
        branch=branch,
        github_access_token=github_access_token,
    )
    try:
        system_test_preparer.connect_to_remote()
        system_test_preparer.prepare_local_env(save_to_path)
    except Exception as exc:
        logger.log("error", "Failed preparing local system test environment", exc=exc)
        raise


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


if __name__ == "__main__":
    main()
