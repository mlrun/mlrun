import click
import json
import paramiko
import pathlib
import requests
import subprocess
import sys
import yaml
import time

import mlrun.utils


logger = mlrun.utils.create_logger(level="debug", name="automation")


class SystemTestPreparer:
    class Constants:
        ssh_username = "iguazio"

        ci_dir_name = "mlrun-automation"
        homedir = pathlib.Path("/home/iguazio/")
        workdir = homedir / ci_dir_name
        mlrun_code_path = workdir / "mlrun"
        system_tests_env_yaml = pathlib.Path("tests") / "system" / "env.yml"

        git_url = "https://github.com/mlrun/mlrun.git"
        provctl_releases = "https://api.github.com/repos/iguazio/provazio/releases"
        provctl_binary_format = "provctl-{release_name}-linux-amd64"

    def __init__(
        self,
        mlrun_version: str,
        override_image_registry: str,
        override_image_repo: str,
        override_mlrun_images: str,
        data_cluster_ip: str,
        data_cluster_ssh_password: str,
        app_cluster_ssh_password: str,
        github_access_token: str,
        mlrun_dbpath: str,
        webapi_direct_http: str,
        username: str,
        access_key: str,
        password: str = None,
        debug: bool = False,
    ):
        self._logger = logger
        self._debug = debug
        self._mlrun_version = mlrun_version
        self._override_image_registry = override_image_registry.strip().strip("/") + "/"
        self._override_image_repo = override_image_repo
        self._override_mlrun_images = override_mlrun_images
        self._data_cluster_ip = data_cluster_ip
        self._data_cluster_ssh_password = data_cluster_ssh_password
        self._app_cluster_ssh_password = app_cluster_ssh_password
        self._github_access_token = github_access_token

        self._env_config = {
            "MLRUN_DBPATH": mlrun_dbpath,
            "V3IO_API": webapi_direct_http,
            "V3IO_USERNAME": username,
            "V3IO_ACCESS_KEY": access_key,
        }
        if password:
            self._env_config["V3IO_PASSWORD"] = password

        self._logger.info("Connecting to data-cluster", data_cluster_ip=data_cluster_ip)
        if not self._debug:
            self._ssh_client = paramiko.SSHClient()
            self._ssh_client.set_missing_host_key_policy(paramiko.WarningPolicy)
            self._ssh_client.connect(
                data_cluster_ip,
                username=self.Constants.ssh_username,
                password=data_cluster_ssh_password,
            )

    def run(self):

        # for sanity clean up before starting the run
        self.clean_up_remote_workdir(close_ssh_client=False)

        self._prepare_test_env()

        if self._override_image_registry:
            self._override_k8s_mlrun_registry()

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
    ) -> str:
        workdir = workdir or str(self.Constants.workdir)
        stdout, stderr, exit_status = "", "", 0

        log_command_location = "locally" if local else "on data cluster"

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
                    command, args, workdir, stdin, live, suppress_errors
                )
            else:
                stdout, stderr, exit_status = self._run_command_remotely(
                    command, args, workdir, stdin, live, suppress_errors
                )
        except (paramiko.SSHException, RuntimeError) as e:
            self._logger.error(
                f"Failed running command {log_command_location}",
                command=command,
                error=e,
                stdout=stdout,
                stderr=stderr,
                exit_status=exit_status,
            )
            raise
        else:
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
        suppress_errors: bool = False,
    ) -> (str, str, int):
        workdir = workdir or self.Constants.workdir
        stdout, stderr, exit_status = "", "", 0
        command = f"cd {workdir}; " + command
        if args:
            command += " " + " ".join(args)

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
        if exit_status != 0 and not suppress_errors:
            raise RuntimeError(f"Command failed with exit status: {exit_status}")

        return stdout, stderr, exit_status

    @staticmethod
    def _run_command_locally(
        command: str,
        args: list = None,
        workdir: str = None,
        stdin: str = None,
        live: bool = True,
        suppress_errors: bool = False,
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
        if exit_status != 0 and not suppress_errors:
            raise RuntimeError(f"Command failed with exit status: {exit_status}")

        return stdout, stderr, exit_status

    def _get_provctl_version_and_url(self):
        response = requests.get(
            self.Constants.provctl_releases,
            headers={"Authorization": f"token {self._github_access_token}"},
        )
        response.raise_for_status()
        provazio_releases = json.loads(response.content)
        stable_provazio_releases = list(
            filter(lambda release: release["tag_name"] != "unstable", provazio_releases)
        )
        latest_provazio_release = stable_provazio_releases[0]
        for asset in latest_provazio_release["assets"]:
            if asset["name"] == self.Constants.provctl_binary_format.format(
                release_name=latest_provazio_release["name"]
            ):
                self._logger.debug(
                    "Got provctl release url",
                    release=latest_provazio_release["name"],
                    name=asset["name"],
                    url=asset["url"],
                )
                return asset["name"], asset["url"]

        raise RuntimeError("provctl binary not found")

    def _prepare_test_env(self):

        self._run_command(
            "mkdir", args=["-p", str(self.Constants.workdir)],
        )
        contents = yaml.safe_dump(self._env_config)
        filepath = str(self.Constants.system_tests_env_yaml)
        self._logger.debug("Populating system tests env.yml", filepath=filepath)
        self._run_command(
            "cat > ", args=[filepath], stdin=contents, local=True,
        )

    def _override_k8s_mlrun_registry(self):
        mlrun_registry_override = self._override_image_registry.strip().strip("/") + "/"
        override_mlrun_registry_manifest = {
            "apiVersion": "v1",
            "data": {"MLRUN_IMAGES_REGISTRY": f"{mlrun_registry_override}"},
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
            "kubectl", args=["apply", "-f", manifest_file_name],
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

    def _patch_mlrun(self, provctl_path):
        time_string = time.strftime("%Y%m%d-%H%M%S")
        self._logger.debug(
            "Creating mlrun patch archive", mlrun_version=self._mlrun_version
        )
        mlrun_archive = f"./mlrun-{self._mlrun_version}.tar"

        override_image_arg = ""
        if self._override_mlrun_images:
            override_image_arg = f"--override-images {self._override_mlrun_images}"

        self._run_command(
            f"./{provctl_path}",
            args=[
                f"--logger-file-path={str(self.Constants.workdir)}/provctl-create-patch-{time_string}.log",
                "create-patch",
                "appservice",
                override_image_arg,
                "mlrun",
                self._mlrun_version,
                mlrun_archive,
            ],
        )

        self._logger.info("Patching MLRun version", mlrun_version=self._mlrun_version)
        self._run_command(
            f"./{provctl_path}",
            args=[
                f"--logger-file-path={str(self.Constants.workdir)}/provctl-patch-mlrun-{time_string}.log",
                "--app-cluster-password",
                self._app_cluster_ssh_password,
                "--data-cluster-password",
                self._data_cluster_ssh_password,
                "patch",
                "appservice",
                "mlrun",
                mlrun_archive,
            ],
        )


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
@click.argument("data-cluster-ip", type=str, required=True)
@click.argument("data-cluster-ssh-password", type=str, required=True)
@click.argument("app-cluster-ssh-password", type=str, required=True)
@click.argument("github-access-token", type=str, required=True)
@click.argument("mlrun-dbpath", type=str, required=True)
@click.argument("webapi-direct-url", type=str, required=True)
@click.argument("username", type=str, required=True)
@click.argument("access-key", type=str, required=True)
@click.argument("password", type=str, default=None, required=False)
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    help="Don't run the ci only show the commands that will be run",
)
def run(
    mlrun_version: str,
    override_image_registry: str,
    override_image_repo: str,
    override_mlrun_images: str,
    data_cluster_ip: str,
    data_cluster_ssh_password: str,
    app_cluster_ssh_password: str,
    github_access_token: str,
    mlrun_dbpath: str,
    webapi_direct_url: str,
    username: str,
    access_key: str,
    password: str,
    debug: bool,
):
    system_test_preparer = SystemTestPreparer(
        mlrun_version,
        override_image_registry,
        override_image_repo,
        override_mlrun_images,
        data_cluster_ip,
        data_cluster_ssh_password,
        app_cluster_ssh_password,
        github_access_token,
        mlrun_dbpath,
        webapi_direct_url,
        username,
        access_key,
        password,
        debug,
    )
    try:
        system_test_preparer.run()
    except Exception as exc:
        logger.error("Failed running system test automation", exc=exc)
        raise


if __name__ == "__main__":
    main()
