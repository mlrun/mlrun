import click
import json
import paramiko
import pathlib
import requests
import subprocess
import sys
import yaml

import mlrun.utils


logger = mlrun.utils.create_logger(level="debug", name="automation")


class SystemTestCIRunner:
    class Constants:
        ssh_username = "iguazio"

        ci_dir_name = "mlrun-automation"
        homedir = pathlib.Path("/home/iguazio/")
        workdir = homedir / ci_dir_name
        mlrun_code_path = workdir / "mlrun"
        system_tests_env_yaml = pathlib.Path("tests") / "system" / "env.yml"

        git_url = "https://github.com/mlrun/mlrun.git"
        provctl_releases = (
            "https://api.github.com/repos/iguazio/provazio/releases/latest"
        )
        provctl_binary_format = "provctl-{release_name}-linux-amd64"

    def __init__(
        self,
        mlrun_version: str,
        mlrun_repo: str,
        data_cluster_ip: str,
        data_cluster_ssh_password: str,
        app_cluster_ssh_password: str,
        github_access_key: str,
        mlrun_dbpath: str,
        v3io_api: str,
        v3io_username: str,
        v3io_access_key: str,
        v3io_password: str = None,
        debug: bool = False,
    ):
        self._logger = logger
        self._debug = debug
        self._mlrun_version = mlrun_version
        self._mlrun_repo = mlrun_repo
        self._data_cluster_ip = data_cluster_ip
        self._data_cluster_ssh_password = data_cluster_ssh_password
        self._app_cluster_ssh_password = app_cluster_ssh_password
        self._github_access_key = github_access_key

        self._env_config = {
            "MLRUN_DBPATH": mlrun_dbpath,
            "V3IO_API": v3io_api,
            "V3IO_USERNAME": v3io_username,
            "V3IO_ACCESS_KEY": v3io_access_key,
        }
        if v3io_password:
            self._env_config["V3IO_PASSWORD"] = v3io_password

        self._logger.info("Connecting to data-cluster")
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
        self.clean_up(close_ssh_client=False)

        self._prepare_test_env()

        if self._mlrun_repo:
            self._override_k8s_mlrun_registry()

        provctl_path = self._download_povctl()
        self._patch_mlrun(provctl_path)

        self._run_command(
            "make", args=["test-system"], local=True,
        )

    def clean_up(self, close_ssh_client: bool = True):
        self._logger.info("Cleaning up")
        self._run_command(
            f"rm -rf {self.Constants.workdir}", workdir=self.Constants.homedir
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
        workdir = workdir or self.Constants.workdir
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

    def _run_command_locally(
        self,
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
            params={"access_token": self._github_access_key},
        )
        response.raise_for_status()
        latest_provazio_release = json.loads(response.content)
        for asset in latest_provazio_release["assets"]:
            if asset["name"] == self.Constants.provctl_binary_format.format(
                release_name=latest_provazio_release["name"]
            ):
                self._logger.debug(
                    "Got provactl release url",
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

        self._logger.debug("Populating system tests env.yml")
        self._run_command(
            "cat > ",
            args=[str(self.Constants.system_tests_env_yaml)],
            stdin=yaml.safe_dump(self._env_config),
            local=True,
        )

    def _override_k8s_mlrun_registry(self):
        mlrun_registry = self._mlrun_repo.split('/')[0]
        override_mlrun_registry_manifest = {
            'apiVersion': 'v1',
            'data': {'MLRUN_IMAGES_REGISTRY': f'{mlrun_registry}/'},
            'kind': 'ConfigMap',
            'metadata': {'name': 'mlrun-override-env', 'namespace': 'default-tenant'},
        }
        manifest_file_name = 'override_mlrun_registry.yml'
        self._run_command(
            "cat > ",
            args=[manifest_file_name],
            stdin=yaml.safe_dump(override_mlrun_registry_manifest),
        )

        self._run_command(
            "kubectl", args=['apply', '-f', manifest_file_name],
        )

    def _download_povctl(self):
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
                f'"Authorization: token {self._github_access_key}"',
                provctl_url,
            ],
        )
        self._run_command("chmod", args=["+x", provctl])
        return provctl

    def _patch_mlrun(self, provctl_path):
        self._logger.debug(
            "Creating mlrun patch archive", mlrun_version=self._mlrun_version
        )
        mlrun_archive = f"./mlrun-{self._mlrun_version}.tar"

        repo_arg = ""
        if self._mlrun_repo:
            repo_arg = f"--override-image-pull-repo {self._mlrun_repo}"

        self._run_command(
            f"./{provctl_path}",
            args=[
                "create-patch",
                "appservice",
                repo_arg,
                "mlrun",
                self._mlrun_version,
                mlrun_archive,
            ],
        )

        self._logger.info("Patching MLRun version", mlrun_version=self._mlrun_version)
        self._run_command(
            f"./{provctl_path}",
            args=[
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
    "--mlrun-repo", "-mr", default=None, help="Override default mlrun image repository."
)
@click.argument("data-cluster-ip", type=str, required=True)
@click.argument("data-cluster-ssh-password", type=str, required=True)
@click.argument("app-cluster-ssh-password", type=str, required=True)
@click.argument("github-access-key", type=str, required=True)
@click.argument("mlrun-dbpath", type=str, required=True)
@click.argument("v3io-api", type=str, required=True)
@click.argument("v3io-username", type=str, required=True)
@click.argument("v3io-access-key", type=str, required=True)
@click.argument("v3io-password", type=str, default=None, required=False)
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    help="Don't run the ci only show the commands that will be run",
)
def run(
    mlrun_version: str,
    mlrun_repo: str,
    data_cluster_ip: str,
    data_cluster_ssh_password: str,
    app_cluster_ssh_password: str,
    github_access_key: str,
    mlrun_dbpath: str,
    v3io_api: str,
    v3io_username: str,
    v3io_password: str,
    v3io_access_key: str,
    debug: bool,
):
    system_test_ci_runner = SystemTestCIRunner(
        mlrun_version,
        mlrun_repo,
        data_cluster_ip,
        data_cluster_ssh_password,
        app_cluster_ssh_password,
        github_access_key,
        mlrun_dbpath,
        v3io_api,
        v3io_username,
        v3io_password,
        v3io_access_key,
        debug,
    )
    try:
        system_test_ci_runner.run()
    except Exception as e:
        logger.error("Failed running system test automation", exception=e)
    finally:
        system_test_ci_runner.clean_up()


if __name__ == "__main__":
    main()
