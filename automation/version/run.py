import json
import pathlib
import subprocess

import click

import logging

# not using mlrun.utils.logger cause I don't want this script to require you to install mlrun
logger = logging.Logger(name="version", level="DEBUG")


@click.group()
def main():
    pass


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("mlrun-version", type=str, required=False, default="unstable")
@click.argument(
    "mlrun-docker-registry", type=str, required=False, default=""
)  # empty represents docker hub
def run(
    mlrun_version: str, mlrun_docker_registry: str,
):
    try:
        out, _, _ = _run_command("git", args=["rev-parse", "HEAD"])
        version_info = {"git_commit": out[:-1].decode("utf-8")}

    except Exception as exc:
        logger.warning("Failed to get version", exc_info=exc)
        version_info = {"git_commit": "unknown"}

    version_info["version"] = mlrun_version
    version_info["docker_registry"] = mlrun_docker_registry

    root = pathlib.Path(__file__).resolve().absolute().parent.parent.parent
    version_file_path = root / "version.json"
    logger.info(f"Writing version info to file: {str(version_info)}")
    with open(version_file_path, "w+") as version_file:
        json.dump(version_info, version_file, sort_keys=False, indent=2)


def _run_command(
    command: str, args: list = None, suppress_errors: bool = False,
) -> (str, str, int):
    if args:
        command += " " + " ".join(args)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        shell=True,
    )

    stdout = process.stdout.read()
    stderr = process.stderr.read()

    exit_status = process.wait()
    if exit_status != 0 and not suppress_errors:
        raise RuntimeError(f"Command failed with exit status: {exit_status}")

    return stdout, stderr, exit_status


if __name__ == "__main__":
    main()
