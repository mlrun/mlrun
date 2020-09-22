import json
import pathlib
import subprocess

import click

import logging

# not using mlrun.utils.logger to not require mlrun package
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("version_file")


@click.group()
def main():
    pass


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("mlrun-version", type=str, required=False, default="unstable")
def create(mlrun_version: str):
    git_commit = "unknown"
    try:
        out, _, _ = _run_command("git", args=["rev-parse", "HEAD"])
        git_commit = out.strip()
        logger.debug(f"Found git commit: {git_commit}")

    except Exception as exc:
        logger.warning("Failed to get version", exc_info=exc)

    version_info = {
        "version": mlrun_version,
        "git_commit": git_commit,
    }

    repo_root = pathlib.Path(__file__).resolve().absolute().parent.parent.parent
    version_file_path = repo_root / "mlrun" / "utils" / "version" / "version.json"
    logger.info(f"Writing version info to file: {str(version_info)}")
    with open(version_file_path, "w+") as version_file:
        json.dump(version_info, version_file, sort_keys=True, indent=2)


def _run_command(
    command: str, args: list = None,
) -> (str, str, int):
    if args:
        command += " " + " ".join(args)

    process = subprocess.run(
        command, shell=True, check=True, capture_output=True, encoding="utf-8",
    )

    return process.stdout, process.stderr, process.returncode


if __name__ == "__main__":
    main()
