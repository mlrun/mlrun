import argparse
import json
import logging
import os.path
import subprocess
import sys

# NOTE
# this script is being used in all build flows before building to add version information to the code
# therefore it needs to be runnable in several environments - GH action, Jenkins, etc...
# therefore this script should be kept python 2 and 3 compatible, and should not require external dependencies
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("version_file")


def main():
    parser = argparse.ArgumentParser(description="Create or update the version file")

    parser.add_argument(
        "--mlrun-version", type=str, required=False, default="0.0.0+unstable"
    )

    args = parser.parse_args()

    create_or_update_version_file(args.mlrun_version)


def create_or_update_version_file(mlrun_version):
    git_commit = "unknown"
    try:
        out = _run_command("git", args=["rev-parse", "HEAD"])
        git_commit = out.strip()
        logger.debug("Found git commit: {}".format(git_commit))

    except Exception as exc:
        logger.warning("Failed to get version", exc_info=exc)

    version_info = {
        "version": mlrun_version,
        "git_commit": git_commit,
    }

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    version_file_path = os.path.join(
        repo_root, "mlrun", "utils", "version", "version.json"
    )
    logger.info("Writing version info to file: {}".format(str(version_info)))
    with open(version_file_path, "w+") as version_file:
        json.dump(version_info, version_file, sort_keys=True, indent=2)


def _run_command(command, args=None):
    if args:
        command += " " + " ".join(args)

    if sys.version_info[0] >= 3:
        process = subprocess.run(
            command, shell=True, check=True, capture_output=True, encoding="utf-8",
        )
        output = process.stdout
    else:
        output = subprocess.check_output(command, shell=True,)

    return output


if __name__ == "__main__":
    main()
