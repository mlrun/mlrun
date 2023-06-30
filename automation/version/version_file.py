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
import json
import logging
import os.path
import re
import subprocess
import sys

import packaging.version

# NOTE
# this script is being used in all build flows before building to add version information to the code
# therefore it needs to be runnable in several environments - GH action, Jenkins, etc...
# therefore this script should be kept python 2 and 3 compatible, and should not require external dependencies
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("version_file")


def main():
    parser = argparse.ArgumentParser(description="Create or update the version file")

    subparsers = parser.add_subparsers(dest="command")
    ensure_parser = subparsers.add_parser(
        "ensure", help="ensure the version file is up to date"
    )
    ensure_parser.add_argument(
        "--mlrun-version", type=str, required=False, default="0.0.0+unstable"
    )

    subparsers.add_parser("current-version", help="bump the version")
    next_version_parser = subparsers.add_parser("next-version", help="bump the version")
    next_version_parser.add_argument(
        "--mode",
        choices=["rc", "patch", "minor", "major"],
        default="rc",
        help="bump the version by the given mode",
    )

    args = parser.parse_args()
    if args.command == "current-version":
        current_version = get_current_version(read_unstable_version_prefix())
        print(current_version)

    elif args.command == "next-version":
        current_version = packaging.version.Version(
            get_current_version(read_unstable_version_prefix())
        )
        next_version = bump_version(args.mode, current_version)
        print(next_version)

    elif args.command == "ensure":
        create_or_update_version_file(args.mlrun_version)


def get_current_version(base_version):
    current_branch = _run_command(
        "git", args=["rev-parse", "--abbrev-ref", "HEAD"]
    ).strip()
    feature_name = (
        resolve_feature_name(current_branch)
        if current_branch.startswith("feature/")
        else ""
    )

    # get last 100 commits
    commits = _run_command("git", args=["log", "--pretty=format:'%H'"]).strip()
    found_tag = None
    for commit in commits.split("\n"):
        # is commit tagged?
        tags = _run_command("git", args=["tag", "--points-at", commit]).strip()
        tags = [tag for tag in tags.split("\n") if tag]
        if not tags:
            continue

        for tag in tags:
            # is tag a semver?
            if not re.match(r"^v[0-9]+\.[0-9]+\.[0-9]+.*$", tag):
                continue
            semver_tag = packaging.version.Version(tag[1:])
            if semver_tag.base_version < base_version.base_version:
                continue

            # is feature branch?
            if feature_name and semver_tag.local and feature_name in semver_tag.local:
                if found_tag and semver_tag < found_tag:
                    continue
                found_tag = semver_tag
                continue

            # we found the feature branch tag, continue because
            # there is no point finding other tags unrelated to feature branch now
            if (
                found_tag
                and found_tag.local
                and feature_name
                and feature_name in found_tag.local
            ):
                continue

            # we might not have found tag or what we found is old one?
            is_rc = semver_tag.pre and semver_tag.pre[0] == "rc"
            if is_rc:
                if found_tag and semver_tag < found_tag:
                    continue
                found_tag = semver_tag
                continue

            # tag is not rc, not feature branch, and not older than current tag. use it
            found_tag = semver_tag

        # stop here because
        # we either have a tag
        # or, moving back in time wont find newer tags on same branch timeline
        break

    # nothing to bump, just return the version
    if not found_tag:
        version = f"{base_version.base_version}-rc1"

        if feature_name:
            version = f"{version}+{feature_name}"
        return version

    found_version = f"{found_tag.major}.{found_tag.minor}.{found_tag.micro}"
    if found_tag.pre and found_tag.pre[0] == "rc":
        found_version += f"-rc{found_tag.pre[1]}"

    if feature_name and found_tag.local:
        found_version += f"+{feature_name}"
    return found_version


def bump_version(mode, current_version):
    current_branch = _run_command(
        "git", args=["rev-parse", "--abbrev-ref", "HEAD"]
    ).strip()
    feature_name = (
        resolve_feature_name(current_branch)
        if current_branch.startswith("feature/")
        else ""
    )

    # bump
    local = current_version.local
    rc = None
    if current_version.pre and current_version.pre[0] == "rc":
        rc = int(current_version.pre[1])
    major, minor, patch = (
        current_version.major,
        current_version.minor,
        current_version.micro,
    )
    if mode == "rc":

        # if current version is not RC, update its patch version
        if rc is None:
            patch = patch + 1
            rc = 1
        else:
            rc += 1
    elif mode == "patch":
        patch = patch + 1
        rc = None
    elif mode == "minor":
        minor = minor + 1
        patch = 0
        rc = None
    elif mode == "major":
        major = major + 1
        minor = 0
        patch = 0
        rc = None

    if feature_name and not rc:
        rc = 1

    new_version = f"{major}.{minor}.{patch}"
    if rc is not None:
        new_version = f"{new_version}-rc{rc}"

    if feature_name and local:
        new_version = f"{new_version}+{local}"
    return new_version


def create_or_update_version_file(mlrun_version):
    git_commit = "unknown"
    try:
        out = _run_command("git", args=["rev-parse", "HEAD"])
        git_commit = out.strip()
        logger.debug("Found git commit: {}".format(git_commit))

    except Exception as exc:
        logger.warning("Failed to get version", exc_info=exc)

    # get feature branch name from git branch
    git_branch = ""
    try:
        out = _run_command("git", args=["rev-parse", "--abbrev-ref", "HEAD"])
        git_branch = out.strip()
        logger.debug("Found git branch: {}".format(git_branch))
    except Exception as exc:
        logger.warning("Failed to get git branch", exc_info=exc)

    # Enrich the version with the feature name (unless version is unstable)
    if (
        "+unstable" not in mlrun_version
        and git_branch
        and git_branch.startswith("feature/")
    ):
        feature_name = resolve_feature_name(git_branch)
        if not mlrun_version.endswith(feature_name):
            mlrun_version = f"{mlrun_version}+{feature_name}"

    # Check if the provided version is a semver and followed by a "-"
    semver_pattern = r"^[0-9]+\.[0-9]+\.[0-9]+"  # e.g. 0.6.0-
    rc_semver_pattern = rf"{semver_pattern}-(a|b|rc)[0-9]+$"

    # In case of semver - do nothing
    if re.match(semver_pattern, mlrun_version):
        pass

    # In case of rc semver - replace the first occurrence of "-" with "+" to align with PEP 440
    # https://peps.python.org/pep-0440/
    elif re.match(rc_semver_pattern, mlrun_version):
        mlrun_version = mlrun_version.replace("-", "+", 1)

    # In case of some free text - check if the provided version matches the semver pattern
    elif not re.match(r"^[0-9]+\.[0-9]+\.[0-9]+.*$", mlrun_version):
        mlrun_version = "0.0.0+" + mlrun_version

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


def resolve_feature_name(branch_name):
    feature_name = branch_name.replace("feature/", "")
    feature_name = feature_name.lower()
    feature_name = re.sub(r"\+\./\\", "-", feature_name)
    return feature_name


def read_unstable_version_prefix():
    with open("automation/version/unstable_version_prefix") as fp:
        return packaging.version.Version(fp.read().strip())


def _run_command(command, args=None):
    if args:
        command += " " + " ".join(args)

    if sys.version_info[0] >= 3:
        process = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            encoding="utf-8",
        )
        output = process.stdout
    else:
        output = subprocess.check_output(
            command,
            shell=True,
        )

    return output


if __name__ == "__main__":
    main()
