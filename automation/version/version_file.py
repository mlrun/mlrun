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

import argparse
import json
import logging
import os
import pathlib
import re
import subprocess
import typing

import packaging.version

logger = logging.getLogger("version_file")
logger.setLevel(os.getenv("LOG_LEVEL", logging.INFO))
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
logger.addHandler(ch)


def main():
    parser = argparse.ArgumentParser(description="Create or update the version file")

    subparsers = parser.add_subparsers(dest="command")
    ensure_parser = subparsers.add_parser(
        "ensure", help="ensure the version file is up to date"
    )
    ensure_parser.add_argument(
        "--mlrun-version", type=str, required=False, default="unstable"
    )

    subparsers.add_parser(
        "is-feature-branch", help="check if the branch is a feature branch"
    )

    is_stable_parser = subparsers.add_parser(
        "is-stable", help="check if the version is stable"
    )
    is_stable_parser.add_argument("version", type=str)

    subparsers.add_parser("current-version", help="get the current version")
    next_version_parser = subparsers.add_parser("next-version", help="get next version")

    # RC        - bump the rc version. if current is not rc, bump patch and set rc to 1
    # RC-GRAD   - bump the rc version to its graduated version (1.0.0-rc1 -> 1.0.0)
    # PATCH     - bump the patch version. reset rc
    # MINOR     - bump the minor version. reset rc / patch
    # MAJOR     - bump the major version. reset rc / patch / minor
    next_version_parser.add_argument(
        "--mode",
        choices=["rc", "rc-grad", "patch", "minor", "major"],
        default="rc",
        help="bump the version by the given mode",
    )

    args = parser.parse_args()
    if args.command == "current-version":
        current_version = get_current_version(read_unstable_version_prefix())
        print(current_version)

    elif args.command == "next-version":
        base_version = read_unstable_version_prefix()
        current_version = get_current_version(base_version)
        next_version = resolve_next_version(
            args.mode,
            packaging.version.Version(current_version),
            base_version,
            get_feature_branch_feature_name(),
        )
        print(next_version)

    elif args.command == "ensure":
        repo_root = pathlib.Path(__file__).parents[2]
        version_file_path = str(
            (repo_root / "mlrun/utils/version/version.json").absolute()
        )
        logger.debug(f"{args.mlrun_version = }")
        create_or_update_version_file(args.mlrun_version, version_file_path)

    elif args.command == "is-stable":
        is_stable = is_stable_version(args.version)
        print(str(is_stable).lower())

    elif args.command == "is-feature-branch":
        print(str(is_feature_branch()).lower())


def get_current_version(
    base_version: packaging.version.Version,
) -> str:
    current_branch = _run_command(
        "git", args=["rev-parse", "--abbrev-ref", "HEAD"]
    ).strip()
    feature_name = (
        resolve_feature_name(current_branch)
        if current_branch.startswith("feature/")
        else ""
    )

    # get last 200 commits, to avoid going over all commits
    commits = _run_command("git", args=["log", "-200", "--pretty=format:'%H'"]).strip()
    found_tag = None

    # most_recent_version is the most recent tag before base version
    most_recent_version = None
    for commit in commits.split("\n"):
        # is commit tagged?
        tags = _run_command("git", args=["tag", "--points-at", commit]).strip()
        tags = [tag for tag in tags.split("\n") if tag]
        if not tags:
            continue

        for tag in tags:
            # work with semvar-like tags only
            if not re.match(r"^v[0-9]+\.[0-9]+\.[0-9]+.*$", tag):
                continue

            semver_tag = packaging.version.parse(tag.removeprefix("v"))

            # compare base versions on both base and current tag
            # if current tag version (e.g.: 1.4.0) is smaller than base version (e.g.: 1.5.0)
            # then, keep that tag version as the most recent version
            # if no base-version tag was made (e.g.: when starting new version (e.g. 1.5.0)
            # but no tag/release was made yet)
            if packaging.version.parse(
                semver_tag.base_version
            ) < packaging.version.parse(base_version.base_version):
                if most_recent_version:
                    if semver_tag > most_recent_version:
                        most_recent_version = semver_tag
                    continue
                most_recent_version = semver_tag
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
        if most_recent_version:
            return version_to_mlrun_version(most_recent_version)

        return version_to_mlrun_version(base_version)

    return version_to_mlrun_version(found_tag)


def resolve_next_version(
    mode: str,
    current_version: packaging.version.Version,
    base_version: packaging.version.Version,
    feature_name: typing.Optional[str] = None,
):
    if (
        base_version.major > current_version.major
        or base_version.minor > current_version.minor
    ):
        # the current version is lower, can be because base version was not tagged yet
        # make current version align with base version
        suffix = ""
        if mode == "rc":
            # index 0 because we increment rc later on
            suffix += "-rc0"
        current_version = packaging.version.Version(base_version.base_version + suffix)

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
    elif mode == "rc-grad":
        rc = None
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

    # when feature name is set, it means we are on a feature branch
    # thus, we ensure rc is set as feature name are not meant to be "stable"
    if feature_name and not rc:
        rc = 1

    new_version = f"{major}.{minor}.{patch}"
    if rc is not None:
        new_version = f"{new_version}-rc{rc}"

    if feature_name:
        new_version = f"{new_version}+{feature_name}"
    return new_version


def create_or_update_version_file(mlrun_version: str, version_file_path: str):
    git_commit = "unknown"
    try:
        git_commit = _run_command("git", args=["rev-parse", "HEAD"]).strip()
        logger.debug(f"Found git commit: {git_commit}")

    except Exception as exc:
        logger.warning("Failed to get version", exc_info=exc)

    # get feature branch name from git branch
    git_branch = ""
    try:
        git_branch = _run_command(
            "git", args=["rev-parse", "--abbrev-ref", "HEAD"]
        ).strip()
        logger.debug(f"Found git branch: {git_branch}")
    except Exception as exc:
        logger.warning("Failed to get git branch", exc_info=exc)

    # Enrich the version with the feature name (unless version is unstable)
    if (
        "unstable" not in mlrun_version
        and git_branch
        and git_branch.startswith("feature/")
    ):
        feature_name = resolve_feature_name(git_branch)
        if not mlrun_version.endswith(feature_name):
            mlrun_version = f"{mlrun_version}+{feature_name}"
            logger.debug(f"With feature_name: {mlrun_version = }")

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

    logger.info(
        f"Writing version info to file: {str(version_info)}, {version_file_path = }"
    )
    with open(version_file_path, "w+") as version_file:
        json.dump(version_info, version_file, sort_keys=True, indent=2)


def resolve_feature_name(branch_name):
    feature_name = branch_name.replace("feature/", "")
    feature_name = feature_name.lower()

    # replace non-alphanumeric characters with "-" to align with PEP 440 and docker tag naming
    feature_name = re.sub(r"\+\./\\", "-", feature_name)
    return feature_name


def read_unstable_version_prefix():
    with open(
        pathlib.Path(__file__).absolute().parent / "unstable_version_prefix"
    ) as fp:
        return packaging.version.Version(fp.read().strip())


def version_to_mlrun_version(version):
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    if version.pre and version.pre[0] == "rc":
        version_str += f"-rc{version.pre[1]}"

    if version.local:
        version_str += f"+{version.local}"
    return version_str


def is_stable_version(mlrun_version: str) -> bool:
    return re.match(r"^\d+\.\d+\.\d+$", mlrun_version) is not None


def is_feature_branch() -> bool:
    return get_feature_branch_feature_name() != ""


def get_feature_branch_feature_name() -> typing.Optional[str]:
    current_branch = _run_command(
        "git", args=["rev-parse", "--abbrev-ref", "HEAD"]
    ).strip()
    return (
        resolve_feature_name(current_branch)
        if current_branch.startswith("feature/")
        else ""
    )


def _run_command(command, args=None):
    if args:
        command += " " + " ".join(args)

    process = subprocess.run(
        command,
        shell=True,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    output = process.stdout

    return output


if __name__ == "__main__":
    main()
