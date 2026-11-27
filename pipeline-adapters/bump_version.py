# Copyright 2026 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import pathlib
import re

PACKAGES = [
    "mlrun-pipelines-kfp-common",
    "mlrun-pipelines-kfp-v1-8",
    "mlrun-pipelines-kfp-v2",
]

VERSION_PATTERN = re.compile(r'^version = "(\d+)\.(\d+)\.(\d+)"$', re.MULTILINE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bump the version of every pipeline-adapters package"
    )
    parser.add_argument("mode", choices=["patch", "minor"])
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).parent
    for package in PACKAGES:
        bump_pyproject(repo_root / package / "pyproject.toml", mode=args.mode)


def bump_pyproject(pyproject_path: pathlib.Path, mode: str) -> None:
    content = pyproject_path.read_text()
    match = VERSION_PATTERN.search(content)
    if not match:
        raise ValueError(f"could not find version in {pyproject_path}")

    current_version = ".".join(match.groups())
    new_version = _bump_version(current_version, mode=mode)
    content = VERSION_PATTERN.sub(f'version = "{new_version}"', content, count=1)
    pyproject_path.write_text(content)
    print(f"{pyproject_path.parent.name}: {current_version} -> {new_version}")


def _bump_version(version: str, mode: str) -> str:
    major, minor, patch = (int(part) for part in version.split("."))
    if mode == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1
    return f"{major}.{minor}.{patch}"


if __name__ == "__main__":
    main()
