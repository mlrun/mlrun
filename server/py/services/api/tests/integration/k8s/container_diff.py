# Copyright 2026 Iguazio
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

"""Thin wrapper around the `container-diff <https://github.com/GoogleContainerTools/container-diff>`_
CLI, used by the Buildah/Kaniko parity integration tests (ML-12889) to assert two images built by
different engines are semantically equivalent. Byte-for-byte comparison doesn't work here - build
timestamps and filesystem metadata always differ between engines - so equivalence is defined as "no
diff in config, files, pip packages, or apt packages", which is what ``container-diff diff`` reports.

Not installed by default; see CONTRIBUTING.md's Testing section for the local install command.
"""

import json
import shutil
import subprocess

_DIFF_TYPES = ("file", "apt", "pip")


def is_installed() -> bool:
    return shutil.which("container-diff") is not None


def assert_images_equivalent(image_a: str, image_b: str, insecure: bool = True) -> None:
    """Run ``container-diff diff`` between two remote images and raise if any of the compared
    dimensions (file/apt/pip) reports a difference.

    :param image_a:  First image reference, e.g. ``registry:5000/some-image:tag``.
    :param image_b:  Second image reference, same registry-reference format.
    :param insecure: Whether the registries are plain HTTP (true for the local test registry).
    """
    args = [
        "container-diff",
        "diff",
        f"remote://{image_a}",
        f"remote://{image_b}",
        "--json",
    ]
    for diff_type in _DIFF_TYPES:
        args += ["--type", diff_type]
    if insecure:
        # container-diff has no bare "insecure" switch - it's opt-in per registry host.
        for registry_host in {image_a.split("/")[0], image_b.split("/")[0]}:
            args += ["--skip-tls-verify-registry", registry_host]

    result = subprocess.run(args, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"container-diff failed (exit {result.returncode}): {result.stderr}"
        )

    diffs = json.loads(result.stdout)
    non_empty_diffs = [d for d in diffs if _diff_is_non_empty(d)]
    if non_empty_diffs:
        raise AssertionError(
            f"container-diff found differences between {image_a} and {image_b}: "
            f"{json.dumps(non_empty_diffs, indent=2)}"
        )


def _diff_is_non_empty(diff_report: dict) -> bool:
    diff = diff_report.get("Diff") or {}
    return any(
        diff.get(key)
        for key in ("Adds", "Dels", "Mods", "Packages1", "Packages2", "InfoDiff")
    )
