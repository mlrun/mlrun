#!/usr/bin/env python3
# Copyright 2023 Iguazio
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
import pathlib
import re
import sys

TARGET_MODULES = {"mlrun", "mlrun_pipelines"}

DISALLOWED_PATTERN = re.compile(r"^\s*from\s+(mlrun|mlrun_pipelines)\s+import")


def check_disallowed_imports(file_path: pathlib.Path) -> list[str]:
    """
    Check for disallowed imports in a file and return a list of violations.
    """
    violations = []
    with open(file_path) as file:
        for line_num, line in enumerate(file, start=1):
            if DISALLOWED_PATTERN.match(line.strip()):
                violations.append(
                    f"{file_path}:{line_num}: Disallowed import '{line.strip()}'."
                    f" Use 'import {line.split()[1]}' instead."
                )
    return violations


def main(files: list[str]):
    """
    Scan all provided files and report disallowed imports.
    """
    all_violations = []
    for file_path in files:
        if file_path.endswith(".py"):
            all_violations.extend(check_disallowed_imports(file_path))

    if all_violations:
        print("The following disallowed imports were found:")
        for violation in all_violations:
            print(f"❌ {violation}")
        sys.exit(1)
    else:
        print("✅ All imports are valid.")
        sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
