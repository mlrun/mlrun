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
# This script checks for nonconforming imports of target modules (mlrun, mlrun_pipelines).
# It can be used standalone or as part of a pre-commit hook. When used as a pre-commit hook,
# it processes file paths passed as arguments (e.g., staged files during a commit).

import re
import sys
from pathlib import Path

TARGET_MODULES = {"mlrun", "mlrun_pipelines"}
# Match any "from mlrun[.something] import x" or "from mlrun_pipelines[.something] import x"
NONCONFORMING_PATTERN = re.compile(
    r"^\s*from\s+({})\b".format("|".join(TARGET_MODULES))
)


def check_nonconforming_imports(file_path: Path) -> list[str]:
    """
    Check for nonconforming imports in a file.

    :param file_path: Path to the file to check.
    :return: A list of violation messages for nonconforming imports.
    """
    violations = []
    try:
        with file_path.open("r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, start=1):
                if NONCONFORMING_PATTERN.match(line.strip()):
                    base_module = line.split()[1].split(".")[0]
                    violations.append(
                        f"{file_path}:{line_num}: Nonconforming import '{line.strip()}'. "
                        f"Use 'import {base_module}' instead."
                    )
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return violations


def main(files: list[str]) -> None:
    """
    Scan all provided files (staged files) and report nonconforming imports.

    :param files: A list of staged file paths as strings.
    :raises SystemExit: Exits with code 1 if violations are found, or 0 if none are found.
    """
    all_violations = []
    for file_path_str in files:
        file_path = Path(file_path_str)
        if file_path.suffix == ".py":
            all_violations.extend(check_nonconforming_imports(file_path))

    if all_violations:
        print("The following nonconforming imports were found:")
        for violation in all_violations:
            print(f"❌ {violation}")
        sys.exit(1)
    else:
        print("✅ All staged imports are valid.")
        sys.exit(0)


if __name__ == "__main__":
    # Pre-commit passes staged files as command-line arguments
    staged_files = sys.argv[1:]
    main(staged_files)
