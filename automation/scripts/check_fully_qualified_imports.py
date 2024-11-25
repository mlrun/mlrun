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
import collections.abc
import pathlib
import re
import sys

IMPORT_REGEX = re.compile(r"from\s+[\w.]+\s+import\s+\w+")


def check_fully_qualified_imports(file_path: pathlib.Path) -> list[tuple[int, str]]:
    """
    Check for non-fully-qualified imports in a given file.

    :param file_path: The path to the file to be checked.
    :type file_path: Path
    :return: A list of tuples, each containing the line number and line content where violations were found.
    :rtype: list[tuple[int, str]]
    """
    violations: list[tuple[int, str]] = []
    try:
        with file_path.open("r") as file:
            for i, line in enumerate(file, start=1):
                if IMPORT_REGEX.search(line):
                    violations.append((i, line.strip()))
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return violations


def iter_python_files(
    file_paths: list[str], excluded: set[pathlib.Path]
) -> collections.abc.Generator[pathlib.Path, None, None]:
    """
    Yield Python files from a list of file paths, excluding specified paths.

    :param file_paths: List of file paths to check.
    :type file_paths: list[str]
    :param excluded: Set of paths to exclude from checking.
    :type excluded: set[Path]
    :yield: Path object for Python files not in excluded paths.
    :rtype: Generator[Path, None, None]
    """
    for file_path_str in file_paths:
        file_path = pathlib.Path(file_path_str)
        if file_path.suffix == ".py" and file_path not in excluded:
            yield file_path


def format_violations(errors: dict[pathlib.Path, list[tuple[int, str]]]) -> None:
    """
    Print formatted violations for non-fully-qualified imports.

    :param errors: Dictionary mapping file paths to lists of violations.
    :type errors: dict[Path, list[tuple[int, str]]]
    """
    print("The following files have non-fully-qualified imports:")
    for file, issues in errors.items():
        for line_no, line in issues:
            # Output in PyCharm-clickable format
            print(f"{file}:{line_no}: {line}")


def main() -> None:
    """
    Main function to check for non-fully-qualified imports in modified files.

    :return: None
    """
    # Parse command-line arguments
    excluded_files: set[pathlib.Path] = {
        pathlib.Path(path) for path in sys.argv[1:] if path.startswith("--exclude=")
    }
    modified_files: list[str] = [
        arg for arg in sys.argv[1:] if not arg.startswith("--exclude=")
    ]

    errors: dict[pathlib.Path, list[tuple[int, str]]] = {}

    for file_path in iter_python_files(modified_files, excluded_files):
        violations = check_fully_qualified_imports(file_path)
        if violations:
            errors[file_path] = violations

    if errors:
        format_violations(errors)
        sys.exit(1)
    else:
        print("All imports are fully qualified!")
        sys.exit(0)


if __name__ == "__main__":
    main()
