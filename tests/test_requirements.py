import builtins
import collections
import json
import pathlib
import re
import unittest.mock

import deepdiff
import setuptools

import tests.conftest


def test_requirement_specifiers_inconsistencies():
    requirements_file_paths = list(
        pathlib.Path(tests.conftest.root_path).rglob("**/*requirements.txt")
    )

    requirement_specifiers = []
    for requirements_file_path in requirements_file_paths:
        requirement_specifiers.extend(_load_requirements(requirements_file_path))

    requirement_specifiers.extend(_import_extras_requirements())

    regex = (
        r"^"
        r"(?P<requirementName>[a-zA-Z\-0-9_]+)"
        r"(?P<requirementExtra>\[[a-zA-Z\-0-9_]+\])?"
        r"(?P<requirementSpecifier>.*)"
    )
    requirement_specifiers_map = collections.defaultdict(list)
    for requirement_specifier in requirement_specifiers:
        match = re.fullmatch(regex, requirement_specifier)
        assert (
            match is not None
        ), f"Requirement specifier did not matched regex. {requirement_specifier}"
        requirement_specifiers_map[match.groupdict()["requirementName"]].append(
            match.groupdict()["requirementSpecifier"]
        )
    inconsistent_specifiers_map = {}
    print(requirement_specifiers_map)
    for requirement_name, requirement_specifiers in requirement_specifiers_map.items():
        if not all(
            requirement_specifier == requirement_specifiers[0]
            for requirement_specifier in requirement_specifiers
        ):
            inconsistent_specifiers_map[requirement_name] = requirement_specifiers

    ignored_inconsistencies_map = {
        # It's ==1.4.1 only in models-gpu because of tensorflow 2.2.0, TF version expected to be changed there soon so
        # ignoring for now
        "scipy": ["~=1.0", "==1.4.1", "~=1.0"],
        # It's ok we have 2 different versions cause they are for different python versions
        "pandas": ["~=1.2; python_version >= '3.7'", "~=1.0; python_version < '3.7'"],
        # The empty specifier is from tests/runtimes/assets/requirements.txt which is there specifically to test the
        # scenario of requirements without version specifiers
        "python-dotenv": ["", "~=0.17.0"],
    }

    for (
        inconsistent_requirement_name,
        inconsistent_specifiers,
    ) in ignored_inconsistencies_map.items():
        if inconsistent_requirement_name in inconsistent_specifiers_map:
            diff = deepdiff.DeepDiff(
                inconsistent_specifiers_map[inconsistent_requirement_name],
                inconsistent_specifiers,
                ignore_order=True,
            )
            if diff == {}:
                del inconsistent_specifiers_map[inconsistent_requirement_name]

    assert inconsistent_specifiers_map == {}


def _import_extras_requirements():
    def mock_file_open(file, *args, **kwargs):
        if "setup.py" not in str(file):
            return unittest.mock.mock_open(
                read_data=json.dumps({"version": "some-ver"})
            ).return_value
        else:
            return original_open(file, *args, **kwargs)

    original_setup = setuptools.setup
    original_open = builtins.open
    setuptools.setup = lambda *args, **kwargs: 0
    builtins.open = mock_file_open

    import setup

    setuptools.setup = original_setup
    builtins.open = original_open

    ignored_extras = [
        "api",
        "complete",
        "complete-api",
    ]

    extras_requirements = []
    for extra_name, extra_requirements in setup.extras_require.items():
        if extra_name not in ignored_extras:
            extras_requirements.extend(extra_requirements)

    return extras_requirements


def _is_ignored_requirement_line(line):
    line = line.strip()
    return (not line) or (line[0] == "#")


def _load_requirements(path):
    with open(path) as file:
        return [line.strip() for line in file if not _is_ignored_requirement_line(line)]
