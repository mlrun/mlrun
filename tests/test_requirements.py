import builtins
import collections
import json
import pathlib
import re
import unittest.mock

import deepdiff
import setuptools

import tests.conftest


def test_extras_requirement_file_aligned():
    """
    See comment in top of "extras-requirements.txt" for explanation for what this test is for
    """
    setup_py_extras_requirements_specifiers = _import_extras_requirements()
    extras_requirements_file_specifiers = _load_requirements(
        pathlib.Path(tests.conftest.root_path) / "extras-requirements.txt"
    )
    setup_py_extras_requirements_specifiers_map = _generate_requirement_specifiers_map(
        setup_py_extras_requirements_specifiers
    )
    extras_requirements_file_specifiers_map = _generate_requirement_specifiers_map(
        extras_requirements_file_specifiers
    )
    assert (
        deepdiff.DeepDiff(
            setup_py_extras_requirements_specifiers_map,
            extras_requirements_file_specifiers_map,
            ignore_order=True,
        )
        == {}
    )


def test_requirement_specifiers_inconsistencies():
    requirements_file_paths = list(
        pathlib.Path(tests.conftest.root_path).rglob("**/*requirements.txt")
    )

    requirement_specifiers = []
    for requirements_file_path in requirements_file_paths:
        requirement_specifiers.extend(_load_requirements(requirements_file_path))

    requirement_specifiers.extend(_import_extras_requirements())

    requirement_specifiers_map = _generate_requirement_specifiers_map(
        requirement_specifiers
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


def test_requirement_from_remote():
    requirement_specifiers_map = _generate_requirement_specifiers_map(
        [
            "some-package~=1.9, <1.17.50",
            "other-package==0.1",
            "git+https://github.com/mlrun/something.git@some-branch#egg=more-package",
        ]
    )
    assert len(requirement_specifiers_map) > 0
    assert requirement_specifiers_map["some-package"] == [
        "~=1.9, <1.17.50",
    ]
    assert requirement_specifiers_map["other-package"] == [
        "==0.1",
    ]
    assert requirement_specifiers_map["more-package"] == [
        "git+https://github.com/mlrun/something.git@some-branch",
    ]


def _generate_requirement_specifiers_map(requirement_specifiers):
    specific_module_regex = (
        r"^"
        r"(?P<requirementName>[a-zA-Z\-0-9_]+)"
        r"(?P<requirementExtra>\[[a-zA-Z\-0-9_]+\])?"
        r"(?P<requirementSpecifier>.*)"
    )
    remote_location_regex = (
        r"^"
        r"(?P<requirementSpecifier>.*)"
        r"#egg="
        r"(?P<requirementName>[a-zA-Z\-0-9_]+)"
    )
    requirement_specifiers_map = collections.defaultdict(list)
    for requirement_specifier in requirement_specifiers:
        regex = (
            remote_location_regex
            if "#egg=" in requirement_specifier
            else specific_module_regex
        )
        match = re.fullmatch(regex, requirement_specifier)
        assert (
            match is not None
        ), f"Requirement specifier did not matched regex. {requirement_specifier}"
        requirement_specifiers_map[match.groupdict()["requirementName"]].append(
            match.groupdict()["requirementSpecifier"]
        )
    return requirement_specifiers_map


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
    """
    Load dependencies from requirements file, exactly like `setup.py`
    """
    with open(path) as fp:
        deps = []
        for line in fp:
            if _is_ignored_requirement_line(line):
                continue
            line = line.strip()

            # e.g.: git+https://github.com/nuclio/nuclio-jupyter.git@some-branch#egg=nuclio-jupyter
            if "#egg=" in line:
                _, package = line.split("#egg=")
                deps.append(f"{package} @ {line}")
                continue

            # append package
            deps.append(line)
        return deps
