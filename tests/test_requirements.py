import builtins
import collections
import json
import pathlib
import re
import typing
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
    setup_py_extras_requirements_specifiers_map = _parse_requirement_specifiers_list(
        setup_py_extras_requirements_specifiers
    )
    extras_requirements_file_specifiers_map = _parse_requirement_specifiers_list(
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


def test_requirement_specifiers_convention():
    """
    This test exists to verify we follow our convention for requirement specifiers which is:
    If the package major is 0, it is considered unstable, and minor changes may include backwards incompatible changes.
    Therefore we limit to patch changes only, the way to do it is to specify X.Y.Z with the ~= operator.
    If the package major is 1 or above, it is considered stable, backwards incompatible changes can only occur together
    with a major bump. Therefore we allow patch and minor changes, the way to do it is to specify X.Y with the ~=
    operator
    """
    requirement_specifiers_map = _generate_all_requirement_specifiers_map()
    print(requirement_specifiers_map)

    invalid_requirement_specifiers_map = collections.defaultdict(set)
    for requirement_name, requirement_specifiers in requirement_specifiers_map.items():
        for requirement_specifier in requirement_specifiers:
            # we don't care about what's coming after the ; (it will be something like "python_version < '3.7'")
            tested_requirement_specifier = requirement_specifier.split(";")[0]
            invalid_requirement = False
            if not tested_requirement_specifier.startswith("~="):
                invalid_requirement = True
            else:
                major_version = int(
                    tested_requirement_specifier[
                        len("~=") : tested_requirement_specifier.find(".")
                    ]
                )
                is_stable_requirement = major_version >= 1
                # if it's stable we want to prevent only major changes, meaning version should be X.Y
                # if it's not stable we want to prevent major and minor changes, meaning version should be X.Y.Z
                wanted_number_of_dot_occurences = 1 if is_stable_requirement else 2
                if (
                    tested_requirement_specifier.count(".")
                    != wanted_number_of_dot_occurences
                ):
                    invalid_requirement = True
            if invalid_requirement:
                invalid_requirement_specifiers_map[requirement_name].add(
                    requirement_specifier
                )

    ignored_invalid_map = {
        # See comment near requirement for why we're limiting to patch changes only for all of these
        "kfp": {"~=1.0.1"},
        "botocore": {">=1.20.106,<1.20.107"},
        "aiobotocore": {"~=1.4.0"},
        "aiohttp": {">=3.6,<3.8"},
        "bokeh": {"~=2.3.0"},
        "storey": {"~=0.8.11, <0.8.12; python_version >= '3.7'"},
        # Black is not stable yet and does not have a release that is not beta, so can't be used with ~=
        "black": {"<=19.10b0"},
        # These 2 are used in a tests that is purposed to test requirement without specifiers
        "faker": {""},
        "python-dotenv": {""},
        # These are not semver
        "opencv-contrib-python": {">=4.2.0.34"},
        "pyhive": {" @ git+https://github.com/v3io/PyHive.git@v0.6.999"},
        "v3io-generator": {
            " @ git+https://github.com/v3io/data-science.git#subdirectory=generator"
        },
        "fsspec": {"~=2021.8.1"},
        "adlfs": {"~=2021.8.1"},
        "s3fs": {"~=2021.8.1"},
        "gcsfs": {"~=2021.8.1"},
        # All of these are actually valid, they just don't use ~= so the test doesn't "understand" that
        # TODO: make test smart enough to understand that
        "urllib3": {">=1.25.4, <1.27"},
        "cryptography": {"~=3.0, <3.4"},
        "chardet": {">=3.0.2, <4.0"},
        "google-auth": {">=1.25.0, <2.0dev"},
        "ipython": {"~=7.0, <7.17"},
        "numpy": {">=1.16.5, <1.22.0"},
        "tabulate": {">=0.8.0, <=0.8.3"},
        "orjson": {">=3,<3.4"},
        "alembic": {"~=1.4,<1.6.0"},
        "distributed": {">=2.23, <3"},
        "boto3": {"~=1.9, <1.17.107"},
        "azure-storage-blob": {"~=12.0, <12.7.0"},
        "dask-ml": {"~=1.4,<1.9.0"},
        "pyarrow": {">=1,<6"},
    }

    for (ignored_requirement_name, ignored_specifiers,) in ignored_invalid_map.items():
        if ignored_requirement_name in invalid_requirement_specifiers_map:
            diff = deepdiff.DeepDiff(
                invalid_requirement_specifiers_map[ignored_requirement_name],
                ignored_specifiers,
                ignore_order=True,
            )
            if diff == {}:
                del invalid_requirement_specifiers_map[ignored_requirement_name]

    assert invalid_requirement_specifiers_map == {}


def test_requirement_specifiers_inconsistencies():
    requirement_specifiers_map = _generate_all_requirement_specifiers_map()
    inconsistent_specifiers_map = {}
    print(requirement_specifiers_map)
    for requirement_name, requirement_specifiers in requirement_specifiers_map.items():
        if not len(requirement_specifiers) == 1:
            inconsistent_specifiers_map[requirement_name] = requirement_specifiers

    ignored_inconsistencies_map = {
        # It's ok we have 2 different versions cause they are for different python versions
        "pandas": {"~=1.2; python_version >= '3.7'", "~=1.0; python_version < '3.7'"},
        "scikit-learn": {
            "~=1.0; python_version >= '3.7'",
            "~=0.23.0; python_version < '3.7'",
        },
        "tensorboard": {
            "~=2.5; python_version >= '3.7'",
            "~=1.0; python_version < '3.7'",
        },
        # The empty specifier is from tests/runtimes/assets/requirements.txt which is there specifically to test the
        # scenario of requirements without version specifiers
        "python-dotenv": {"", "~=0.17.0"},
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
    requirement_specifiers_map = _parse_requirement_specifiers_list(
        [
            "some-package~=1.9, <1.17.50",
            "other-package==0.1",
            "git+https://github.com/mlrun/something.git@some-branch#egg=more-package",
        ]
    )
    assert len(requirement_specifiers_map) > 0
    assert requirement_specifiers_map["some-package"] == {
        "~=1.9, <1.17.50",
    }
    assert requirement_specifiers_map["other-package"] == {
        "==0.1",
    }
    assert requirement_specifiers_map["more-package"] == {
        "git+https://github.com/mlrun/something.git@some-branch",
    }


def _generate_all_requirement_specifiers_map() -> typing.Dict[str, typing.Set]:
    requirements_file_paths = list(
        pathlib.Path(tests.conftest.root_path).rglob("**/*requirements.txt")
    )
    venv_path = pathlib.Path(tests.conftest.root_path) / "venv"
    requirements_file_paths = list(
        filter(lambda path: str(venv_path) not in str(path), requirements_file_paths)
    )

    requirement_specifiers = []
    for requirements_file_path in requirements_file_paths:
        requirement_specifiers.extend(_load_requirements(requirements_file_path))

    requirement_specifiers.extend(_import_extras_requirements())

    return _parse_requirement_specifiers_list(requirement_specifiers)


def _parse_requirement_specifiers_list(
    requirement_specifiers,
) -> typing.Dict[str, typing.Set]:
    specific_module_regex = (
        r"^"
        r"(?P<requirementName>[a-zA-Z\-0-9_]+)"
        r"(?P<requirementExtra>\[[a-zA-Z\-0-9_]+\])?"
        r"(?P<requirementSpecifier>.*)"
    )
    remote_location_regex = (
        r"^(?P<requirementSpecifier>.*)#egg=(?P<requirementName>[^#]+)"
    )
    requirement_specifiers_map = collections.defaultdict(set)
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
        requirement_specifiers_map[match.groupdict()["requirementName"].lower()].add(
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
