import re
import tests.conftest
import pathlib
import collections
import deepdiff


def test_requirement_specifiers_inconsistencies():
    requirements_file_paths = list(
        pathlib.Path(tests.conftest.root_path).rglob("**/*requirements.txt")
    )

    requirement_specifiers = []
    for requirements_file_path in requirements_file_paths:
        requirement_specifiers.extend(load_requirements(requirements_file_path))

    regex = (
        r"^"
        r"(?P<requirementName>[a-zA-Z\-0-9_]+)"
        r"(?P<requirementExtra>\[[a-zA-Z\-0-9_]+\])?"
        r"(?P<requirementSpecifier>.*)"
    )
    requirement_specifiers_map = collections.defaultdict(list)
    for requirement_specifier in requirement_specifiers:
        match = re.fullmatch(regex, requirement_specifier)
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
        "scipy": ["~=1.0", "==1.4.1", "~=1.0"]
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


def is_ignored_requirement_line(line):
    line = line.strip()
    return (not line) or (line[0] == "#")


def load_requirements(path):
    with open(path) as file:
        return [line.strip() for line in file if not is_ignored_requirement_line(line)]
