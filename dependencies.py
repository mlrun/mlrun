# Copyright 2023 MLRun Authors
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
import typing


def install_requirements():
    return list(_load_dependencies_from_file("requirements.txt"))


def dev_requirements():
    return list(_load_dependencies_from_file("dev-requirements.txt"))


def extra_requirements():

    # NOTE:
    #     - These are tested in `automation/package_test/test.py`. If you modify these, make sure to change the
    #       corresponding line there.
    #     - We have a copy of these in extras-requirements.txt. If you modify these, make sure to change it there as well
    extras_require = {
        # from 1.17.107 boto3 requires botocore>=1.20.107,<1.21.0 which
        # conflicts with s3fs 2021.8.1 that has aiobotocore~=1.4.0
        # which so far (1.4.1) has botocore>=1.20.106,<1.20.107
        # boto3 1.17.106 has botocore>=1.20.106,<1.21.0, so we must add botocore explicitly
        "s3": [
            "boto3~=1.9, <1.17.107",
            "botocore>=1.20.106,<1.20.107",
            "aiobotocore~=1.4.0",
            "s3fs~=2021.8.1",
        ],
        "azure-blob-storage": [
            "msrest~=0.6.21",
            "azure-core~=1.24",
            "azure-storage-blob~=12.13",
            "adlfs~=2021.8.1",
        ],
        "azure-key-vault": ["azure-identity~=1.5", "azure-keyvault-secrets~=4.2"],
        "bokeh": [
            # >=2.4.2 to force having a security fix done in 2.4.2
            "bokeh~=2.4, >=2.4.2",
        ],
        # plotly artifact body in 5.12.0 may contain chars that are not encodable in 'latin-1' encoding
        # so, it cannot be logged as artifact (raised UnicodeEncode error - ML-3255)
        "plotly": ["plotly~=5.4, <5.12.0"],
        # used to generate visualization nuclio/serving graph steps
        "graphviz": ["graphviz~=0.20.0"],
        # google-cloud is mainly used for QA, that is why we are not including it in complete
        "google-cloud": [
            # because of kfp 1.8.13 requiring google-cloud-storage<2.0.0, >=1.20.0
            "google-cloud-storage~=1.20",
            # because of storey which isn't compatible with google-cloud-bigquery >3.2, conflicting grpcio
            # google-cloud-bigquery 3.3.0 has grpcio >= 1.47.0, < 2.0dev while storey 1.2.2 has grpcio<1.42 and >1.34.0
            "google-cloud-bigquery[pandas]~=3.2",
            "google-cloud~=0.34",
        ],
        "google-cloud-storage": ["gcsfs~=2021.8.1"],
        "google-cloud-bigquery": ["google-cloud-bigquery[pandas]~=3.2"],
        "kafka": ["kafka-python~=2.0"],
        "redis": ["redis~=4.3"],
    }

    # see above why we are excluding google-cloud
    exclude_from_complete = ["bokeh", "google-cloud"]
    api_deps = list(
        _load_dependencies_from_file("dockerfiles/mlrun-api/requirements.txt")
    )
    extras_require.update(
        {
            "api": api_deps,
            "all": _get_extra_dependencies(extras_require=extras_require),
            "complete": _get_extra_dependencies(
                exclude=exclude_from_complete,
                extras_require=extras_require,
            ),
            "complete-api": _get_extra_dependencies(
                exclude=exclude_from_complete,
                base_deps=api_deps,
                extras_require=extras_require,
            ),
        }
    )
    return extras_require


def _is_ignored(line: str) -> bool:
    line = line.strip()
    return (not line) or (line[0] == "#")


def _extract_package_from_egg(line: str) -> str:
    """Extract egg name from line"""
    if "#egg=" in line:
        _, package = line.split("#egg=")
        return f"{package} @ {line}"
    return line


def _load_dependencies_from_file(path: str) -> typing.List[str]:
    """Load dependencies from requirements file"""
    with open(path) as fp:
        return [
            _extract_package_from_egg(line.strip())
            for line in fp
            if not _is_ignored(line)
        ]


def _get_extra_dependencies(
    include: typing.List[str] = None,
    exclude: typing.List[str] = None,
    base_deps: typing.List[str] = None,
    extras_require: typing.Dict[str, typing.List[str]] = None,
) -> typing.List[str]:
    """Get list of dependencies for given extras categories

    :param include: list of extras categories to include
    :param exclude: list of extras categories to exclude
    :param base_deps: list of base dependencies to include
    :return: list of dependencies
    """
    include = include or []
    exclude = exclude or []
    base_deps = base_deps or []
    extras_require = extras_require or {}
    extra_deps = {
        requirement
        for extra_key, requirement_list in extras_require.items()
        for requirement in requirement_list
        if extra_key not in exclude and (not include or extra_key in include)
    }
    extra_deps.update(base_deps)
    return list(sorted(extra_deps))
