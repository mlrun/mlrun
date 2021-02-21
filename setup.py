# Copyright 2018 Iguazio
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

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlrun-setup")


def version():
    try:
        with open("mlrun/utils/version/version.json") as version_file:
            version_metadata = json.load(version_file)
            return version_metadata["version"]
    except (ValueError, KeyError, FileNotFoundError):
        # When installing un-released version (e.g. by doing
        # pip install git+https://github.com/mlrun/mlrun@development)
        # it won't have a version file, so adding some sane default
        logger.warning("Failed resolving version. Ignoring and using unstable")
        return "unstable"


def is_ignored(line):
    line = line.strip()
    return (not line) or (line[0] == "#")


def load_deps(path):
    """Load dependencies from requirements file"""
    with open(path) as fp:
        return [line.strip() for line in fp if not is_ignored(line)]


with open("README.md") as fp:
    long_desc = fp.read()

install_requires = list(load_deps("requirements.txt"))
tests_require = list(load_deps("dev-requirements.txt"))
api_deps = list(load_deps("dockerfiles/mlrun-api/requirements.txt"))

# NOTE: These are tested in `automation/package_test/test.py` If
# you modify these, make sure to change the corresponding line there.
extras_require = {
    # from 1.16.53 it requires botocore<1.20.0,>=1.19.53 which conflicts with s3fs 0.5.2 that has aiobotocore>=1.0.1
    # which resolves to 1.2.1 which has botocore >=1.19.52,<1.19.53
    # boto3 1.16.53 has botocore<1.20.0, >=1.19.53, so we must add botocore explictly
    "s3": ["boto3~=1.9, <1.16.53", "botocore>=1.19.52, <1.19.53", "s3fs~=0.5"],
    # <12.7.0 from adlfs 0.6.3
    "azure-blob-storage": ["azure-storage-blob~=12.0, <12.7.0", "adlfs~=0.6"],
}
extras_require["complete"] = sorted(
    {
        requirement
        for requirement_list in extras_require.values()
        for requirement in requirement_list
    }
)
extras_require["api"] = api_deps
complete_api_deps = set(api_deps)
complete_api_deps.update(extras_require["complete"])
extras_require["complete-api"] = sorted(complete_api_deps)


setup(
    name="mlrun",
    version=version(),
    description="Tracking and config of machine learning runs",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    author="Yaron Haviv",
    author_email="yaronh@iguazio.com",
    license="MIT",
    url="https://github.com/mlrun/mlrun",
    packages=[
        "mlrun",
        "mlrun.runtimes",
        "mlrun.runtimes.mpijob",
        "mlrun.serving",
        "mlrun.db",
        "mlrun.mlutils",
        "mlrun.platforms",
        "mlrun.projects",
        "mlrun.artifacts",
        "mlrun.utils",
        "mlrun.utils.version",
        "mlrun.datastore",
        "mlrun.data_types",
        "mlrun.feature_store",
        "mlrun.feature_store.retrieval",
        "mlrun.api",
        "mlrun.api.api",
        "mlrun.api.api.endpoints",
        "mlrun.api.db",
        "mlrun.api.db.sqldb",
        "mlrun.api.db.filedb",
        "mlrun.api.schemas",
        "mlrun.api.crud",
        "mlrun.api.utils",
        "mlrun.api.utils.singletons",
        "mlrun.api.utils.clients",
        "mlrun.api.utils.projects",
        "mlrun.api.utils.projects.remotes",
    ],
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    tests_require=tests_require,
    zip_safe=False,
    include_package_data=True,
    entry_points={"console_scripts": ["mlrun=mlrun.__main__:main"]},
)
