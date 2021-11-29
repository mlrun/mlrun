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
        logger.warning("Failed resolving version. Ignoring and using 0.0.0+unstable")
        return "0.0.0+unstable"


def is_ignored(line):
    line = line.strip()
    return (not line) or (line[0] == "#")


def load_deps(path):
    """Load dependencies from requirements file"""
    with open(path) as fp:
        deps = []
        for line in fp:
            if is_ignored(line):
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


with open("README.md") as fp:
    long_desc = fp.read()

install_requires = list(load_deps("requirements.txt"))
tests_require = list(load_deps("dev-requirements.txt"))
api_deps = list(load_deps("dockerfiles/mlrun-api/requirements.txt"))

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
    "azure-blob-storage": ["azure-storage-blob~=12.0", "adlfs~=2021.8.1"],
    "azure-key-vault": ["azure-identity~=1.5", "azure-keyvault-secrets~=4.2"],
    # bokeh 2.4.0 requires typing-extensions>=3.10.0 but all tensorflow versions that compatible with our
    # tensorflow~=2.4.1 requirement requiring typing-extensions~=3.7.4 so limiting to 2.3.x
    "bokeh": ["bokeh~=2.3.0"],
    "plotly": ["plotly~=5.4"],
    "google-cloud-storage": ["gcsfs~=2021.8.1"],
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
        "mlrun.api",
        "mlrun.api.api",
        "mlrun.api.api.endpoints",
        "mlrun.api.crud",
        "mlrun.api.db",
        "mlrun.api.db.filedb",
        "mlrun.api.db.sqldb",
        "mlrun.api.db.sqldb.models",
        "mlrun.api.migrations_sqlite",
        "mlrun.api.migrations_sqlite.versions",
        "mlrun.api.migrations_mysql",
        "mlrun.api.migrations_mysql.versions",
        "mlrun.api.schemas",
        "mlrun.api.utils",
        "mlrun.api.utils.auth",
        "mlrun.api.utils.auth.providers",
        "mlrun.api.utils.clients",
        "mlrun.api.utils.db",
        "mlrun.api.utils.projects",
        "mlrun.api.utils.projects.remotes",
        "mlrun.api.utils.singletons",
        "mlrun.artifacts",
        "mlrun.data_types",
        "mlrun.datastore",
        "mlrun.db",
        "mlrun.feature_store",
        "mlrun.feature_store.retrieval",
        "mlrun.frameworks",
        "mlrun.frameworks._common",
        "mlrun.frameworks._dl_common",
        "mlrun.frameworks._dl_common.loggers",
        "mlrun.frameworks._ml_common",
        "mlrun.frameworks.auto_mlrun",
        "mlrun.frameworks.lgbm",
        "mlrun.frameworks.onnx",
        "mlrun.frameworks.pytorch",
        "mlrun.frameworks.pytorch.callbacks",
        "mlrun.frameworks.sklearn",
        "mlrun.frameworks.tf_keras",
        "mlrun.frameworks.tf_keras.callbacks",
        "mlrun.frameworks.xgboost",
        "mlrun.mlutils",
        "mlrun.model_monitoring",
        "mlrun.platforms",
        "mlrun.projects",
        "mlrun.runtimes",
        "mlrun.runtimes.mpijob",
        "mlrun.runtimes.sparkjob",
        "mlrun.serving",
        "mlrun.utils",
        "mlrun.utils.version",
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
