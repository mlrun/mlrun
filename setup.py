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

import dependencies
import packages

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


with open("README.md") as fp:
    long_desc = fp.read()


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
    packages=packages.packages(
        exclude_packages=[
            "mlrun.api.migrations_sqlite.tests",
            "mlrun.api.proto",
        ]
    ),
    install_requires=dependencies.base_requirements(),
    tests_require=dependencies.dev_requirements(),
    extras_require=dependencies.extra_requirements(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    zip_safe=False,
    include_package_data=True,
    entry_points={"console_scripts": ["mlrun=mlrun.__main__:main"]},
)
