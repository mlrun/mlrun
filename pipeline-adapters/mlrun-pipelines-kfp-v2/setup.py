# Copyright 2024 Iguazio
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

import logging

from setuptools import find_namespace_packages, setup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlrun-kfp-setup")

setup(
    name="mlrun-pipelines-kfp-v2-experiment",
    version="0.2.1",
    description="MLRun Pipelines package for providing KFP 2.* compatibility",
    author="Yaron Haviv",
    author_email="yaronh@iguazio.com",
    license="Apache License 2.0",
    url="https://github.com/mlrun/mlrun",
    packages=find_namespace_packages(
        where="src/",
        include=[
            "mlrun_pipelines",
        ],
    ),
    package_dir={"": "src"},
    keywords=[
        "mlrun",
        "kfp",
    ],
    python_requires=">=3.9, <3.12",
    install_requires=[
        "kfp[kubernetes]~=2.5.0",
    ],
)
