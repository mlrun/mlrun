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

import json
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlrun-setup")


def version():
    try:
        with open("mlrun/utils/version/version.json") as version_file:
            version_metadata = json.load(version_file)
            version_ = version_metadata["version"]
            # replace "1.4.0-rc1+rca" with "1.4.0rc1+rca"
            return re.sub(r"(\d+\.\d+\.\d+)-rc(\d+)", r"\1rc\2", version_)
    except (ValueError, KeyError, FileNotFoundError):
        # When installing un-released version (e.g. by doing
        # pip install git+https://github.com/mlrun/mlrun@development)
        # it won't have a version file, so adding some sane default
        logger.warning("Failed resolving version. Ignoring and using 0.0.0+unstable")
        return "0.0.0+unstable"


__pypi_version__ = version()
