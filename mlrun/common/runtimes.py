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
import abc


class AbstractFindMeAName(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def verify_base_image(runtime):
        """resolves whether build is required and sets the runtime base image"""
        pass

    @staticmethod
    @abc.abstractmethod
    def save(runtime):
        """store the function to the db"""
        pass

    @staticmethod
    def run(runtime):
        """run the function from the server/client[local/remote]"""
        pass

    @staticmethod
    @abc.abstractmethod
    def _enrich_and_validate(runtime):
        """
        enrich the function with:
            1. default values
            2. mlrun config values
            3. project context values
            4. run specific parameters
         and validate the function
        """
        pass
