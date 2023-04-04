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


class _BaseLauncher(abc.ABC):
    """
    Abstract class for managing and running functions in different contexts
    This class is designed to encapsulate the logic of running a function in different contexts
    i.e. running a function locally, remotely or in a server
    Each context will have its own implementation of the abstract methods
    """

    @staticmethod
    @abc.abstractmethod
    def verify_base_image(runtime):
        """resolves and sets the build base image if build is needed"""
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
    def _enrich_runtime(runtime):
        pass

    @staticmethod
    @abc.abstractmethod
    def _validate_runtime(runtime):
        pass
