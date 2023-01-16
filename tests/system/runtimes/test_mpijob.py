# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import json
import os
import time
import uuid

import pandas as pd
import pytest
import requests
import v3io
from storey import MapClass
from v3io.dataplane import RaiseForStatus

import mlrun
import tests.system.base
from mlrun import feature_store as fstore
from mlrun.datastore.sources import KafkaSource
from mlrun.datastore.targets import ParquetTarget


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestMpiJobRuntime(tests.system.base.TestMLRunSystem):
    project_name = "does-not-exist-mpijob"

    def test_run_state_completion(self):
        code_path = str(self.assets_path / "mpijob_function.py")

        # project = mlrun.get_or_create_project(name=self.project_name, context="./", user_project=False)

        # Create the open mpi function:
        mpijob_function = mlrun.code_to_function(
            name="mpijob_test",
            kind="mpijob",
            handler="handler",
            project=self.project_name,
            filename=code_path,
            image="mlrun/ml-models",
            requirements=["mpi4py"]
        )
        mpijob_function.spec.replicas = 4
        mpijob_function.deploy()  # In order to build the image with `mpi4py`.

        mpijob_run = mpijob_function.run()

        mpijob_time = mpijob_run.status.results['time']
        mpijob_result = mpijob_run.status.results['result']
        assert mpijob_time is not None
        assert mpijob_result == 1000
