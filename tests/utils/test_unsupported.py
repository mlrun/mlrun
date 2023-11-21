# Copyright 2023 Iguazio
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

from kfp.dsl import ParallelFor
from pytest import raises

from mlrun.utils import unsupported


def test_unsupported_parallelfor():
    # Create an object with an invalid parameter to trigger the expected Kubeflow behaviour
    with raises(ValueError):
        ParallelFor(loop_args=[1], parallelism=-1)

    with unsupported.disable_unsupported_external_features():
        # Creation of an unsupported class must raise a NotImplementedError when done in context
        with raises(NotImplementedError):
            ParallelFor(loop_args=[1], parallelism=-1)

    # Interact with the native Kubeflow class again to confirm that the context has been cleaned up
    with raises(ValueError):
        ParallelFor(loop_args=[1], parallelism=-1)
