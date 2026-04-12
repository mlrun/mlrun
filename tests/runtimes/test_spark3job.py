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

from unittest.mock import patch

import pytest

import mlrun.runtimes.pod
import mlrun.runtimes.sparkjob.spark3job


@pytest.mark.parametrize("attr_name", ["driver_affinity", "executor_affinity"])
def test_affinity_setter_passes_correct_attribute_name(attr_name):
    """
    Verify that each affinity setter passes its own attribute name
    to transform_attribute_to_k8s_class_instance.
    Regression test for a copy-paste error where the driver_affinity
    setter was passing "executor_affinity" instead of "driver_affinity".
    """
    spec = mlrun.runtimes.sparkjob.spark3job.Spark3JobSpec()

    with patch.object(
        mlrun.runtimes.pod,
        "transform_attribute_to_k8s_class_instance",
        wraps=mlrun.runtimes.pod.transform_attribute_to_k8s_class_instance,
    ) as mock_transform:
        test_affinity = {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "test-node",
                                    "operator": "In",
                                    "values": ["true"],
                                }
                            ]
                        }
                    ]
                }
            }
        }

        setattr(spec, attr_name, test_affinity)

        mock_transform.assert_called_once_with(attr_name, test_affinity)
