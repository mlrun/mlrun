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
import pandas
import pytest

import mlrun.mlutils.data


def test_get_sample_failure_label_not_exist():
    data = {"col1": [1, 2], "col2": [3, 4]}
    data_frame = pandas.DataFrame(data=data)
    with pytest.raises(ValueError):
        mlrun.mlutils.data.get_sample(data_frame, 2, "non_existing_label")
    with pytest.raises(ValueError):
        mlrun.mlutils.data.get_sample(data_frame, -2, "non_existing_label")
