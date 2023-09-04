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

import mlrun.model_monitoring.api


def test_read_dataset_as_dataframe():
    # Test list with feature columns
    dataset = [[5.8, 2.8, 5.1, 2.4], [6.0, 2.2, 4.0, 1.0]]
    feature_columns = ["feature_1", "feature_2", "feature_3", "feature_4"]

    df, _ = mlrun.model_monitoring.api.read_dataset_as_dataframe(
        dataset=dataset,
        feature_columns=feature_columns,
    )
    assert list(df.columns) == feature_columns
    assert df["feature_1"].to_list() == [dataset[0][0], dataset[1][0]]

    # Test dictionary
    dataset_dict = {}
    for i in range(len(feature_columns)):
        dataset_dict[feature_columns[i]] = [dataset[0][i], dataset[1][i]]
    df, _ = mlrun.model_monitoring.api.read_dataset_as_dataframe(
        dataset=dataset_dict, drop_columns="feature_2"
    )
    feature_columns.remove("feature_2")
    assert list(df.columns) == feature_columns
