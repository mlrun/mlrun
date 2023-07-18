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
#
import pandas as pd

features = {
    "feature_1": [
        {"a": 4, "b": 8, "c": 5},
        {"a": 5, "b": 6, "c": 3},
        {"a": 3, "b": 10, "c": 2},
    ],
    "feature_2": [
        {"a": 3, "b": 2, "c": 10},
        {"a": 9, "b": 10, "c": 9},
        {"a": 4, "b": 9, "c": 2},
        {"a": 3, "b": 6, "c": 4},
    ],
}


def log_dataset(context):
    for dataset_name, dataset_content in (features or {}).items():
        df = pd.DataFrame(dataset_content)
        context.log_dataset(
            dataset_name,
            df=df,
            format="csv",
        )
