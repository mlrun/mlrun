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
import pytest

from mlrun.data_types import InferOptions
from mlrun.data_types.infer import infer_schema_from_df
from mlrun.features import Feature
from mlrun.model import ObjectList


@pytest.mark.parametrize(
    "df, features, push_at_start",
    [
        (
            pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}),
            ObjectList.from_list(Feature, [Feature(name="c"), Feature(name="d")]),
            True,
        ),
        (
            pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}),
            ObjectList.from_list(Feature, [Feature(name="c"), Feature(name="d")]),
            False,
        ),
    ],
)
def test_infer_schema_from_df(df, features, push_at_start):
    infer_schema_from_df(
        df, features, {}, push_at_start=push_at_start, options=InferOptions.Features
    )
    features_by_order = list(features.keys())
    if push_at_start:
        assert features_by_order == ["a", "b", "c", "d"]
    else:
        assert features_by_order == ["c", "d", "a", "b"]
