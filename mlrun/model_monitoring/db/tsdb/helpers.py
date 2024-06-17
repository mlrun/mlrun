# Copyright 2024 Iguazio
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
import pandas as pd

import mlrun.common.schemas.model_monitoring as mm_schemas
from mlrun.utils import logger


def _get_result_kind(result_df: pd.DataFrame) -> mm_schemas.ResultKindApp:
    kind_series = result_df[mm_schemas.ResultData.RESULT_KIND]
    unique_kinds = kind_series.unique()
    if len(unique_kinds) > 1:
        logger.warning(
            "The result has more than one kind",
            kinds=list(unique_kinds),
            application_name=result_df[mm_schemas.WriterEvent.APPLICATION_NAME],
            result_name=result_df[mm_schemas.ResultData.RESULT_NAME],
        )
    return unique_kinds[0]
