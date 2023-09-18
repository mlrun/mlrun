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

import mlrun
from mlrun.model_monitoring.batch_application import BatchApplicationProcessor


def handler(context: mlrun.run.MLClientCtx):
    """
    RunS model monitoring batch application

    :param context: the MLRun context
    """
    batch_processor = BatchApplicationProcessor(
        context=context,
        project=context.project,
    )
    batch_processor.run()
    if batch_processor.endpoints_exceptions:
        print(batch_processor.endpoints_exceptions)
