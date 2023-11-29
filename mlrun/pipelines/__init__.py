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

# TODO: Fetch currently installed KFP version to determine what package to enable
pipeline_compatibility_mode = "kfp-v1.8"

# fmt: off
if pipeline_compatibility_mode == "kfp-v1.8":
    import mlrun.pipelines.kfp.v1_8.patcher  # noqa

    # TODO: encapsulate import list in the __init__ of the target module
    from mlrun.pipelines.kfp.v1_8 import *  # noqa

    # TODO: we need to namespace the following imports
    from mlrun.pipelines.kfp.v1_8.helpers import *  # noqa
    from mlrun.pipelines.kfp.v1_8.ops import *  # noqa
# fmt: on
