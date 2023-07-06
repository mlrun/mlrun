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
# serving runtime hooks, used in empty serving functions
from mlrun.runtimes import nuclio_init_hook


def init_context(context):
    nuclio_init_hook(context, globals(), "serving_v2")


def handler(context, event):
    return context.mlrun_handler(context, event)
