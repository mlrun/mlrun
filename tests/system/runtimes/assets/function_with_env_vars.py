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
import datetime
import os


def handler(context, env_vars_names=["ENV_VAR1", "ENV_VAR2"]):
    print("started", str(datetime.datetime.now()))
    for env_var_name in env_vars_names:
        context.log_result(env_var_name, os.environ.get(env_var_name))
    context.log_result("finished", str(datetime.datetime.now()))
    print("finished", str(datetime.datetime.now()))
