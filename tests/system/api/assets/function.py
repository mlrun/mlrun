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
def secret_test_function(context, secrets: list = []):
    """Validate that given secrets exists

    :param context: the MLRun context
    :param secrets: name of the secrets that we want to look at
    """
    context.logger.info("running function")
    for sec_name in secrets:
        sec_value = context.get_secret(sec_name)
        context.logger.info("Secret: {} ==> {}".format(sec_name, sec_value))
        context.log_result(sec_name, sec_value)
    return True
