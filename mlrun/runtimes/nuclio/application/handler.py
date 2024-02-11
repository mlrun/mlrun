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

##############################################
#  Written in python for testing purposes    #
##############################################
import os

import nuclio_sdk
import requests


def handler(context: nuclio_sdk.Context, event: nuclio_sdk.Event):
    context.logger.info("This is an unstructured log")
    # get the encryption key
    app_port = os.environ.get("SERVING_PORT", "8080")
    response = requests.get(f"http://localhost:{app_port}")

    return context.Response(
        body=response.text,
        headers={},
        content_type="text/plain",
        status_code=response.status_code,
    )
