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
#
import json
import pathlib
import typing

import requests

import mlrun.projects

assets_path = pathlib.Path(__file__).parent.parent / "assets"


def deploy_notification_nuclio(
    project: mlrun.projects.MlrunProject, image: str = None
) -> str:
    nuclio_function = project.set_function(
        name="notification-nuclio-function",
        func=str(assets_path / "notification_nuclio_function.py"),
        image="mlrun/mlrun" if image is None else image,
        kind="nuclio",
    )
    nuclio_function.deploy()
    return nuclio_function.spec.command


def get_notifications_from_nuclio_and_reset_notification_cache(
    nuclio_function_url: str,
) -> typing.Generator[dict, None, None]:
    response = requests.post(nuclio_function_url, json={"operation": "list"})
    response_data = json.loads(response.text)

    # Extract notification data from the response
    notifications = response_data["data_list"]

    yield from notifications

    requests.post(nuclio_function_url, json={"operation": "reset"})
