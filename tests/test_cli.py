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
import pathlib

import mlrun.projects
from mlrun.__main__ import add_notification_to_project, line2keylist
from mlrun.utils import list2dict


def test_add_notification_to_cli_from_file():
    input_file_path = str(pathlib.Path(__file__).parent / "notification")
    notification = (f"file={input_file_path}",)
    notifications = line2keylist(notification, keyname="type", valname="params")
    project = mlrun.projects.MlrunProject(name="test")
    for notification in notifications:
        if notification["type"] == "file":
            with open(notification["params"]) as fp:
                lines = fp.read().splitlines()
                notification = list2dict(lines)
                add_notification_to_project(notification, project)

        else:
            add_notification_to_project(
                {notification["type"]: notification["params"]}, project
            )

    assert project._notifiers._notifications["slack"].params.get("webhook") == "123234"
    assert project._notifiers._notifications["ipython"].params.get("webhook") == "1232"


def test_add_notification_to_cli_from_dict():
    notification = ('slack={"webhook":"123234"}', 'ipython={"webhook":"1232"}')
    notifications = line2keylist(notification, keyname="type", valname="params")
    print(notifications)
    project = mlrun.projects.MlrunProject(name="test")
    for notification in notifications:
        if notification["type"] == "file":
            with open(notification["params"]) as fp:
                lines = fp.read().splitlines()
                notification = list2dict(lines)
                add_notification_to_project(notification, project)

        else:
            add_notification_to_project(
                {notification["type"]: notification["params"]}, project
            )

    assert project._notifiers._notifications["slack"].params.get("webhook") == "123234"
    assert project._notifiers._notifications["ipython"].params.get("webhook") == "1232"
