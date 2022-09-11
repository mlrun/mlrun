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

from .console import ConsoleNotification
from .slack import SlackNotification
from .ipython import IPythonNotification
from .git import GitNotification


class NotificationTypes:
    types = {
        "slack": SlackNotification,
        "console": ConsoleNotification,
        "ipython": IPythonNotification,
        "git": GitNotification,
    }

    @classmethod
    def get(cls, notification_type):
        return cls.types.get(notification_type)

    @classmethod
    def all(cls):
        return cls.types.keys()
