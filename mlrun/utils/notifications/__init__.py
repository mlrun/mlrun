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

"""
Module for handling and sending notifications.
It is used by the SDK to send pipeline notifications, and by the cmdline to send publish git comments.
It is also used by the API to send run notifications via the run monitor.
For this reason, the module is in the general utils package, so it can be used by the SDK, CMD and the API.
"""

from .notification import *  # noqa
from .notification_pusher import *  # noqa
