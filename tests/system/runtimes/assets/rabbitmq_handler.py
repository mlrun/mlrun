# Copyright 2026 Iguazio
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

from storey import MapClass


class ProcessMessage(MapClass):
    """Simple step that processes RabbitMQ messages and adds a marker."""

    def do(self, event):
        self.context.logger.info(f"Processing message: {event}")

        if isinstance(event, bytes):
            import json

            event = json.loads(event.decode("utf-8"))

        # Add processing marker
        event["_processed"] = True

        return event
