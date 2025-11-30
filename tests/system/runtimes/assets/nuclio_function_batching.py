# Copyright 2025 Iguazio
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


def handler(context, batch: list):
    context.logger.info_with("Got batched event!")
    batched_response = []
    for item in batch:
        event_id = item.id
        batched_response.append(
            context.Response(
                body=f"Hello {event_id}",
                headers={},
                content_type="text/plain",
                status_code=200,
                event_id=event_id,
            )
        )
    return batched_response
