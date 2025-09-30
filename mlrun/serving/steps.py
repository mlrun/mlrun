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

import storey

import mlrun


class ChoiceByField(storey.Choice):
    """
    Choosing downstream outlets using custom event field.
    :param field_name: event field name to derive outlets.
    """

    def __init__(self, field_name):
        self.field_name = field_name
        super().__init__()

    def select_outlets(self, event):
        if self.field_name not in event.keys():
            raise mlrun.MLRunInvalidArgumentError(
                f"Field name {self.field_name} is not contained in the event keys {list(event.keys())}."
            )
        outlets = (
            [event[self.field_name]]
            if isinstance(event[self.field_name], str)
            else event[self.field_name]
        )
        if not outlets:
            raise mlrun.MLRunNotFoundError(
                f"Steps not found for given field name {self.field_name}."
            )
        return outlets
