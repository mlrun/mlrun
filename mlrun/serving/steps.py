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

from typing import Union

import storey

import mlrun.errors


class ChoiceByField(storey.Choice):
    """
    Selects downstream outlets to route each event based on a predetermined field.
    :param field_name: event field name that contains the step name or names of the desired outlet or outlets
    """

    def __init__(self, field_name: Union[str, list[str]], **kwargs):
        self.field_name = field_name
        super().__init__(**kwargs)

    def select_outlets(self, event):
        # Case 1: Missing field
        if self.field_name not in event:
            raise mlrun.errors.MLRunRuntimeError(
                f"Field '{self.field_name}' is not contained in the event keys {list(event.keys())}."
            )

        outlet = event[self.field_name]

        # Case 2: Field exists but is None
        if outlet is None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Field '{self.field_name}' exists but its value is None."
            )

        # Case 3: Invalid type
        if not isinstance(outlet, (str, list, tuple)):
            raise mlrun.errors.MLRunInvalidArgumentTypeError(
                f"Field '{self.field_name}' must be a string or list of strings "
                f"but is instead of type '{type(outlet).__name__}'."
            )

        outlets = [outlet] if isinstance(outlet, str) else outlet

        # Case 4: Empty list or tuple
        if not outlets:
            raise mlrun.errors.MLRunRuntimeError(
                f"The value of the key '{self.field_name}' cannot be an empty {type(outlets).__name__}."
            )

        return outlets
