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
from copy import copy
from typing import Union

from mlrun.serving import Model, ModelSelector


class DummyModel(Model):
    def predict(self, body, **kwargs):
        body["extra"] = 123
        return body


class MyModelSelector(ModelSelector):
    def __init__(self, models: Union[list[str], list[Model]]):
        super().__init__()
        self.models = copy(models)

    def select(
        self, event, available_models: list[Model]
    ) -> Union[list[str], list[Model]]:
        current_models = event.body.pop("models", [])
        if current_models and set(current_models).issubset(set(self.models)):
            return current_models
        return []
