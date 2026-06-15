from typing import Union

from mlrun.serving import Model, ModelSelector


class MyModelSelector(ModelSelector):
    def select(
        self, event, available_models: list[Model]
    ) -> Union[list[str], list[Model]]:
        return [event.body.get("model_name", "")]
