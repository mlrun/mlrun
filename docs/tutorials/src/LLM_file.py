from mlrun.serving import ModelSelector, Model, LLModel
from typing import Union

class MyModelSelector(ModelSelector):
    def select(
        self, event, available_models: list[Model]
    ) -> Union[list[str], list[Model]]:
        return [event.body.get("model_name", "")]
        