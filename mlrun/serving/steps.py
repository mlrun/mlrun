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
        outlets = [event[self.field_name]] if isinstance(event[self.field_name],
                                                         str) else event[self.field_name]
        if not outlets:
            raise mlrun.MLRunNotFoundError(
                f"Steps not found for given field name {self.field_name}."
            )
        return outlets
