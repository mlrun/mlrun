import mlrun.common.types


class EventsMode(mlrun.common.types.StrEnum):
    enabled = "enabled"
    disabled = "disabled"


class EventClientsTypes(mlrun.common.types.StrEnum):
    iguazio = "iguazio"
    nop = "nop"
