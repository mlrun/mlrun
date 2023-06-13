import mlrun.common.types


class EventsModes(mlrun.common.types.StrEnum):
    enabled = "enabled"
    disabled = "disabled"


class EventClientKinds(mlrun.common.types.StrEnum):
    iguazio = "iguazio"
    nop = "nop"


class SecretEventActions(mlrun.common.types.StrEnum):
    created = "created"
    updated = "updated"
    deleted = "deleted"


class AuthSecretEventActions(mlrun.common.types.StrEnum):
    created = "created"
    updated = "updated"
