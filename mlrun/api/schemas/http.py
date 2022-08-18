import enum


class HTTPSessionRetryMode(str, enum.Enum):
    enabled = "enabled"
    disabled = "disabled"
