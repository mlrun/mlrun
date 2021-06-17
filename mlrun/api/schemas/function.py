import enum


class FunctionState(str, enum.Enum):
    unknown = "unknown"
    ready = "ready"
    error = "error"  # represents deployment error

    deploying = "deploying"
    # there is currently an  usage of the builder (lower) pod state as the function state, ideally these two would map
    # to deploying but for backwards compatibility reasons we have to keep them
    running = "running"
    pending = "pending"
