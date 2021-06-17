import enum


class FunctionState(str, enum.Enum):
    unknown = "unknown"
    ready = "ready"
    error = "error"  # represents deployment error

    deploying = "deploying"
    # there is currently an abuse usage of the builder (lower) pod state as the function state, ideally these two would
    # map to deploying but for backwards compatibility reasons we have to keep them
    running = "running"
    pending = "pending"
    # same goes for the build which is not coming from the pod, but is used and we can't just omit it for BC reasons
    build = "build"
