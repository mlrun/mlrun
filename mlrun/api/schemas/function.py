# Ideally we would want this to be class FunctionState(str, enum.Enum) which is the "FastAPI-compatible" way of creating
# schemas
# But, when we save a function to the DB, we pickle the body, which saves the state as an instance of this class (and
# not just a string), then if for some reason we downgrade to 0.6.4, before we had this class, we fail reading (pickle
# load) the function from the DB.
# Note that the problems are happening only if the state is assigned in the API side. When it's in the client side it
# anyways passes through JSON in the HTTP request body and come up as a string in the API side.
# For now I'm simply making the class a simple string consts class
# 2 other solutions I thought of:
# 1. Changing the places where we set the state in the UI to use set the actual enum value (FunctionState.x.value) - too
# fragile, tomorrow someone will set the state using the enum
# 2. Changing the function to be saved into a JSON field instead of pickled inside BLOB field - looks like the ideal we
# should go to, but too complicated and needed something fast and simple.
class FunctionState:
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
