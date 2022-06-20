import mlrun
import mlrun.api.schemas


def run_only_on_chief():
    def decorator(function):
        def wrapper(*args, **kwargs):
            if (
                mlrun.mlconf.httpdb.clusterization.role
                == mlrun.api.schemas.ClusterizationRole.chief
            ):
                return function(*args, **kwargs)

            message = f"{function.__name__} is supposed to run only on chief, re-route."
            raise mlrun.errors.MLRunIncompatibleVersionError(message)

        return wrapper

    return decorator
