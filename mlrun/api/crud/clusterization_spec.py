import mlrun
import mlrun.api.schemas
import mlrun.utils.singleton


class ClusterizationSpec(
    metaclass=mlrun.utils.singleton.Singleton,
):
    @staticmethod
    def get_clusterization_spec():
        is_chief = mlrun.mlconf.httpdb.clusterization.role == "chief"
        return mlrun.api.schemas.ClusterizationSpec(
            chief_api_state=mlrun.mlconf.httpdb.state if is_chief else None,
            chief_version=mlrun.mlconf.version if is_chief else None,
        )
