import mlrun.api.utils.events.iguazio
import mlrun.utils.singleton


class EventsFactory(object):
    @staticmethod
    def get_events_client(kind: str = None, **kwargs):
        if not kind:
            if mlrun.mlconf.get_parsed_igz_version():
                kind = "iguazio"

        if kind == "iguazio":
            if not mlrun.mlconf.get_parsed_igz_version():
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Iguazio events client can only be used in Iguazio environment"
                )
            return mlrun.api.utils.events.iguazio.Client(**kwargs)

        raise mlrun.errors.MLRunInvalidArgumentError(f"Unsupported events kind: {kind}")
