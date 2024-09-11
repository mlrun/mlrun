
import typing
import mlrun.common.types
from .base import ObjectFormat


class FeatureSetFormat(ObjectFormat, mlrun.common.types.StrEnum):
    minimal = "minimal"

    @staticmethod
    def format_method(_format: str) -> typing.Optional[typing.Callable]:
        return {
            FeatureSetFormat.full: None,
            FeatureSetFormat.minimal:
                None
            #     ArtifactFormat.filter_obj_method(
            #     [
            #         "kind",
            #         "metadata",
            #         "status",
            #         "project",
            #         "spec.producer",
            #         "spec.db_key",
            #         "spec.size",
            #         "spec.framework",
            #         "spec.algorithm",
            #         "spec.metrics",
            #         "spec.target_path",
            #     ]
            # ),
        }[_format]
