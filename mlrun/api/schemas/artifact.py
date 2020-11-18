import enum
import typing


class ArtifactCategories(str, enum.Enum):
    model = "model"
    dataset = "dataset"
    other = "other"

    def to_kinds_filter(self) -> typing.Tuple[typing.List[str], bool]:
        # FIXME: these artifact definitions (or at least the kinds enum) should sit in a dedicated module
        # import here to prevent import cycle
        import mlrun.artifacts.dataset
        import mlrun.artifacts.model

        if self.value == ArtifactCategories.model.value:
            return [mlrun.artifacts.model.ModelArtifact.kind], False
        if self.value == ArtifactCategories.dataset.value:
            return [mlrun.artifacts.dataset.DatasetArtifact.kind], False
        if self.value == ArtifactCategories.other.value:
            return (
                [
                    mlrun.artifacts.model.ModelArtifact.kind,
                    mlrun.artifacts.dataset.DatasetArtifact.kind,
                ],
                True,
            )
