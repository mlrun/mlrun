from enum import Enum

import mergedeep

import mlrun.errors


class Format(str, Enum):
    full = "full"
    name_only = "name_only"


class PatchMode(str, Enum):
    replace = "replace"
    additive = "additive"

    def to_mergedeep_strategy(self) -> mergedeep.Strategy:
        if self.value == PatchMode.replace:
            return mergedeep.Strategy.REPLACE
        elif self.value == PatchMode.additive:
            return mergedeep.Strategy.ADDITIVE
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unknown patch mode: {self.value}"
            )


headers_prefix = "x-mlrun-"


class HeaderNames:
    patch_mode = f"{headers_prefix}patch-mode"
