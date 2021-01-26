from enum import Enum

import mergedeep

import mlrun.errors


class Format(str, Enum):
    full = "full"
    name_only = "name_only"
    metadata_only = "metadata_only"


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


class DeletionStrategy(str, Enum):
    restrict = "restrict"
    cascade = "cascade"

    @staticmethod
    def default():
        return DeletionStrategy.restrict

    def to_nuclio_deletion_strategy(self) -> str:
        if self.value == DeletionStrategy.restrict:
            return "restricted"
        elif self.value == DeletionStrategy.cascade:
            return "cascading"
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unknown deletion strategy: {self.value}"
            )


headers_prefix = "x-mlrun-"


class HeaderNames:
    patch_mode = f"{headers_prefix}patch-mode"
    deletion_strategy = f"{headers_prefix}deletion-strategy"
    secret_store_token = f"{headers_prefix}secret-store-token"
