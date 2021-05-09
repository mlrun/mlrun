from enum import Enum

import mergedeep

import mlrun.errors


class Format(str, Enum):
    full = "full"
    name_only = "name_only"
    metadata_only = "metadata_only"
    summary = "summary"


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


class FeatureStorePartitionByField(str, Enum):
    name = "name"  # Supported for feature-store objects

    def to_partition_by_db_field(self, db_cls):
        if self.value == FeatureStorePartitionByField.name:
            return db_cls.name
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unknown group by field: {self.value}"
            )


# For now, we only support sorting by updated field
class SortField(str, Enum):
    updated = "updated"


class OrderType(str, Enum):
    asc = "asc"
    desc = "desc"

    def to_order_by_predicate(self, db_field):
        if self.value == OrderType.asc:
            return db_field.asc()
        else:
            return db_field.desc()


labels_prefix = "mlrun/"


class LabelNames:
    schedule_name = f"{labels_prefix}schedule-name"
