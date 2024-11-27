# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import mergedeep

import mlrun.common.types
import mlrun.errors


class PatchMode(mlrun.common.types.StrEnum):
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


class DeletionStrategy(mlrun.common.types.StrEnum):
    restrict = "restrict"
    restricted = "restricted"
    cascade = "cascade"
    cascading = "cascading"
    check = "check"

    @staticmethod
    def default():
        return DeletionStrategy.restricted

    def is_restricted(self):
        if self.value in [DeletionStrategy.restrict, DeletionStrategy.restricted]:
            return True
        return False

    def is_cascading(self):
        if self.value in [DeletionStrategy.cascade, DeletionStrategy.cascading]:
            return True
        return False

    def strategy_to_check(self):
        if self.value in [
            DeletionStrategy.check,
            DeletionStrategy.restrict,
            DeletionStrategy.restricted,
        ]:
            return True
        return False

    def to_nuclio_deletion_strategy(self) -> str:
        if self.is_restricted():
            return "restricted"
        elif self.is_cascading():
            return "cascading"
        elif self.value == DeletionStrategy.check.value:
            return "check"
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unknown deletion strategy: {self.value}"
            )

    def to_iguazio_deletion_strategy(self) -> str:
        if self.is_restricted():
            return "restricted"
        elif self.is_cascading():
            return "cascading"
        elif self.value == DeletionStrategy.check.value:
            raise NotImplementedError(
                "Iguazio does not support the check deletion strategy"
            )
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unknown deletion strategy: {self.value}"
            )


headers_prefix = "x-mlrun-"


class HeaderNames:
    projects_role = "x-projects-role"
    patch_mode = f"{headers_prefix}patch-mode"
    deletion_strategy = f"{headers_prefix}deletion-strategy"
    secret_store_token = f"{headers_prefix}secret-store-token"
    pipeline_arguments = f"{headers_prefix}pipeline-arguments"
    client_version = f"{headers_prefix}client-version"
    python_version = f"{headers_prefix}client-python-version"
    backend_version = f"{headers_prefix}be-version"
    ui_version = f"{headers_prefix}ui-version"
    ui_clear_cache = f"{headers_prefix}ui-clear-cache"


class FeatureStorePartitionByField(mlrun.common.types.StrEnum):
    name = "name"  # Supported for feature-store objects

    def to_partition_by_db_field(self, db_cls):
        if self.value == FeatureStorePartitionByField.name:
            return db_cls.name
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unknown group by field: {self.value}"
            )


class RunPartitionByField(mlrun.common.types.StrEnum):
    name = "name"  # Supported for runs objects
    project_and_name = "project_and_name"  # Supported for runs objects

    def to_partition_by_db_field(self, db_cls):
        if self.value == RunPartitionByField.name:
            return db_cls.name
        elif self.value == RunPartitionByField.project_and_name:
            return db_cls.project, db_cls.name
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unknown group by field: {self.value}"
            )


class ArtifactPartitionByField(mlrun.common.types.StrEnum):
    name = "name"  # Supported for artifacts objects
    project_and_name = "project_and_name"  # Supported for artifacts objects

    def to_partition_by_db_field(self, db_cls):
        if self.value == ArtifactPartitionByField.name:
            return db_cls.key
        elif self.value == ArtifactPartitionByField.project_and_name:
            return db_cls.project, db_cls.key
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unknown group by field: {self.value}"
            )


class SortField(mlrun.common.types.StrEnum):
    created = "created"
    updated = "updated"

    def to_db_field(self, db_cls):
        if self.value == SortField.created:
            # not doing type check to prevent import that will cause a cycle
            if db_cls.__name__ == "Run":
                return db_cls.start_time
            return db_cls.created
        elif self.value == SortField.updated:
            return db_cls.updated
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unknown sort by field: {self.value}"
            )


class OrderType(mlrun.common.types.StrEnum):
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


class APIStates:
    online = "online"
    waiting_for_migrations = "waiting_for_migrations"
    migrations_in_progress = "migrations_in_progress"
    migrations_failed = "migrations_failed"
    migrations_completed = "migrations_completed"
    offline = "offline"
    waiting_for_chief = "waiting_for_chief"

    @staticmethod
    def terminal_states():
        return [APIStates.online, APIStates.offline]

    @staticmethod
    def description(state: str):
        return {
            APIStates.online: "API is online",
            APIStates.waiting_for_migrations: "API is waiting for migrations to be triggered. "
            "Send POST request to /api/operations/migrations to trigger it",
            APIStates.migrations_in_progress: "Migrations are in progress",
            APIStates.migrations_failed: "Migrations failed, API can't be started",
            APIStates.migrations_completed: "Migrations completed, API is waiting to become online",
            APIStates.offline: "API is offline",
            APIStates.waiting_for_chief: "API is waiting for chief to be ready",
        }.get(state, f"Unknown API state '{state}'")


class ClusterizationRole:
    chief = "chief"
    worker = "worker"


class LogsCollectorMode:
    legacy = "legacy"
    sidecar = "sidecar"
    best_effort = "best-effort"
