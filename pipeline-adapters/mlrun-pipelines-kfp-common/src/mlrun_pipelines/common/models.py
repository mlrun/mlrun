# Copyright 2024 Iguazio
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
import enum


class StrEnum(str, enum.Enum):
    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class RunStatuses(StrEnum):
    """
    Class for different types of statuses a 'PipelineRun' can have using an enum type.
    Beyond enumerating all possible statuses, this class ensures comparisons are case-insensitive.

    Statuses commonly used by MLRun and associated with KFP 1.8:
        - succeeded: Indicates that the run has successfully completed.
        - failed: Indicates that the run has failed to complete.
        - skipped: Indicates that the run was skipped.
        - error: Indicates that an error occurred during the run.
        - running: Indicates that the run is currently ongoing.

    The statuses specific to KFP 2.0:
        - runtime_state_unspecified: Indicates that the run's status is not specified; similar to "" (Unknown) on Argo.
        - pending: Indicates that the run is pending, waiting to be executed.
        - canceling: Indicates that a cancel request has been made for the run.
        - canceled: Indicates that the run has been canceled.
        - paused: Indicates that the run is currently paused or on hold.

    This class also includes methods for computing all statuses, stable statuses
    (ones that will no longer change), and transient statuses (ones that may still change).
    """

    # States available on KFP 1.8 and traditionally used by MLRun
    succeeded = "Succeeded"
    failed = "Failed"
    skipped = "Skipped"
    error = "Error"  # available only on KFP 1.8 or lower
    running = "Running"

    # States available only on KFP 2.0
    runtime_state_unspecified = "Runtime_State_Unspecified"
    pending = "Pending"
    canceling = "Canceling"
    canceled = "Canceled"
    paused = "Paused"

    def __eq__(self, other):
        return self.value.casefold() == str(other).casefold()

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return self.value

    @classmethod
    def _missing_(cls, value: str):
        value = value.casefold()
        for member in cls:
            if member.value.casefold() == value:
                return member
        return None

    @staticmethod
    def all():
        return [
            RunStatuses.succeeded,
            RunStatuses.failed,
            RunStatuses.skipped,
            RunStatuses.error,
            RunStatuses.running,
            RunStatuses.runtime_state_unspecified,
            RunStatuses.pending,
            RunStatuses.canceling,
            RunStatuses.canceled,
            RunStatuses.paused,
        ]

    @staticmethod
    def stable_statuses():
        return [
            RunStatuses.succeeded,
            RunStatuses.failed,
            RunStatuses.skipped,
            RunStatuses.error,
            RunStatuses.canceled,
        ]

    @staticmethod
    def transient_statuses():
        return [
            status
            for status in RunStatuses.all()
            if status not in RunStatuses.stable_statuses()
        ]
