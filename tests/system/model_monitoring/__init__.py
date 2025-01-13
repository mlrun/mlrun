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

import os

from mlrun.datastore.datastore_profile import (
    DatastoreProfile,
    DatastoreProfileV3io,
    TDEngineDatastoreProfile,
)


# TODO: Remove after ML-9020
def get_tsdb_datastore_profile_from_env() -> DatastoreProfile:
    tsdb_connection = os.getenv("MLRUN_MODEL_ENDPOINT_MONITORING__TSDB_CONNECTION")
    if tsdb_connection == "v3io":
        return DatastoreProfileV3io(name="v3io-tsdb-profile")
    elif tsdb_connection.startswith("taosws://"):
        return TDEngineDatastoreProfile.from_dsn(
            tsdb_connection, profile_name="tdengine-profile"
        )
    else:
        raise ValueError(f"Unsupported {tsdb_connection=}")
