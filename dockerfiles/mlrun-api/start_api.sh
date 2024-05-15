# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Check if the MLRUN_MEMRAY environment variable is set and not 0
if [ -n "$MLRUN_MEMRAY" ] && [ "$MLRUN_MEMRAY" != 0 ]; then
    echo "Installing memray..."
    python -m pip install memray==1.12.0
    if [ -n "$MLRUN_MEMRAY_LIVE_PORT" ]; then
        echo "Starting API with live memray profiling on port $MLRUN_MEMRAY_LIVE_PORT..."
        python -m memray run --live-remote --live-port $MLRUN_MEMRAY_LIVE_PORT -m server.api db
    elif [ -n "$MLRUN_MEMRAY_OUTPUT_FILE" ]; then
        echo "Starting API with memray profiling output file $MLRUN_MEMRAY_OUTPUT_FILE..."
        python -m memray run --output $MLRUN_MEMRAY_OUTPUT_FILE --force -m server.api db
    else
        echo "Starting API with memray profiling..."
        python -m memray run -m server.api db
    fi
else
    python -m server.api db
fi
