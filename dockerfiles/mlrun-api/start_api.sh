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

: '
         _\|/_
         (o o)
 +----oOO-{_}-OOo----------------------------------------------------------------------------------------------------+
 |                                                                                                                   |
 |  This script runs with tini to ensure capturing zombie processes and signal handling.                             |
 |  It is important to run the API with exec so that the bash process will be replaced with                          |
 |   the API process and for zombie processes reaping.                                                               |
 |                                                                                                                   |
 +-------------------------------------------------------------------------------------------------------------------+
'

# Lower case the MLRUN_MEMRAY env var
MLRUN_MEMRAY_LOWER=$(echo "$MLRUN_MEMRAY" | tr '[:upper:]' '[:lower:]')
# Ensure 1 leading space
MLRUN_MEMRAY_EXTRA_FLAGS=$(echo "${MLRUN_MEMRAY_EXTRA_FLAGS# }" | sed 's/^/ /')

# Check if the mlrun memray is set to a true value
if [[ -n "$MLRUN_MEMRAY_LOWER"  && ( "$MLRUN_MEMRAY_LOWER" == "1" || "$MLRUN_MEMRAY_LOWER" == "true" || "$MLRUN_MEMRAY_LOWER" == "yes" || "$MLRUN_MEMRAY_LOWER" == "on" )]]; then
    if [[ -n "$MLRUN_MEMRAY_OUTPUT_FILE" ]]; then
        echo "Starting API with memray profiling output file $MLRUN_MEMRAY_OUTPUT_FILE..."
        exec python -m memray run${MLRUN_MEMRAY_EXTRA_FLAGS% } --output "$MLRUN_MEMRAY_OUTPUT_FILE" --force -m server.api.main
    else
        echo "Starting API with memray profiling..."
        exec python -m memray run${MLRUN_MEMRAY_EXTRA_FLAGS% } -m server.api.main
    fi
else
    exec uvicorn server.api.main:app \
        --proxy-headers \
        --host 0.0.0.0 \
        --log-config server/api/uvicorn_log_config.yaml
fi
