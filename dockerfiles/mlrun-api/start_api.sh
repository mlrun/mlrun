#!/bin/bash

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
