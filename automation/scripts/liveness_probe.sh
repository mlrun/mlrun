#!/bin/bash

MLRUN_API_PROCESS_LINE="/usr/local/bin/python -m mlrun.api.main"
MLRUN_API_PROCESS_LINE_AMOUNT=2  # including the grep command

ps -fe | test $MLRUN_API_PROCESS_LINE_AMOUNT -eq $(grep -c "$MLRUN_API_PROCESS_LINE")
exit $?