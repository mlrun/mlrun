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

import asyncio
import json
import os
import time

import nuclio_sdk
import storey

from mlrun.runtimes import nuclio_init_hook
from mlrun.serving import V2ModelServer

# This benchmark depends on V3IO_API and V3IO_ACCESS_KEY env vars being set

access_key = os.getenv("V3IO_ACCESS_KEY")

serving_spec = {
    "function_uri": "test-app-flow/model-serving",
    "version": "v2",
    "parameters": {"stream_args": {"access_key": access_key}},
    "graph": {
        "kind": "router",
        "routes": {
            "classification": {
                "class_args": {
                    "model_path": "store://models/test-app-flow/classification:latest"
                },
                "kind": "task",
                "class_name": "ClassifierModel",
            }
        },
    },
    "load_mode": None,
    "functions": {},
    "graph_initializer": None,
    "error_stream": None,
    "track_models": True,
    "tracking_policy": {
        "stream_image": "mlrun/mlrun",
        "default_batch_image": "mlrun/mlrun",
        "default_batch_intervals": {
            "year": None,
            "month": None,
            "day": None,
            "week": None,
            "day_of_week": None,
            "hour": "*/1",
            "minute": 0,
            "second": None,
            "start_date": None,
            "end_date": None,
            "timezone": None,
            "jitter": None,
        },
    },
    "default_content_type": None,
}

os.environ["SERVING_SPEC_ENV"] = json.dumps(serving_spec)

preprocess_seconds = 0.2
prediction_seconds = 0.2
postprocess_seconds = 0.2
num_requests = 30


class ClassifierModel(V2ModelServer):
    def load(self):
        pass

    def preprocess(self, request: dict, operation) -> dict:
        time.sleep(preprocess_seconds)
        return request

    def predict(self, body: dict):
        time.sleep(prediction_seconds)
        return []

    def postprocess(self, request: dict):
        time.sleep(postprocess_seconds)
        return request


class Logger:
    def debug(self, message, *args, **kw_args):
        print(message)

    def info(self, message, *args, **kw_args):
        print(message)

    def warn(self, message, *args, **kw_args):
        print(message)

    def error(self, message, *args, **kw_args):
        print(message)

    def debug_with(self, message, *args, **kw_args):
        print(message)

    def info_with(self, message, *args, **kw_args):
        print(message)

    def warn_with(self, message, *args, **kw_args):
        print(message)

    def error_with(self, message, *args, **kw_args):
        print(message)


class Context:
    logger = Logger()
    Response = nuclio_sdk.Response
    worker_id = 0


context = Context()


async def main():
    nuclio_init_hook(context, globals(), "serving_v2")
    d = {"inputs": []}

    start = time.monotonic()
    for i in range(num_requests):
        res = context.mlrun_handler(context, storey.Event(d))
        print(f"{i}: {res}")
    end = time.monotonic()
    print(f"Runtime {end - start} seconds")


asyncio.run(main())
