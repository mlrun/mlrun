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

# for this function to run in nuclio you need to:
# - set the python base image e.g.:
#     python:3.6-jessie
# - add mlrun package install to the Build commands:
#     pip install mlrun


import time

from mlrun import get_or_create_ctx


def handler(context, event):
    ctx = get_or_create_ctx("myfunc", event=event)
    p1 = ctx.get_param("p1", 1)
    p2 = ctx.get_param("p2", "a-string")

    context.logger.info(
        f"Run: {ctx.name} uid={ctx.uid}:{ctx.iteration} Params: p1={p1}, p2={p2}"
    )

    time.sleep(1)

    # log scalar values (KFP metrics)
    ctx.log_result("accuracy", p1 * 2)
    ctx.log_result("latency", p1 * 3)

    # log various types of artifacts (and set UI viewers)
    ctx.log_artifact("test", body=b"abc is 123", local_path="test.txt")
    ctx.log_artifact("test_html", body=b"<b> Some HTML <b>", format="html")

    context.logger.info("run complete!")
    return ctx.to_json()
