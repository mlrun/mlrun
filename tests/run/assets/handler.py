# Copyright 2018 Iguazio
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
import os

import mlrun


def myhandler(context: mlrun.MLClientCtx, p1, p2=3, p3=4):
    print(f"Run: {context.name} (uid={context.uid})")
    context.logger.info(f"iter={context.iteration} p2={p2}")
    context.log_result("accuracy", p2 * 2)
    context.log_result("loss", p3 * 3)
    context.log_artifact("file_result", body=b"abc123", local_path="result.txt")


def env_file_test(context: mlrun.MLClientCtx):
    context.log_result("ENV_ARG1", os.environ.get("ENV_ARG1"))
    context.log_result("kfp_ttl", mlrun.mlconf.kfp_ttl)


class mycls:
    def __init__(self, context=None, a1=1):
        self.context = context
        self.a1 = a1

    def mtd(self, context, x=0, y=0):
        print(f"x={x}, y={y}, a1={self.a1}")
        context.log_result("rx", x)
        context.log_result("ry", y)
        context.log_result("ra1", self.a1)
