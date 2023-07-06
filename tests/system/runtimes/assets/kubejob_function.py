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
def hello_world(context):
    context.logger.info("hello world")


class mycls:
    def __init__(self, context=None, a1=1):
        self.context = context
        self.a1 = a1

    def mtd(self, context, x=0, y=0):
        print(f"x={x}, y={y}, a1={self.a1}")
        context.log_result("rx", x)
        context.log_result("ry", y)
        context.log_result("ra1", self.a1)
