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
#
import datetime


def handler(context, file_path="/tmp/args.txt"):
    print("started", str(datetime.datetime.now()))
    with open(file_path, "r") as file:
        lines = file.read().split("\n")
    lines = [line for line in lines if line]
    context.log_result("results", lines)
    print("finished", str(datetime.datetime.now()))
