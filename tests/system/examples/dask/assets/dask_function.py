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
# function that will be distributed
def inc(x):
    return x + 2


# wrapper function, uses the dask client object
def main(context, x=1, y=2):
    context.logger.info(f"params: x={x},y={y}")
    print(f"params: x={x},y={y}")
    x = context.dask_client.submit(inc, x)
    print(x)
    print(x.result())
    context.log_result("y", x.result())
