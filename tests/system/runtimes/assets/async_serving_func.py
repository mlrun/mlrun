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

import asyncio

import storey


class PreprocessStep(storey.MapClass):
    async def do(self, event):
        return {"preprocessed_item": event}


async def func_handler(context, event):
    handler = context.mlru
    await asyncio.sleep(5)
    return {"message": "Hello from async function!", "input_event": event}
