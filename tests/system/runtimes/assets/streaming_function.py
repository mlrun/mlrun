# Copyright 2026 Iguazio
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

from mlrun.serving import Model


class StreamingChoice(storey.Choice):
    """Routes events based on URL path."""

    def select_outlets(self, event):
        # event.path is the URL path, e.g., "/step" or "/model"
        path = getattr(event, "path", "/step")
        route = path.lstrip("/") or "step"
        return [route]


class StreamingModel(Model):
    """A model that yields streaming chunks from predict()."""

    def __init__(self, num_chunks=3, **kwargs):
        super().__init__(**kwargs)
        self.num_chunks = num_chunks

    def predict(self, body, **kwargs):
        if isinstance(body, bytes):
            body = body.decode("utf-8")

        for i in range(self.num_chunks):
            yield f"{body}_chunk_{i}"

    async def predict_async(self, body, **kwargs):
        if isinstance(body, bytes):
            body = body.decode("utf-8")

        for i in range(self.num_chunks):
            yield f"{body}_chunk_{i}"


class Echo:
    def do(self, x):
        return x


class StreamingStep:
    """A step that yields streaming chunks."""

    def __init__(self, context=None, name=None, num_chunks=3):
        self.context = context
        self.name = name
        self.num_chunks = num_chunks

    async def do(self, x):
        """Yield multiple chunks for a single input."""
        if isinstance(x, bytes):
            x = x.decode("utf-8")

        for i in range(self.num_chunks):
            await asyncio.sleep(1)
            yield f"{x}_chunk_{i}"


class ErrorStreamingStep:
    """A step that yields one chunk then raises mid-stream."""

    def __init__(self, context=None, name=None):
        self.context = context
        self.name = name

    async def do(self, x):
        if isinstance(x, bytes):
            x = x.decode("utf-8")
        yield f"{x}_chunk_0"
        await asyncio.sleep(0.5)
        raise ValueError("Simulated mid-stream error")
