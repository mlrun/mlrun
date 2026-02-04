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

from openai import OpenAI

from mlrun.agentic.chains.base import ChainRunner


class LanguageGuardrail(ChainRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI()

    def _run(self, event):
        answer = event.results.get("answer", "")
        if not answer:
            return {}

        response = self.client.moderations.create(
            model="omni-moderation-latest",
            input=answer,
        )

        result = response.results[0]

        if result.flagged:
            return {
                "stop": True,
                "error_message": "The response contains unsafe language.",
            }

        return {"language guardrail": "valid"}
