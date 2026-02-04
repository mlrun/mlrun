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

from transformers import pipeline

from mlrun.agentic.chains.base import ChainRunner


class SentimentAnalysisStep(ChainRunner):
    DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

    def __init__(
        self,
        tokenizer: str = None,
        model: str = None,
        pipeline_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or self.DEFAULT_MODEL
        self.model = model or self.DEFAULT_MODEL
        self.sentiment_classifier = pipeline(
            "sentiment-analysis",
            tokenizer=self.tokenizer,
            model=self.model,
            **(pipeline_kwargs or {}),
        )

    def _run(self, event):
        query = event.query
        sentiment = self.sentiment_classifier(query)
        return {
            "answer": sentiment[0]["label"],
            "sources": "",
        }
