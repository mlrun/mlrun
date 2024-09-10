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

from mlrun.genai.chains.base import ChainRunner


class SentimentAnalysisStep(ChainRunner):
    """
    Processes sentiment analysis on a given text.
    """

    # Default model to use as model and tokenizer if not given
    DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

    def __init__(
        self,
        tokenizer: str = None,
        model: str = None,
        pipeline_kwargs: dict = None,
        **kwargs,
    ):
        """
        Initialize the sentiment analysis step.

        :param model:           The name of the model to use, if not given, the default model will be used, has to be
                                from the roberta model family.
        :param tokenizer:       The name of the tokenizer to use, if not given, the default tokenizer will be used,
                                has to be compatible with the model.
        :param pipeline_kwargs: Additional keyword arguments to pass to the HuggingFace pipeline.
        """
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or self.DEFAULT_MODEL
        self.model = model or self.DEFAULT_MODEL
        # Load the HuggingFace sentiment analysis pipeline
        self.sentiment_classifier = pipeline(
            "sentiment-analysis",
            tokenizer=self.tokenizer,
            model=self.model,
            **pipeline_kwargs,
        )

    def _run(self, event):
        """
        Run the sentiment analysis step.

        :param event: The event to process.

        :return: The processed event with the sentiment analysis result.
        """
        transcription = event.query
        sentiment = self.sentiment_classifier(
            transcription
        )  # Is a list of dictionaries (in tested examples)
        return {
            "answer": sentiment[0]["label"],
            "sources": "",
        }  # TODO: Can only return string
