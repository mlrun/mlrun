# Copyright 2019 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import Dict

import numpy as np
import pytest

import mlrun

CLASS_NAME = "mlrun.frameworks.huggingface.HuggingFaceModelServer"

# The PIPELINES list below contains all the NLP tasks that exists in the huggingface framework.
# Each example is a dictionary that include:
# - task        = the name of task
# - example     = input for prediction
# - result_keys = the keys that need to appear in the prediction.

PIPELINES = [
    {
        "task": "sentiment-analysis",
        "example": "We are very happy to show you the 🤗 Transformers library.",
        "result_keys": ["label", "score"],
    },
    {
        "task": "text-generation",
        "example": {
            "text_inputs": "Hello, I'm a language model",
            "max_length": 20,
            "num_return_sequences": 1,
        },
        "result_keys": ["generated_text"],
    },
    {
        "task": "ner",
        "example": "My name is Wolfgang",
        "result_keys": ["entity", "score", "index", "word", "start", "end"],
    },
    {
        "task": "question-answering",
        "example": {
            "question": "Where do I live?",
            "context": "My name is Merve and I live in İstanbul.",
        },
        "result_keys": ["score", "start", "end", "answer"],
    },
    {
        "task": "fill-mask",
        "example": "Paris is the <mask> of France.",
        "result_keys": ["score", "token", "token_str", "sequence"],
    },
    {
        "task": "summarization",
        "example": "Paris is the capital and most populous city of France,"
        " with an estimated population of 2,175,601 residents as of 2018,"
        " in an area of more than 105 square kilometres (41 square miles)."
        " The City of Paris is the centre and seat of government of the region"
        " and province of Île-de-France, or Paris Region, which has an estimated population of 12,174,880,"
        " or about 18 percent of the population of France as of 2017.",
        "result_keys": ["summary_text"],
    },
    {
        "task": "translation_en_to_fr",
        "example": "How old are you?",
        "result_keys": ["translation_text"],
    },
]


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_default_models(pipeline: Dict):
    """
    Test the HuggingFaceModelServer over all the NLP tasks in pipelines.
    :param pipeline: A Dict that contains the name of the task,
    an input for prediction and keys in the result prediction
    """
    serving_function = mlrun.new_function(
        name="serving",
        image="mlrun/ml-models",
        kind="serving",
        # requirements=['transformers'],
    )
    serving_function.add_model(
        pipeline["task"],
        class_name=CLASS_NAME,
        model_path="123",  # This is not used, just for enabling the process.
        task=pipeline["task"],
    )
    server = serving_function.to_mock_server()
    result = server.test(
        f'/v2/models/{pipeline["task"]}', body={"inputs": [pipeline["example"]]}
    )
    prediction = result["outputs"][0]
    assert all(
        result_key in prediction.keys() for result_key in pipeline["result_keys"]
    )


def test_local_model_serving():
    """
    Testing the model server with a specific model and tokenizer.
    """
    # Creating the serving function:
    serving_function = mlrun.new_function(
        name="serving",
        image="mlrun/ml-models",
        kind="serving",
        # requirements=['transformers'],
    )

    # Adding model:
    serving_function.add_model(
        "model1",
        class_name=CLASS_NAME,
        model_path="123",  # This is not used, just for enabling the process.
        task="sentiment-analysis",
        model_class="TFAutoModelForSequenceClassification",
        model_name="nlptown/bert-base-multilingual-uncased-sentiment",
        tokenizer_class="AutoTokenizer",
        tokenizer_name="nlptown/bert-base-multilingual-uncased-sentiment",
    )

    # Creating a mock server and predicting:
    server = serving_function.to_mock_server()
    result = server.test(
        "/v2/models/model1",
        body={
            "inputs": [
                "Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers."
            ]
        },
    )

    # Checking the prediction has the required value:
    prediction = result["outputs"][0]
    assert prediction["label"] == "5 stars" and np.isclose(prediction["score"], 0.72727)
