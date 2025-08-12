# Copyright 2025 Iguazio
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
import pytest

import mlrun
from mlrun.datastore.model_provider.huggingface_provider import HuggingFaceProvider


@pytest.mark.parametrize(
    "response, expected_str_response",
    [
        (
            [{"generated_text": "The capital of Germany is Berlin."}],
            "The capital of Germany is Berlin.",
        ),
    ],
)
def test_response_to_str(response, expected_str_response):
    extracted_string = HuggingFaceProvider._extract_string_output(response=response)
    assert extracted_string == expected_str_response


def test_response_to_str_error():
    # This response can be reproduced with Hugging Face Provider by invoking with num_return_sequences=2:
    response = [
        {"generated_text": "The capital of Germany is Berlin."},
        {"generated_text": "The capital of Japan is Tokyo"},
    ]
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match="HuggingFaceProvider: extracting string from response is only"
        " supported for single-response outputs",
    ):
        HuggingFaceProvider._extract_string_output(response=response)
