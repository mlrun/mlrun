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

import os
import pathlib
import subprocess
from subprocess import PIPE, Popen

import pytest

import mlrun

relative_asset_path = "mlrun/frameworks/langchain/llm_model_server.py"
langchain_model_server_path = str(
    pathlib.Path(__file__).absolute().parent.parent.parent.parent / relative_asset_path
)


#: if true, delete the model after the test
_OLLAMA_DELETE_MODEL_POST_TEST = False

# Model names
_OLLAMA_MODEL = "qwen:0.5b"
_OPENAI_MODEL = "gpt-3.5-turbo-instruct"
_HUGGINGFACE_MODEL = "Qwen/Qwen2-0.5B-Instruct"

# General test configs
PROMPT = "How far is the moon"
QUESTION_LEN = len(PROMPT.split(" "))
MAX_TOKENS = 10
TEMPERATURE = 0.0000000001

# To run this test, you need to:
# 1. install ollama on your computer
# 2. pull the desired model from the ollama repository (for example, in terminal: ollama pull llama3)
# 3. run ollama with said model in terminal (for example, in terminal: ollama run llama3)


def ollama_check_skip():
    """
    Check if ollama is installed
    """
    try:
        result = subprocess.run(["ollama", "--help"], stdout=PIPE)
    except Exception as e:
        print(f"Error checking ollama: {e}")
        return True
    return result.returncode != 0


@pytest.fixture
def ollama_fixture():
    """
    Do the setup and cleanup for the ollama test
    """
    # pull or make sure the model is available
    subprocess.run(["ollama", "pull", _OLLAMA_MODEL], stdout=PIPE)

    # start the ollama server
    ollama_serve_process = Popen(["ollama", "serve"], stdout=PIPE)

    yield

    ollama_serve_process.kill()
    # delete the model after the test if requested
    global _OLLAMA_DELETE_MODEL_POST_TEST
    if _OLLAMA_DELETE_MODEL_POST_TEST:
        subprocess.run(["ollama", "rm", _OLLAMA_MODEL], stdout=PIPE)


@pytest.mark.skipif(ollama_check_skip(), reason="Ollama not installed")
def test_ollama(
    ollama_fixture,
    prompt=PROMPT,
    question_len=QUESTION_LEN,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
):
    """
    Test the langchain model server with an ollama model
    """

    project = mlrun.get_or_create_project(
        name="ollama-model-server-example", context="./"
    )
    serving_func = project.set_function(
        func=langchain_model_server_path,
        name="ollama-langchain-model-server",
        kind="serving",
        image="mlrun/mlrun",
    )
    serving_func.add_model(
        "ollama-langchain-model",
        llm="Ollama",
        class_name="LangChainModelServer",
        init_kwargs={"model": _OLLAMA_MODEL},
        model_path=".",
    )
    server = serving_func.to_mock_server()
    predict_result = server.test(
        "/v2/models/ollama-langchain-model/predict", {"inputs": [prompt]}
    )
    assert predict_result
    print("ollama predict successful predict_result", predict_result)
    invoke_result = server.test(
        "/v2/models/ollama-langchain-model/predict",
        {"inputs": [prompt], "usage": "invoke"},
    )
    assert invoke_result
    print("ollama invoke successful invoke_result", invoke_result)
    invoke_result_params = server.test(
        "/v2/models/ollama-langchain-model/predict",
        {
            "inputs": [prompt],
            "stop": ["<eos>"],
            "generation_kwargs": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
            "usage": "invoke",
        },
    )
    assert invoke_result_params
    assert len(invoke_result_params["outputs"].split(" ")) <= max_tokens + question_len
    print("ollama invoke with params successful result", invoke_result_params)
    batch_result = server.test(
        "/v2/models/ollama-langchain-model/predict",
        {"inputs": [prompt, prompt], "usage": "batch"},
    )
    assert batch_result
    print("ollama batch successful batch_result", batch_result)
    batch_result_params = server.test(
        "/v2/models/ollama-langchain-model/predict",
        {
            "inputs": [prompt, prompt],
            "generation_kwargs": {"num_predict": max_tokens},
            "usage": "batch",
        },
    )
    assert batch_result_params
    for result in batch_result_params["outputs"]:
        assert len(result.split(" ")) <= max_tokens + question_len
    print("ollama batch with params successful batch_result", batch_result_params)


def skip_openai():
    """
    Check if the OpenAI API credentials are set
    """
    return not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_BASE_URL")


@pytest.mark.skipif(skip_openai(), reason="OpenAI API credentials not set")
def test_openai(prompt=PROMPT, question_len=QUESTION_LEN, max_tokens=MAX_TOKENS):
    """
    Test the langchain model server with an openai model
    """

    project = mlrun.get_or_create_project(
        name="openai-model-server-example", context="./"
    )
    serving_func = project.set_function(
        func=langchain_model_server_path,
        name="openai-langchain-model-server",
        kind="serving",
        image="mlrun/mlrun",
    )
    serving_func.add_model(
        "openai-langchain-model",
        llm="OpenAI",
        class_name="LangChainModelServer",
        init_kwargs={"model": _OPENAI_MODEL},
        model_path=".",
    )
    server = serving_func.to_mock_server()
    predict_result = server.test(
        "/v2/models/openai-langchain-model/predict",
        {"inputs": [prompt], "usage": "invoke"},
    )
    assert predict_result
    print("openai predict successful predict_result", predict_result)
    invoke_result = server.test(
        "/v2/models/openai-langchain-model/predict",
        {"inputs": [prompt], "usage": "invoke"},
    )
    assert invoke_result
    print("openai invoke successful invoke_result", invoke_result)
    invoke_result_params = server.test(
        "/v2/models/openai-langchain-model/predict",
        {
            "inputs": [prompt],
            "generation_kwargs": {"max_tokens": max_tokens},
            "usage": "invoke",
        },
    )
    assert (
        invoke_result_params
        and len(invoke_result_params["outputs"].split(" ")) <= max_tokens + question_len
    )
    batch_result = server.test(
        "/v2/models/openai-langchain-model/predict",
        {"inputs": [prompt, prompt], "usage": "batch"},
    )
    assert batch_result
    print("openai batch successful batch_result", batch_result)
    batch_result_params = server.test(
        "/v2/models/openai-langchain-model/predict",
        {
            "inputs": [prompt, prompt],
            "generation_kwargs": {"max_tokens": max_tokens},
            "usage": "batch",
        },
    )
    assert batch_result_params
    for result in batch_result_params["outputs"]:
        assert len(result.split(" ")) <= max_tokens + question_len
    print("openai batch with params successful batch_result", batch_result_params)


os.environ["HUGGINGFACE_API_KEY"] = "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"


def test_huggingface(
    model_id=_HUGGINGFACE_MODEL,
    prompt=PROMPT,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    question_len=QUESTION_LEN,
):
    """
    Test the langchain model server with a huggingface model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_tokens
    )

    project = mlrun.get_or_create_project(
        name="huggingface-model-server-example", context="./"
    )
    # Create a serving function with a local pipeline
    serving_func = project.set_function(
        func=langchain_model_server_path,
        name="huggingface-langchain-model-server",
        kind="serving",
        image="mlrun/mlrun",
    )
    serving_func.add_model(
        "huggingface-langchain-model",
        class_name="LangChainModelServer",
        llm="HuggingFacePipeline",
        init_kwargs={"pipeline": pipe},
        model_path=".",
    )
    server1 = serving_func.to_mock_server()

    # Create a second serving function with a model id
    serving_func2 = project.set_function(
        func=langchain_model_server_path,
        name="huggingface-langchain-model-server2",
        kind="serving",
        image="mlrun/mlrun",
    )
    serving_func2.add_model(
        "huggingface-langchain-model",
        class_name="LangChainModelServer",
        llm="HuggingFacePipeline",
        init_method="from_model_id",
        init_kwargs={
            "model_id": model_id,
            "task": "text-generation",
            "pipeline_kwargs": {"max_new_tokens": max_tokens},
        },
        model_path=".",
    )
    server2 = serving_func2.to_mock_server()
    for server in [server1, server2]:
        predict_result = server.test(
            "/v2/models/huggingface-langchain-model/predict",
            {"inputs": [prompt]},
        )
        assert predict_result
        print("huggingface successful predict predict_result", predict_result)
        invoke_result1 = server.test(
            "/v2/models/huggingface-langchain-model/predict",
            {
                "inputs": [prompt],
                "usage": "invoke",
                "generation_kwargs": {
                    "return_full_text": False,
                    "temperature": temperature,
                },
            },
        )

        assert (
            invoke_result1
            and len(invoke_result1["outputs"].lstrip().split(" "))
            <= max_tokens + question_len
        )
        print("huggingface successful invoke invoke_result", invoke_result1)
        invoke_result3 = server.test(
            "/v2/models/huggingface-langchain-model/predict",
            {
                "inputs": [prompt],
                "stop": "<eos>",
                "usage": "invoke",
                "generation_kwargs": {
                    "return_full_text": False,
                    "temperature": temperature,
                },
            },
        )
        assert (
            invoke_result3
            and len(invoke_result3["outputs"].lstrip().split(" "))
            <= max_tokens + question_len
        )
        print("huggingface successful invoke invoke_result", invoke_result3)
        batch_result1 = server.test(
            "/v2/models/huggingface-langchain-model/predict",
            {
                "inputs": [prompt, prompt],
                "usage": "batch",
            },
        )
        assert batch_result1
        print("huggingface successful batch batch_result", batch_result1)
        batch_result2 = server.test(
            "/v2/models/huggingface-langchain-model/predict",
            {
                "inputs": [prompt, prompt],
                "usage": "batch",
                "generation_kwargs": {
                    "return_full_text": False,
                    "temperature": temperature,
                },
            },
        )
        assert batch_result2
        for result in batch_result2["outputs"]:
            assert len(result.lstrip().split(" ")) <= max_tokens + question_len
        print("huggingface successful batch batch_result", batch_result2)
