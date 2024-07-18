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
import os
import subprocess
from subprocess import PIPE, Popen

import pytest

import mlrun

langchain_model_server_path = os.path.abspath(
    "../../../mlrun/frameworks/langchain/model_server.py"
)

#: if true, delete the model after the test
_OLLAMA_DELETE_MODEL_POST_TEST = False
_OLLAMA_MODEL = "qwen:0.5b"
_OPENAI_MODEL = "gpt-3.5-turbo-instruct"
_HUGGINGFACE_MODEL = "gpt2"


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
        return 0
    return result.returncode != 0


@pytest.fixture
@pytest.mark.skipif(ollama_check_skip(), reason="Ollama not installed")
def ollama_fixture():
    """
    Do the setup and cleanup for the ollama test
    :return:
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
def test_ollama(ollama_fixture):
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
        model_class="Ollama",
        class_name="LangChainModelServer",
        llm=_OLLAMA_MODEL,
        init_kwargs={"model": _OLLAMA_MODEL},
        model_path=".",
    )
    server = serving_func.to_mock_server()
    predict_result = server.test(
        "/v2/models/ollama-langchain-model/predict", {"inputs": ["how old are you?"]}
    )
    assert predict_result
    print("ollama predict successful predict_result", predict_result)
    invoke_result = server.test(
        "/v2/models/ollama-langchain-model/invoke", {"inputs": ["how old are you?"]}
    )
    assert invoke_result
    print("ollama invoke successful invoke_result", invoke_result)
    batch_result = server.test(
        "/v2/models/ollama-langchain-model/batch",
        {"inputs": ["how old are you?", "how old are you?"]},
    )
    assert batch_result
    print("ollama batch successful batch_result", batch_result)


@pytest.mark.skipif(ollama_check_skip(), reason="OpenAI API credentials not set")
@pytest.mark.asyncio
async def test_ollama_async(ollama_fixture):
    """
    Test the langchain model server with an ollama model asynchronously
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
        model_class="Ollama",
        class_name="LangChainModelServer",
        llm=_OLLAMA_MODEL,
        init_kwargs={"model": _OLLAMA_MODEL},
        model_path=".",
    )
    server = serving_func.to_mock_server()
    ainvoke_result = await asyncio.gather(
        server.test(
            "/v2/models/ollama-langchain-model/ainvoke",
            {"inputs": ["how old are you?"]},
        )
    )
    ainvoke_result = ainvoke_result[0]
    assert ainvoke_result
    print("ollama ainvoke successful ainvoke_result", ainvoke_result)
    abatch_result = await asyncio.gather(
        server.test(
            "/v2/models/ollama-langchain-model/abatch",
            {"inputs": ["how old are you?", "how old are you?"]},
        )
    )
    abatch_result = abatch_result[0]
    assert abatch_result
    print("ollama abatch successful abatch_result", abatch_result)


def skip_openai():
    """
    Check if the OpenAI API credentials are set
    """
    return not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_BASE_URL")


@pytest.mark.skipif(skip_openai(), reason="OpenAI API credentials not set")
def test_openai():
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
        model_class="OpenAI",
        class_name="LangChainModelServer",
        llm="OpenAI",
        init_kwargs={"model": _OPENAI_MODEL},
        model_path=".",
    )
    server = serving_func.to_mock_server()
    predict_result = server.test(
        "/v2/models/openai-langchain-model/predict", {"inputs": ["how old are you?"]}
    )
    assert predict_result
    print("openai predict successful predict_result", predict_result)
    invoke_result = server.test(
        "/v2/models/openai-langchain-model/invoke", {"inputs": ["how old are you?"]}
    )
    assert invoke_result
    print("openai invoke successful invoke_result", invoke_result)
    batch_result = server.test(
        "/v2/models/openai-langchain-model/batch",
        {"inputs": ["how old are you?", "how old are you?"]},
    )
    assert batch_result
    print("openai batch successful batch_result", batch_result)


@pytest.mark.skipif(skip_openai(), reason="OpenAI API credentials not set")
@pytest.mark.asyncio
async def test_openai_async():
    """
    Test the langchain model server with an openai model asynchronously
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
        model_class="OpenAI",
        class_name="LangChainModelServer",
        llm="OpenAI",
        init_kwargs={"model": _OPENAI_MODEL},
        model_path=".",
    )
    server = serving_func.to_mock_server()
    ainvoke_result = await asyncio.gather(
        server.test(
            "/v2/models/openai-langchain-model/ainvoke",
            {"inputs": ["how old are you?"]},
        )
    )
    ainvoke_result = ainvoke_result[0]
    assert ainvoke_result
    print("openai ainvoke successful ainvoke_result", ainvoke_result)
    abatch_result = await asyncio.gather(
        server.test(
            "/v2/models/openai-langchain-model/abatch",
            {"inputs": ["how old are you?", "how old are you?"]},
        )
    )
    abatch_result = abatch_result[0]
    assert abatch_result
    print("openai abatch successful abatch_result", abatch_result)


def test_huggingface():
    """
    Test the langchain model server with a huggingface model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_id = _HUGGINGFACE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
    )

    project = mlrun.get_or_create_project(
        name="huggingface-model-server-example", context="./"
    )
    serving_func = project.set_function(
        func=langchain_model_server_path,
        name="huggingface-langchain-model-server",
        kind="serving",
        image="mlrun/mlrun",
    )
    serving_func.add_model(
        "huggingface-langchain-model",
        model_class="HuggingFacePipeline",
        class_name="LangChainModelServer",
        llm=_HUGGINGFACE_MODEL,
        init_kwargs={"pipeline": pipe},
        model_path=".",
    )
    server = serving_func.to_mock_server()
    predict_result = server.test(
        "/v2/models/huggingface-langchain-model/predict",
        {"inputs": ["how old are you?"]},
    )
    assert predict_result
    print("huggingface successful predict predict_result", predict_result)
    invoke_result = server.test(
        "/v2/models/huggingface-langchain-model/invoke",
        {"inputs": ["how old are you?"]},
    )
    assert invoke_result
    print("huggingface successful invoke invoke_result", invoke_result)
    batch_result = server.test(
        "/v2/models/huggingface-langchain-model/batch",
        {"inputs": ["how old are you?", "how old are you?"]},
    )
    assert batch_result
    print("huggingface successful batch batch_result", batch_result)


@pytest.mark.asyncio
async def test_huggingface_async():
    """
    Test the langchain model server with a huggingface model asynchronously
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_id = _HUGGINGFACE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
    )

    project = mlrun.get_or_create_project(
        name="huggingface-model-server-example", context="./"
    )
    serving_func = project.set_function(
        func=langchain_model_server_path,
        name="huggingface-langchain-model-server",
        kind="serving",
        image="mlrun/mlrun",
    )
    serving_func.add_model(
        "huggingface-langchain-model",
        model_class="HuggingFacePipeline",
        class_name="LangChainModelServer",
        llm=_HUGGINGFACE_MODEL,
        init_kwargs={"pipeline": pipe},
        model_path=".",
    )
    server = serving_func.to_mock_server()
    ainvoke_result = await asyncio.gather(
        server.test(
            "/v2/models/huggingface-langchain-model/ainvoke",
            {"inputs": ["how old are you?"]},
        )
    )
    ainvoke_result = ainvoke_result[0]
    assert ainvoke_result
    print("huggingface successful ainvoke ainvoke_result", ainvoke_result)

    abatch_result = await asyncio.gather(
        server.test(
            "/v2/models/huggingface-langchain-model/abatch",
            {"inputs": ["how old are you?", "how old are you?"]},
        )
    )
    abatch_result = abatch_result[0]
    assert abatch_result
    print("huggingface successful abatch abatch_result", abatch_result)
