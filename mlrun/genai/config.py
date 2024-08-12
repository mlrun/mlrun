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

import importlib
import logging
import os
from pathlib import Path

import dotenv
import yaml
from pydantic import BaseModel

root_path = Path(__file__).parent.parent.parent
dotenv.load_dotenv(os.environ.get("AGENT_ENV_PATH", str(root_path / ".env")))
default_data_path = os.environ.get("AGENT_DATA_PATH", str(root_path / "data"))


class AppConfig(BaseModel):
    """Configuration for the agent."""

    api_url: str = "http://localhost:8001"  # url of the controller API
    verbose: bool = True
    log_level: str = "DEBUG"
    use_local_db: bool = True

    chunk_size: int = 1024
    chunk_overlap: int = 20

    # Embeddings
    embeddings: dict = {"class_name": "huggingface", "model_name": "all-MiniLM-L6-v2"}

    # Default LLM
    default_llm: dict = {
        "class_name": "langchain_openai.ChatOpenAI",
        "temperature": 0,
        "model_name": "gpt-3.5-turbo",
    }
    # Vector store
    default_vector_store: dict = {
        "class_name": "milvus",
        "collection_name": "default",
        "connection_args": {"address": "localhost:19530"},
    }

    # Pipeline kwargs
    pipeline_args: dict = {}

    def default_collection(self):
        return self.default_vector_store.get("collection_name", "default")

    def print(self):
        print(yaml.dump(self.dict()))

    @classmethod
    def load_from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.parse_obj(data)

    @classmethod
    def local_config(cls):
        """Create a local config for testing oe local deployment."""
        config = cls()
        config.verbose = True
        config.default_vector_store = {
            "class_name": "chroma",
            "collection_name": "default",
            "persist_directory": str((Path(default_data_path) / "chroma").absolute()),
        }
        return config


is_local_config = os.environ.get("IS_LOCAL_CONFIG", "0").lower().strip() in [
    "true",
    "1",
]
config_path = os.environ.get("AGENT_CONFIG_PATH")

if config_path:
    config = AppConfig.load_from_yaml(config_path)
elif is_local_config:
    config = AppConfig.local_config()
else:
    config = AppConfig()

logger = logging.getLogger("llmagent")
logger.setLevel(config.log_level.upper())
logger.addHandler(logging.StreamHandler())
logger.info("Logger initialized...")
# logger.info(f"Using config:\n {yaml.dump(config.model_dump())}")


embeddings_shortcuts = {
    "huggingface": "langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings",
    "openai": "langchain_openai.embeddings.base.OpenAIEmbeddings",
}

vector_db_shortcuts = {
    "milvus": "langchain_community.vectorstores.Milvus",
    "chroma": "langchain_community.vectorstores.chroma.Chroma",
}

llm_shortcuts = {
    "chat": "langchain_openai.ChatOpenAI",
    "gpt": "langchain_community.chat_models.GPT",
}


def get_embedding_function(config: AppConfig, embeddings_args: dict = None):
    return get_object_from_dict(
        embeddings_args or config.embeddings, embeddings_shortcuts
    )


def get_llm(config: AppConfig, llm_args: dict = None):
    """Get a language model instance."""
    return get_object_from_dict(llm_args or config.default_llm, llm_shortcuts)


def get_vector_db(
    config: AppConfig,
    data_source_name: str = None,
    vector_store_args: dict = None,
):
    """Get a vector database instance.

    Args:
        config: An AppConfig instance.
        data_source_name: The name of the collection to use (if not default).
        vector_store_args: class_name and arguments to pass to the vector store class (None will use the config).
    """
    embeddings = get_embedding_function(config=config)
    vector_store_args = vector_store_args or config.default_vector_store
    vector_store_args = vector_store_args.copy()
    if data_source_name:
        vector_store_args["collection_name"] = data_source_name
    vector_store_args["embedding_function"] = embeddings
    return get_object_from_dict(vector_store_args, vector_db_shortcuts)


def get_class_from_string(class_path, shortcuts: dict = {}) -> type:
    if class_path in shortcuts:
        class_path = shortcuts[class_path]
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def get_object_from_dict(obj_dict: dict, shortcuts: dict = {}):
    if not isinstance(obj_dict, dict):
        return obj_dict
    obj_dict = obj_dict.copy()
    class_name = obj_dict.pop("class_name")
    class_ = get_class_from_string(class_name, shortcuts)
    return class_(**obj_dict)
