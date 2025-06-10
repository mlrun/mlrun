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

import os
from typing import cast

import pytest
import yaml

import mlrun
from mlrun import model_provider_manager
from mlrun.datastore.datastore_profile import (
    DatastoreProfileOpenAI,
    register_temporary_client_datastore_profile,
)
from mlrun.datastore.model_providers import OpenAIProvider

here = os.path.dirname(__file__)
config = {}
config_file_path = os.path.join(here, "test-openai.yml")
if os.path.exists(config_file_path):
    with open(config_file_path) as yaml_file:
        config = yaml.safe_load(yaml_file)


def openai_configured():
    if not config.get("OPENAI_API_KEY") or not config.get("OPENAI_BASE_URL"):
        return False
    return True


@pytest.mark.skipif(
    openai_configured(),
    reason="Requires OPENAI_API_KEY and OPENAI_BASE_URL to be set under test-openai.yml",
)
@pytest.mark.parametrize("use_datastore_profile", [True, False])
class TestOpenAIProvider:
    profile_name = "openai_profile"

    @classmethod
    def setup_class(cls):
        cls.env_secrets = config["env"]
        cls.basic_llm_model = "gpt-4"

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self, use_datastore_profile):
        if use_datastore_profile:
            self.profile = DatastoreProfileOpenAI(
                name=self.profile_name,
                api_key=self.env_secrets.get("OPENAI_API_KEY"),
                organization=self.env_secrets.get("OPENAI_ORG_ID"),
                project=self.env_secrets.get("OPENAI_PROJECT_ID"),
                base_url=self.env_secrets.get("OPENAI_BASE_URL"),
                timeout=self.env_secrets.get("OPENAI_TIMEOUT"),
                max_retries=self.env_secrets.get("OPENAI_MAX_RETRIES"),
            )
            register_temporary_client_datastore_profile(self.profile)
            self.url_prefix = f"ds://{self.profile_name}/"
            self.reset_env()
        else:
            for key, env_param in self.env_secrets.items():
                if env_param:
                    os.environ[key] = env_param
            model_provider_manager.reset_secrets()
            self.url_prefix = "openai://"

    @classmethod
    def reset_env(cls):
        for key, env_param in cls.env_secrets.items():
            if env_param:
                os.environ.pop(key, None)

    @staticmethod
    def check_basic_invoke(model_url: str, secrets: dict, model_name: str):
        model_provider = mlrun.get_model_provider(url=model_url, secrets=secrets)
        model_provider = cast(OpenAIProvider, model_provider)
        response = model_provider.basic_llm_invoke(
            prompt="what is the capital of france?"
        )
        assert "paris" in response.choices[0].message.content.lower()
        assert model_provider.model == model_name

    def test_basic_invoke(self):
        model_url = self.url_prefix + self.basic_llm_model
        #  env check
        self.check_basic_invoke(
            model_url=model_url, secrets={}, model_name=self.basic_llm_model
        )
        # secrets check
        self.reset_env()
        self.check_basic_invoke(
            model_url=model_url,
            secrets=self.env_secrets,
            model_name=self.basic_llm_model,
        )

    def test_configurable_model(self):
        configurable_model = mlrun.mlconf.model_providers.openai_default_model
        if not configurable_model:
            pytest.skip(
                "model_providers.openai_default_model is not configured in conf, cannot perform the test"
            )

        #  checking default model usage:
        model_url = self.url_prefix
        #  env check
        self.check_basic_invoke(
            model_url=model_url, secrets={}, model_name=configurable_model
        )
        # secrets check
        self.reset_env()
        self.check_basic_invoke(
            model_url=model_url, secrets=self.env_secrets, model_name=configurable_model
        )
