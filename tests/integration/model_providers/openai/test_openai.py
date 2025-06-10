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
import mlrun
import pytest
import yaml
from mlrun.datastore.model_providers import OpenAIProvider
from typing import cast

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
class TestOpenAIProvider:

    @classmethod
    def setup_class(cls):
        cls.env_secrets = config["env"]

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self):
        for key, env_param in self.env_secrets.items():
            if env_param:
                os.environ[key] = env_param
        mlrun.model_provider_manager.reset_secrets()

    @classmethod
    def reset_env(cls):
        for key, env_param in cls.env_secrets.items():
            if env_param:
                os.environ.pop(key, None)

    @staticmethod
    def check_basic_invoke(model_url, secrets):
        model_provider = mlrun.get_model_provider(url=model_url, secrets=secrets)
        model_provider = cast(OpenAIProvider, model_provider)
        response = model_provider.basic_llm_invoke(prompt="what is the capital of france?")
        assert "paris" in response.choices[0].message.content.lower()

    @pytest.mark.parametrize(
        "model_url",
        ["openai://gpt-4"],
    )
    def test_basic_invoke(self, model_url):
        #  env check
        self.check_basic_invoke(model_url=model_url, secrets={})
        # secrets check
        self.reset_env()
        self.check_basic_invoke(model_url=model_url, secrets=self.env_secrets)
