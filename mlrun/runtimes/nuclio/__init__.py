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

import mlrun.runtimes.nuclio.serving as nuclio_serving  # noqa
import mlrun.runtimes.nuclio.nuclio as nuclio_nuclio  # noqa
import mlrun.runtimes.nuclio.function as nuclio_function  # noqa
import mlrun.runtimes.nuclio.api_gateway as nuclio_api_gateway  # noqa

ServingRuntime = nuclio_serving.ServingRuntime
new_v2_model_server = nuclio_serving.new_v2_model_server
nuclio_init_hook = nuclio_nuclio.nuclio_init_hook
min_nuclio_versions = nuclio_function.min_nuclio_versions
multiple_port_sidecar_is_supported = nuclio_function.multiple_port_sidecar_is_supported
RemoteRuntime = nuclio_function.RemoteRuntime
APIGateway = nuclio_api_gateway.APIGateway
