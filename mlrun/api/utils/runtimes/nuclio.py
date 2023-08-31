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
#
import mlrun.api.utils.clients.nuclio
from mlrun.config import config
from mlrun.errors import err_to_str
from mlrun.utils import logger

cached_nuclio_version = None


# if nuclio version specified on mlrun config set it likewise,
# if not specified, get it from nuclio api client
# since this is a heavy operation (sending requests to API), and it's unlikely that the version
# will change - cache it (this means if we upgrade nuclio, we need to restart mlrun to re-fetch the new version)
def resolve_nuclio_version():
    global cached_nuclio_version

    if not cached_nuclio_version:

        # config override everything
        nuclio_version = config.nuclio_version
        if not nuclio_version and config.nuclio_dashboard_url:
            try:
                nuclio_client = mlrun.api.utils.clients.nuclio.Client()
                nuclio_version = nuclio_client.get_dashboard_version()
            except Exception as exc:
                logger.warning("Failed to resolve nuclio version", exc=err_to_str(exc))

        cached_nuclio_version = nuclio_version

    return cached_nuclio_version
