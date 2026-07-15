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
"""Environment-dispatched schema package.

On import this always pulls the version-agnostic ``_shared`` layer, then binds the
model face for the current **environment** — ``_v2`` (native Pydantic 2) in the
mlrun API server, else ``_v1`` (``pydantic.v1``) for the client/SDK. Selection is by
environment, not by the installed Pydantic major: a client may run on Pydantic 2 yet
still needs ``_v1`` because its call-sites use the v1 instance API (see
``_dispatch``). Everything is re-exported under the unchanged public path
``mlrun.common.schemas.*``; the per-topic submodules (``mlrun.common.schemas.<topic>``)
are preserved by thin facades. See the Backend HLD (ML-12736) §2.1.2.
"""

from ._dispatch import USE_V2_SCHEMAS as _USE_V2
from ._shared import *  # noqa: F401,F403
from ._shared import __all__ as _shared_all

if _USE_V2:
    from ._v2 import *  # noqa: F401,F403
    from ._v2 import __all__ as _face_all
else:
    from ._v1 import *  # noqa: F401,F403
    from ._v1 import __all__ as _face_all

# Import the per-topic facades so the ``mlrun.common.schemas.<topic>`` submodule
# paths keep resolving as attributes of the package (callers rely on both the flat
# names above and the per-topic submodules).
from . import (  # noqa: E402, F401
    alert,
    api_gateway,
    artifact,
    auth,
    background_task,
    client_spec,
    clusterization_spec,
    common,
    constants,
    datastore_profile,
    events,
    feature_store,
    frontend_spec,
    function,
    http,
    hub,
    k8s,
    memory_reports,
    model_monitoring,
    notification,
    object,
    pagination,
    partition_interval,
    pipeline,
    project,
    regex,
    runs,
    runtime_resource,
    schedule,
    secret,
    serving,
    tag,
    workflow,
)

__all__ = [*_shared_all, *_face_all]
