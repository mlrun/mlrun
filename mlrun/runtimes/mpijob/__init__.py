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

# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx

from dependency_injector import containers, providers

from mlrun.config import config

from .. import MPIJobCRDVersions
from .v1 import MpiRuntimeV1
from .v1alpha1 import MpiRuntimeV1Alpha1


def _resolve_mpijob_crd_version():
    # config is expected to get enriched from the API through the client-spec
    return config.mpijob_crd_version or MPIJobCRDVersions.default()


class MpiRuntimeContainer(containers.DeclarativeContainer):
    resolver = providers.Callable(
        _resolve_mpijob_crd_version,
    )

    selector = providers.Selector(
        resolver,
        v1=providers.Object(MpiRuntimeV1),
        v1alpha1=providers.Object(MpiRuntimeV1Alpha1),
    )

    # An empty selector to be overriden by the API
    handler_selector = providers.Selector(
        resolver,
        v1=providers.Object(None),
        v1alpha1=providers.Object(None),
    )
