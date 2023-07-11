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
import mlrun.api.utils.events.base
import mlrun.api.utils.events.iguazio
import mlrun.api.utils.events.nop
import mlrun.common.schemas
import mlrun.utils.singleton


class EventsFactory(object):
    @staticmethod
    def get_events_client(
        kind: mlrun.common.schemas.EventClientKinds = None, **kwargs
    ) -> mlrun.api.utils.events.base.BaseEventClient:
        if mlrun.mlconf.events.mode == mlrun.common.schemas.EventsModes.disabled:
            return mlrun.api.utils.events.nop.NopClient()

        if not kind:
            if mlrun.mlconf.get_parsed_igz_version():
                kind = mlrun.common.schemas.EventClientKinds.iguazio

        if kind == mlrun.common.schemas.EventClientKinds.iguazio:
            if not mlrun.mlconf.get_parsed_igz_version():
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Iguazio events client can only be used in Iguazio environment"
                )
            return mlrun.api.utils.events.iguazio.Client(**kwargs)

        return mlrun.api.utils.events.nop.NopClient()
