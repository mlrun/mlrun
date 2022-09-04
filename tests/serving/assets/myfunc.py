# Copyright 2018 Iguazio
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
import storey

from mlrun.serving import GraphContext


def myhand(x, context=None):
    assert isinstance(context, GraphContext), "didnt get a valid context"
    return x * 2


class Mycls(storey.MapClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def do(self, event):
        return event * 2
