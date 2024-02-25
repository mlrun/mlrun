# Copyright 2024 Iguazio
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

import tempfile

from kfp.compiler import Compiler


def compile_pipeline(pipeline, **kwargs):
    # TODO: how to set a cleanup ttl on KFP 2.0?
    pipe_file = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False).name
    Compiler().compile(pipeline, pipe_file, type_check=False)
    return pipe_file
