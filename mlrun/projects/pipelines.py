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


import mlrun
from mlrun.utils import parse_versioned_object_uri


class FunctionsDict(dict):
    def __init__(self, project, functions=None, decorator=None):
        dict.__init__(self, functions)
        self.project = project
        self._decorator = decorator

    def load_or_set_function(self, key, default=None):
        if key in dict.keys(self):
            return enrich_function_object(
                self.project, dict.__getitem__(self, key), self._decorator
            )

        function = default or get_db_function(self.project, key)
        dict.__setitem__(self, key, function)
        return enrich_function_object(self.project, function, self._decorator)

    def get(self, key, default=None):
        return self.load_or_set_function(key, default)

    def __getitem__(self, key):
        return self.load_or_set_function(key)

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)


def get_db_function(project, key):
    project_instance, name, tag, hash_key = parse_versioned_object_uri(
        key, project.metadata.name
    )
    runtime = mlrun.get_run_db().get_function(name, project_instance, tag, hash_key)
    return mlrun.new_function(runtime=runtime)


def enrich_function_object(project, function, decorator=None):
    f = function.copy()
    f.metadata.project = project.metadata.name
    src = f.spec.build.source
    if project.spec.source and src and src in [".", "./"]:
        if project.spec.mountdir:
            f.spec.workdir = project.spec.mountdir
            f.spec.build.source = ""
        else:
            f.spec.build.source = project.spec.source
            f.spec.build.load_source_on_run = project.spec.load_source_on_run
    if decorator:
        decorator(f)
    return f
