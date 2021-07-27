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
    def __init__(self, project, functions=None, decorator=None, db=None):
        dict.__init__(self, functions)
        self.project = project
        self._db = db or mlrun.get_run_db()
        self._decorator = decorator

    def _enrich(self, function):
        function = _enrich_function_source(self.project, function)
        if self._decorator:
            self._decorator(function)
        return function

    def load_or_set_function(self, key, default=None):
        if key in dict.keys(self):
            return self._enrich(dict.__getitem__(self, key))

        if default:
            function = default
        else:
            project_name = self.project.metadata.name
            project_instance, name, tag, hash_key = parse_versioned_object_uri(
                key, project_name
            )
            runtime = self._db.get_function(name, project_instance, tag, hash_key)
            function = mlrun.new_function(runtime=runtime)

        dict.__setitem__(self, key, function)
        return self._enrich(function)

    def get(self, key, default=None):
        return self.load_or_set_function(key, default)

    def __getitem__(self, key):
        return self.load_or_set_function(key)

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)


def _enrich_function_source(project, func):
    f = func.copy()
    f.metadata.project = project.metadata.name
    src = f.spec.build.source
    if project.spec.source and src and src in [".", "./"]:
        if project.spec.mountdir:
            f.spec.workdir = project.spec.mountdir
            f.spec.build.source = ""
        else:
            f.spec.build.source = project.spec.source
            f.spec.build.load_source_on_run = project.spec.load_source_on_run
    return f
