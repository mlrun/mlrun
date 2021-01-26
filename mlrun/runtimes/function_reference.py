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

import os
import mlrun
from base64 import b64encode
from nuclio.build import mlrun_footer


from ..model import ModelObj
from ..utils import generate_object_uri


class FunctionReference(ModelObj):
    """function reference/template"""

    def __init__(
        self,
        url=None,
        image=None,
        requirements=None,
        code=None,
        spec=None,
        kind=None,
        name=None,
    ):
        self.url = url
        self.kind = kind
        self.image = image
        self.requirements = requirements
        self.name = name
        self.spec = spec
        self.code = code

        self._function = None
        self._address = None

    def fullname(self, parent):
        return f"{parent.metadata.name}-{self.name}"

    def uri(self, parent, tag=None, hash_key=None, fullname=True):
        name = self.fullname(parent) if fullname else self.name
        return generate_object_uri(
            parent.metadata.project,
            name,
            tag=tag or parent.metadata.tag,
            hash_key=hash_key,
        )

    @property
    def function_object(self):
        return self._function

    def to_function(self, default_kind=None):
        if self.url and "://" not in self.url:
            if not os.path.isfile(self.url):
                raise OSError("{} not found".format(self.url))

        kind = self.kind or default_kind
        if self.spec:
            func = mlrun.new_function(self.name, runtime=self.spec)
        elif self.url:
            if (
                self.url.endswith(".yaml")
                or self.url.startswith("db://")
                or self.url.startswith("hub://")
            ):
                func = mlrun.import_function(self.url)
                if self.image:
                    func.spec.image = self.image
            elif self.url.endswith(".ipynb"):
                func = mlrun.code_to_function(
                    self.name, filename=self.url, image=self.image, kind=kind
                )
            elif self.url.endswith(".py"):
                # todo: support code text as input (for UI)
                if not self.image:
                    raise ValueError(
                        "image must be provided with py code files, "
                        "use function object for more control/settings"
                    )
                func = mlrun.code_to_function(
                    self.name, filename=self.url, image=self.image, kind=kind
                )
            else:
                raise ValueError(
                    "unsupported function url {} or no spec".format(self.url)
                )
        elif self.code is not None:
            code = self.code
            if kind == mlrun.runtimes.RuntimeKinds.serving:
                code = code + mlrun_footer.format(
                    mlrun.runtimes.serving.serving_subkind
                )
            func = mlrun.new_function(self.name, kind=kind, image=self.image)
            data = b64encode(code.encode("utf-8")).decode("utf-8")
            func.spec.build.functionSourceCode = data
            if kind not in mlrun.runtimes.RuntimeKinds.nuclio_runtimes():
                func.spec.default_handler = "handler"
        else:
            raise ValueError("url or spec or code must be specified")

        if self.requirements:
            func.with_requirements(self.requirements)
        self._function = func
        return func

    @property
    def address(self):
        return self._address

    def deploy(self, **kwargs):
        self._address = self._function.deploy(**kwargs)
        return self._address
