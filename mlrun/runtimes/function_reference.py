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

import os

from nuclio.build import mlrun_footer

import mlrun
import mlrun.utils.helpers

from ..model import ModelObj
from ..utils import generate_object_uri
from .utils import enrich_function_from_dict


class FunctionReference(ModelObj):
    """function reference/template, point to function and add/override resources"""

    def __init__(
        self,
        url=None,
        image=None,
        requirements=None,
        code=None,
        spec=None,
        kind=None,
        name=None,
        track_models=None,
    ):
        self.url = url
        self.kind = kind
        self.image = image
        self.requirements = requirements
        self.name = name
        if hasattr(spec, "to_dict"):
            spec = spec.to_dict()
        self.spec = spec
        self.code = code
        self.track_models = track_models

        self._function = None
        self._address = None

    def is_empty(self):
        if self.url or self.code or self.spec:
            return False
        return True

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
        """get the generated function object"""
        return self._function

    def to_function(self, default_kind=None, default_image=None):
        """generate a function object from the ref definitions"""
        if self.url and "://" not in self.url:
            if not os.path.isfile(self.url):
                raise OSError(f"{self.url} not found")

        kind = self.kind or default_kind
        if self.url:
            if (
                self.url.endswith(".yaml")
                or self.url.startswith("db://")
                or self.url.startswith("hub://")
            ):
                func = mlrun.import_function(self.url)
                func.spec.image = self.image or func.spec.image or default_image
            elif self.url.endswith(".ipynb"):
                func = mlrun.code_to_function(
                    self.name, filename=self.url, image=self.image, kind=kind
                )
                func.spec.image = func.spec.image or default_image
            elif self.url.endswith(".py"):
                # todo: support code text as input (for UI)
                image = self.image or default_image
                if not image:
                    raise ValueError(
                        "image must be provided with py code files, "
                        "use function object for more control/settings"
                    )
                func = mlrun.code_to_function(
                    self.name, filename=self.url, image=image, kind=kind
                )
            else:
                raise ValueError(f"unsupported function url {self.url} or no spec")
            if self.spec:
                func = enrich_function_from_dict(func, self.spec)
        elif self.code is not None:
            code = self.code
            if kind == mlrun.runtimes.RuntimeKinds.serving:
                code = code + mlrun_footer.format(
                    mlrun.runtimes.nuclio.serving.serving_subkind
                )
            func = mlrun.new_function(
                self.name, kind=kind, image=self.image or default_image
            )
            data = mlrun.utils.helpers.encode_user_code(code)
            func.spec.build.functionSourceCode = data
            if kind not in mlrun.runtimes.RuntimeKinds.nuclio_runtimes():
                func.spec.default_handler = "handler"
            if self.spec:
                func = enrich_function_from_dict(func, self.spec)
        elif self.spec:
            func = mlrun.new_function(self.name, runtime=self.spec)
        else:
            raise ValueError("url or spec or code must be specified")

        if self.requirements:
            func.with_requirements(self.requirements)
        self._function = func
        func.spec.track_models = self.track_models
        return func

    @property
    def address(self):
        return self._address

    def deploy(self, **kwargs):
        """deploy the function"""
        self._address = self._function.deploy(**kwargs)
        return self._address
