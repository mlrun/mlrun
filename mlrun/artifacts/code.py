# Copyright 2026 Iguazio
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

import mlrun.common.types

from .base import Artifact, ArtifactSpec


class CodeArtifactCodeType(mlrun.common.types.StrEnum):
    function = "function"
    workflow = "workflow"


class CodeArtifactSpec(ArtifactSpec):
    _dict_fields = ArtifactSpec._dict_fields + [
        "language",
        "code_type",
    ]

    def __init__(
        self,
        src_path=None,
        target_path=None,
        viewer=None,
        is_inline=False,
        format=None,
        size=None,
        db_key=None,
        extra_data=None,
        body=None,
        language=None,
        code_type=None,
    ):
        super().__init__(
            src_path=src_path,
            target_path=target_path,
            viewer=viewer,
            is_inline=is_inline,
            format=format,
            size=size,
            db_key=db_key,
            extra_data=extra_data,
            body=body,
        )
        self.language = language
        self.code_type = code_type


class CodeArtifact(Artifact):
    """Code Artifact

    Store a code file or archive for use as a function or workflow source.
    Supports a single code file or a single archive (.zip, .tar.gz).
    """

    kind = "code"

    def __init__(
        self,
        key=None,
        body=None,
        format=None,
        target_path=None,
        src_path=None,
        language=None,
        code_type=None,
        **kwargs,
    ):
        """
        :param key:          Artifact key
        :param body:         Inline code content
        :param format:       Optional file format
        :param target_path:  Absolute target path
        :param src_path:     Path to the local code file or archive
        :param language:     Programming language and version (e.g. "python:3.9")
        :param code_type:    Type of code: "function" or "workflow" (default: "function")
        """
        super().__init__(
            key,
            body,
            format=format,
            target_path=target_path,
            src_path=src_path,
            **kwargs,
        )
        self.spec.language = language
        self.spec.code_type = CodeArtifactCodeType(
            code_type or CodeArtifactCodeType.function
        )

    @property
    def spec(self) -> CodeArtifactSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", CodeArtifactSpec)
