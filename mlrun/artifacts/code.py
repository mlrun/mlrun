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
        "requirements",
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
        language: str | None = None,
        code_type: CodeArtifactCodeType | None = None,
        requirements: list[str] | None = None,
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
        self.requirements = requirements

    @classmethod
    def from_dict(cls, struct=None, fields=None, deprecated_fields=None):
        # Coerce serialized string to enum so the code_type attribute is
        # always a CodeArtifactCodeType instance post-deserialization.
        if struct and isinstance(struct.get("code_type"), str):
            struct = {**struct, "code_type": CodeArtifactCodeType(struct["code_type"])}
        return super().from_dict(
            struct=struct, fields=fields, deprecated_fields=deprecated_fields
        )


class CodeArtifact(Artifact):
    """Code Artifact

    Store source code for use as a function or workflow source. The artifact
    payload is a source file or an archive (``.zip`` / ``.tar.gz``) whose
    members are extracted on resolution. The payload may be carried inline
    as ``body`` (subject to the inline-artifact size limit) or uploaded to
    ``target_path`` like any other artifact.
    """

    kind = "code"

    def __init__(
        self,
        key=None,
        body=None,
        format=None,
        target_path=None,
        src_path=None,
        language: str | None = None,
        code_type: str | CodeArtifactCodeType | None = None,
        requirements: list[str] | None = None,
        **kwargs,
    ):
        """
        :param key:          Artifact key
        :param body:         Inline code content
        :param format:       Optional file format
        :param target_path:  Absolute target path
        :param src_path:     Path to the local code file or archive
        :param language:     Programming language (e.g. ``"python"``).
                             Free-text advisory metadata — no validation or
                             enforcement is applied, and the value is not consulted
                             at resolution or execution time.
                             When ``None``, derived from the ``target_path`` (or
                             ``src_path``) suffix: ``.py``/``.ipynb`` → ``"python"``,
                             archives/unknown → ``""``, no path → stays ``None``.
        :param code_type:    Type of code: "function" or "workflow" (default: "function")
        :param requirements: List of dependency strings (e.g. ["pandas>=2.0", "numpy"])
        """
        super().__init__(
            key,
            body,
            format=format,
            target_path=target_path,
            src_path=src_path,
            **kwargs,
        )
        if language is None:
            language = _derive_language_from_path(target_path or src_path)
        self.spec.language = language
        self.spec.code_type = CodeArtifactCodeType(
            code_type or CodeArtifactCodeType.function
        )
        self.spec.requirements = requirements

    @property
    def spec(self) -> CodeArtifactSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", CodeArtifactSpec)


_PYTHON_SUFFIXES = (".py", ".ipynb")


def _derive_language_from_path(path: str | None) -> str | None:
    """Derive a language value from a file path's suffix.

    Returns ``"python"`` for ``.py``/``.ipynb``, ``""`` for archives and unknown
    suffixes (caller knows there's a path but can't infer the language), and
    ``None`` when no path is available (nothing to infer from).
    """
    if not path:
        return None
    if path.lower().endswith(_PYTHON_SUFFIXES):
        return "python"
    return ""
