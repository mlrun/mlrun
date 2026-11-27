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
"""L1 structural parity between the ``_v1`` (pydantic.v1) and ``_v2`` (native pydantic 2)
schema faces (ML-12891).

Both faces are imported directly (not through the environment-dispatched facade) and compared
model-by-model: same model set per module, and per field the same required-ness, default,
alias, and extra policy. This is the structural gate the pydantic-2 flip is built on; L2
behavioural / wire parity lives in ML-12900.

Runs on the pydantic-2 baseline (both ``pydantic`` and ``pydantic.v1`` importable in-process).
"""

import importlib
import pkgutil

import pydantic as v2
import pydantic.v1 as v1
import pytest

import mlrun.common.schemas._v1 as v1_pkg

# The native ``_v2`` face only behaves as pydantic 2 on the pydantic-2 baseline (the API-server /
# parity CI job). Under a pydantic-1 install the ``_v2`` models degrade to v1 and lack
# ``model_fields``, so skip rather than error there.
if int(v2.VERSION.split(".")[0]) < 2:
    pytest.skip(
        "L1 schema parity requires the pydantic 2 baseline",
        allow_module_level=True,
    )

_REQUIRED = object()


def _module_suffixes(pkg):
    """Every non-package submodule under ``pkg``, as a dotted suffix (e.g. ``alert``,
    ``model_monitoring.model_endpoints``)."""
    prefix = pkg.__name__ + "."
    return sorted(
        info.name[len(prefix) :]
        for info in pkgutil.walk_packages(pkg.__path__, prefix)
        if not info.ispkg
    )


def _models_defined_in(module, base):
    """BaseModel subclasses *defined* in ``module`` (not imported into it)."""
    return {
        name: obj
        for name in dir(module)
        if isinstance(obj := getattr(module, name), type)
        and issubclass(obj, base)
        and obj.__module__ == module.__name__
    }


def _normalized_alias(alias, field_name):
    # pydantic v1 defaults a field's alias to its name; v2 leaves it None. Treat both as "no alias".
    return None if alias in (None, field_name) else alias


def _v1_contract(field_name, model_field):
    required = bool(model_field.required)
    if required:
        default = _REQUIRED
    elif model_field.default_factory is not None:
        default = ("factory", repr(model_field.default_factory()))
    else:
        default = ("value", repr(model_field.default))
    return required, default, _normalized_alias(model_field.alias, field_name)


def _v2_contract(field_name, field_info):
    required = field_info.is_required()
    if required:
        default = _REQUIRED
    elif field_info.default_factory is not None:
        default = ("factory", repr(field_info.default_factory()))
    else:
        default = ("value", repr(field_info.default))
    return required, default, _normalized_alias(field_info.alias, field_name)


def _extra_policy(cls_v1, cls_v2):
    e1 = getattr(cls_v1.__config__, "extra", None)
    e1 = getattr(e1, "value", e1) or "ignore"
    e2 = cls_v2.model_config.get("extra") or "ignore"
    return str(e1), str(e2)


_SUFFIXES = _module_suffixes(v1_pkg)


def _load_pair(suffix):
    m1 = importlib.import_module(f"mlrun.common.schemas._v1.{suffix}")
    m2 = importlib.import_module(f"mlrun.common.schemas._v2.{suffix}")
    return m1, m2


# (suffix, model_name) for every model present in both faces — the per-model field-parity cases.
_MODEL_CASES = []
for _suffix in _SUFFIXES:
    _m1, _m2 = _load_pair(_suffix)
    _shared_models = set(_models_defined_in(_m1, v1.BaseModel)) & set(
        _models_defined_in(_m2, v2.BaseModel)
    )
    _MODEL_CASES.extend((_suffix, _name) for _name in sorted(_shared_models))


@pytest.mark.parametrize("suffix", _SUFFIXES)
def test_same_model_set(suffix):
    m1, m2 = _load_pair(suffix)
    v1_models = set(_models_defined_in(m1, v1.BaseModel))
    v2_models = set(_models_defined_in(m2, v2.BaseModel))
    assert v1_models == v2_models, (
        f"{suffix}: model set differs "
        f"(v1-only={sorted(v1_models - v2_models)}, v2-only={sorted(v2_models - v1_models)})"
    )


@pytest.mark.parametrize("suffix,model_name", _MODEL_CASES)
def test_model_field_parity(suffix, model_name):
    m1, m2 = _load_pair(suffix)
    cls_v1 = _models_defined_in(m1, v1.BaseModel)[model_name]
    cls_v2 = _models_defined_in(m2, v2.BaseModel)[model_name]

    f1 = {n: _v1_contract(n, f) for n, f in cls_v1.__fields__.items()}
    f2 = {n: _v2_contract(n, f) for n, f in cls_v2.model_fields.items()}

    assert set(f1) == set(f2), (
        f"{suffix}.{model_name}: field set differs "
        f"(v1-only={sorted(set(f1) - set(f2))}, v2-only={sorted(set(f2) - set(f1))})"
    )
    mismatches = {name: (f1[name], f2[name]) for name in f1 if f1[name] != f2[name]}
    assert not mismatches, (
        f"{suffix}.{model_name}: field contract mismatches {mismatches}"
    )

    extra_v1, extra_v2 = _extra_policy(cls_v1, cls_v2)
    assert extra_v1 == extra_v2, (
        f"{suffix}.{model_name}: extra policy differs (v1={extra_v1}, v2={extra_v2})"
    )
