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
import json
from importlib import import_module
from inspect import _empty, signature

from deprecated import deprecated

# for backwards compatibility - can be removed when we separate the hub branches for 0.6.x ad 0.5.x
from .plots import eval_class_model, eval_model_v2  # noqa: F401

# TODO: remove mlutils in 1.5.0


@deprecated(
    version="1.3.0",
    reason="'mlrun.mlutils' will be removed in 1.5.0, use 'mlrun.framework' instead",
    category=FutureWarning,
)
def get_class_fit(module_pkg_class: str):
    """generate a model config
    :param module_pkg_class:  str description of model, e.g.
        `sklearn.ensemble.RandomForestClassifier`
    """
    splits = module_pkg_class.split(".")
    model_ = getattr(import_module(".".join(splits[:-1])), splits[-1])
    f = list(signature(model_().fit).parameters.items())
    d = {}
    for i in range(len(f)):
        d.update({f[i][0]: None if f[i][1].default is _empty else f[i][1].default})

    return {
        "CLASS": model_().get_params(),
        "FIT": d,
        "META": {
            "pkg_version": import_module(splits[0]).__version__,
            "class": module_pkg_class,
        },
    }


@deprecated(
    version="1.3.0",
    reason="'mlrun.mlutils' will be removed in 1.5.0, use 'mlrun.framework' instead",
    category=FutureWarning,
)
def gen_sklearn_model(model_pkg, skparams):
    """generate an sklearn model configuration

    input can be either a "package.module.class" or
    a json file
    """
    if model_pkg.endswith("json"):
        model_config = json.load(open(model_pkg, "r"))
    else:
        model_config = get_class_fit(model_pkg)

    # we used to use skparams as is (without .items()) so supporting both cases for backwards compatibility
    skparams = skparams.items() if isinstance(skparams, dict) else skparams
    for k, v in skparams:
        if k.startswith("CLASS_"):
            model_config["CLASS"][k[6:]] = v
        if k.startswith("FIT_"):
            model_config["FIT"][k[4:]] = v

    return model_config
