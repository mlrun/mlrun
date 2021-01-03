import json
from inspect import signature, _empty
from importlib import import_module

# for backwards compatibility - can be removed when we separate the hub branches for 0.6.x ad 0.5.x
from .plots import eval_model_v2, eval_class_model  # noqa: F401


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


def gen_sklearn_model(model_pkg, skparams):
    """generate an sklearn model configuration

    input can be either a "package.module.class" or
    a json file
    """
    if model_pkg.endswith("json"):
        model_config = json.load(open(model_pkg, "r"))
    else:
        model_config = get_class_fit(model_pkg)

    for k, v in skparams:
        if k.startswith("CLASS_"):
            model_config["CLASS"][k[6:]] = v
        if k.startswith("FIT_"):
            model_config["FIT"][k[4:]] = v

    return model_config
