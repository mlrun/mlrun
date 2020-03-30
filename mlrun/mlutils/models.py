from inspect import signature, _empty
from importlib import import_module


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
        d.update({f[i][0]: None if f[i][1].default is _empty
                  else f[i][1].default})

    return ({"CLASS": model_().get_params(),
             "FIT": d,
             "META": {"pkg_version": import_module(splits[0]).__version__,
                      "class": module_pkg_class}})
