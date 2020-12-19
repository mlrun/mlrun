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

# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx

import mlrun
from mlrun.config import config
from mlrun.model import DataClass
from mlrun.utils.helpers import parse_function_uri
from mlrun.datastore import store_manager
from mlrun.artifacts import dict_to_artifact


def get_data_resource(kind, uri, db=None, secrets=None):
    db = db or mlrun.get_run_db().connect(secrets)

    if kind == DataClass.FeatureSet:
        project, name, tag, uid = parse_function_uri(uri, config.default_project)
        obj = db.get_feature_set(name, project, tag, uid)
        return mlrun.featurestore.FeatureSet.from_dict(obj)

    elif kind == DataClass.FeatureVector:
        project, name, tag, uid = parse_function_uri(uri, config.default_project)
        obj = db.get_feature_vector(name, project, tag, uid)
        return mlrun.featurestore.FeatureVector.from_dict(obj)

    elif DataClass.is_artifact(kind):
        project, name, tag, uid = parse_function_uri(uri, config.default_project)
        resp = db.read_artifact(name, project=project, tag=tag or uid)
        if resp:
            return dict_to_artifact(resp)

    elif DataClass.Object:
        stores = store_manager.set(secrets, db=db)
        return stores.object(url=uri)
    else:
        raise ValueError(f"illegal kind {kind}")
