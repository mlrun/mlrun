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
import sys

# import pickle5 on clients with python < 3.8 since the server is using pickle protocol 5 (python 3.9)
# TODO: remove this when we drop support for python 3.7
if sys.version_info < (3, 8):
    try:
        import pickle5 as pickle
    except ImportError:
        import pickle
else:
    import pickle


def dumps(x):
    return pickle.dumps(x)


def loads(x):
    return pickle.loads(x)


def load(x):
    return pickle.load(x)
