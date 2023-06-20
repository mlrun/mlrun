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
import hashlib

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


def _hash_list(*list_to_hash):
    list_to_hash = [str(element) for element in list_to_hash]
    str_concatted = "".join(list_to_hash)
    sha1 = hashlib.sha1()
    sha1.update(str_concatted.encode("utf8"))
    return sha1.hexdigest()


def _redis_stringify_key(*args):
    if len(args) == 1:
        key_list = args[0]
    else:
        key_list = list(args)
    suffix = "}:static"
    if isinstance(key_list, list):
        if len(key_list) >= 3:
            return str(key_list[0]) + "." + _hash_list(*key_list[1:]) + suffix
        if len(key_list) == 2:
            return str(key_list[0]) + "." + str(key_list[1]) + suffix
        return str(key_list[0]) + suffix
    return str(key_list) + suffix


hash_and_concat_v3io_udf = udf(_hash_list, StringType())
hash_and_concat_redis_udf = udf(_redis_stringify_key, StringType())
