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
#
import pickle
from datetime import datetime

from sqlalchemy.orm import class_mapper


class BaseModel:
    def to_dict(self, exclude=None):
        """
        NOTE - this function (currently) does not handle serializing relationships
        """
        exclude = exclude or []
        mapper = class_mapper(self.__class__)
        columns = [column.key for column in mapper.columns if column.key not in exclude]
        get_key_value = (
            lambda c: (c, getattr(self, c).isoformat())
            if isinstance(getattr(self, c), datetime)
            else (c, getattr(self, c))
        )
        return dict(map(get_key_value, columns))


class HasStruct(BaseModel):
    @property
    def struct(self):
        return pickle.loads(self.body)

    @struct.setter
    def struct(self, value):
        self.body = pickle.dumps(value)

    def to_dict(self, exclude=None):
        """
        NOTE - this function (currently) does not handle serializing relationships
        """
        exclude = exclude or []
        exclude.append("body")
        return super().to_dict(exclude)
