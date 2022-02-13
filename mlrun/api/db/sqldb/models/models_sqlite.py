# Copyright 2019 Iguazio
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

import json
import pickle
import warnings
from datetime import datetime, timezone

import orjson
from sqlalchemy import (
    BLOB,
    JSON,
    TIMESTAMP,
    Column,
    ForeignKey,
    Integer,
    String,
    Table,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import class_mapper, relationship

from mlrun.api import schemas
from mlrun.api.utils.db.sql_collation import SQLCollationUtil

Base = declarative_base()
NULL = None  # Avoid flake8 issuing warnings when comparing in filter
run_time_fmt = "%Y-%m-%dT%H:%M:%S.%fZ"


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


def make_label(table):
    class Label(Base, BaseModel):
        __tablename__ = f"{table}_labels"
        __table_args__ = (
            UniqueConstraint("name", "parent", name=f"_{table}_labels_uc"),
        )

        id = Column(Integer, primary_key=True)
        name = Column(String(255, collation=SQLCollationUtil.collation()))
        value = Column(String(255, collation=SQLCollationUtil.collation()))
        parent = Column(Integer, ForeignKey(f"{table}.id"))

    return Label


def make_tag(table):
    class Tag(Base, BaseModel):
        __tablename__ = f"{table}_tags"
        __table_args__ = (
            UniqueConstraint("project", "name", "obj_id", name=f"_{table}_tags_uc"),
        )

        id = Column(Integer, primary_key=True)
        project = Column(String(255, collation=SQLCollationUtil.collation()))
        name = Column(String(255, collation=SQLCollationUtil.collation()))
        obj_id = Column(Integer, ForeignKey(f"{table}.id"))

    return Tag


# TODO: don't want to refactor everything in one PR so splitting this function to 2 versions - eventually only this one
#  should be used
def make_tag_v2(table):
    class Tag(Base, BaseModel):
        __tablename__ = f"{table}_tags"
        __table_args__ = (
            UniqueConstraint("project", "name", "obj_name", name=f"_{table}_tags_uc"),
        )

        id = Column(Integer, primary_key=True)
        project = Column(String(255, collation=SQLCollationUtil.collation()))
        name = Column(String(255, collation=SQLCollationUtil.collation()))
        obj_id = Column(Integer, ForeignKey(f"{table}.id"))
        obj_name = Column(
            String(255, collation=SQLCollationUtil.collation()),
            ForeignKey(f"{table}.name"),
        )

    return Tag


# quell SQLAlchemy warnings on duplicate class name (Label)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    class Artifact(Base, HasStruct):
        __tablename__ = "artifacts"
        __table_args__ = (
            UniqueConstraint("uid", "project", "key", name="_artifacts_uc"),
        )

        Label = make_label(__tablename__)
        Tag = make_tag(__tablename__)

        id = Column(Integer, primary_key=True)
        key = Column(String(255, collation=SQLCollationUtil.collation()))
        project = Column(String(255, collation=SQLCollationUtil.collation()))
        uid = Column(String(255, collation=SQLCollationUtil.collation()))
        updated = Column(TIMESTAMP)
        # TODO: change to JSON, see mlrun/api/schemas/function.py::FunctionState for reasoning
        body = Column(BLOB)
        labels = relationship(Label)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.key}/{self.uid}"

    class Function(Base, HasStruct):
        __tablename__ = "functions"
        __table_args__ = (
            UniqueConstraint("name", "project", "uid", name="_functions_uc"),
        )

        Label = make_label(__tablename__)
        Tag = make_tag_v2(__tablename__)

        id = Column(Integer, primary_key=True)
        name = Column(String(255, collation=SQLCollationUtil.collation()))
        project = Column(String(255, collation=SQLCollationUtil.collation()))
        uid = Column(String(255, collation=SQLCollationUtil.collation()))
        # TODO: change to JSON, see mlrun/api/schemas/function.py::FunctionState for reasoning
        body = Column(BLOB)
        updated = Column(TIMESTAMP)
        labels = relationship(Label)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}/{self.uid}"

    class Log(Base, BaseModel):
        __tablename__ = "logs"

        id = Column(Integer, primary_key=True)
        uid = Column(String(255, collation=SQLCollationUtil.collation()))
        project = Column(String(255, collation=SQLCollationUtil.collation()))
        # TODO: change to JSON, see mlrun/api/schemas/function.py::FunctionState for reasoning
        body = Column(BLOB)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.uid}"

    class Run(Base, HasStruct):
        __tablename__ = "runs"
        __table_args__ = (
            UniqueConstraint("uid", "project", "iteration", name="_runs_uc"),
        )

        Label = make_label(__tablename__)
        Tag = make_tag(__tablename__)

        id = Column(Integer, primary_key=True)
        uid = Column(String(255, collation=SQLCollationUtil.collation()))
        project = Column(String(255, collation=SQLCollationUtil.collation()))
        name = Column(
            String(255, collation=SQLCollationUtil.collation()), default="no-name"
        )
        iteration = Column(Integer)
        state = Column(String(255, collation=SQLCollationUtil.collation()))
        # TODO: change to JSON, see mlrun/api/schemas/function.py::FunctionState for reasoning
        body = Column(BLOB)
        start_time = Column(TIMESTAMP)
        updated = Column(TIMESTAMP, default=datetime.utcnow)
        labels = relationship(Label)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.uid}/{self.iteration}"

    class Schedule(Base, BaseModel):
        __tablename__ = "schedules_v2"
        __table_args__ = (UniqueConstraint("project", "name", name="_schedules_v2_uc"),)

        Label = make_label(__tablename__)

        id = Column(Integer, primary_key=True)
        project = Column(
            String(255, collation=SQLCollationUtil.collation()), nullable=False
        )
        name = Column(
            String(255, collation=SQLCollationUtil.collation()), nullable=False
        )
        kind = Column(String(255, collation=SQLCollationUtil.collation()))
        desired_state = Column(String(255, collation=SQLCollationUtil.collation()))
        state = Column(String(255, collation=SQLCollationUtil.collation()))
        creation_time = Column(TIMESTAMP)
        cron_trigger_str = Column(String(255, collation=SQLCollationUtil.collation()))
        last_run_uri = Column(String(255, collation=SQLCollationUtil.collation()))
        # TODO: change to JSON, see mlrun/api/schemas/function.py::FunctionState for reasoning
        struct = Column(BLOB)
        labels = relationship(Label, cascade="all, delete-orphan")
        concurrency_limit = Column(Integer, nullable=False)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}"

        @property
        def scheduled_object(self):
            return pickle.loads(self.struct)

        @scheduled_object.setter
        def scheduled_object(self, value):
            self.struct = pickle.dumps(value)

        @property
        def cron_trigger(self) -> schemas.ScheduleCronTrigger:
            return orjson.loads(self.cron_trigger_str)

        @cron_trigger.setter
        def cron_trigger(self, trigger: schemas.ScheduleCronTrigger):
            self.cron_trigger_str = orjson.dumps(trigger.dict(exclude_unset=True))

    # Define "many to many" users/projects
    project_users = Table(
        "project_users",
        Base.metadata,
        Column("project_id", Integer, ForeignKey("projects.id")),
        Column("user_id", Integer, ForeignKey("users.id")),
    )

    class User(Base, BaseModel):
        __tablename__ = "users"
        __table_args__ = (UniqueConstraint("name", name="_users_uc"),)

        id = Column(Integer, primary_key=True)
        name = Column(String(255, collation=SQLCollationUtil.collation()))

    class Project(Base, BaseModel):
        __tablename__ = "projects"
        # For now since we use project name a lot
        __table_args__ = (UniqueConstraint("name", name="_projects_uc"),)

        id = Column(Integer, primary_key=True)
        name = Column(String(255, collation=SQLCollationUtil.collation()))
        description = Column(String(255, collation=SQLCollationUtil.collation()))
        owner = Column(String(255, collation=SQLCollationUtil.collation()))
        source = Column(String(255, collation=SQLCollationUtil.collation()))
        # the attribute name used to be _spec which is just a wrong naming, the attribute was renamed to _full_object
        # leaving the column as is to prevent redundant migration
        # TODO: change to JSON, see mlrun/api/schemas/function.py::FunctionState for reasoning
        _full_object = Column("spec", BLOB)
        created = Column(TIMESTAMP, default=datetime.utcnow)
        state = Column(String(255, collation=SQLCollationUtil.collation()))
        users = relationship(User, secondary=project_users)

        Label = make_label(__tablename__)

        labels = relationship(Label, cascade="all, delete-orphan")

        def get_identifier_string(self) -> str:
            return f"{self.name}"

        @property
        def full_object(self):
            if self._full_object:
                return pickle.loads(self._full_object)

        @full_object.setter
        def full_object(self, value):
            self._full_object = pickle.dumps(value)

    class Feature(Base, BaseModel):
        __tablename__ = "features"
        id = Column(Integer, primary_key=True)
        feature_set_id = Column(Integer, ForeignKey("feature_sets.id"))

        name = Column(String(255, collation=SQLCollationUtil.collation()))
        value_type = Column(String(255, collation=SQLCollationUtil.collation()))

        Label = make_label(__tablename__)
        labels = relationship(Label, cascade="all, delete-orphan")

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}"

    class Entity(Base, BaseModel):
        __tablename__ = "entities"
        id = Column(Integer, primary_key=True)
        feature_set_id = Column(Integer, ForeignKey("feature_sets.id"))

        name = Column(String(255, collation=SQLCollationUtil.collation()))
        value_type = Column(String(255, collation=SQLCollationUtil.collation()))

        Label = make_label(__tablename__)
        labels = relationship(Label, cascade="all, delete-orphan")

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}"

    class FeatureSet(Base, BaseModel):
        __tablename__ = "feature_sets"
        __table_args__ = (
            UniqueConstraint("name", "project", "uid", name="_feature_set_uc"),
        )

        id = Column(Integer, primary_key=True)
        name = Column(String(255, collation=SQLCollationUtil.collation()))
        project = Column(String(255, collation=SQLCollationUtil.collation()))
        created = Column(TIMESTAMP, default=datetime.now(timezone.utc))
        updated = Column(TIMESTAMP, default=datetime.now(timezone.utc))
        state = Column(String(255, collation=SQLCollationUtil.collation()))
        uid = Column(String(255, collation=SQLCollationUtil.collation()))

        _full_object = Column("object", JSON)

        Label = make_label(__tablename__)
        Tag = make_tag_v2(__tablename__)

        labels = relationship(Label, cascade="all, delete-orphan")

        features = relationship(Feature, cascade="all, delete-orphan")
        entities = relationship(Entity, cascade="all, delete-orphan")

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}/{self.uid}"

        @property
        def full_object(self):
            if self._full_object:
                return json.loads(self._full_object)

        @full_object.setter
        def full_object(self, value):
            self._full_object = json.dumps(value)

    class FeatureVector(Base, BaseModel):
        __tablename__ = "feature_vectors"
        __table_args__ = (
            UniqueConstraint("name", "project", "uid", name="_feature_vectors_uc"),
        )

        id = Column(Integer, primary_key=True)
        name = Column(String(255, collation=SQLCollationUtil.collation()))
        project = Column(String(255, collation=SQLCollationUtil.collation()))
        created = Column(TIMESTAMP, default=datetime.now(timezone.utc))
        updated = Column(TIMESTAMP, default=datetime.now(timezone.utc))
        state = Column(String(255, collation=SQLCollationUtil.collation()))
        uid = Column(String(255, collation=SQLCollationUtil.collation()))

        _full_object = Column("object", JSON)

        Label = make_label(__tablename__)
        Tag = make_tag_v2(__tablename__)

        labels = relationship(Label, cascade="all, delete-orphan")

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}/{self.uid}"

        @property
        def full_object(self):
            if self._full_object:
                return json.loads(self._full_object)

        @full_object.setter
        def full_object(self, value):
            self._full_object = json.dumps(value)

    class MarketplaceSource(Base, BaseModel):
        __tablename__ = "marketplace_sources"
        __table_args__ = (UniqueConstraint("name", name="_marketplace_sources_uc"),)

        id = Column(Integer, primary_key=True)
        name = Column(String(255, collation=SQLCollationUtil.collation()))
        index = Column(Integer)
        created = Column(TIMESTAMP, default=datetime.now(timezone.utc))
        updated = Column(TIMESTAMP, default=datetime.now(timezone.utc))

        _full_object = Column("object", JSON)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}"

        @property
        def full_object(self):
            if self._full_object:
                return json.loads(self._full_object)

        @full_object.setter
        def full_object(self, value):
            self._full_object = json.dumps(value)

    class DataVersion(Base, BaseModel):
        __tablename__ = "data_versions"
        __table_args__ = (UniqueConstraint("version", name="_versions_uc"),)

        id = Column(Integer, primary_key=True)
        version = Column(String(255, collation=SQLCollationUtil.collation()))
        created = Column(TIMESTAMP, default=datetime.now(timezone.utc))


# Must be after all table definitions
_tagged = [cls for cls in Base.__subclasses__() if hasattr(cls, "Tag")]
_labeled = [cls for cls in Base.__subclasses__() if hasattr(cls, "Label")]
_classes = [cls for cls in Base.__subclasses__()]
_table2cls = {cls.__table__.name: cls for cls in Base.__subclasses__()}
