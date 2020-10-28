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

import orjson
import pickle
import warnings
from datetime import datetime

from sqlalchemy import (
    BLOB,
    TIMESTAMP,
    Column,
    ForeignKey,
    Integer,
    String,
    Table,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from mlrun.api import schemas

Base = declarative_base()
NULL = None  # Avoid flake8 issuing warnings when comparing in filter
run_time_fmt = "%Y-%m-%dT%H:%M:%S.%fZ"


class HasStruct:
    @property
    def struct(self):
        return pickle.loads(self.body)

    @struct.setter
    def struct(self, value):
        self.body = pickle.dumps(value)


def make_label(table):
    class Label(Base):
        __tablename__ = f"{table}_labels"
        __table_args__ = (
            UniqueConstraint("name", "parent", name=f"_{table}_labels_uc"),
        )

        id = Column(Integer, primary_key=True)
        name = Column(String)
        value = Column(String)
        parent = Column(Integer, ForeignKey(f"{table}.id"))

    return Label


def make_tag(table):
    class Tag(Base):
        __tablename__ = f"{table}_tags"
        __table_args__ = (
            UniqueConstraint("project", "name", "obj_id", name=f"_{table}_tags_uc"),
        )

        id = Column(Integer, primary_key=True)
        project = Column(String)
        name = Column(String)
        obj_id = Column(Integer, ForeignKey(f"{table}.id"))

    return Tag


# TODO: don't want to refactor everything in one PR so splitting this function to 2 versions - eventually only this one
#  should be used
def make_tag_v2(table):
    class Tag(Base):
        __tablename__ = f"{table}_tags"
        __table_args__ = (
            UniqueConstraint("project", "name", "obj_name", name=f"_{table}_tags_uc"),
        )

        id = Column(Integer, primary_key=True)
        project = Column(String)
        name = Column(String)
        obj_id = Column(Integer, ForeignKey(f"{table}.id"))
        obj_name = Column(Integer, ForeignKey(f"{table}.name"))

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
        key = Column(String)
        project = Column(String)
        uid = Column(String)
        updated = Column(TIMESTAMP)
        body = Column(BLOB)
        labels = relationship(Label)

    class Function(Base, HasStruct):
        __tablename__ = "functions"
        __table_args__ = (
            UniqueConstraint("name", "project", "uid", name="_functions_uc"),
        )

        Label = make_label(__tablename__)
        Tag = make_tag_v2(__tablename__)

        id = Column(Integer, primary_key=True)
        name = Column(String)
        project = Column(String)
        uid = Column(String)
        body = Column(BLOB)
        updated = Column(TIMESTAMP)
        labels = relationship(Label)

    class Log(Base):
        __tablename__ = "logs"

        id = Column(Integer, primary_key=True)
        uid = Column(String)
        project = Column(String)
        body = Column(BLOB)

    class Run(Base, HasStruct):
        __tablename__ = "runs"
        __table_args__ = (
            UniqueConstraint("uid", "project", "iteration", name="_runs_uc"),
        )

        Label = make_label(__tablename__)
        Tag = make_tag(__tablename__)

        id = Column(Integer, primary_key=True)
        uid = Column(String)
        project = Column(String)
        iteration = Column(Integer)
        state = Column(String)
        body = Column(BLOB)
        start_time = Column(TIMESTAMP)
        labels = relationship(Label)

    class Schedule(Base):
        __tablename__ = "schedules_v2"
        project = Column(String, primary_key=True)
        name = Column(String, primary_key=True)
        kind = Column(String)
        desired_state = Column(String)
        state = Column(String)
        creation_time = Column(TIMESTAMP)
        cron_trigger_str = Column(String)
        struct = Column(BLOB)

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

    class User(Base):
        __tablename__ = "users"
        __table_args__ = (UniqueConstraint("name", name="_users_uc"),)

        id = Column(Integer, primary_key=True)
        name = Column(String)

    class Project(Base):
        __tablename__ = "projects"
        # For now since we use project name a lot
        __table_args__ = (UniqueConstraint("name", name="_projects_uc"),)

        id = Column(Integer, primary_key=True)
        name = Column(String)
        description = Column(String)
        owner = Column(String)
        source = Column(String)
        _spec = Column("spec", BLOB)
        created = Column(TIMESTAMP, default=datetime.utcnow)
        state = Column(String)
        users = relationship(User, secondary=project_users)

        @property
        def spec(self):
            if self._spec:
                return pickle.loads(self._spec)

        @spec.setter
        def spec(self, value):
            self._spec = pickle.dumps(value)


# Must be after all table definitions
_tagged = [cls for cls in Base.__subclasses__() if hasattr(cls, "Tag")]
_labeled = [cls for cls in Base.__subclasses__() if hasattr(cls, "Label")]
_classes = [cls for cls in Base.__subclasses__()]
_table2cls = {cls.__table__.name: cls for cls in Base.__subclasses__()}
