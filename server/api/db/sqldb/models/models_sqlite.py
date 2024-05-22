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

import json
import pickle
import warnings
from datetime import datetime, timezone

import orjson
from sqlalchemy import (
    BLOB,
    BOOLEAN,
    JSON,
    TIMESTAMP,
    Column,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

import mlrun.common.schemas
import mlrun.utils.db
from server.api.utils.db.sql_collation import SQLCollationUtil

from .common import post_table_definitions

Base = declarative_base()
NULL = None  # Avoid flake8 issuing warnings when comparing in filter


def make_label(table):
    class Label(Base, mlrun.utils.db.BaseModel):
        __tablename__ = f"{table}_labels"
        __table_args__ = (
            UniqueConstraint("name", "parent", name=f"_{table}_labels_uc"),
            Index(f"idx_{table}_labels_name_value", "name", "value"),
        )

        id = Column(Integer, primary_key=True)
        name = Column(String(255, collation=SQLCollationUtil.collation()))
        value = Column(String(255, collation=SQLCollationUtil.collation()))
        parent = Column(Integer, ForeignKey(f"{table}.id"))

        def get_identifier_string(self) -> str:
            return f"{self.parent}/{self.name}/{self.value}"

    return Label


def make_tag(table):
    class Tag(Base, mlrun.utils.db.BaseModel):
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
    class Tag(Base, mlrun.utils.db.BaseModel):
        __tablename__ = f"{table}_tags"
        __table_args__ = (
            UniqueConstraint("project", "name", "obj_name", name=f"_{table}_tags_uc"),
        )

        id = Column(Integer, primary_key=True)
        project = Column(String(255, collation=SQLCollationUtil.collation()))
        name = Column(String(255, collation=SQLCollationUtil.collation()))
        obj_id = Column(Integer, ForeignKey(f"{table}.id"))
        obj_name = Column(String(255, collation=SQLCollationUtil.collation()))

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}"

    return Tag


def make_artifact_tag(table):
    class ArtifactTag(Base, mlrun.utils.db.BaseModel):
        __tablename__ = f"{table}_tags"
        __table_args__ = (
            UniqueConstraint("project", "name", "obj_id", name=f"_{table}_tags_uc"),
        )

        id = Column(Integer, primary_key=True)
        project = Column(String(255, collation=SQLCollationUtil.collation()))
        name = Column(String(255, collation=SQLCollationUtil.collation()))
        obj_id = Column(Integer, ForeignKey(f"{table}.id"))
        obj_name = Column(String(255, collation=SQLCollationUtil.collation()))

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}"

    return ArtifactTag


def make_notification(table):
    class Notification(Base, mlrun.utils.db.BaseModel):
        __tablename__ = f"{table}_notifications"
        __table_args__ = (
            UniqueConstraint(
                "name",
                "parent_id",
                name=f"_{table}_notifications_uc",
            ),
        )

        id = Column(Integer, primary_key=True)
        project = Column(String(255, collation=SQLCollationUtil.collation()))
        name = Column(
            String(255, collation=SQLCollationUtil.collation()), nullable=False
        )
        kind = Column(
            String(255, collation=SQLCollationUtil.collation()), nullable=False
        )
        message = Column(
            String(255, collation=SQLCollationUtil.collation()), nullable=False
        )
        severity = Column(
            String(255, collation=SQLCollationUtil.collation()), nullable=False
        )
        when = Column(
            String(255, collation=SQLCollationUtil.collation()), nullable=False
        )
        condition = Column(
            String(255, collation=SQLCollationUtil.collation()), nullable=False
        )
        secret_params = Column("secret_params", JSON)
        params = Column("params", JSON)
        parent_id = Column(Integer, ForeignKey(f"{table}.id"))
        sent_time = Column(
            TIMESTAMP(),
            nullable=True,
        )
        status = Column(
            String(255, collation=SQLCollationUtil.collation()), nullable=False
        )
        reason = Column(
            String(255, collation=SQLCollationUtil.collation()), nullable=True
        )

    return Notification


# quell SQLAlchemy warnings on duplicate class name (Label)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # deprecated, use ArtifactV2 instead
    # TODO: remove in 1.8.0. Note that removing it will require upgrading mlrun in at least 2 steps:
    #  1. upgrade to 1.6.x which will create the new table
    #  2. upgrade to 1.7.x which will remove the old table
    class Artifact(Base, mlrun.utils.db.HasStruct):
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
        # TODO: change to JSON, see mlrun/common/schemas/function.py::FunctionState for reasoning
        body = Column(BLOB)
        labels = relationship(Label)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.key}/{self.uid}"

    class ArtifactV2(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "artifacts_v2"
        __table_args__ = (
            UniqueConstraint("uid", "project", "key", name="_artifacts_uc"),
        )

        Label = make_label(__tablename__)
        Tag = make_artifact_tag(__tablename__)

        id = Column(Integer, primary_key=True)
        key = Column(String(255, collation=SQLCollationUtil.collation()), index=True)
        project = Column(String(255, collation=SQLCollationUtil.collation()))
        kind = Column(String(255, collation=SQLCollationUtil.collation()), index=True)
        producer_id = Column(String(255, collation=SQLCollationUtil.collation()))
        iteration = Column(Integer)
        best_iteration = Column(BOOLEAN, default=False, index=True)
        uid = Column(String(255, collation=SQLCollationUtil.collation()))
        created = Column(TIMESTAMP, default=datetime.now(timezone.utc))
        updated = Column(TIMESTAMP, default=datetime.now(timezone.utc))
        _full_object = Column("object", BLOB)

        labels = relationship(Label, cascade="all, delete-orphan")
        tags = relationship(Tag, cascade="all, delete-orphan")

        @property
        def full_object(self):
            if self._full_object:
                return pickle.loads(self._full_object)

        @full_object.setter
        def full_object(self, value):
            self._full_object = pickle.dumps(value)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.key}/{self.uid}"

    class Function(Base, mlrun.utils.db.HasStruct):
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
        # TODO: change to JSON, see mlrun/common/schemas/function.py::FunctionState for reasoning
        body = Column(BLOB)
        updated = Column(TIMESTAMP)
        labels = relationship(Label)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}/{self.uid}"

    class Log(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "logs"

        id = Column(Integer, primary_key=True)
        uid = Column(String(255, collation=SQLCollationUtil.collation()))
        project = Column(String(255, collation=SQLCollationUtil.collation()))
        # TODO: change to JSON, see mlrun/common/schemas/function.py::FunctionState for reasoning
        body = Column(BLOB)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.uid}"

    class Run(Base, mlrun.utils.db.HasStruct):
        __tablename__ = "runs"
        __table_args__ = (
            UniqueConstraint("uid", "project", "iteration", name="_runs_uc"),
            Index("idx_runs_project_id", "id", "project"),
        )

        Label = make_label(__tablename__)
        Tag = make_tag(__tablename__)
        Notification = make_notification(__tablename__)

        id = Column(Integer, primary_key=True)
        uid = Column(String(255, collation=SQLCollationUtil.collation()))
        project = Column(String(255, collation=SQLCollationUtil.collation()))
        name = Column(
            String(255, collation=SQLCollationUtil.collation()), default="no-name"
        )
        iteration = Column(Integer)
        state = Column(String(255, collation=SQLCollationUtil.collation()))
        # TODO: change to JSON, see mlrun/common/schemas/function.py::FunctionState for reasoning
        body = Column(BLOB)
        start_time = Column(TIMESTAMP)
        # requested logs column indicates whether logs were requested for this run
        # None - old runs prior to the column addition, logs were already collected for them, so no need to collect them
        # False - logs were not requested for this run
        # True - logs were requested for this run
        requested_logs = Column(BOOLEAN)
        updated = Column(TIMESTAMP, default=datetime.utcnow)
        labels = relationship(Label)
        notifications = relationship(Notification, cascade="all, delete-orphan")

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.uid}/{self.iteration}"

    class BackgroundTask(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "background_tasks"
        __table_args__ = (
            UniqueConstraint("name", "project", name="_background_tasks_uc"),
        )

        id = Column(Integer, primary_key=True)
        name = Column(
            String(255, collation=SQLCollationUtil.collation()), nullable=False
        )
        project = Column(
            String(255, collation=SQLCollationUtil.collation()), nullable=False
        )
        created = Column(TIMESTAMP, default=datetime.now(timezone.utc))
        updated = Column(TIMESTAMP, default=datetime.now(timezone.utc))
        state = Column(String(255, collation=SQLCollationUtil.collation()))
        error = Column(String(255, collation=SQLCollationUtil.collation()))
        timeout = Column(Integer)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}"

    class Schedule(Base, mlrun.utils.db.BaseModel):
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
        # TODO: change to JSON, see mlrun/common/schemas/function.py::FunctionState for reasoning
        struct = Column(BLOB)
        labels = relationship(Label, cascade="all, delete-orphan")
        concurrency_limit = Column(Integer, nullable=False)
        next_run_time = Column(TIMESTAMP)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}"

        @property
        def scheduled_object(self):
            return pickle.loads(self.struct)

        @scheduled_object.setter
        def scheduled_object(self, value):
            self.struct = pickle.dumps(value)

        @property
        def cron_trigger(self) -> mlrun.common.schemas.ScheduleCronTrigger:
            return orjson.loads(self.cron_trigger_str)

        @cron_trigger.setter
        def cron_trigger(self, trigger: mlrun.common.schemas.ScheduleCronTrigger):
            self.cron_trigger_str = orjson.dumps(trigger.dict(exclude_unset=True))

    # Define "many to many" users/projects
    project_users = Table(
        "project_users",
        Base.metadata,
        Column("project_id", Integer, ForeignKey("projects.id")),
        Column("user_id", Integer, ForeignKey("users.id")),
    )

    class User(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "users"
        __table_args__ = (UniqueConstraint("name", name="_users_uc"),)

        id = Column(Integer, primary_key=True)
        name = Column(String(255, collation=SQLCollationUtil.collation()))

        def get_identifier_string(self) -> str:
            return f"{self.name}"

    class Project(Base, mlrun.utils.db.BaseModel):
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
        # TODO: change to JSON, see mlrun/common/schemas/function.py::FunctionState for reasoning
        _full_object = Column("spec", BLOB)
        created = Column(TIMESTAMP, default=datetime.utcnow)
        state = Column(String(255, collation=SQLCollationUtil.collation()))
        default_function_node_selector = Column("default_function_node_selector", JSON)
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

    class Feature(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "features"
        id = Column(Integer, primary_key=True)
        feature_set_id = Column(Integer, ForeignKey("feature_sets.id"))

        name = Column(String(255, collation=SQLCollationUtil.collation()))
        value_type = Column(String(255, collation=SQLCollationUtil.collation()))

        Label = make_label(__tablename__)
        labels = relationship(Label, cascade="all, delete-orphan")

        def get_identifier_string(self) -> str:
            return f"{self.feature_set_id}/{self.name}"

    class Entity(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "entities"
        id = Column(Integer, primary_key=True)
        feature_set_id = Column(Integer, ForeignKey("feature_sets.id"))

        name = Column(String(255, collation=SQLCollationUtil.collation()))
        value_type = Column(String(255, collation=SQLCollationUtil.collation()))

        Label = make_label(__tablename__)
        labels = relationship(Label, cascade="all, delete-orphan")

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}"

    class FeatureSet(Base, mlrun.utils.db.BaseModel):
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
            self._full_object = json.dumps(value, default=str)

    class FeatureVector(Base, mlrun.utils.db.BaseModel):
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
            self._full_object = json.dumps(value, default=str)

    class HubSource(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "hub_sources"
        __table_args__ = (UniqueConstraint("name", name="_hub_sources_uc"),)

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
            self._full_object = json.dumps(value, default=str)

    class DataVersion(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "data_versions"
        __table_args__ = (UniqueConstraint("version", name="_versions_uc"),)

        id = Column(Integer, primary_key=True)
        version = Column(String(255, collation=SQLCollationUtil.collation()))
        created = Column(TIMESTAMP, default=datetime.now(timezone.utc))

        def get_identifier_string(self) -> str:
            return f"{self.version}"

    class DatastoreProfile(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "datastore_profiles"
        __table_args__ = (
            UniqueConstraint("name", "project", name="_datastore_profiles_uc"),
        )

        id = Column(Integer, primary_key=True, autoincrement=True)
        name = Column(String(255, collation=SQLCollationUtil.collation()))
        type = Column(String(255, collation=SQLCollationUtil.collation()))
        project = Column(String(255, collation=SQLCollationUtil.collation()))

        _full_object = Column("object", JSON)

        @property
        def full_object(self):
            if self._full_object:
                return json.loads(self._full_object)

        @full_object.setter
        def full_object(self, value):
            self._full_object = json.dumps(value, default=str)

    class PaginationCache(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "pagination_cache"

        key = Column(
            String(255, collation=SQLCollationUtil.collation()), primary_key=True
        )
        user = Column(String(255, collation=SQLCollationUtil.collation()))
        function = Column(String(255, collation=SQLCollationUtil.collation()))
        current_page = Column(Integer)
        page_size = Column(Integer)
        kwargs = Column(JSON)
        last_accessed = Column(TIMESTAMP, default=datetime.now(timezone.utc))

    class AlertState(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "alert_states"
        __table_args__ = (UniqueConstraint("id", "parent_id", name="alert_states_uc"),)

        id = Column(Integer, primary_key=True)
        count = Column(Integer)
        created = Column(
            TIMESTAMP(),
            default=datetime.now(timezone.utc),
        )
        last_updated = Column(
            TIMESTAMP(),
            default=None,
        )
        active = Column(BOOLEAN, default=False)

        parent_id = Column(Integer, ForeignKey("alert_configs.id"))

        _full_object = Column("object", JSON)

        @property
        def full_object(self):
            if self._full_object:
                return json.loads(self._full_object)

        @full_object.setter
        def full_object(self, value):
            self._full_object = json.dumps(value, default=str)

    class AlertConfig(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "alert_configs"
        __table_args__ = (
            UniqueConstraint("project", "name", name="_alert_configs_uc"),
        )

        Notification = make_notification(__tablename__)

        id = Column(Integer, primary_key=True)
        name = Column(
            String(255, collation=SQLCollationUtil.collation()), nullable=False
        )
        project = Column(
            String(255, collation=SQLCollationUtil.collation()), nullable=False
        )

        notifications = relationship(Notification, cascade="all, delete-orphan")
        alerts = relationship(AlertState, cascade="all, delete-orphan")

        _full_object = Column("object", JSON)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}"

        @property
        def full_object(self):
            if self._full_object:
                return json.loads(self._full_object)

        @full_object.setter
        def full_object(self, value):
            self._full_object = json.dumps(value, default=str)

    class AlertTemplate(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "alert_templates"
        __table_args__ = (UniqueConstraint("name", name="_alert_templates_uc"),)

        id = Column(Integer, primary_key=True)
        name = Column(
            String(255, collation=SQLCollationUtil.collation()), nullable=False
        )

        _full_object = Column("object", JSON)

        def get_identifier_string(self) -> str:
            return f"{self.name}"

        @property
        def full_object(self):
            if self._full_object:
                return json.loads(self._full_object)

        @full_object.setter
        def full_object(self, value):
            self._full_object = json.dumps(value, default=str)


# Must be after all table definitions
post_table_definitions(base_cls=Base)
