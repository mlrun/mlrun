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
import os
import pickle
import uuid
import warnings
from datetime import datetime, timezone

import orjson
from sqlalchemy import (
    BOOLEAN,
    JSON,
    Column,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    PrimaryKeyConstraint,
    Table,
    UniqueConstraint,
    event,
    text,
)
from sqlalchemy.engine import Connection
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapper, declared_attr, relationship

import mlrun.common.db.dialects
import mlrun.common.schemas
import mlrun.db.sql_types
import mlrun.utils.db

Base = declarative_base()
NULL = None  # Avoid flake8 issuing warnings when comparing in filter

_tagged = None
_labeled = None
_with_notifications = None
_classes = None


def post_table_definitions(base_cls):
    global _tagged
    global _labeled
    global _with_notifications
    global _classes
    _tagged = [cls for cls in base_cls.__subclasses__() if hasattr(cls, "Tag")]
    _labeled = [cls for cls in base_cls.__subclasses__() if hasattr(cls, "Label")]
    _with_notifications = [
        cls for cls in base_cls.__subclasses__() if hasattr(cls, "Notification")
    ]
    _classes = [cls for cls in base_cls.__subclasses__()]


def make_label(parent_cls):
    table = parent_cls.__tablename__

    class Label(Base, mlrun.utils.db.BaseModel):
        __tablename__ = f"{table}_labels"
        __table_args__ = (
            UniqueConstraint("name", "parent", name=f"_{table}_labels_uc"),
            Index(f"idx_{table}_labels_name_value", "name", "value"),
        )

        id = Column(Integer, primary_key=True)
        name = Column(mlrun.db.sql_types.Utf8BinText)
        value = Column(mlrun.db.sql_types.Utf8BinText)

        parent = Column(
            Integer,
            ForeignKey(
                f"{table}.id",
                name=f"_{table}_labels_parent_fk",
                ondelete="CASCADE",
            ),
            nullable=True,
        )

        parent_rel = relationship(
            parent_cls,
            back_populates="labels",
            passive_deletes=True,
        )

        def get_identifier_string(self) -> str:
            return f"{self.parent}/{self.name}/{self.value}"

    return Label


def make_tag(parent_cls):
    table = parent_cls.__tablename__

    class Tag(Base, mlrun.utils.db.BaseModel):
        __tablename__ = f"{table}_tags"
        __table_args__ = (
            UniqueConstraint("project", "name", "obj_id", name=f"_{table}_tags_uc"),
        )

        id = Column(Integer, primary_key=True)
        project = Column(mlrun.db.sql_types.Utf8BinText)
        name = Column(mlrun.db.sql_types.Utf8BinText)

        obj_id = Column(
            Integer,
            ForeignKey(
                f"{table}.id",
                name=f"_{table}_tags_obj_id_fk",
                ondelete="CASCADE",
            ),
            nullable=True,
        )

        parent_rel = relationship(
            parent_cls,
            back_populates="tags",
            passive_deletes=True,
        )

    return Tag


class TagMixin:
    __tablename__ = None

    @declared_attr
    def Tag(cls):  # noqa: N805 N802
        return make_tag(cls)

    @declared_attr
    def tags(cls):  # noqa: N805
        return relationship(
            cls.Tag,
            back_populates="parent_rel",
            cascade="all, delete-orphan",
            passive_deletes=True,
        )


def make_tag_v2(parent_cls):
    table = parent_cls.__tablename__

    class TagV2(Base, mlrun.utils.db.BaseModel):
        __tablename__ = f"{table}_tags"
        __table_args__ = (
            UniqueConstraint("project", "name", "obj_name", name=f"_{table}_tags_uc"),
        )

        id = Column(Integer, primary_key=True)
        project = Column(mlrun.db.sql_types.Utf8BinText)
        name = Column(mlrun.db.sql_types.Utf8BinText)
        obj_id = Column(
            Integer,
            ForeignKey(f"{table}.id", ondelete="CASCADE"),
            nullable=True,
        )
        obj_name = Column(mlrun.db.sql_types.Utf8BinText)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}"

        parent_rel = relationship(
            parent_cls,
            back_populates="tags",
            passive_deletes=True,
        )

    return TagV2


class TagV2Mixin:
    __tablename__ = None

    @declared_attr
    def Tag(cls):  # noqa: N805 N802
        return make_tag_v2(cls)

    @declared_attr
    def tags(cls):  # noqa: N805
        return relationship(
            cls.Tag,
            back_populates="parent_rel",
            cascade="all, delete-orphan",
            passive_deletes=True,
        )


def make_artifact_tag(cls):
    """
    For artifacts, we cannot use tag_v2 because different artifacts with the same key can have the same tag.
    therefore we need to use the obj_id as the unique constraint.
    """
    table = cls.__tablename__  # "artifacts_v2"

    class ArtifactTag(Base, mlrun.utils.db.BaseModel):
        __tablename__ = f"{table}_tags"
        __table_args__ = (
            UniqueConstraint("project", "name", "obj_id", name=f"_{table}_tags_uc"),
            Index(
                f"idx_{table}_tags_project_name_obj_name",
                "project",
                "name",
                "obj_name",
            ),
            ForeignKeyConstraint(
                ["obj_id"],
                [f"{table}.id"],
                name="artifacts_v2_tags_ibfk_1",
                ondelete="CASCADE",
            ),
        )

        id = Column(Integer, primary_key=True)
        project = Column(mlrun.db.sql_types.Utf8BinText)
        name = Column(mlrun.db.sql_types.Utf8BinText)
        obj_id = Column(
            Integer,
            nullable=True,
        )
        obj_name = Column(mlrun.db.sql_types.Utf8BinText)

        parent_rel = relationship(
            cls,
            back_populates="tags",
            passive_deletes=True,
        )

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}"

    return ArtifactTag


class ArtifactTagMixin:
    __tablename__ = None

    @declared_attr
    def Tag(cls):  # noqa: N805 N802
        return make_artifact_tag(cls)

    @declared_attr
    def tags(cls):  # noqa: N805
        return relationship(
            cls.Tag,
            back_populates="parent_rel",
            cascade="all, delete-orphan",
            passive_deletes=True,
        )


def make_notification(cls):
    table = cls.__tablename__

    class Notification(Base, mlrun.utils.db.BaseModel):
        __tablename__ = f"{table}_notifications"
        __table_args__ = (
            UniqueConstraint("name", "parent_id", name=f"_{table}_notifications_uc"),
        )

        id = Column(Integer, primary_key=True)
        project = Column(mlrun.db.sql_types.Utf8BinText)
        name = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)
        kind = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)
        message = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)
        severity = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)
        when = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)
        condition = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)
        secret_params = Column("secret_params", JSON)
        params = Column("params", JSON)
        parent_id = Column(
            Integer,
            ForeignKey(f"{table}.id", ondelete="CASCADE"),
            nullable=False,
        )
        parent_rel = relationship(cls, back_populates="notifications")

        # TODO: Separate table for notification state.
        #   Currently, we are only supporting one notification being sent per DB row (either on completion or on error).
        #   In the future, we might want to support multiple notifications per DB row, and we might want to support on
        #   start, therefore we need to separate the state from the notification itself (e.g. this table can be  table
        #   with notification_id, state, when, last_sent, etc.). This will require some refactoring in the code.
        sent_time = Column(mlrun.db.sql_types.DateTime, nullable=True)
        status = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)
        reason = Column(mlrun.db.sql_types.Utf8BinText, nullable=True)

    return Notification


class NotificationMixin:
    __tablename__ = None

    @declared_attr
    def Notification(cls):  # noqa: N805 N802
        return make_notification(cls)

    @declared_attr
    def notifications(cls):  # noqa: N805
        return relationship(
            cls.Notification,
            back_populates="parent_rel",
            cascade="all, delete-orphan",
            passive_deletes=True,
        )


class LabelMixin:
    __tablename__ = None

    @declared_attr
    def Label(cls):  # noqa: N805 N802
        return make_label(cls)

    @declared_attr
    def labels(cls):  # noqa: N805
        return relationship(
            cls.Label,
            back_populates="parent_rel",
            cascade="all, delete-orphan",
            passive_deletes=True,
        )


# quell SQLAlchemy warnings on duplicate class name (Label)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # deprecated, use ArtifactV2 instead
    # TODO: Remove once data migration v5 is obsolete and add schema migration to remove this table
    class Artifact(Base, LabelMixin, TagMixin, mlrun.utils.db.HasStruct):
        __tablename__ = "artifacts"
        __table_args__ = (
            UniqueConstraint("uid", "project", "key", name="_artifacts_uc"),
        )

        id = Column(Integer, primary_key=True)
        key = Column(mlrun.db.sql_types.Utf8BinText)
        project = Column(mlrun.db.sql_types.Utf8BinText)
        uid = Column(mlrun.db.sql_types.Utf8BinText)
        updated = Column(mlrun.db.sql_types.DateTime)
        # TODO: change to JSON, see mlrun/common/schemas/function.py::FunctionState for reasoning
        body = Column(mlrun.db.sql_types.Blob)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.key}/{self.uid}"

    class ArtifactV2(Base, LabelMixin, ArtifactTagMixin, mlrun.utils.db.BaseModel):
        __tablename__ = "artifacts_v2"
        __table_args__ = (
            UniqueConstraint("uid", "project", "key", name="_artifacts_v2_uc"),
            # Used when enriching workflow status with run artifacts. See https://iguazio.atlassian.net/browse/ML-6770
            Index(
                "idx_artifacts_producer_id_best_iteration_and_project",
                "project",
                "producer_id",
                "best_iteration",
            ),
            # Used to speed up querying artifact tags which is frequently done by UI with project and category.
            # See https://iguazio.atlassian.net/browse/ML-7266
            Index(
                "idx_project_kind",
                "project",
                "kind",
            ),
            # Used for calculating the project counters more efficiently.
            # See https://iguazio.atlassian.net/browse/ML-8556
            Index("idx_project_kind_key", "project", "kind", "key"),
            # Used explicitly in list_artifacts, as most of the queries request best_iteration, and all always sort by
            # updated. See https://iguazio.atlassian.net/browse/ML-9189
            Index("idx_project_bi_updated", "project", "best_iteration", "updated"),
        )

        id = Column(Integer, primary_key=True)
        key = Column(mlrun.db.sql_types.Utf8BinText, index=True)
        project = Column(mlrun.db.sql_types.Utf8BinText)
        kind = Column(mlrun.db.sql_types.Utf8BinText, index=True)
        producer_id = Column(mlrun.db.sql_types.Utf8BinText)
        producer_uri = Column(mlrun.db.sql_types.Utf8BinText)
        iteration = Column(Integer)
        best_iteration = Column(BOOLEAN, default=False, index=True)
        uid = Column(mlrun.db.sql_types.Utf8BinText)
        parent_id = Column(
            Integer,
            ForeignKey("artifacts_v2.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        )
        created = Column(
            mlrun.db.sql_types.DateTime,
            default=lambda: datetime.now(timezone.utc),
        )
        updated = Column(
            mlrun.db.sql_types.DateTime,
            default=lambda: datetime.now(timezone.utc),
        )
        _full_object = Column("object", mlrun.db.sql_types.Blob)
        parent = relationship(
            "ArtifactV2",
            remote_side=[id],
            backref="child_artifacts",
            passive_deletes=True,
        )

        @property
        def full_object(self):
            if self._full_object:
                artifact_struct = pickle.loads(self._full_object)

                # These fields are saved in full_object as timestamps with fsp=6, while the corresponding columns
                # in the database have fsp=3. Since 'ORDER BY' is applied to the column, we return the value from
                # the column (not from the full_object) to ensure the ordering is correct.
                # In SQLite, the updated and created columns return timestamps with fsp=6.
                artifact_struct["metadata"]["updated"] = mlrun.utils.format_datetime(
                    self.updated
                )
                artifact_struct["metadata"]["created"] = mlrun.utils.format_datetime(
                    self.created
                )
                return artifact_struct

        @full_object.setter
        def full_object(self, value):
            self._full_object = pickle.dumps(value)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.key}/{self.uid}"

    class Function(Base, LabelMixin, TagV2Mixin, mlrun.utils.db.HasStruct):
        __tablename__ = "functions"
        __table_args__ = (
            UniqueConstraint("name", "project", "uid", name="_functions_uc"),
            Index("idx_project_state", "project", "state"),
        )

        id = Column(Integer, primary_key=True)
        name = Column(mlrun.db.sql_types.Utf8BinText)
        project = Column(mlrun.db.sql_types.Utf8BinText)
        uid = Column(mlrun.db.sql_types.Utf8BinText)
        kind = Column(mlrun.db.sql_types.Utf8BinText)
        state = Column(mlrun.db.sql_types.Utf8BinText)
        # TODO: change to JSON, see mlrun/common/schemas/function.py::FunctionState for reasoning
        body = Column(mlrun.db.sql_types.Blob)
        updated = Column(mlrun.db.sql_types.DateTime)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}/{self.uid}"

    class Run(Base, LabelMixin, TagMixin, NotificationMixin, mlrun.utils.db.HasStruct):
        __tablename__ = "runs"
        __table_args__ = (
            UniqueConstraint("uid", "project", "iteration", name="_runs_uc"),
            Index("idx_runs_project_id", "id", "project", unique=True),
        )

        id = Column(Integer, primary_key=True)
        uid = Column(mlrun.db.sql_types.Utf8BinText)
        project = Column(mlrun.db.sql_types.Utf8BinText)
        name = Column(mlrun.db.sql_types.Utf8BinText, default="no-name")
        iteration = Column(Integer)
        state = Column(mlrun.db.sql_types.Utf8BinText)
        # TODO: change to JSON, see mlrun/common/schemas/function.py::FunctionState for reasoning
        body = Column(mlrun.db.sql_types.Blob)
        start_time = Column(mlrun.db.sql_types.DateTime)
        end_time = Column(mlrun.db.sql_types.MicroSecondDateTime)
        updated = Column(mlrun.db.sql_types.DateTime, default=datetime.utcnow)
        # requested logs column indicates whether logs were requested for this run
        # None - old runs prior to the column addition, logs were already collected for them, so no need to collect them
        # False - logs were not requested for this run
        # True - logs were requested for this run
        requested_logs = Column(BOOLEAN, default=False, index=True)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.uid}/{self.iteration}"

    class BackgroundTask(
        Base,
        mlrun.utils.db.BaseModel,
    ):
        __tablename__ = "background_tasks"
        __table_args__ = (
            UniqueConstraint("name", "project", name="_background_tasks_uc"),
        )

        id = Column(Integer, primary_key=True)
        name = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)
        project = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)
        created = Column(
            mlrun.db.sql_types.DateTime,
            default=lambda: datetime.now(timezone.utc),
        )
        updated = Column(
            mlrun.db.sql_types.DateTime,
            default=lambda: datetime.now(timezone.utc),
            onupdate=lambda: datetime.now(timezone.utc),
        )
        state = Column(mlrun.db.sql_types.Utf8BinText, index=True)
        error = Column(mlrun.db.sql_types.Utf8BinText)
        timeout = Column(Integer)

        labels = relationship(
            "BackgroundTaskLabel",
            back_populates="task",
            cascade="all, delete-orphan",
            passive_deletes=True,
        )

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}"

    class BackgroundTaskLabel(Base):
        __tablename__ = "background_task_labels"
        __table_args__ = (
            UniqueConstraint(
                "task_id", "name", name="uq_bg_task_labels_task_id_and_name"
            ),
        )

        id = Column(Integer, primary_key=True)
        task_id = Column(
            Integer,
            ForeignKey("background_tasks.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
        name = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)
        value = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)

        task = relationship(
            "BackgroundTask",
            back_populates="labels",
        )
        project = association_proxy("task", "project")

    class Schedule(Base, LabelMixin, mlrun.utils.db.BaseModel):
        __tablename__ = "schedules_v2"
        __table_args__ = (UniqueConstraint("project", "name", name="_schedules_v2_uc"),)

        id = Column(Integer, primary_key=True)
        project = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)
        name = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)
        kind = Column(mlrun.db.sql_types.Utf8BinText)
        desired_state = Column(mlrun.db.sql_types.Utf8BinText)
        state = Column(mlrun.db.sql_types.Utf8BinText)
        creation_time = Column(mlrun.db.sql_types.DateTime)
        cron_trigger_str = Column(mlrun.db.sql_types.Utf8BinText)
        last_run_uri = Column(mlrun.db.sql_types.Utf8BinText)
        # TODO: change to JSON, see mlrun/common/schemas/function.py::FunctionState for reasoning
        struct = Column(mlrun.db.sql_types.Blob)

        concurrency_limit = Column(Integer, nullable=False)
        next_run_time = Column(mlrun.db.sql_types.DateTime)

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
        name = Column(mlrun.db.sql_types.Utf8BinText)

        def get_identifier_string(self) -> str:
            return f"{self.name}"

    class Project(Base, LabelMixin, mlrun.utils.db.BaseModel):
        __tablename__ = "projects"
        # For now since we use project name a lot
        __table_args__ = (UniqueConstraint("name", name="_projects_uc"),)

        id = Column(Integer, primary_key=True)
        name = Column(mlrun.db.sql_types.Utf8BinText)
        description = Column(mlrun.db.sql_types.Utf8BinText)
        owner = Column(mlrun.db.sql_types.Utf8BinText)
        source = Column(mlrun.db.sql_types.Utf8BinText)
        # the attribute name used to be _spec which is just a wrong naming, the attribute was renamed to _full_object
        # leaving the column as is to prevent redundant migration
        # TODO: change to JSON, see mlrun/common/schemas/function.py::FunctionState for reasoning
        _full_object = Column("spec", mlrun.db.sql_types.Blob)
        created = Column(mlrun.db.sql_types.DateTime, default=datetime.utcnow)
        default_function_node_selector = Column("default_function_node_selector", JSON)
        state = Column(mlrun.db.sql_types.Utf8BinText)
        users = relationship(User, secondary=project_users)

        def get_identifier_string(self) -> str:
            return f"{self.name}"

        @property
        def full_object(self):
            if self._full_object:
                return pickle.loads(self._full_object)

        @full_object.setter
        def full_object(self, value):
            self._full_object = pickle.dumps(value)

    class Feature(Base, LabelMixin, mlrun.utils.db.BaseModel):
        __tablename__ = "features"
        id = Column(Integer, primary_key=True)
        feature_set_id = Column(
            Integer, ForeignKey("feature_sets.id", ondelete="CASCADE")
        )

        name = Column(mlrun.db.sql_types.Utf8BinText)
        value_type = Column(mlrun.db.sql_types.Utf8BinText)

        feature_set = relationship(
            "FeatureSet",
            back_populates="features",
        )

        def get_identifier_string(self) -> str:
            return f"{self.feature_set_id}/{self.name}"

    class Entity(Base, LabelMixin, mlrun.utils.db.BaseModel):
        __tablename__ = "entities"
        id = Column(Integer, primary_key=True)
        feature_set_id = Column(
            Integer, ForeignKey("feature_sets.id", ondelete="CASCADE")
        )

        name = Column(mlrun.db.sql_types.Utf8BinText)
        value_type = Column(mlrun.db.sql_types.Utf8BinText)

        feature_set = relationship(
            "FeatureSet",
            back_populates="entities",
        )

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}"

    class FeatureSet(Base, LabelMixin, TagV2Mixin, mlrun.utils.db.BaseModel):
        __tablename__ = "feature_sets"
        __table_args__ = (
            UniqueConstraint("name", "project", "uid", name="_feature_set_uc"),
        )

        id = Column(Integer, primary_key=True)
        name = Column(mlrun.db.sql_types.Utf8BinText)
        project = Column(mlrun.db.sql_types.Utf8BinText)
        created = Column(
            mlrun.db.sql_types.DateTime,
            default=lambda: datetime.now(timezone.utc),
        )
        updated = Column(
            mlrun.db.sql_types.DateTime,
            default=lambda: datetime.now(timezone.utc),
        )
        state = Column(mlrun.db.sql_types.Utf8BinText)
        uid = Column(mlrun.db.sql_types.Utf8BinText)

        _full_object = Column("object", JSON)

        features = relationship(
            Feature,
            cascade="all, delete-orphan",
            back_populates="feature_set",
            passive_deletes=True,
        )
        entities = relationship(
            Entity,
            cascade="all, delete-orphan",
            back_populates="feature_set",
            passive_deletes=True,
        )

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}/{self.uid}"

        @property
        def full_object(self):
            if self._full_object:
                return json.loads(self._full_object)

        @full_object.setter
        def full_object(self, value):
            # TODO - convert to pickle, to avoid issues with non-json serializable fields such as datetime
            self._full_object = json.dumps(value, default=str)

    class FeatureVector(Base, LabelMixin, TagV2Mixin, mlrun.utils.db.BaseModel):
        __tablename__ = "feature_vectors"
        __table_args__ = (
            UniqueConstraint("name", "project", "uid", name="_feature_vectors_uc"),
        )

        id = Column(Integer, primary_key=True)
        name = Column(mlrun.db.sql_types.Utf8BinText)
        project = Column(mlrun.db.sql_types.Utf8BinText)
        created = Column(
            mlrun.db.sql_types.DateTime,
            default=lambda: datetime.now(timezone.utc),
        )
        updated = Column(
            mlrun.db.sql_types.DateTime,
            default=lambda: datetime.now(timezone.utc),
        )
        state = Column(mlrun.db.sql_types.Utf8BinText)
        uid = Column(mlrun.db.sql_types.Utf8BinText)

        _full_object = Column("object", JSON)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}/{self.uid}"

        @property
        def full_object(self):
            if self._full_object:
                return json.loads(self._full_object)

        @full_object.setter
        def full_object(self, value):
            # TODO - convert to pickle, to avoid issues with non-json serializable fields such as datetime
            self._full_object = json.dumps(value, default=str)

    class HubSource(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "hub_sources"
        __table_args__ = (UniqueConstraint("name", name="_hub_sources_uc"),)

        id = Column(Integer, primary_key=True)
        name = Column(mlrun.db.sql_types.Utf8BinText)
        index = Column(Integer)
        created = Column(
            mlrun.db.sql_types.DateTime,
            default=lambda: datetime.now(timezone.utc),
        )
        updated = Column(
            mlrun.db.sql_types.DateTime,
            default=lambda: datetime.now(timezone.utc),
        )

        _full_object = Column("object", JSON)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}"

        @property
        def full_object(self):
            if self._full_object:
                return json.loads(self._full_object)

        @full_object.setter
        def full_object(self, value):
            # TODO - convert to pickle, to avoid issues with non-json serializable fields such as datetime
            self._full_object = json.dumps(value, default=str)

    class DataVersion(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "data_versions"
        __table_args__ = (UniqueConstraint("version", name="_versions_uc"),)

        id = Column(Integer, primary_key=True)
        version = Column(mlrun.db.sql_types.Utf8BinText)
        created = Column(
            mlrun.db.sql_types.DateTime,
            default=lambda: datetime.now(timezone.utc),
        )

        def get_identifier_string(self) -> str:
            return f"{self.version}"

    class DatastoreProfile(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "datastore_profiles"
        __table_args__ = (
            UniqueConstraint("name", "project", name="_datastore_profiles_uc"),
        )

        id = Column(Integer, primary_key=True)
        name = Column(mlrun.db.sql_types.Utf8BinText)
        project = Column(mlrun.db.sql_types.Utf8BinText)
        type = Column(mlrun.db.sql_types.Utf8BinText)
        _full_object = Column("object", JSON)

        @property
        def full_object(self):
            if self._full_object:
                return json.loads(self._full_object)

        @full_object.setter
        def full_object(self, value):
            self._full_object = json.dumps(value, default=str)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}"

    class PaginationCache(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "pagination_cache"

        key = Column(mlrun.db.sql_types.Utf8BinText, primary_key=True)
        user = Column(mlrun.db.sql_types.Utf8BinText)
        function = Column(mlrun.db.sql_types.Utf8BinText)
        current_page = Column(Integer)
        page_size = Column(Integer)
        kwargs = Column(JSON)
        last_accessed = Column(
            mlrun.db.sql_types.DateTime,  # TODO: change to `datetime`, see ML-6921
            default=lambda: datetime.now(timezone.utc),
        )

        def get_identifier_string(self) -> str:
            return f"{self.key}"

    class AlertState(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "alert_states"
        __table_args__ = (UniqueConstraint("parent_id", name="_alert_state_parent_uc"),)

        id = Column(Integer, primary_key=True)
        count = Column(Integer)
        created = Column(
            mlrun.db.sql_types.DateTime,  # TODO: change to `datetime`, see ML-6921
            default=lambda: datetime.now(timezone.utc),
        )
        last_updated = Column(
            mlrun.db.sql_types.DateTime,  # TODO: change to `datetime`, see ML-6921
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

        def get_identifier_string(self) -> str:
            return f"{self.id}"

    class AlertConfig(Base, NotificationMixin, mlrun.utils.db.BaseModel):
        __tablename__ = "alert_configs"
        __table_args__ = (
            UniqueConstraint("project", "name", name="_alert_configs_uc"),
        )

        id = Column(Integer, primary_key=True)
        name = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)
        project = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)

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
        name = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)

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

    class AlertActivation(Base):
        __tablename__ = "alert_activations"

        # partition setup at import
        _interval_name = os.getenv("PARTITION_INTERVAL", "YEARWEEK").upper()
        if not mlrun.common.schemas.partition.PartitionInterval.is_valid(
            _interval_name
        ):
            raise ValueError(
                f"Partition interval must be one of: "
                f"{mlrun.common.schemas.partition.PartitionInterval.valid_intervals()}"
            )
        _interval = mlrun.common.schemas.partition.PartitionInterval(_interval_name)
        _expr = _interval.get_partition_expression(column_name="activation_time")
        _pname, _pval = _interval.get_partition_info(datetime.utcnow())[0]

        __table_args__ = (
            PrimaryKeyConstraint("id", "activation_time", name="_alert_activation_uc"),
            Index("ix_alert_activation_project_name", "project", "name"),
            Index(
                "ix_alert_activation_project_activation_time",
                "project",
                "activation_time",
            ),
            {
                "mysql_engine": "InnoDB",
                "mysql_charset": "utf8mb4",
                "mysql_partition_by": f"RANGE ({_expr})",
                "mysql_partition_options": f"(PARTITION p{_pname} VALUES LESS THAN ({_pval}))",
                "postgresql_partition_by": f"RANGE ({_expr})",
            },
        )

        id = Column(Integer, autoincrement=True, primary_key=True)
        # Keep fsp=3 for activation_time as it is part of the primary key and partitioning logic,
        # ensuring stable indexing and avoiding potential inconsistencies.
        # This must remain unchanged to maintain compatibility with existing logic
        # and prevent unintended precision changes.
        activation_time = Column(
            mlrun.db.sql_types.DateTime(timezone=True), nullable=False
        )
        name = Column(mlrun.db.sql_types.Utf8BinText(), nullable=False)
        project = Column(mlrun.db.sql_types.Utf8BinText(), nullable=False)
        data = Column(JSON)
        entity_id = Column(mlrun.db.sql_types.Utf8BinText(), nullable=False)
        entity_kind = Column(mlrun.db.sql_types.Utf8BinText(), nullable=False)
        event_kind = Column(mlrun.db.sql_types.Utf8BinText(), nullable=False)
        severity = Column(mlrun.db.sql_types.Utf8BinText(), nullable=False)
        number_of_events = Column(Integer, nullable=False)

        # Similarly, keep fsp=3 for reset_time to ensure consistency with activation_time
        # and maintain compatibility with the existing system behavior.
        reset_time = Column(mlrun.db.sql_types.DateTime(timezone=True), nullable=True)

        def get_identifier_string(self) -> str:
            return f"{self.project}/{self.name}/{self.id}"

    class ProjectSummary(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "project_summaries"
        __table_args__ = (UniqueConstraint("project", name="_project_summaries_uc"),)

        id = Column(Integer, primary_key=True)
        project = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)
        updated = Column(mlrun.db.sql_types.MicroSecondDateTime)
        summary = Column(JSON)

        def get_identifier_string(self) -> str:
            return f"{self.project}"

    class TimeWindowTracker(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "time_window_trackers"

        key = Column(mlrun.db.sql_types.Utf8BinText, primary_key=True)
        timestamp = Column(
            mlrun.db.sql_types.MicroSecondDateTime,
            nullable=False,
            default=lambda: datetime.now(timezone.utc),
        )
        max_window_size_seconds = Column(Integer)

        def get_identifier_string(self) -> str:
            return f"{self.key}"

    class ModelEndpoint(Base, LabelMixin, TagV2Mixin, mlrun.utils.db.HasStruct):
        __tablename__ = "model_endpoints"

        id = Column(Integer, primary_key=True)
        uid = Column(
            mlrun.db.sql_types.UuidType, default=lambda: uuid.uuid4().hex, unique=True
        )
        name = Column(mlrun.db.sql_types.Utf8BinText)
        endpoint_type = Column(Integer, nullable=False)
        project = Column(mlrun.db.sql_types.Utf8BinText)
        body = Column(mlrun.db.sql_types.Blob)
        created = Column(
            mlrun.db.sql_types.DateTime,
            default=lambda: datetime.now(timezone.utc),
        )
        updated = Column(
            mlrun.db.sql_types.DateTime,
            default=lambda: datetime.now(timezone.utc),
        )
        function_id = Column(
            Integer,
            ForeignKey("functions.id", ondelete="SET NULL"),
            nullable=True,
        )
        function = relationship(Function)

        model_id = Column(
            Integer,
            ForeignKey("artifacts_v2.id"),
            nullable=True,
        )
        model = relationship(ArtifactV2)

        def get_identifier_string(self) -> str:
            return f"{self.project}_{self.name}_{self.created}"

    class SystemMetadata(Base, mlrun.utils.db.BaseModel):
        __tablename__ = "system_metadata"
        __table_args__ = (UniqueConstraint("key", name="_system_metadata_uc"),)

        id = Column(Integer, primary_key=True)
        key = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)
        # This column stores a string value, when extracting or manipulating it, ensure to handle it appropriately
        value = Column(mlrun.db.sql_types.Utf8BinText, nullable=False)

        def get_identifier_string(self) -> str:
            return f"{self.key}"


@event.listens_for(AlertActivation.__table__, "before_create")
def _disable_autoinc_on_sqlite(table, connection, **kw):
    if connection.dialect.name.startswith(mlrun.common.db.dialects.Dialects.SQLITE):
        # disable SQLAlchemy's AUTOINCREMENT flag
        table.c.id.autoincrement = False


@event.listens_for(AlertActivation, "before_insert")
def _sqlite_autoincrement(
    mapper: Mapper, connection: Connection, target: AlertActivation
) -> None:
    if (
        connection.dialect.name.startswith(mlrun.common.db.dialects.Dialects.SQLITE)
        and target.id is None
    ):
        next_id: int = connection.execute(
            text("SELECT COALESCE(MAX(id),0) + 1 FROM alert_activations")
        ).scalar_one()
        target.id = next_id


def get_partitioned_table_names():
    return [
        AlertActivation.__tablename__,
    ]


# Must be after all table definitions
post_table_definitions(base_cls=Base)
