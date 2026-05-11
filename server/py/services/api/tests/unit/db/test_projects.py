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

import datetime
import unittest.mock

import deepdiff
import pytest

import mlrun
import mlrun.common.formatters
import mlrun.common.schemas
import mlrun.config
import mlrun.errors

import framework.utils.project_formats
from framework.db.sqldb.models import Project
from framework.tests.unit.db.common_fixtures import TestDatabaseBase


class TestProjects(TestDatabaseBase):
    def test_get_project(self):
        project_name = "project-name"
        project_description = "some description"
        project_labels = {
            "some-label": "some-label-value",
        }
        project_default_node_selector = {"gpu": "true"}
        self._db.create_project(
            self._db_session,
            mlrun.common.schemas.Project(
                metadata=mlrun.common.schemas.ProjectMetadata(
                    name=project_name, labels=project_labels
                ),
                spec=mlrun.common.schemas.ProjectSpec(
                    description=project_description,
                    default_function_node_selector=project_default_node_selector,
                ),
            ),
        )

        project_output = self._db.get_project(self._db_session, project_name)
        assert project_output.metadata.name == project_name
        assert project_output.spec.description == project_description
        assert (
            deepdiff.DeepDiff(
                project_default_node_selector,
                project_output.spec.default_function_node_selector,
                ignore_order=True,
            )
            == {}
        )
        assert (
            deepdiff.DeepDiff(
                project_labels,
                project_output.metadata.labels,
                ignore_order=True,
            )
            == {}
        )

    def test_list_project(self):
        expected_projects = [
            {"name": "project-name-1"},
            {"name": "project-name-2", "description": "project-description-2"},
            {"name": "project-name-3", "labels": {"key": "value"}},
            {
                "name": "project-name-4",
                "description": "project-description-4",
                "labels": {"key2": "value2"},
            },
        ]
        for project in expected_projects:
            self._db.create_project(
                self._db_session,
                mlrun.common.schemas.Project(
                    metadata=mlrun.common.schemas.ProjectMetadata(
                        name=project["name"], labels=project.get("labels")
                    ),
                    spec=mlrun.common.schemas.ProjectSpec(
                        description=project.get("description")
                    ),
                ),
            )
        projects_output = self._db.list_projects(self._db_session)
        for index, project in enumerate(projects_output.projects):
            assert project.metadata.name == expected_projects[index]["name"]
            assert project.spec.description == expected_projects[index].get(
                "description"
            )
            assert (
                deepdiff.DeepDiff(
                    expected_projects[index].get("labels"),
                    project.metadata.labels,
                    ignore_order=True,
                )
                == {}
            )

    def test_list_project_minimal(self):
        expected_projects = ["project-name-1", "project-name-2", "project-name-3"]
        for project in expected_projects:
            self._db.create_project(
                self._db_session,
                mlrun.common.schemas.Project(
                    metadata=mlrun.common.schemas.ProjectMetadata(
                        name=project,
                    ),
                    spec=mlrun.common.schemas.ProjectSpec(
                        description="some-proj",
                        artifacts=[{"key": "value"}],
                        workflows=[{"key": "value"}],
                        functions=[{"key": "value"}],
                    ),
                ),
            )
        projects_output = self._db.list_projects(
            self._db_session, format_=mlrun.common.formatters.ProjectFormat.minimal
        )
        for index, project in enumerate(projects_output.projects):
            assert project.metadata.name == expected_projects[index]
            assert project.spec.artifacts is None
            assert project.spec.workflows is None
            assert project.spec.functions is None

        projects_output = self._db.list_projects(self._db_session)
        for index, project in enumerate(projects_output.projects):
            assert project.metadata.name == expected_projects[index]
            assert project.spec.artifacts == [{"key": "value"}]
            assert project.spec.workflows == [{"key": "value"}]
            assert project.spec.functions == [{"key": "value"}]

    def test_list_project_names_filter(self):
        project_names = [
            "project-1",
            "project-2",
            "project-3",
            "project-4",
            "project-5",
        ]
        for project in project_names:
            self._db.create_project(
                self._db_session,
                mlrun.common.schemas.Project(
                    metadata=mlrun.common.schemas.ProjectMetadata(name=project),
                ),
            )
        filter_names = [project_names[0], project_names[3], project_names[4]]
        projects_output = self._db.list_projects(
            self._db_session,
            format_=mlrun.common.formatters.ProjectFormat.name_only,
            names=filter_names,
        )

        assert (
            deepdiff.DeepDiff(
                filter_names,
                projects_output.projects,
                ignore_order=True,
            )
            == {}
        )

        projects_output = self._db.list_projects(
            self._db_session,
            format_=mlrun.common.formatters.ProjectFormat.name_only,
            names=[],
        )

        assert projects_output.projects == []

    def test_list_project_name_and_created_only(self):
        project_names = [
            "project-1",
            "project-2",
        ]
        for project in project_names:
            self._db.create_project(
                self._db_session,
                mlrun.common.schemas.Project(
                    metadata=mlrun.common.schemas.ProjectMetadata(name=project),
                ),
            )
        projects_output = self._db.list_projects(
            self._db_session,
            format_=mlrun.common.formatters.ProjectFormat.name_and_creation_time,
        )
        projects_output_names = [project[0] for project in projects_output.projects]

        assert projects_output_names == project_names

        # Assert creation times
        for project in projects_output.projects:
            project_name, creation_time = project

            # Ensure creation_time is a datetime object
            assert isinstance(creation_time, datetime.datetime)

            # Ensure creation_time is today's date
            assert creation_time.date() == datetime.datetime.today().date()

    def test_create_project(self):
        project = self._generate_project()
        project_summary = self._generate_project_summary()
        self._db.create_project(
            self._db_session,
            project.copy(deep=True),
        )
        self._assert_project(project)
        self._assert_project_summary(project_summary)

    def test_store_project_creation(self):
        project = self._generate_project()
        self._db.store_project(
            self._db_session,
            project.metadata.name,
            project.copy(deep=True),
        )
        self._assert_project(project)

    def test_store_project_update(self):
        project = self._generate_project()
        self._db.create_project(
            self._db_session,
            project.copy(deep=True),
        )

        self._db.store_project(
            self._db_session,
            project.metadata.name,
            mlrun.common.schemas.Project(
                metadata=mlrun.common.schemas.ProjectMetadata(
                    name=project.metadata.name
                ),
            ),
        )
        project_output = self._db.get_project(self._db_session, project.metadata.name)
        assert project_output.metadata.name == project.metadata.name
        assert project_output.spec.description is None
        assert project_output.metadata.labels == {}
        # Created in request body should be ignored and set by the DB layer
        assert project_output.metadata.created != project.metadata.created

    def test_patch_project(self):
        project = self._generate_project()
        self._db.create_project(
            self._db_session,
            project.copy(deep=True),
        )

        patched_project_description = "some description 2"
        patched_project_labels = {
            "some-label": "some-label-value",
        }
        self._db.patch_project(
            self._db_session,
            project.metadata.name,
            {
                "metadata": {
                    "created": project.metadata.created,
                    "labels": patched_project_labels,
                },
                "spec": {"description": patched_project_description},
            },
        )
        project_output = self._db.get_project(self._db_session, project.metadata.name)
        assert project_output.metadata.name == project.metadata.name
        assert project_output.spec.description == patched_project_description
        # Created in request body should be ignored and set by the DB layer
        assert project_output.metadata.created != project.metadata.created
        assert (
            deepdiff.DeepDiff(
                patched_project_labels,
                project_output.metadata.labels,
                ignore_order=True,
            )
            == {}
        )

    def test_delete_project(self):
        project_name = "project-name"
        project_description = "some description"
        self._db.create_project(
            self._db_session,
            mlrun.common.schemas.Project(
                metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
                spec=mlrun.common.schemas.ProjectSpec(description=project_description),
            ),
        )
        self._db.delete_project(self._db_session, project_name)

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self._db.get_project(self._db_session, project_name)

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self._db.get_project_summary(self._db_session, project_name)

    def test_refresh_project_summaries(self):
        project_summaries = [
            self._generate_project_summary("project-summary-1"),
            self._generate_project_summary("project-summary-2"),
        ]

        for summary in project_summaries:
            project = self._generate_project(summary.name)
            self._db.create_project(self._db_session, project)

        # Delete one of the projects without deleting its summary
        with unittest.mock.patch.object(self._db, "_delete_project_summary"):
            self._db.delete_project(self._db_session, "project-summary-2")

        # Create project without project summary
        summary = self._generate_project_summary("project-summary-3")
        project_summaries.append(summary)
        project = self._generate_project(summary.name)
        with unittest.mock.patch.object(self._db, "_append_project_summary"):
            self._db.create_project(self._db_session, project)

        self._db_session.delete = unittest.mock.MagicMock()
        self._db_session.add = unittest.mock.MagicMock()
        self._db_session.commit = unittest.mock.MagicMock()

        self._db.refresh_project_summaries(self._db_session, project_summaries)

        # Assert that 'project-summary-1' was updated
        assert self._db_session.add.call_count == 1
        added_summary = self._db_session.add.call_args[0][0]
        assert added_summary.project == "project-summary-1"

        # Assert that 'project-summary-2' was deleted
        assert self._db_session.delete.call_count == 1
        deleted_summary = self._db_session.delete.call_args[0][0]
        assert deleted_summary.project == "project-summary-2"

    def test_projects_crud(self):
        project = mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(name="p1"),
            spec=mlrun.common.schemas.ProjectSpec(
                description="banana", other_field="value"
            ),
            status=mlrun.common.schemas.ObjectStatus(state="active"),
        )
        self._db.create_project(self._db_session, project)
        project_output = self._db.get_project(
            self._db_session, name=project.metadata.name
        )
        assert (
            deepdiff.DeepDiff(
                project.dict(),
                project_output.dict(exclude={"id"}),
                ignore_order=True,
            )
            == {}
        )

        project_patch = {"spec": {"description": "lemon"}}
        self._db.patch_project(self._db_session, project.metadata.name, project_patch)
        project_output = self._db.get_project(
            self._db_session, name=project.metadata.name
        )
        assert project_output.spec.description == project_patch["spec"]["description"]

        project_2 = mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(name="p2"),
        )
        self._db.create_project(self._db_session, project_2)
        projects_output = self._db.list_projects(
            self._db_session, format_=mlrun.common.formatters.ProjectFormat.name_only
        )
        assert [
            project.metadata.name,
            project_2.metadata.name,
        ] == projects_output.projects

    def test_list_projects_custom_selection_name_and_owner(self):
        """
        Verify that ProjectFormatCustomSelection queries only the requested
        columns from the DB and returns a minimal Project schema with
        metadata.name and spec.owner — without loading the full pickle blob.
        """
        project_name = "custom-sel-project"
        owner = "the-owner"
        self._db.create_project(
            self._db_session,
            mlrun.common.schemas.Project(
                metadata=mlrun.common.schemas.ProjectMetadata(
                    name=project_name,
                    labels={"env": "test"},
                ),
                spec=mlrun.common.schemas.ProjectSpec(
                    description="heavy description",
                    owner=owner,
                    artifacts=[{"key": "value"}],
                    workflows=[{"key": "value"}],
                    functions=[{"key": "value"}],
                ),
            ),
        )

        custom_format = framework.utils.project_formats.ProjectFormatCustomSelection(
            [
                framework.utils.project_formats.ProjectFormatCustom.name,
                framework.utils.project_formats.ProjectFormatCustom.owner,
            ]
        )
        projects_output = self._db.list_projects(
            self._db_session,
            format_=custom_format,
            names=[project_name],
        )

        assert len(projects_output.projects) == 1
        project = projects_output.projects[0]

        # Requested columns are populated
        assert project.metadata.name == project_name
        assert project.spec.owner == owner

        # Fields NOT in the custom selection should be empty/default —
        # proving we didn't deserialize the full pickle blob
        assert project.spec.description is None or project.spec.description == ""
        assert not project.spec.artifacts
        assert not project.spec.workflows
        assert not project.spec.functions
        assert not project.metadata.labels

    def test_list_projects_custom_selection_filters_by_name(self):
        """Verify names= filter works with custom selection format."""
        for name in ["proj-a", "proj-b", "proj-c"]:
            self._db.create_project(
                self._db_session,
                mlrun.common.schemas.Project(
                    metadata=mlrun.common.schemas.ProjectMetadata(name=name),
                    spec=mlrun.common.schemas.ProjectSpec(owner=f"owner-{name}"),
                ),
            )

        custom_format = framework.utils.project_formats.ProjectFormatCustomSelection(
            [
                framework.utils.project_formats.ProjectFormatCustom.name,
                framework.utils.project_formats.ProjectFormatCustom.owner,
            ]
        )
        projects_output = self._db.list_projects(
            self._db_session,
            format_=custom_format,
            names=["proj-a", "proj-c"],
        )

        returned_names = {p.metadata.name for p in projects_output.projects}
        assert returned_names == {"proj-a", "proj-c"}

    def test_list_stale_projects_empty(self):
        output = self._db.list_stale_projects(
            self._db_session,
            format_=mlrun.common.formatters.ProjectFormat.name_only,
        )
        assert output.projects == []

    def test_list_stale_projects_phase_none_excluded(self):
        # Very old, state matches a TTL bucket, but phase IS NULL → not stale.
        mlrun.mlconf.httpdb.projects.stale_resource_ttl_create = "60s"

        self._seed_stale_project(
            "p-no-phase", state="creating", phase=None, age_seconds=600
        )
        output = self._db.list_stale_projects(
            self._db_session,
            format_=mlrun.common.formatters.ProjectFormat.name_only,
        )
        assert output.projects == []

    def test_list_stale_projects_creating_above_and_below_ttl(self):
        mlrun.mlconf.httpdb.projects.stale_resource_ttl_create = "60s"

        self._seed_stale_project(
            "creating-stale", state="creating", phase=1, age_seconds=600
        )
        self._seed_stale_project(
            "creating-fresh", state="creating", phase=1, age_seconds=5
        )

        output = self._db.list_stale_projects(
            self._db_session,
            format_=mlrun.common.formatters.ProjectFormat.name_only,
        )
        assert output.projects == ["creating-stale"]

        # Default-format smoke check: confirm _full_object deserialization
        # still works on the seeded record (covers format_-passthrough too).
        full_output = self._db.list_stale_projects(
            self._db_session,
            format_=mlrun.common.formatters.ProjectFormat.full,
        )
        assert len(full_output.projects) == 1
        assert full_output.projects[0].metadata.name == "creating-stale"

    def test_list_stale_projects_online_above_and_below_ttl(self):
        mlrun.mlconf.httpdb.projects.stale_resource_ttl_update = "120s"

        self._seed_stale_project(
            "online-stale", state="online", phase=1, age_seconds=600
        )
        self._seed_stale_project("online-fresh", state="online", phase=1, age_seconds=5)

        output = self._db.list_stale_projects(
            self._db_session,
            format_=mlrun.common.formatters.ProjectFormat.name_only,
        )
        assert output.projects == ["online-stale"]

    def test_list_stale_projects_deleting_above_and_below_ttl(self):
        mlrun.mlconf.httpdb.projects.stale_resource_ttl_delete = "300s"

        self._seed_stale_project(
            "deleting-stale", state="deleting", phase=1, age_seconds=1200
        )
        self._seed_stale_project(
            "deleting-fresh", state="deleting", phase=1, age_seconds=5
        )

        output = self._db.list_stale_projects(
            self._db_session,
            format_=mlrun.common.formatters.ProjectFormat.name_only,
        )
        assert output.projects == ["deleting-stale"]

    def test_list_stale_projects_unknown_state_excluded(self):
        # State outside {creating, online, deleting} hits else_=False even
        # with a non-null phase and an updated_at far in the past.
        self._seed_stale_project(
            "p-archived", state="archived", phase=1, age_seconds=10_000
        )
        output = self._db.list_stale_projects(
            self._db_session,
            format_=mlrun.common.formatters.ProjectFormat.name_only,
        )
        assert output.projects == []

    def test_list_stale_projects_state_specific_ttl_not_cross_applied(self):
        # state="deleting", age=200s. ttl_update=120s but ttl_delete=300s.
        # Older than update TTL, newer than delete TTL → not stale.
        mlrun.mlconf.httpdb.projects.stale_resource_ttl_update = "120s"
        mlrun.mlconf.httpdb.projects.stale_resource_ttl_delete = "300s"

        self._seed_stale_project(
            "deleting-young", state="deleting", phase=1, age_seconds=200
        )
        output = self._db.list_stale_projects(
            self._db_session,
            format_=mlrun.common.formatters.ProjectFormat.name_only,
        )
        assert output.projects == []

    def test_list_stale_projects_returns_only_stale_subset(self):
        mlrun.mlconf.httpdb.projects.stale_resource_ttl_create = "60s"
        mlrun.mlconf.httpdb.projects.stale_resource_ttl_update = "120s"

        self._seed_stale_project(
            "stale-creating", state="creating", phase=1, age_seconds=600
        )
        self._seed_stale_project(
            "fresh-creating", state="creating", phase=1, age_seconds=5
        )
        self._seed_stale_project(
            "stale-online", state="online", phase=1, age_seconds=600
        )
        self._seed_stale_project(
            "phase-null", state="deleting", phase=None, age_seconds=10_000
        )
        self._seed_stale_project(
            "archived", state="archived", phase=1, age_seconds=10_000
        )

        output = self._db.list_stale_projects(
            self._db_session,
            format_=mlrun.common.formatters.ProjectFormat.name_only,
        )
        assert set(output.projects) == {"stale-creating", "stale-online"}

    def test_list_stale_projects_format_name_only(self):
        mlrun.mlconf.httpdb.projects.stale_resource_ttl_create = "60s"

        self._seed_stale_project("x", state="creating", phase=1, age_seconds=600)

        output = self._db.list_stale_projects(
            self._db_session,
            format_=mlrun.common.formatters.ProjectFormat.name_only,
        )
        assert all(isinstance(p, str) for p in output.projects)
        assert output.projects == ["x"]

    def _generate_and_insert_pre_060_record(self, project_name: str):
        pre_060_record = Project(name=project_name)
        self._db_session.add(pre_060_record)
        self._db_session.commit()

    @staticmethod
    def _generate_project(name="project-name"):
        return mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(
                name=name,
                created=datetime.datetime.now() - datetime.timedelta(seconds=1),
                labels={
                    "some-label": "some-label-value",
                },
            ),
            spec=mlrun.common.schemas.ProjectSpec(
                description="some description", owner="owner-name"
            ),
        )

    @staticmethod
    def _generate_project_summary(
        project="project-name",
    ) -> mlrun.common.schemas.ProjectSummary:
        return mlrun.common.schemas.ProjectSummary(
            name=project,
            updated=datetime.datetime.now(),
        )

    def _assert_project(
        self,
        expected_project: mlrun.common.schemas.Project,
    ):
        project_output = self._db.get_project(
            self._db_session, expected_project.metadata.name
        )
        assert project_output.metadata.name == expected_project.metadata.name
        assert project_output.spec.description == expected_project.spec.description
        assert project_output.spec.owner == expected_project.spec.owner
        # Created in request body should be ignored and set by the DB layer
        assert project_output.metadata.created != expected_project.metadata.created
        assert (
            deepdiff.DeepDiff(
                expected_project.metadata.labels,
                project_output.metadata.labels,
                ignore_order=True,
            )
            == {}
        )

    def _assert_project_summary(
        self,
        expected_project_summary: mlrun.common.schemas.ProjectSummary,
    ):
        project_summary_output = self._db.get_project_summary(
            self._db_session, expected_project_summary.name
        )
        assert (
            deepdiff.DeepDiff(
                expected_project_summary,
                project_summary_output,
                ignore_order=True,
                exclude_paths="root.updated",
            )
            == {}
        )

    def _seed_stale_project(
        self,
        name: str,
        *,
        state: str | None,
        phase: int | None,
        age_seconds: int,
    ) -> None:
        # Going through create_project populates _full_object/ProjectSummary;
        # state/phase/updated_at are then patched directly because
        # create_project doesn't expose them.
        self._db.create_project(
            self._db_session,
            mlrun.common.schemas.Project(
                metadata=mlrun.common.schemas.ProjectMetadata(name=name),
            ),
        )
        record = self._db_session.query(Project).filter(Project.name == name).one()
        record.state = state
        record.phase = phase
        record.updated_at = datetime.datetime.now(datetime.UTC) - datetime.timedelta(
            seconds=age_seconds
        )
        self._db_session.commit()
