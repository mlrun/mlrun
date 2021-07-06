import abc
import datetime
import typing

import mergedeep

import mlrun.api.schemas
from mlrun.utils import logger


class Member(abc.ABC):
    @abc.abstractmethod
    def create_project(
        self,
        session: str,
        project: mlrun.api.schemas.Project,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.api.schemas.Project, bool]:
        pass

    @abc.abstractmethod
    def store_project(
        self,
        session: str,
        name: str,
        project: mlrun.api.schemas.Project,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.api.schemas.Project, bool]:
        pass

    @abc.abstractmethod
    def delete_project(
        self,
        session: str,
        name: str,
        deletion_strategy: mlrun.api.schemas.DeletionStrategy = mlrun.api.schemas.DeletionStrategy.default(),
        wait_for_completion: bool = True,
    ) -> bool:
        pass

    @abc.abstractmethod
    def list_projects(
        self, session: str, updated_after: typing.Optional[datetime.datetime] = None,
    ) -> typing.Tuple[
        typing.List[mlrun.api.schemas.Project], typing.Optional[datetime.datetime]
    ]:
        pass

    @abc.abstractmethod
    def get_project(self, session: str, name: str,) -> mlrun.api.schemas.Project:
        pass

    def patch_project(
        self,
        session: str,
        name: str,
        project: dict,
        patch_mode: mlrun.api.schemas.PatchMode = mlrun.api.schemas.PatchMode.replace,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.api.schemas.Project, bool]:
        logger.debug("Patching project in leader", name=name, project=project)
        current_project = self.get_project(session, name)
        strategy = patch_mode.to_mergedeep_strategy()
        current_project_dict = current_project.dict(exclude_unset=True)
        mergedeep.merge(current_project_dict, project, strategy=strategy)
        patched_project = mlrun.api.schemas.Project(**current_project_dict)
        return self.store_project(session, name, patched_project, wait_for_completion)
