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
import warnings
from copy import copy
from typing import List

import pandas as pd

import mlrun
import mlrun.frameworks

from .artifacts import Artifact, dict_to_artifact
from .config import config
from .render import artifacts_to_html, runs_to_html
from .utils import flatten, get_artifact_target, get_in, is_legacy_artifact

list_header = [
    "project",
    "uid",
    "iter",
    "start",
    "state",
    "name",
    "labels",
    "inputs",
    "parameters",
    "results",
    "artifacts",
    "error",
]

iter_index = list_header.index("iter")
state_index = list_header.index("state")
parameters_index = list_header.index("parameters")
results_index = list_header.index("results")


class RunList(list):
    def to_rows(self, extend_iterations=False):
        """return the run list as flattened rows"""
        rows = []
        for run in self:
            iterations = get_in(run, "status.iterations", "")
            row = [
                get_in(run, "metadata.project", config.default_project),
                get_in(run, "metadata.uid", ""),
                get_in(run, "metadata.iteration", ""),
                get_in(run, "status.start_time", ""),
                get_in(run, "status.state", ""),
                get_in(run, "metadata.name", ""),
                get_in(run, "metadata.labels", ""),
                get_in(run, "spec.inputs", ""),
                get_in(run, "spec.parameters", ""),
                get_in(run, "status.results", ""),
                get_in(run, "status.artifacts", []),
                get_in(run, "status.error", ""),
            ]
            if extend_iterations and iterations:
                parameters_dict = {
                    key[len("param.") :]: i
                    for i, key in enumerate(iterations[0])
                    if key.startswith("param.")
                }
                results_dict = {
                    key[len("output.") :]: i
                    for i, key in enumerate(iterations[0])
                    if key.startswith("output.")
                }
                for iter in iterations[1:]:
                    row[state_index] = iter[0]
                    row[iter_index] = iter[1]
                    row[parameters_index] = {
                        key: iter[col] for key, col in parameters_dict.items()
                    }
                    row[results_index] = {
                        key: iter[col] for key, col in results_dict.items()
                    }
                    rows.append(copy(row))
            else:
                rows.append(row)

        return [list_header] + rows

    def to_df(self, flat=False, extend_iterations=False, cache=True):
        """convert the run list to a dataframe"""
        if hasattr(self, "_df") and cache:
            return self._df
        rows = self.to_rows(extend_iterations=extend_iterations)
        df = pd.DataFrame(rows[1:], columns=rows[0])  # .set_index('iter')
        df["start"] = pd.to_datetime(df["start"])

        if flat:
            df = flatten(df, "labels")
            df = flatten(df, "parameters", "param.")
            df = flatten(df, "results", "output.")
        self._df = df
        return df

    def show(self, display=True, classes=None, short=False, extend_iterations=False):
        """show the run list as a table in Jupyter"""
        html = runs_to_html(
            self.to_df(extend_iterations=extend_iterations),
            display,
            classes=classes,
            short=short,
        )
        if not display:
            return html

    def to_objects(self) -> List["mlrun.RunObject"]:
        """Return a list of Run Objects"""
        return [mlrun.RunObject.from_dict(run) for run in self]

    def compare(
        self,
        hide_identical: bool = True,
        exclude: list = None,
        show: bool = None,
        extend_iterations=True,
        filename=None,
        colorscale: str = None,
    ):
        """return/show parallel coordinates plot + table to compare between the list of runs

        example:

            # return a list of runs in the project, and compare them (show charts)
            runs = project.list_runs(name='download', labels='owner=admin')
            runs.compare()

        :param hide_identical: hide columns with identical values
        :param exclude:        User-provided list of parameters to be excluded from the plot
        :param show:           Allows the user to display the plot within the notebook
        :param extend_iterations: include the iteration (hyper-param) results
        :param filename:       Output filename to save the plot html file
        :param colorscale:     colors used for the lines in the parallel coordinate plot
        :return:  plot html
        """
        return mlrun.frameworks.parallel_coordinates.compare_run_objects(
            self.to_objects(),
            hide_identical=hide_identical,
            exclude=exclude,
            show=show,
            extend_iterations=extend_iterations,
            filename=filename,
            colorscale=colorscale,
        )


class ArtifactList(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.tag = ""

    def to_rows(self):
        """return the artifact list as flattened rows"""
        rows = []
        head = {
            "tree": ["tree", "metadata.tree"],
            "key": ["key", "metadata.key"],
            "iter": ["iter", "metadata.iter"],
            "kind": ["kind", "kind"],
            "path": ["target_path", "spec.target_path"],
            "hash": ["hash", "metadata.hash"],
            "viewer": ["viewer", "spec.viewer"],
            "updated": ["updated", "metadata.updated"],
            "description": ["description", "metadata.description"],
            "producer": ["producer", "spec.producer"],
            "sources": ["sources", "spec.sources"],
            "labels": ["labels", "metadata.labels"],
        }
        for artifact in self:
            fields_index = 0 if is_legacy_artifact(artifact) else 1
            row = [get_in(artifact, v[fields_index], "") for k, v in head.items()]
            rows.append(row)

        return [head.keys()] + rows

    def to_df(self, flat=False):
        """convert the artifact list to a dataframe"""
        rows = self.to_rows()
        df = pd.DataFrame(rows[1:], columns=rows[0])
        df["updated"] = pd.to_datetime(df["updated"])

        if flat:
            df = flatten(df, "producer", "prod_")
            df = flatten(df, "sources", "src_")

        return df

    def show(self, display=True, classes=None):
        """show the artifact list as a table in Jupyter"""
        df = self.to_df()
        if self.tag != "*":
            df.drop("tree", axis=1, inplace=True)
        html = artifacts_to_html(df, display, classes=classes)
        if not display:
            return html

    def to_objects(self) -> List[Artifact]:
        """return as a list of artifact objects"""
        return [dict_to_artifact(artifact) for artifact in self]

    def objects(self) -> List[Artifact]:
        """return as a list of artifact objects"""
        warnings.warn(
            "'objects' is deprecated in 1.3.0 and will be removed in 1.5.0. "
            "Use 'to_objects' instead.",
            # TODO: remove in 1.5.0
            FutureWarning,
        )
        return [dict_to_artifact(artifact) for artifact in self]

    def dataitems(self) -> List["mlrun.DataItem"]:
        """return as a list of DataItem objects"""
        dataitems = []
        for item in self:
            artifact = get_artifact_target(item)
            if artifact:
                dataitems.append(mlrun.get_dataitem(artifact))
        return dataitems


class FunctionList(list):
    def __init__(self):
        pass
        # TODO
