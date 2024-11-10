# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional


def packages(exclude_packages: Optional[list[str]] = None) -> list[str]:
    """Get list of project packages"""
    _exclude_packages = set(exclude_packages or [])
    all_packages = _flatten_packages(
        _get_package_dict(f"{os.path.dirname(__file__)}/mlrun"), parent_key="mlrun"
    )
    return list(sorted(all_packages.difference(_exclude_packages)))


def _get_package_dict(starting_path, exclude: Optional[list[str]] = None) -> dict:
    """Get hierarchical dict of packages from starting path"""
    package_dict = {}
    exclude = exclude or ["__pycache__"]

    for dir_path, dir_names, _ in os.walk(starting_path):
        key_path = dir_path.replace(starting_path, "")
        sub_package_dict = package_dict
        for sub_package in key_path.split("/"):
            if sub_package and sub_package not in exclude:
                sub_package_dict = sub_package_dict[sub_package]

        for dir_name in dir_names:
            if dir_name not in exclude:
                sub_package_dict[dir_name] = {}

    return package_dict


def _flatten_packages(
    package_dict: dict, parent_key: str = "", sep: str = "."
) -> set[str]:
    """Flatten hierarchical dict of packages to set of packages"""
    items = [parent_key] if parent_key else []
    for key, value in package_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        items.append(new_key)
        if isinstance(value, dict) and value:
            items.extend(_flatten_packages(value, new_key, sep=sep))

    return set(items)
