# Copyright 2024 Iguazio
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

import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.utils.singleton
import server.api.utils.singletons.db


class FunctionImageCache(metaclass=mlrun.utils.singleton.Singleton):
    @staticmethod
    def store_function_image_cache_record(
        session: sqlalchemy.orm.Session,
        function_name: str,
        image: str,
        mlrun_version: str = None,
        nuclio_version: str = None,
        base_image: str = None,
    ):
        db = server.api.utils.singletons.db.get_db()
        return db.store_function_image_cache_record(
            session, function_name, image, mlrun_version, nuclio_version, base_image
        )

    @staticmethod
    def get_function_image_cache_record(
        session: sqlalchemy.orm.Session,
        function_name: str,
        image: str = None,
        mlrun_version: str = None,
        nuclio_version: str = None,
        base_image: str = None,
    ):
        db = server.api.utils.singletons.db.get_db()
        return db.get_function_image_cache_record(
            session, function_name, image, mlrun_version, nuclio_version, base_image
        )

    @staticmethod
    def list_function_image_cache_records(
        session: sqlalchemy.orm.Session,
        function_name: str,
        image: str,
        mlrun_version: str = None,
        nuclio_version: str = None,
        base_image: str = None,
        as_query: bool = False,
    ):
        db = server.api.utils.singletons.db.get_db()
        return db.list_function_image_cache_records(
            session,
            function_name,
            image,
            mlrun_version,
            nuclio_version,
            base_image,
            as_query,
        )

    @staticmethod
    def delete_function_image_cache_record(
        session: sqlalchemy.orm.Session,
        function_name: str,
        image: str,
        mlrun_version: str = None,
        nuclio_version: str = None,
        base_image: str = None,
    ):
        db = server.api.utils.singletons.db.get_db()
        db.delete_function_image_cache_record(
            session, function_name, image, mlrun_version, nuclio_version, base_image
        )
