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

from server.api.db.base import DBInterface


def test_function_image_cache_crud(db: DBInterface, db_session: sqlalchemy.orm.Session):
    function_name = "function_name"
    image = "image"
    mlrun_version = "mlrun_version"
    nuclio_version = "nuclio_version"
    base_image = "base_image"

    # store record
    db.store_function_image_cache_record(
        session=db_session,
        function_name=function_name,
        image=image,
        mlrun_version=mlrun_version,
        nuclio_version=nuclio_version,
        base_image=base_image,
    )

    # get record
    record = db.get_function_image_cache_record(
        session=db_session,
        function_name=function_name,
        image=image,
        mlrun_version=mlrun_version,
        nuclio_version=nuclio_version,
        base_image=base_image,
    )
    assert record.function_name == function_name
    assert record.image == image
    assert record.mlrun_version == mlrun_version
    assert record.nuclio_version == nuclio_version
    assert record.base_image == base_image

    # list records
    records = db.list_function_image_cache_records(
        session=db_session,
        function_name=function_name,
        image=image,
        mlrun_version=mlrun_version,
        nuclio_version=nuclio_version,
        base_image=base_image,
    )
    assert len(records) == 1
    assert records[0].function_name == function_name
    assert records[0].image == image
    assert records[0].mlrun_version == mlrun_version
    assert records[0].nuclio_version == nuclio_version
    assert records[0].base_image == base_image


def test_update_function_image_cache_record(
    db: DBInterface, db_session: sqlalchemy.orm.Session
):
    function_name = "function_name"
    image_1 = "image"
    image_2 = "image_2"
    mlrun_version = "mlrun_version"
    nuclio_version = "nuclio_version"
    base_image = "base_image"

    # store record
    db.store_function_image_cache_record(
        session=db_session,
        function_name=function_name,
        image=image_1,
        mlrun_version=mlrun_version,
        nuclio_version=nuclio_version,
        base_image=base_image,
    )

    # update record
    db.store_function_image_cache_record(
        session=db_session,
        function_name=function_name,
        image=image_2,
        mlrun_version=mlrun_version,
        nuclio_version=nuclio_version,
        base_image=base_image,
    )

    # get record
    record = db.get_function_image_cache_record(
        session=db_session,
        function_name=function_name,
        image=image_2,
        mlrun_version=mlrun_version,
        nuclio_version=nuclio_version,
        base_image=base_image,
    )
    assert record.function_name == function_name
    assert record.image == image_2
    assert record.mlrun_version == mlrun_version
    assert record.nuclio_version == nuclio_version
    assert record.base_image == base_image


def test_multiple_images_for_function(
    db: DBInterface, db_session: sqlalchemy.orm.Session
):
    function_name = "function_name"
    function_name_2 = "function_name_2"
    mlrun_version = "mlrun_version"
    nuclio_version = "nuclio_version"
    image_1 = "image_1"
    image_2 = "image_2"
    base_image_1 = "base_image_1"
    base_image_2 = "base_image_2"

    # store record
    db.store_function_image_cache_record(
        session=db_session,
        function_name=function_name,
        image=image_1,
        mlrun_version=mlrun_version,
        nuclio_version=nuclio_version,
        base_image=base_image_1,
    )

    db.store_function_image_cache_record(
        session=db_session,
        function_name=function_name,
        image=image_2,
        mlrun_version=mlrun_version,
        nuclio_version=nuclio_version,
        base_image=base_image_2,
    )

    # store record for another function
    db.store_function_image_cache_record(
        session=db_session,
        function_name=function_name_2,
        image=image_1,
        mlrun_version=mlrun_version,
        nuclio_version=nuclio_version,
        base_image=base_image_1,
    )

    # list records with specific base image
    records = db.list_function_image_cache_records(
        session=db_session,
        function_name=function_name,
        base_image=base_image_1,
    )
    assert len(records) == 1
    assert records[0].image == image_1

    records = db.list_function_image_cache_records(
        session=db_session,
        function_name=function_name,
        base_image=base_image_2,
    )
    assert len(records) == 1
    assert records[0].image == image_2

    # list records
    records = db.list_function_image_cache_records(
        session=db_session,
        function_name=function_name,
    )
    assert len(records) == 2

    records = db.list_function_image_cache_records(
        session=db_session,
        base_image=base_image_1,
    )
    assert len(records) == 2

    records = db.list_function_image_cache_records(
        session=db_session,
    )
    assert len(records) == 3


def test_delete_function_image_cache_records(
    db: DBInterface, db_session: sqlalchemy.orm.Session
):
    function_name = "function_name"
    function_name_2 = "function_name_2"
    mlrun_version = "mlrun_version"
    nuclio_version = "nuclio_version"
    image_1 = "image_1"
    base_image_1 = "base_image_1"

    # store records
    db.store_function_image_cache_record(
        session=db_session,
        function_name=function_name,
        image=image_1,
        mlrun_version=mlrun_version,
        nuclio_version=nuclio_version,
        base_image=base_image_1,
    )

    db.store_function_image_cache_record(
        session=db_session,
        function_name=function_name_2,
        image=image_1,
        mlrun_version=mlrun_version,
        nuclio_version=nuclio_version,
        base_image=base_image_1,
    )

    records = db.list_function_image_cache_records(
        session=db_session,
    )
    assert len(records) == 2

    # delete record
    db.delete_function_image_cache_records(
        session=db_session,
        function_name=function_name,
    )

    records = db.list_function_image_cache_records(
        session=db_session,
    )
    assert len(records) == 1
