# Copyright 2021 The Feast Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pytz
import pyarrow as pa
from pydantic import StrictStr
from pydantic.schema import Literal

from feast import Entity, FeatureTable
from feast.feature_view import FeatureView
from feast.infra.key_encoding_utils import serialize_entity_key, serialize_entity_keys
from feast.infra.online_stores.online_store import OnlineStore
from feast.protos.feast.types.EntityKey_pb2 import EntityKey as EntityKeyProto
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.repo_config import FeastConfigBaseModel, RepoConfig
from feast.usage import log_exceptions_and_usage, tracing_span
from feast.infra.provider import _get_column_names

class SqliteOnlineStoreConfig(FeastConfigBaseModel):
    """ Online store config for local (SQLite-based) store """

    type: Literal[
        "sqlite", "feast.infra.online_stores.sqlite.SqliteOnlineStore"
    ] = "sqlite"
    """ Online store type selector"""

    path: StrictStr = "data/online.db"
    """ (optional) Path to sqlite db """


class SqliteOnlineStore(OnlineStore):
    """
    OnlineStore is an object used for all interaction between Feast and the service used for offline storage of
    features.
    """

    _conn: Optional[sqlite3.Connection] = None

    @staticmethod
    def _get_db_path(config: RepoConfig) -> str:
        assert (
            config.online_store.type == "sqlite"
            or config.online_store.type.endswith("SqliteOnlineStore")
        )

        if config.repo_path and not Path(config.online_store.path).is_absolute():
            db_path = str(config.repo_path / config.online_store.path)
        else:
            db_path = config.online_store.path
        return db_path

    def _get_conn(self, config: RepoConfig):
        if not self._conn:
            db_path = self._get_db_path(config)
            Path(db_path).parent.mkdir(exist_ok=True)
            self._conn = sqlite3.connect(
                db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
        return self._conn

    @log_exceptions_and_usage(online_store="sqlite")
    def online_write_batch(
        self,
        config: RepoConfig,
        table: Union[FeatureTable, FeatureView],
        entities: List[Entity],
        data: Union[pa.Table, pa.RecordBatch],
        progress: Optional[Callable[[int], Any]],
    ) -> None:
        if hasattr(data, "from_batches"):
            data = data.from_batches([data])

        conn = self._get_conn(config)

        project = config.project

        (
            entity_cols,
            feature_cols,
            event_ts_col,
            created_ts_col,
        ) = _get_column_names(table, entities)

        entity_key_bin = serialize_entity_keys(data.select([entity_cols]))
        features_bin = [batch.serialize() for batch in data.select([feature_cols]).to_batches(1)]
        event_ts = [_to_naive_utc(ts) for ts in data.column(event_ts_col).to_pylist()]
        created_ts = [_to_naive_utc(ts) if ts is not None else ts for ts in data.column(created_ts_col).to_pylist()]

        with conn:
            for entity_key, value, event, created in zip(entity_key_bin, features_bin, event_ts, created_ts):
                conn.execute(
                    f"""
                        UPDATE {_table_id(project, table)}
                        SET value = ?, event_ts = ?, created_ts = ?
                        WHERE (entity_key = ?)
                    """,
                    (
                        # SET
                        value,
                        event,
                        created,
                        # WHERE
                        entity_key,
                    ),
                )

                conn.execute(
                    f"""INSERT OR IGNORE INTO {_table_id(project, table)}
                        (entity_key, value, event_ts, created_ts)
                        VALUES (?, ?, ?, ?)""",
                    (
                        entity_key,
                        value,
                        event,
                        created,
                    ),
                )
            if progress:
                progress(1)

    @log_exceptions_and_usage(online_store="sqlite")
    def online_read(
        self,
        config: RepoConfig,
        table: Union[FeatureTable, FeatureView],
        entity_keys: Union[pa.Table, pa.RecordBatch],
        requested_features: Optional[List[str]] = None,
    ) -> List[Tuple[Optional[datetime], Optional[Dict[str, ValueProto]]]]:
        if hasattr(entity_keys, "from_batches"):
            entity_keys = entity_keys.from_batches([entity_keys])

        conn = self._get_conn(config)
        cur = conn.cursor()

        result: List[Tuple[Optional[datetime], Optional[Dict[str, ValueProto]]]] = []

        with tracing_span(name="remote_call"):
            # Fetch all entities in one go
            cur.execute(
                f"SELECT entity_key, value "
                f"FROM {_table_id(config.project, table)} "
                f"WHERE entity_key IN ({','.join('?' * len(entity_keys))}) "
                f"ORDER BY entity_key",
                [serialize_entity_key(entity_key) for entity_key in entity_keys],
            )
            rows = cur.fetchall()

        rows = {
            k: v for k, v in itertools.groupby(rows, key=lambda r: r[0])
        }
        for entity_key in entity_keys:
            entity_key_bin = serialize_entity_key(entity_key)
            res = {}
            res_ts = None
            for _, feature_name, val_bin, ts in rows.get(entity_key_bin, []):
                val = ValueProto()
                val.ParseFromString(val_bin)
                res[feature_name] = val
                res_ts = ts

            if not res:
                result.append((None, None))
            else:
                result.append((res_ts, res))
        return result

    @log_exceptions_and_usage(online_store="sqlite")
    def update(
        self,
        config: RepoConfig,
        tables_to_delete: Sequence[Union[FeatureTable, FeatureView]],
        tables_to_keep: Sequence[Union[FeatureTable, FeatureView]],
        entities_to_delete: Sequence[Entity],
        entities_to_keep: Sequence[Entity],
        partial: bool,
    ):
        conn = self._get_conn(config)
        project = config.project

        for table in tables_to_keep:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {_table_id(project, table)} (entity_key BLOB,  value BLOB, event_ts timestamp, created_ts timestamp,  PRIMARY KEY(entity_key))"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {_table_id(project, table)}_ek ON {_table_id(project, table)} (entity_key);"
            )

        for table in tables_to_delete:
            conn.execute(f"DROP TABLE IF EXISTS {_table_id(project, table)}")

    def teardown(
        self,
        config: RepoConfig,
        tables: Sequence[Union[FeatureTable, FeatureView]],
        entities: Sequence[Entity],
    ):
        try:
            os.unlink(self._get_db_path(config))
        except FileNotFoundError:
            pass


def _table_id(project: str, table: Union[FeatureTable, FeatureView]) -> str:
    return f"{project}_{table.name}"


def _to_naive_utc(ts: datetime):
    if ts.tzinfo is None:
        return ts
    else:
        return ts.astimezone(pytz.utc).replace(tzinfo=None)
