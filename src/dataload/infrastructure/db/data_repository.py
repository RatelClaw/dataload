from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from asyncpg.exceptions import PostgresError
from dataload.domain.entities import (
    TableSchema,
    DBOperationError,
    DataValidationError,
)
from dataload.infrastructure.db.db_connection import DBConnection
from dataload.config import logger, DEFAULT_DIMENSION
from dataload.embedding_config import VectorStoreConfig, create_vector_store_config
from tenacity import retry, stop_after_attempt, wait_fixed
import json


class DataRepositoryInterface(ABC):
    @abstractmethod
    async def get_table_schema(self, table_name: str) -> TableSchema:
        pass

    @abstractmethod
    async def insert_data(
        self, table_name: str, df: pd.DataFrame, pk_columns: List[str]
    ):
        pass

    @abstractmethod
    async def update_data(
        self, table_name: str, df: pd.DataFrame, pk_columns: List[str]
    ):
        pass

    @abstractmethod
    async def set_inactive(
        self, table_name: str, pks: List[tuple], pk_columns: List[str]
    ):
        pass

    @abstractmethod
    async def get_active_data(
        self, table_name: str, columns: List[str]
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    async def get_embed_columns_names(self, table_name: str) -> List[str]:
        pass

    @abstractmethod
    async def get_data_columns(self, table_name: str) -> List[str]:
        pass

    @abstractmethod
    async def create_table(
        self,
        table_name: str,
        df: pd.DataFrame,
        pk_columns: List[str],
        embed_type: str = "combined",
        embed_columns_names: List[str] = [],
    ) -> Dict[str, str]:
        pass

    @abstractmethod
    async def add_column(self, table_name: str, column_name: str, column_type: str):
        pass


class PostgresDataRepository(DataRepositoryInterface):
    """Postgres implementation of data repository using asyncpg."""

    EXTRA_COLUMNS = [
        "embed_columns_names",
        "embed_columns_value",
        "embeddings",
        "is_active",
    ]

    def __init__(self, db_connection: DBConnection, config: Optional[Dict[str, any]] = None):
        self.db = db_connection
        # Initialize configuration with defaults
        self.config: VectorStoreConfig = create_vector_store_config("postgres", config)
        logger.info(f"Initialized PostgresDataRepository with dimension: {self.config.dimension}, index_type: {self.config.index_type}")
    
    def _get_distance_operator(self, distance_metric: str) -> str:
        """Get the appropriate distance operator for pgvector."""
        operators = {
            "cosine": "vector_cosine_ops",
            "euclidean": "vector_l2_ops", 
            "dot_product": "vector_ip_ops"
        }
        return operators.get(distance_metric, "vector_cosine_ops")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def get_table_schema(self, table_name: str) -> TableSchema:
        query = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = $1;
        """
        async with self.db.get_connection() as conn:
            rows = await conn.fetch(query, table_name)
        if not rows:
            raise DBOperationError(f"Table {table_name} not found")
        columns = {row["column_name"]: row["data_type"] for row in rows}
        nullables = {row["column_name"]: row["is_nullable"] == "YES" for row in rows}
        return TableSchema(columns=columns, nullables=nullables)

    async def create_table(
        self,
        table_name: str,
        df: pd.DataFrame,
        pk_columns: List[str],
        embed_type: str = "combined",
        embed_columns_names: List[str] = [],
    ) -> Dict[str, str]:
        """Create table based on DataFrame schema and additional columns, return column types."""
        # Validate primary key columns
        if not all(col in df.columns for col in pk_columns):
            raise DataValidationError(
                f"Primary key columns {pk_columns} not in DataFrame columns {list(df.columns)}"
            )
        for col in pk_columns:
            if df[col].isnull().any():
                raise DataValidationError(
                    f"Primary key column {col} contains null values"
                )

        pd_to_pg = {
            "object": "text",
            "float64": "double precision",
            "float32": "double precision",
            "int64": "bigint",
            "int32": "integer",
            "bool": "boolean",
            "datetime64": "timestamp",
            "timedelta64": "interval",
        }
        column_types = {}
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            # Check if column contains lists (e.g., project_type, source_urls)
            if df[col].apply(lambda x: isinstance(x, (list, dict, np.ndarray))).any():
                pg_type = "jsonb"
            else:
                # Force text for ID columns
                if "id" in col.lower() or "index" in col.lower():
                    pg_type = "text"
                else:
                    # For object columns, check if they are numeric-like
                    non_null = df[col].dropna()
                    if dtype == "object" and len(non_null) > 0:
                        try:
                            if (
                                non_null.apply(
                                    lambda x: isinstance(x, (str, float))
                                    and str(x).replace(".", "").isdigit()
                                ).all()
                                and not non_null.apply(
                                    lambda x: isinstance(x, str) and len(x) > 20
                                ).any()
                            ):
                                pg_type = (
                                    "bigint"
                                    if non_null.astype(float).max() > 2**31 - 1
                                    else "integer"
                                )
                            else:
                                pg_type = "text"
                        except (ValueError, TypeError):
                            pg_type = "text"
                    else:
                        pg_type = pd_to_pg.get(dtype, "text")
            not_null = " NOT NULL" if col in pk_columns else ""
            columns.append(f'"{col}" {pg_type}{not_null}')
            column_types[col] = pg_type
        # Add embed_columns_names always
        columns = [c.strip() for c in columns if c and c.strip()]
        columns.append("embed_columns_names jsonb")
        column_types["embed_columns_names"] = "jsonb"
        if embed_type == "combined":
            columns.append("embed_columns_value text")
            columns.append(f"embeddings vector({self.config.dimension})")
            column_types["embed_columns_value"] = "text"
            column_types["embeddings"] = f"vector({self.config.dimension})"
        elif embed_type == "separated":
            for col in embed_columns_names:
                enc_col = f"{col}_enc"
                columns.append(f'"{enc_col}" vector({self.config.dimension})')
                column_types[enc_col] = f"vector({self.config.dimension})"
        columns.append("is_active boolean DEFAULT true")
        column_types["is_active"] = "boolean"
        # Add primary key constraint
        quoted_pk_columns = [f'"{col}"' for col in pk_columns]
        columns.append(f"PRIMARY KEY ({', '.join(quoted_pk_columns)})")

        # dataload/infrastructure/db/data_repository.py (Inside create_table method)

        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
        try:
            async with self.db.get_connection() as conn:
                async with conn.transaction():
                    await conn.execute(query)

                    # Create vector indexes based on configuration
                    if embed_type == "combined":
                        embed_cols = ["embeddings"]
                    else:
                        embed_cols = [f"{col}_enc" for col in embed_columns_names]

                    for ec in embed_cols:
                        if self.config.index_type == "ivfflat":
                            # Use IVFFlat index for high dimensions or when explicitly configured
                            distance_op = self._get_distance_operator(self.config.distance_metric)
                            index_query = f"""
                                CREATE INDEX IF NOT EXISTS idx_{table_name}_{ec} 
                                ON {table_name} 
                                USING ivfflat ("{ec}" {distance_op}) 
                                WITH (lists = {self.config.ivfflat_lists})
                            """
                        else:
                            # Use HNSW index (limited to 2000 dimensions)
                            if self.config.dimension > 2000:
                                logger.warning(f"HNSW index doesn't support dimensions > 2000. Using IVFFlat instead.")
                                distance_op = self._get_distance_operator(self.config.distance_metric)
                                index_query = f"""
                                    CREATE INDEX IF NOT EXISTS idx_{table_name}_{ec} 
                                    ON {table_name} 
                                    USING ivfflat ("{ec}" {distance_op}) 
                                    WITH (lists = {self.config.ivfflat_lists})
                                """
                            else:
                                distance_op = self._get_distance_operator(self.config.distance_metric)
                                index_query = f"""
                                    CREATE INDEX IF NOT EXISTS idx_{table_name}_{ec} 
                                    ON {table_name} 
                                    USING hnsw ("{ec}" {distance_op}) 
                                    WITH (m = {self.config.hnsw_m}, ef_construction = {self.config.hnsw_ef_construction})
                                """
                        
                        await conn.execute(index_query)

            return column_types
        except PostgresError as e:
            logger.error(f"Table creation error: {e}")
            raise DBOperationError(f"Failed to create table {table_name}: {e}")
        # query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
        # try:
        #     async with self.db.get_connection() as conn:
        #         async with conn.transaction():
        #             await conn.execute(query)
        #             # Create hnsw indexes for vector columns
        #             if embed_type == "combined":
        #                 embed_cols = ["embeddings"]
        #             else:
        #                 embed_cols = [f"{col}_enc" for col in embed_columns_names]
        #             for ec in embed_cols:
        #                 index_query = f'CREATE INDEX IF NOT EXISTS idx_{table_name}_{ec} ON {table_name} USING hnsw ("{ec}" vector_cosine_ops) WITH (m = 16, ef_construction = 64)'
        #                 await conn.execute(index_query)
        #     return column_types
        # except PostgresError as e:
        #     logger.error(f"Table creation error: {e}")
        #     raise DBOperationError(f"Failed to create table {table_name}: {e}")

    async def add_column(self, table_name: str, column_name: str, column_type: str):
        query = f'ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS "{column_name}" {column_type}'
        try:
            async with self.db.get_connection() as conn:
                async with conn.transaction():
                    await conn.execute(query)
        except PostgresError as e:
            logger.error(f"Add column error: {e}")
            raise DBOperationError(
                f"Failed to add column {column_name} to {table_name}: {e}"
            )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def insert_data(
        self, table_name: str, df: pd.DataFrame, pk_columns: List[str]
    ):
        if df.empty:
            return
        pandas_columns = list(df.columns)
        sql_columns = [f'"{col}"' for col in pandas_columns]
        # Get column types from schema (or created table)
        try:
            schema = await self.get_table_schema(table_name)
            column_types = schema.columns
            nullables = schema.nullables
        except DBOperationError:
            raise DBOperationError(
                f"Table {table_name} must be created before insertion"
            )

        # Convert DataFrame values to match Postgres types
        df_converted = df.copy()
        for col in pandas_columns:
            pg_type = column_types.get(col, "text")
            if pg_type in ("integer", "bigint"):
                df_converted[col] = pd.to_numeric(df_converted[col], errors="coerce")
                if nullables.get(col, True):
                    df_converted[col] = df_converted[col].apply(
                        lambda x: int(x) if pd.notnull(x) else None
                    )
                else:
                    df_converted[col] = df_converted[col].fillna(0).apply(int)
            elif pg_type == "double precision":
                df_converted[col] = pd.to_numeric(df_converted[col], errors="coerce")
                if nullables.get(col, True):
                    df_converted[col] = df_converted[col].apply(
                        lambda x: float(x) if pd.notnull(x) else None
                    )
                else:
                    df_converted[col] = df_converted[col].fillna(0.0).apply(float)
            elif pg_type == "timestamp":
                df_converted[col] = pd.to_datetime(df_converted[col], errors="coerce")
                df_converted[col] = df_converted[col].apply(
                    lambda x: x if pd.notnull(x) else None
                )
            elif pg_type == "boolean":
                df_converted[col] = df_converted[col].apply(
                    lambda x: bool(x) if pd.notnull(x) else None
                )
            elif pg_type == "text":
                df_converted[col] = df_converted[col].apply(
                    lambda x: str(x) if pd.notnull(x) else None
                )
            elif pg_type.startswith("vector("):
                df_converted[col] = df_converted[col].apply(
                    lambda x: [float(v) for v in x] if x is not None else None
                )
            elif pg_type == "jsonb":
                df_converted[col] = df_converted[col].apply(
                    lambda x: (
                        json.dumps(x.tolist() if isinstance(x, np.ndarray) else x)
                        if isinstance(x, (list, dict, np.ndarray))
                        else (json.dumps(x) if pd.notnull(x) else None)
                    )
                )

        values = [tuple(row) for row in df_converted.itertuples(index=False, name=None)]
        query = f"INSERT INTO {table_name} ({', '.join(sql_columns)}) VALUES ({', '.join(f'${i+1}' for i in range(len(sql_columns)))})"
        try:
            async with self.db.get_connection() as conn:
                async with conn.transaction():
                    await conn.executemany(query, values)
        except PostgresError as e:
            logger.error(f"Data insertion error: {e}")
            raise DBOperationError(f"Failed to insert data into {table_name}: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def update_data(
        self, table_name: str, df: pd.DataFrame, pk_columns: List[str]
    ):
        if df.empty:
            return
        pandas_columns = list(df.columns)
        sql_columns = [f'"{col}"' for col in pandas_columns]
        sql_pk_columns = [f'"{col}"' for col in pk_columns]
        set_clause = ", ".join(
            [
                f'"{col}" = EXCLUDED."{col}"'
                for col in pandas_columns
                if col not in pk_columns
            ]
        )
        query = f"""
        INSERT INTO {table_name} ({', '.join(sql_columns)})
        VALUES ({', '.join(f'${i+1}' for i in range(len(sql_columns)))})
        ON CONFLICT ({', '.join(sql_pk_columns)}) DO UPDATE SET {set_clause}
        """
        # Convert DataFrame values as in insert_data
        schema = await self.get_table_schema(table_name)
        column_types = schema.columns
        nullables = schema.nullables
        df_converted = df.copy()
        for col in pandas_columns:
            pg_type = column_types.get(col, "text")
            if pg_type in ("integer", "bigint"):
                df_converted[col] = pd.to_numeric(df_converted[col], errors="coerce")
                if nullables.get(col, True):
                    df_converted[col] = df_converted[col].apply(
                        lambda x: int(x) if pd.notnull(x) else None
                    )
                else:
                    df_converted[col] = df_converted[col].fillna(0).apply(int)
            elif pg_type == "double precision":
                df_converted[col] = pd.to_numeric(df_converted[col], errors="coerce")
                if nullables.get(col, True):
                    df_converted[col] = df_converted[col].apply(
                        lambda x: float(x) if pd.notnull(x) else None
                    )
                else:
                    df_converted[col] = df_converted[col].fillna(0.0).apply(float)
            elif pg_type == "timestamp":
                df_converted[col] = pd.to_datetime(df_converted[col], errors="coerce")
                df_converted[col] = df_converted[col].apply(
                    lambda x: x if pd.notnull(x) else None
                )
            elif pg_type == "jsonb":
                df_converted[col] = df_converted[col].apply(
                    lambda x: (
                        json.dumps(x.tolist() if isinstance(x, np.ndarray) else x)
                        if isinstance(x, (list, dict, np.ndarray))
                        else (json.dumps(x) if pd.notnull(x) else None)
                    )
                )
            elif pg_type == "boolean":
                df_converted[col] = df_converted[col].apply(
                    lambda x: bool(x) if pd.notnull(x) else None
                )
            elif pg_type == "text":
                df_converted[col] = df_converted[col].apply(
                    lambda x: str(x) if pd.notnull(x) else None
                )
            elif pg_type.startswith("vector("):
                df_converted[col] = df_converted[col].apply(
                    lambda x: [float(v) for v in x] if x is not None else None
                )

        values = [tuple(row) for row in df_converted.itertuples(index=False, name=None)]
        async with self.db.get_connection() as conn:
            async with conn.transaction():
                await conn.executemany(query, values)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def set_inactive(
        self, table_name: str, pks: List[tuple], pk_columns: List[str]
    ):
        if not pks:
            return
        sql_pk_columns = [f'"{col}"' for col in pk_columns]
        num_pk_fields = len(pk_columns)
        placeholders = [
            f'({", ".join(f"${i + j * num_pk_fields}" for i in range(1, num_pk_fields + 1))})'
            for j in range(len(pks))
        ]
        query = f"UPDATE {table_name} SET is_active = FALSE WHERE ({', '.join(sql_pk_columns)}) IN (VALUES {', '.join(placeholders)}) AND is_active = TRUE"
        params = [val for pk in pks for val in pk]
        try:
            async with self.db.get_connection() as conn:
                async with conn.transaction():
                    await conn.execute(query, *params)
        except PostgresError as e:
            logger.error(f"Set inactive error: {e}")
            raise DBOperationError(f"Failed to set inactive in {table_name}: {e}")

    async def get_active_data(
        self, table_name: str, columns: List[str]
    ) -> pd.DataFrame:
        sql_columns = [f'"{col}"' for col in columns]
        query = f"SELECT {', '.join(sql_columns)} FROM {table_name} WHERE is_active"
        async with self.db.get_connection() as conn:
            rows = await conn.fetch(query)
        return pd.DataFrame(rows, columns=columns)

    async def get_embed_columns_names(self, table_name: str) -> List[str]:
        query = f"SELECT embed_columns_names FROM {table_name} WHERE is_active LIMIT 1"
        async with self.db.get_connection() as conn:
            row = await conn.fetchrow(query)
        if row:
            return row["embed_columns_names"]  # array as list
        raise DBOperationError(f"No active rows in {table_name}")

    async def get_data_columns(self, table_name: str) -> List[str]:
        schema = await self.get_table_schema(table_name)
        return [
            col
            for col in schema.columns
            if col not in self.EXTRA_COLUMNS and not col.endswith("_enc")
        ]

    # Insert this method into your PostgresDataRepository class
    # in src/dataload/infrastructure/db/data_repository.py

    async def search(
        self,
        table_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        embed_column: str = "embeddings",
        id_column: Optional[str] = None,
    ) -> List[Dict]:
        """Performs vector similarity search using the pgvector <-> operator."""

        # 1. Get relevant columns for the final result/metadata
        try:
            # data_columns excludes 'embeddings', 'embed_columns_names', etc.
            data_columns = await self.get_data_columns(table_name)
        except DBOperationError:
            # Fallback for search on an empty table (less likely in real scenarios)
            data_columns = ["id", "name", "description"]

        # Determine id_column if not provided
        if id_column is None:
            id_candidates = [
                col
                for col in data_columns
                if "id" in col.lower() or "index" in col.lower()
            ]
            id_column = (
                id_candidates[0]
                if id_candidates
                else (data_columns[0] if data_columns else "id")
            )

        # Ensure id_column is included for the result ID
        if id_column not in data_columns:
            data_columns.append(id_column)

        # 2. Construct the SQL query
        # The query retrieves all data columns, plus the calculated distance.
        # The ORDER BY clause is crucial for k-NN search using the '<->' operator.
        data_sql = [f'"{col}"' for col in data_columns]
        query = f"""
        SELECT 
            {', '.join(data_sql)}, 
            "{embed_column}" <-> $1 AS distance
        FROM {table_name}
        WHERE is_active = TRUE
        ORDER BY "{embed_column}" <-> $1
        LIMIT $2
        """

        try:
            async with self.db.get_connection() as conn:
                rows = await conn.fetch(query, query_embedding, top_k)

            results = []
            for row in rows:
                row_dict = dict(row)

                # Construct metadata dictionary by excluding control columns
                metadata = {
                    k: v
                    for k, v in row_dict.items()
                    if k in data_columns and k != id_column
                }

                # Determine the original text column that was embedded
                # For 'description_enc', the document text should come from 'description'
                document_column = (
                    embed_column.replace("_enc", "")
                    if embed_column.endswith("_enc")
                    else "embed_columns_value"
                )
                document_text = row_dict.get(document_column, "N/A")

                results.append(
                    {
                        "id": str(row_dict.get(id_column, "N/A")),
                        "document": document_text,
                        "distance": row_dict.get("distance", -1.0),
                        "metadata": metadata,
                    }
                )
            return results

        except PostgresError as e:
            logger.error(f"Postgres search error in {table_name}: {e}")
            raise DBOperationError(f"Postgres search failed: {e}")
