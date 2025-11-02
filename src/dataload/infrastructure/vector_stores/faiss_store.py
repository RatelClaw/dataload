import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from dataload.interfaces.vector_store import VectorStoreInterface
from dataload.domain.entities import DataValidationError, DBOperationError, TableSchema
from dataload.config import DEFAULT_DIMENSION, logger
from dataload.embedding_config import VectorStoreConfig, create_vector_store_config

# --- Persistence Configuration ---
FAISS_STORAGE_PATH = "./faiss_storage"


class FaissVectorStore(VectorStoreInterface):
    """
    FAISS in-memory vector store implementation with simulated disk persistence.
    Supports combined and separated embeddings for flexible retrieval.
    """

    EXTRA_COLUMNS = [
        "embeddings",
        "embed_columns_value",
        "embed_columns_names",
        "is_active",
    ]

    def __init__(self, persistence_path: str = FAISS_STORAGE_PATH, config: Optional[Dict[str, Any]] = None):
        """Initialize FAISS Vector Store with lazy imports and persistence setup."""
        self.faiss_module = self._lazy_import_faiss()
        
        # Initialize configuration with defaults
        self.config: VectorStoreConfig = create_vector_store_config("faiss", config)
        
        self.persistence_path = persistence_path

        # Internal in-memory structures
        self.indexes: Dict[str, Any] = {}
        self.enc_indexes: Dict[str, Any] = {}
        self.data: Dict[str, "pd.DataFrame"] = {}
        self.schemas: Dict[str, TableSchema] = {}
        self.table_map: Dict[str, str] = {}

        os.makedirs(self.persistence_path, exist_ok=True)
        self._load_all_data()
        
        logger.info(f"Initialized FaissVectorStore with dimension: {self.config.dimension}, index_type: {self.config.faiss_index_type}")

    # --------------------------------------------------------------------------
    # Utility / Lazy Import Helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _lazy_import_faiss():
        try:
            import faiss

            return faiss
        except ImportError as exc:
            raise DBOperationError(
                "FAISS is not installed. Please install with: pip install vector-dataloader[faiss]"
            ) from exc

    def _get_index_file(self, index_key: str) -> str:
        """Get the file path for a FAISS index."""
        return os.path.join(self.persistence_path, f"{index_key}_faiss.bin")

    def _get_data_file(self, table_name: str) -> str:
        """Get the file path for the DataFrame."""
        return os.path.join(self.persistence_path, f"{table_name}_data.csv")

    def _save_index(self, index, index_key: str):
        """Save a FAISS index to disk."""
        self.faiss_module.write_index(index, self._get_index_file(index_key))
        logger.info(f"FAISS index saved to {self._get_index_file(index_key)}")

    def _load_index(self, index_key: str) -> object:
        """Load a FAISS index from disk."""
        file_path = self._get_index_file(index_key)
        if os.path.exists(file_path):
            return self.faiss_module.read_index(file_path)
        raise FileNotFoundError(f"Index file not found for {index_key}")

    def _save_data(self, df: pd.DataFrame, table_name: str):
        """Save a DataFrame (metadata/schema) to disk."""
        # Convert non-primitive types like lists/vectors (embeddings, lists in metadata) to strings/JSON before saving
        df_save = df.copy()
        for col in df_save.columns:
            # Simple check for list/numpy arrays to serialize for CSV
            if (
                df_save[col].dtype == object
                and df_save[col]
                .apply(lambda x: isinstance(x, (list, np.ndarray)))
                .any()
            ):
                df_save[col] = df_save[col].apply(
                    lambda x: (
                        json.dumps(x.tolist())
                        if isinstance(x, np.ndarray)
                        else json.dumps(x)
                    )
                )

        df_save.to_csv(self._get_data_file(table_name), index=False)

    # In FaissVectorStore class in faiss_store.py

    def _load_data(self, table_name: str) -> pd.DataFrame:
        """Load a DataFrame (metadata/schema) from disk, ensuring text columns load correctly."""
        file_path = self._get_data_file(table_name)
        if os.path.exists(file_path):

            # --- FIX: Use dtype=str to load all columns as strings initially ---
            # This prevents pandas from crashing when trying to convert 'name' (a string) to a float.
            df = pd.read_csv(file_path, dtype=str)

            # Deserialize columns that were JSON strings (like embeddings/lists)
            for col in df.columns:
                # Explicitly deserialize embedded columns
                if (
                    col.endswith("_enc")
                    or col == "embeddings"
                    or col == "embed_columns_names"
                ):
                    try:
                        # Load the JSON string back into a list/array
                        df[col] = df[col].apply(
                            lambda x: (
                                np.array(json.loads(x)).astype("float32").tolist()
                                if pd.notnull(x) and x != "nan"
                                else None
                            )
                        )
                    except (
                        json.JSONDecodeError,
                        AttributeError,
                        TypeError,
                        ValueError,
                    ):
                        logger.warning(
                            f"Failed to deserialize embedding column {col} for {table_name}."
                        )
                        # If deserialization fails, set to None to prevent downstream issues
                        df[col] = None

                # Optionally convert primary key 'id' to a numeric type if needed for lookup stability
                # if col == 'id' and 'id' in df.columns:
                #     df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)

            return df

        raise FileNotFoundError(f"Data file not found for {table_name}")

    # In FaissVectorStore class in faiss_store.py

    def _load_all_data(self):
        """Load all data and indexes from disk at startup."""

        # First, load all DataFrames (metadata)
        for filename in os.listdir(self.persistence_path):
            if filename.endswith("_data.csv"):
                table_name = filename.replace("_data.csv", "")
                try:
                    self.data[table_name] = self._load_data(table_name)
                    logger.info(f"Loaded DataFrame for table: {table_name}")
                    self.schemas[table_name] = TableSchema(
                        columns={col: "text" for col in self.data[table_name].columns},
                        nullables={col: True for col in self.data[table_name].columns},
                    )
                except Exception as e:
                    logger.warning(f"Failed to load data for {table_name}: {e}")

        # Second, load all FAISS Indexes
        for filename in os.listdir(self.persistence_path):
            if filename.endswith("_faiss.bin"):
                index_key = filename.replace("_faiss.bin", "")

                # Check for Combined Index (index_key == table_name)
                if index_key in self.data:
                    try:
                        index = self._load_index(index_key)
                        self.indexes[index_key] = (
                            index  # Store in the combined dictionary
                        )
                        logger.info(f"Successfully loaded COMBINED index: {index_key}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to load combined index {index_key}: {e}"
                        )

                # Check for Separated Index (e.g., table_name_col_enc)
                elif index_key.endswith("_enc"):
                    parts = index_key.rsplit("_", 1)
                    table_name = index_key.rsplit("_", 2)[0]  # Extract table name part

                    try:
                        index = self._load_index(index_key)
                        self.enc_indexes[index_key] = (
                            index  # Store in the separated dictionary
                        )
                        self.table_map[index_key] = (
                            table_name  # Map it back to the table name (needed for search)
                        )
                        logger.info(f"Successfully loaded SEPARATED index: {index_key}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to load separated index {index_key}: {e}"
                        )

                else:
                    logger.warning(
                        f"Index {index_key} found but does not match combined or separated format."
                    )
        # ... (create_table, insert_data, and search implementations from the previous response)

    async def get_table_schema(self, table_name: str) -> TableSchema:
        """Retrieve the stored schema for a given table."""
        if table_name in self.schemas:
            return self.schemas[table_name]
        # Raise an error if the table (schema) doesn't exist
        raise DBOperationError(
            f"Table schema for '{table_name}' not found in FaissVectorStore."
        )

    async def create_table(
        self,
        table_name: str,
        df: pd.DataFrame,
        pk_columns: List[str],
        embed_type: str = "combined",
        embed_columns_names: List[str] = [],
    ) -> Dict[str, str]:
        # ... (implementation from previous response)
        faiss = self.faiss_module
        if table_name in self.schemas:
            logger.info(f"Table {table_name} already exists. Skipping creation.")
            return self.schemas[table_name].columns

        if not all(col in df.columns for col in pk_columns):
            raise DataValidationError(
                f"Primary key columns {pk_columns} not in DataFrame"
            )

        column_types = {col: "text" for col in df.columns}
        column_types["embed_columns_names"] = "text[]"
        column_types["is_active"] = "boolean"

        if embed_type == "combined":
            column_types["embed_columns_value"] = "text"
            column_types["embeddings"] = f"vector({self.config.dimension})"
            
            # Create FAISS index based on configuration
            if self.config.faiss_index_type == "IndexFlatIP":
                self.indexes[table_name] = faiss.IndexFlatIP(self.config.dimension)
            else:
                self.indexes[table_name] = faiss.IndexFlatL2(self.config.dimension)
        else:
            for col in embed_columns_names:
                col_enc = f"{col}_enc"
                column_types[col_enc] = f"vector({self.config.dimension})"
                index_key = f"{table_name}_{col_enc}"
                
                # Create FAISS index based on configuration
                if self.config.faiss_index_type == "IndexFlatIP":
                    self.enc_indexes[index_key] = faiss.IndexFlatIP(self.config.dimension)
                else:
                    self.enc_indexes[index_key] = faiss.IndexFlatL2(self.config.dimension)
                    
                self.table_map[index_key] = table_name

        self.schemas[table_name] = TableSchema(
            columns=column_types, nullables={col: True for col in column_types}
        )
        self.data[table_name] = pd.DataFrame(columns=list(column_types.keys()))
        self._save_data(self.data[table_name], table_name)
        return column_types

    async def insert_data(
        self, table_name: str, df: pd.DataFrame, pk_columns: List[str]
    ):
        # ... (implementation from previous response)
        if table_name not in self.data:
            raise DBOperationError(f"Table {table_name} not found")

        if not all(col in df.columns for col in pk_columns):
            raise DataValidationError(f"Primary keys {pk_columns} missing in DataFrame")

        # 1. Update in-memory DataFrame (metadata)
        self.data[table_name] = pd.concat(
            [self.data[table_name], df], ignore_index=True
        ).reset_index(drop=True)
        self._save_data(self.data[table_name], table_name)  # Save the updated data

        # 2. Add embeddings to FAISS
        enc_cols = [col for col in df.columns if col.endswith("_enc")]

        if "embeddings" in df.columns:
            embeddings = np.array(df["embeddings"].tolist()).astype("float32")
            if embeddings.size > 0:
                self.indexes[table_name].add(embeddings)
                self._save_index(self.indexes[table_name], table_name)  # Save index
        elif enc_cols:
            for enc_col in enc_cols:
                index_key = f"{table_name}_{enc_col}"
                if index_key in self.enc_indexes:
                    embeddings = np.array(df[enc_col].tolist()).astype("float32")
                    if embeddings.size > 0:
                        self.enc_indexes[index_key].add(embeddings)
                        self._save_index(
                            self.enc_indexes[index_key], index_key
                        )  # Save index

    async def search(
        self,
        table_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        embed_column: str = None,
    ) -> List[Dict[str, Any]]:
        # ... (implementation from previous response)
        index_key = table_name

        if embed_column and embed_column.endswith("_enc"):
            index_key = f"{table_name}_{embed_column}"
            index = self.enc_indexes.get(index_key)
            data_table_name = table_name
        elif table_name in self.indexes:
            index = self.indexes[table_name]
            data_table_name = table_name
        else:
            raise DBOperationError(f"FAISS index for '{index_key}' not initialized.")

        if not index:
            raise DBOperationError(f"FAISS index for '{index_key}' not initialized.")

        if index.ntotal == 0:
            return []

        query = np.array([query_embedding]).astype("float32")
        distances, indices = index.search(query, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.data[data_table_name]):
                row = self.data[data_table_name].iloc[idx].to_dict()

                filtered_metadata = {
                    k: v
                    for k, v in row.items()
                    if not k.endswith("_enc") and k not in self.EXTRA_COLUMNS
                }

                original_col_name = (
                    embed_column.replace("_enc", "")
                    if embed_column
                    else row.get("embed_columns_value", "")
                )
                document_text = str(row.get(original_col_name, original_col_name))

                results.append(
                    {
                        "id": str(
                            row.get("id", idx)
                        ),  # Assuming 'id' is the PK from your main.py
                        "document": document_text,
                        "metadata": filtered_metadata,
                        "distance": float(dist),
                    }
                )

        return results

    # ---------------------------
    # Implementation of Abstract Methods (TO RESOLVE TypeError)
    # ---------------------------

    async def add_column(self, table_name: str, column_name: str, column_type: str):
        """Add a column to the schema and DataFrame."""
        if table_name not in self.schemas:
            raise DBOperationError(f"Table {table_name} not found")

        self.schemas[table_name].columns[column_name] = column_type
        # Add column to the in-memory DataFrame
        if column_name not in self.data[table_name].columns:
            self.data[table_name][column_name] = None
        # Save the updated data frame to disk
        self._save_data(self.data[table_name], table_name)

    async def get_active_data(
        self, table_name: str, columns: List[str] = None
    ) -> pd.DataFrame:
        """Retrieve active data from in-memory DataFrame."""
        if table_name not in self.data:
            return pd.DataFrame(columns=columns or [])

        active_data = self.data[table_name]
        if "is_active" in active_data.columns:
            active_data = active_data[active_data["is_active"] != False]

        if columns:
            # Filter columns and reset index
            return active_data[
                [col for col in columns if col in active_data.columns]
            ].reset_index(drop=True)
        return active_data.reset_index(drop=True)

    async def get_data_columns(self, table_name: str) -> List[str]:
        """Get data columns excluding extra and embedding columns."""
        if table_name in self.schemas:
            return [
                col
                for col in self.schemas[table_name].columns.keys()
                if col not in self.EXTRA_COLUMNS and not col.endswith("_enc")
            ]
        return []

    async def get_embed_columns_names(self, table_name: str) -> List[str]:
        """Get embedding column names from schema (the actual columns containing the vectors)."""
        if table_name not in self.schemas:
            return []

        # Returns the actual columns containing the vector data
        return [
            c
            for c in self.schemas[table_name].columns.keys()
            if "_enc" in c or c == "embeddings"
        ]

    async def set_inactive(
        self, table_name: str, pks: List[tuple], pk_columns: List[str]
    ):
        """Mark records as inactive by deleting from the data (FAISS removal is complex)."""
        if table_name not in self.data or not pks or not pk_columns:
            return

        pk_col = pk_columns[0]
        pk_values = [pk[0] for pk in pks]

        # Set 'is_active' to False in the data frame
        if "is_active" in self.data[table_name].columns:
            self.data[table_name].loc[
                self.data[table_name][pk_col].isin(pk_values), "is_active"
            ] = False
            # Save the updated data frame to disk
            self._save_data(self.data[table_name], table_name)

        # NOTE: In a production FAISS setup, deletion requires a separate index management strategy
        # (e.g., IndexIDMap + filtering) or full periodic re-indexing. This in-memory
        # implementation only handles the metadata update.

    async def update_data(
        self, table_name: str, df: pd.DataFrame, pk_columns: List[str]
    ):
        """Update data in the DataFrame (metadata only) and re-index the FAISS index if embeddings changed."""
        if table_name not in self.data:
            raise DBOperationError(f"Table {table_name} not found")

        if not pk_columns:
            raise DataValidationError(
                "Primary key columns must be provided for update."
            )

        existing = self.data[table_name]
        pk = pk_columns[0]

        new_rows_df = []
        for _, row in df.iterrows():
            existing_idx = existing[existing[pk] == row[pk]].index

            if not existing_idx.empty:
                # Update existing row (metadata)
                existing.loc[existing_idx, list(row.index)] = row.values
                # NOTE: FAISS vectors are NOT updated here to avoid complex re-indexing.
            else:
                new_rows_df.append(row)

        # Concatenate new rows
        if new_rows_df:
            self.data[table_name] = pd.concat(
                [existing, pd.DataFrame(new_rows_df)], ignore_index=True
            ).reset_index(drop=True)
            # NOTE: If embeddings are present, they should ideally be added to FAISS here,
            # but this simplifies the in-memory update process.

        self.data[table_name] = existing.reset_index(drop=True)
        self._save_data(self.data[table_name], table_name)
