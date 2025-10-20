import os
import pandas as pd
import boto3
from io import StringIO
from typing import Union, Dict, Optional

from dataload.interfaces.storage_loader import StorageLoaderInterface
from dataload.config import logger
from dataload.domain.entities import DBOperationError


class S3Loader(StorageLoaderInterface):
    """Loads CSV files from AWS S3."""

    def __init__(self):
        self.s3 = boto3.client("s3")

    def load_csv(self, uri: str) -> pd.DataFrame:
        if not uri.startswith("s3://"):
            raise ValueError("URI must start with s3://")

        try:
            bucket, key = uri[5:].split("/", 1)
            obj = self.s3.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
            logger.info(f"Loaded CSV from S3: {bucket}/{key}, rows={len(df)}")
            return df
        except Exception as e:
            logger.error(f"S3 load error for {uri}: {e}")
            raise DBOperationError(f"Failed to load CSV from S3: {e}")

    def load_json(self, source: Union[str, dict], config: Optional[Dict] = None) -> pd.DataFrame:
        """Load JSON data from S3 and return a pandas DataFrame.
        
        Note: This is a basic implementation for backward compatibility.
        For full API/JSON functionality, use APIJSONStorageLoader.
        
        Args:
            source (Union[str, dict]): S3 URI for JSON file (s3://bucket/key)
            config (Optional[Dict]): Not used in this basic implementation
            
        Returns:
            pd.DataFrame: DataFrame containing the JSON data
            
        Raises:
            ValueError: If source is not a valid S3 URI or not supported
            DBOperationError: If S3 operation fails
        """
        if isinstance(source, dict):
            # Convert dict to DataFrame directly
            try:
                df = pd.json_normalize(source)
                logger.info(f"Loaded JSON from dict, rows={len(df)}")
                return df
            except Exception as e:
                logger.error(f"JSON dict load error: {e}")
                raise DBOperationError(f"Failed to load JSON from dict: {e}")
        
        if not isinstance(source, str) or not source.startswith("s3://"):
            raise ValueError("S3Loader only supports S3 URIs (s3://) or dict objects for JSON loading")

        try:
            bucket, key = source[5:].split("/", 1)
            obj = self.s3.get_object(Bucket=bucket, Key=key)
            json_content = obj["Body"].read().decode("utf-8")
            df = pd.read_json(StringIO(json_content))
            logger.info(f"Loaded JSON from S3: {bucket}/{key}, rows={len(df)}")
            return df
        except Exception as e:
            logger.error(f"S3 JSON load error for {source}: {e}")
            raise DBOperationError(f"Failed to load JSON from S3: {e}")


class LocalLoader(StorageLoaderInterface):
    """Loads CSV files from the local filesystem."""

    def __init__(self):
        pass

    def load_csv(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise ValueError(f"Local file not found: {path}")

        try:
            df = pd.read_csv(path)
            logger.info(f"Loaded local CSV: {path}, rows={len(df)}")
            return df
        except Exception as e:
            logger.error(f"Local load error for {path}: {e}")
            raise DBOperationError(f"Failed to load local CSV: {e}")

    def load_json(self, source: Union[str, dict], config: Optional[Dict] = None) -> pd.DataFrame:
        """Load JSON data from local file or dict and return a pandas DataFrame.
        
        Note: This is a basic implementation for backward compatibility.
        For full API/JSON functionality, use APIJSONStorageLoader.
        
        Args:
            source (Union[str, dict]): Local file path for JSON file or dict object
            config (Optional[Dict]): Not used in this basic implementation
            
        Returns:
            pd.DataFrame: DataFrame containing the JSON data
            
        Raises:
            ValueError: If source format is invalid
            FileNotFoundError: If JSON file cannot be found
            DBOperationError: If JSON loading fails
        """
        if isinstance(source, dict):
            # Convert dict to DataFrame directly
            try:
                df = pd.json_normalize(source)
                logger.info(f"Loaded JSON from dict, rows={len(df)}")
                return df
            except Exception as e:
                logger.error(f"JSON dict load error: {e}")
                raise DBOperationError(f"Failed to load JSON from dict: {e}")
        
        if not isinstance(source, str):
            raise ValueError("LocalLoader only supports file paths (str) or dict objects for JSON loading")
        
        if source.startswith(('http://', 'https://')):
            raise ValueError("LocalLoader does not support API URLs. Use APIJSONStorageLoader for API endpoints.")

        if not os.path.exists(source):
            raise FileNotFoundError(f"Local JSON file not found: {source}")

        try:
            df = pd.read_json(source)
            logger.info(f"Loaded local JSON: {source}, rows={len(df)}")
            return df
        except Exception as e:
            logger.error(f"Local JSON load error for {source}: {e}")
            raise DBOperationError(f"Failed to load local JSON: {e}")
