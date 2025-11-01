from abc import ABC, abstractmethod
from typing import Union, Dict, Optional
import pandas as pd


class StorageLoaderInterface(ABC):
    """Abstract interface for loading data from different sources."""

    @abstractmethod
    def load_csv(self, path: str) -> pd.DataFrame:
        """Load a CSV file and return a pandas DataFrame.
        
        Args:
            path (str): Path to the CSV file to load
            
        Returns:
            pd.DataFrame: DataFrame containing the loaded CSV data
            
        Raises:
            FileNotFoundError: If the CSV file cannot be found
            ValueError: If the CSV file cannot be parsed
        """
        pass

    @abstractmethod
    async def load_json(self, source: Union[str, dict], config: Optional[Dict] = None) -> pd.DataFrame:
        """Load JSON data from API, file, or raw data and return a pandas DataFrame.
        
        This method supports multiple JSON data sources:
        - API endpoints (URLs starting with http/https)
        - Local JSON files (file paths)
        - Raw JSON data (dict objects)
        
        Args:
            source (Union[str, dict]): The JSON data source. Can be:
                - URL string for API endpoints
                - File path string for local JSON files  
                - Dict object for raw JSON data
            config (Optional[Dict], optional): Configuration parameters for JSON processing.
                May include authentication, pagination, flattening options, etc.
                Defaults to None.
                
        Returns:
            pd.DataFrame: DataFrame containing the processed JSON data
            
        Raises:
            ValueError: If the source format is invalid or JSON cannot be parsed
            ConnectionError: If API endpoint cannot be reached
            AuthenticationError: If API authentication fails
            FileNotFoundError: If JSON file cannot be found
        """
        pass
