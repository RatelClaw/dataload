import json
import boto3
from botocore.exceptions import ClientError
from typing import List, Optional, Dict, Any
from dataload.interfaces.embedding_provider import EmbeddingProviderInterface
from dataload.config import (
    AWS_REGION,
    EMBEDDING_MODEL,
    CONTENT_TYPE,
    DEFAULT_VECTOR_VALUE,
    DEFAULT_DIMENSION,
    logger,
)
from dataload.embedding_config import BedrockEmbeddingConfig, create_embedding_config


class BedrockEmbeddingProvider(EmbeddingProviderInterface):
    """Bedrock embedding provider."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Initialize configuration with defaults
        self.config: BedrockEmbeddingConfig = create_embedding_config("bedrock", config)
        
        self.client = self._create_bedrock_client()
        logger.info(f"Initialized Bedrock provider with model: {self.config.model_id}, dimension: {self.config.dimension}")

    def _create_bedrock_client(self):
        """Create Bedrock client."""
        try:
            region = self.config.region or AWS_REGION
            return boto3.client("bedrock-runtime", region_name=region)
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {str(e)}")
            raise

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = []
        for text in texts:
            emb = self._create_description_embedding(text)
            embeddings.append(emb)
        return embeddings

    def _create_description_embedding(self, desc: str) -> list:
        """Create text embeddings."""
        if not desc or not isinstance(desc, str):
            logger.warning(f"Invalid description: {desc}")
            return [DEFAULT_VECTOR_VALUE] * self.config.dimension
        try:
            payload = {"inputText": desc}
            body = json.dumps(payload)
            response = self.client.invoke_model(
                body=body,
                modelId=self.config.model_id,
                accept=self.config.content_type,
                contentType=self.config.content_type,
            )
            response_body = json.loads(response.get("body").read())
            embedding = response_body.get(
                "embedding", [DEFAULT_VECTOR_VALUE] * self.config.dimension
            )
            
            # Validate dimension
            if len(embedding) != self.config.dimension:
                logger.warning(
                    f"Bedrock model {self.config.model_id} returned dimension {len(embedding)}. Expected {self.config.dimension}."
                )
            
            return embedding
        except ClientError as e:
            logger.error(f"Error creating embedding: {str(e)}")
            return [DEFAULT_VECTOR_VALUE] * self.config.dimension
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode Bedrock response JSON: {str(e)}", exc_info=True
            )
            return [DEFAULT_VECTOR_VALUE] * self.config.dimension
        except Exception as e:
            logger.error(
                f"Unexpected error while creating embedding: {str(e)}", exc_info=True
            )
            return [DEFAULT_VECTOR_VALUE] * self.config.dimension
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.config.dimension
