import asyncio
from dataload.infrastructure.db.db_connection import DBConnection
from dataload.infrastructure.db.data_repository import PostgresDataRepository
from dataload.infrastructure.vector_stores.chroma_store import ChromaVectorStore
from dataload.infrastructure.vector_stores.faiss_store import FaissVectorStore
from dataload.infrastructure.storage.loaders import LocalLoader
from dataload.application.services.embedding.gemini_provider import (
    GeminiEmbeddingProvider,
)
from dataload.application.use_cases.data_loader_use_case import dataloadUseCase


async def main():
    # Initialize Postgres (optional, if using Postgres instead of Chroma)
    # db_conn = DBConnection()
    # await db_conn.initialize()
    # repo = PostgresDataRepository(db_connection=db_conn)

    # Initialize Chroma vector store (persistent or in-memory)
    repo = ChromaVectorStore(
        mode="persistent", path="./my_chroma_db"
    )  # Or mode='in-memory'
    # repo = FaissVectorStore()

    # Initialize embedding provider and loader
    embedding = GeminiEmbeddingProvider()
    loader = LocalLoader()
    use_case = dataloadUseCase(repo, embedding, loader)

    # Load data from CSV
    await use_case.execute(
        "data_to_load/sample.csv",
        "test_table_2",
        ["name", "description"],
        ["id"],
        create_table_if_not_exists=True,
        embed_type="combined",  # 'separated'  # Or 'combined'
    )

    # Retrieval example
    # query_text = "example project description"
    # query_embedding = embedding.get_embeddings([query_text])[0]

    # For combined mode (uses 'embeddings' column)
    # results = await repo.search('test_table_2', query_embedding, top_k=5)
    # print("Combined mode retrieval results:")
    # for result in results:
    #     print(f"ID: {result['id']}, Document: {result['document']}, Distance: {result['distance']}, Metadata: {result['metadata']}")

    # For separated mode (query specific _enc column, e.g., 'name_enc' or 'description_enc')
    # results = await repo.search('test_table_2', query_embedding, top_k=5, embed_column='description_enc')
    # print("Separated mode retrieval results (description_enc):")
    # for result in results:
    #     print(f"ID: {result['id']}, Document: {result['document']}, Distance: {result['distance']}, Metadata: {result['metadata']}")


if __name__ == "__main__":
    asyncio.run(main())


# uv run main_pg_st.py  done
# uv run main_pg_gemni.py  done
# uv run main_chroma_gemni.py  done
# uv run main_faiss_gemni.py  done
# uv run main_faiss_st.py  done
# uv run main_chroma_st.py  done
