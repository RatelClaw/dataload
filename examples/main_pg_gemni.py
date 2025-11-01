import asyncio
from dataload.infrastructure.db.db_connection import DBConnection
from dataload.infrastructure.db.data_repository import PostgresDataRepository
from dataload.infrastructure.storage.loaders import LocalLoader
from dataload.application.services.embedding.gemini_provider import (
    GeminiEmbeddingProvider,
)
from dataload.application.use_cases.data_loader_use_case import dataloadUseCase


async def main():
    db_conn = DBConnection()
    await db_conn.initialize()
    repo = PostgresDataRepository(db_conn)
    embedding = GeminiEmbeddingProvider()
    loader = LocalLoader()
    use_case = dataloadUseCase(repo, embedding, loader)

    # await use_case.execute(
    #     'data_to_load/sample_2.csv',
    #     'test_table_com_pg_gemini_st',
    #     ['Name', 'Description'],
    #     ['Index'],
    #     create_table_if_not_exists=True,
    #     embed_type=  'combined' #'separated'  # or 'combined'
    # )

    # await use_case.execute(
    #     'data_to_load/sample.csv',
    #     'test_table_v4',
    #     ['name', 'description'],
    #     ['id'],
    #     create_table_if_not_exists=True,
    #     embed_type='separated'  # or 'combined'
    # )

    # # Retrieval example
    query_text = "example project description"
    query_embedding = embedding.get_embeddings([query_text])[0]

    # For combined mode (uses 'embeddings' column)
    results = await repo.search("test_table_com_pg_gemini_st", query_embedding, top_k=5)
    print("Combined mode retrieval results:")
    for result in results:
        print(
            f"ID: {result['id']}, Document: {result['document']}, Distance: {result['distance']}, Metadata: {result['metadata']}"
        )

    # # For separated mode (query specific _enc column, e.g., 'name_enc' or 'description_enc')
    # results = await repo.search('test_table_v4', query_embedding, top_k=2, embed_column='description_enc')
    # print("Separated mode retrieval results (description_enc):")
    # for result in results:
    #     print(f"ID: {result['id']}, Document: {result['document']}, Distance: {result['distance']}, Metadata: {result['metadata']}")


if __name__ == "__main__":
    asyncio.run(main())
