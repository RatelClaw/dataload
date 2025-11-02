import asyncio
from dataload.infrastructure.vector_stores.chroma_store import ChromaVectorStore
from dataload.infrastructure.storage.loaders import LocalLoader
from dataload.application.services.embedding.sentence_transformers_provider import (
    SentenceTransformersProvider,
)
from dataload.application.use_cases.data_loader_use_case import dataloadUseCase


async def main():
    repo = ChromaVectorStore(
        mode="persistent", path="./my_chroma_db"
    )  # or mode='in-memory'
    embedding = SentenceTransformersProvider()
    loader = LocalLoader()
    use_case = dataloadUseCase(repo, embedding, loader)

    await use_case.execute(
        "data_to_load/sample_2.csv",
        "test_table",
        ["Name", "Description"],
        ["Index"],
        create_table_if_not_exists=True,
        embed_type="separated",  # or 'combined'
    )

    # # # # Retrieval example (commented)
    # query_text = "example query"
    # query_embedding = embedding.get_embeddings([query_text])[0]
    # results = await repo.search('test_table', query_embedding, top_k=5, embed_column='Description_enc')  # For separated mode
    # print("Retrieval results:")
    # for result in results:
    #     print(f"ID: {result['id']}, Document: {result['document']}, Distance: {result['distance']}, Metadata: {result['metadata']}")


if __name__ == "__main__":
    asyncio.run(main())
