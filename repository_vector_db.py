import os
from typing import List

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (HnswAlgorithmConfiguration,
                                                   SearchField,
                                                   SearchFieldDataType,
                                                   SearchIndex,
                                                   SearchableField,
                                                   SimpleField,
                                                   VectorSearch,
                                                   VectorSearchProfile)
from azure.search.documents.models import VectorizedQuery

# Azure AI Search configurations
SEARCH_ENDPOINT = os.getenv("AI_SEARCH_ENDPOINT")
SEARCH_API_KEY = os.getenv("AI_SEARCH_API_KEY")
SEARCH_INDEX_NAME = os.getenv("AI_SEARCH_INDEX_NAME")

ai_search_index_client = SearchIndexClient(
    endpoint=SEARCH_ENDPOINT,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

ai_search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)


async def create_search_index():
    # Check if the index exists
    try:
        await ai_search_index_client.get_index(name=SEARCH_INDEX_NAME)
    except ResourceNotFoundError:
        # Create the index
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="filename", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                hidden=False,
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="default"  # use default `myHnswProfile`
            ),
        ]
        vector_search = VectorSearch(
            profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="default")],
            algorithms=[HnswAlgorithmConfiguration(name="default")]
        )

        index = SearchIndex(
            name=SEARCH_INDEX_NAME,
            fields=fields,
            vector_search=vector_search
        )

        # Create the index
        search_index = await ai_search_index_client.create_index(index)
        print(f"Created index: {search_index.name}")
    except Exception as e:
        print(f"An error occurred while creating the index: {e}")
        raise


async def search_chunks(question: str, embedding: List[float], top_k: int = 3) -> List[str]:
    # Prepare the vector query
    vector = VectorizedQuery(
        vector=embedding,
        fields="embedding",
        k_nearest_neighbors=5,
    )

    # Perform vector search in Azure Cognitive Search
    search_results = await ai_search_client.search(
        search_text=question,
        vector_queries=[vector],
        top=top_k,
        select="filename,content",
    )

    # Extract the content from the search results
    chunks = []
    async for result in search_results:
        filename = result.get('filename', 'Unknown Filename')
        chunks.append(filename + "->\n" + result['content'])
    return chunks
