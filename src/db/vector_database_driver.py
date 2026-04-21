import chromadb
import torch
import uuid

from typing import List, Any
from chromadb.api.types import (
    Embedding,
    Metadata
)

class PovaryoshkaVectorDatabaseDriver:
    def __init__(
        self,
        collection_name: str,
        host="localhost",
        port=8000,
        tenant_name = "povaryoshka",
        database_name = "assistant"
    ):
        self.__client = chromadb.HttpClient(
            tenant=tenant_name,
            database=database_name,
            host=host,
            port=port
        )
        self.__collection = self.__client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None
        )

    def add(
        self,
        document_list: list[str],
        embedding_list: list[Embedding],
        metadata_list: list[Metadata] | None = None,
        id_list: list[str] | None = None,
    ):
        if id_list is None:
            id_list = [str(uuid.uuid4()) for _ in document_list] 
        self.__collection.add(
            documents=document_list,
            embeddings=embedding_list,
            metadatas=metadata_list,
            ids=id_list,
        )

    def search(
        self,
        embedding_tensor: torch.Tensor,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> List[dict[str, Any]]:
        embedding_list = embedding_tensor.tolist()
        results = self.__collection.query(
            query_embeddings=[embedding_list],
            n_results=top_k,
            where=where
        )
        if not results["documents"] or len(results["documents"][0]) == 0:
            return []
        output = []
        for i in range(len(results["documents"][0])):
            output.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i], # type: ignore
                "distance": results["distances"][0][i], # type: ignore
            })
        return output
