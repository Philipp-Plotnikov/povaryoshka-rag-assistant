from db.vector_database_driver import PovaryoshkaVectorDatabaseDriver
from models.encoder.encoder import PovaryoshkaEncoder
from typing import Any


class PovaryoshkaRetriever:
  def __init__(
    self,
    encoder: PovaryoshkaEncoder,
    persistent_db_driver: PovaryoshkaVectorDatabaseDriver
  ):
    super().__init__()
    self.__encoder = encoder
    self.__persistent_db_driver = persistent_db_driver

  def build_index(self, chunk_list: list[str]):
    embeddings_tensor = self.__encoder.encode(chunk_list)
    self.__persistent_db_driver.add(
      chunk_list,
      [embedding.cpu().numpy() for embedding in embeddings_tensor]
    )

  def get_chunk_list(
    self,
    query: str,
    top_k=5
  ) -> list[dict[str, Any]]:
    query_embedding = self.__encoder.encode(
      [query],
      "search_query"
    )[0]
    results = self.__persistent_db_driver.search(
      query_embedding,
      top_k
    )
    return results
