import numpy as np
import torch

from db.vector_database_driver import PovaryoshkaVectorDatabaseDriver


class PovaryoshkaContextSufficiencyClassifier:
    def __init__(
        self,
        retriever_persistent_db_driver: PovaryoshkaVectorDatabaseDriver,
        threshold: float = 0.48,
        top_k: int = 10
    ):
        self.__retriever_persistent_db_driver = retriever_persistent_db_driver
        self.__threshold = threshold
        self.__top_k = top_k

    def __score(self, query_embedding_tensor: torch.Tensor) -> float:
        results = self.__retriever_persistent_db_driver.search(
            embedding_tensor=query_embedding_tensor,
            top_k=self.__top_k
        )
        if not results:
            return 0.0
        distances = np.array([r["distance"] for r in results])
        sims = 1.0 - distances
        max_sim = float(np.max(sims))
        sorted_sims = np.sort(sims)
        margin = float(sorted_sims[-1] - sorted_sims[-2]) if len(sims) > 1 else 0.0
        score = 0.85 * max_sim + 0.15 * margin
        return score

    def is_need_context_manager(self, query_embedding_tensor: torch.Tensor) -> bool:
        return self.__score(query_embedding_tensor) < self.__threshold
