from typing import Literal

from models.context_sufficiency_classifier.context_sufficiency_classifier import PovaryoshkaContextSufficiencyClassifier
from models.encoder.encoder import PovaryoshkaEncoder


class PovaryoshkaQueryRouter:
    def __init__(
        self,
        encoder: PovaryoshkaEncoder,
        context_sufficiency_classifier: PovaryoshkaContextSufficiencyClassifier,
    ):
        super().__init__()
        self.__encoder = encoder
        self.__context_sufficiency_classifier = context_sufficiency_classifier
    
    def route_query(
        self,
        context: str
    ) -> Literal["context_manager", "retriever"]:
        query_embedding = self.__encoder.encode([context], "search_query")[0]
        if self.__context_sufficiency_classifier.is_need_context_manager(query_embedding):
            return "context_manager"
        return "retriever"
