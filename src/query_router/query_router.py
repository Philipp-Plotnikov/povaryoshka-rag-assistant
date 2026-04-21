from typing import Any, Literal

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
        full_context: dict[str, Any]
    ) -> Literal["context_manager", "retriever"]:
        context = f"- {full_context['query']}\n" + "\n".join(f"- {context}" for context in full_context['context_history'])
        query_embedding = self.__encoder.encode([context], "search_query")[0]
        if self.__context_sufficiency_classifier.is_need_context_manager(query_embedding):
            return "context_manager"
        return "retriever"
