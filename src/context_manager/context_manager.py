import math
import time
import torch
import numpy as np
from typing import Any

from db.in_memory_database_driver import PovaryoshkaInMemoryDatabaseDriver
from models.encoder.encoder import PovaryoshkaEncoder
from db.vector_database_driver import PovaryoshkaVectorDatabaseDriver
from models.encoder.utils import load_encoder
from models.llm.llm import PovaryoshkaLLM
from retriever.retriever import PovaryoshkaRetriever
from training.encoder_train_loop.utils import get_train_chunk_list, get_val_chunk_list
from training.llm_train_loop.utils import build_prompt_for_query_rewriting, build_prompt_for_summarization


class PovaryoshkaContextManager:
    def __init__(
        self,
        encoder: PovaryoshkaEncoder,
        llm: PovaryoshkaLLM,
        in_memory_db_driver: PovaryoshkaInMemoryDatabaseDriver,
        persistent_db_driver: PovaryoshkaVectorDatabaseDriver,
        top_k: int = 2,
        max_history_length: int = 5,
    ):
        self.__encoder = encoder
        self.__llm = llm
        self.__in_memory_db_driver = in_memory_db_driver
        self.__persistent_db_driver = persistent_db_driver
        self.__top_k = top_k
        self.__max_history_length = max_history_length

    def __summarize(self, text_list: list[str]) -> str:
        text = "\n".join(text_list)
        print(f"Контекст: {text}")
        prompt = build_prompt_for_summarization(text_list)
        result = self.__llm.generate(prompt, "summary").strip()
        print(f"Сжатый запрос: '{result}'")
        return result

    def add_context(self, user_id: str, query: str):
        context_history = self.__in_memory_db_driver.get(user_id)
        if len(context_history) == 0:
            self.__in_memory_db_driver.add(
                user_id,
                {
                    'text': query,
                    'timestamp': time.time()
                }
            )
            return
        similarity_score = self.similarity(query, context_history)
        if len(context_history) >= self.__max_history_length or similarity_score < 0.43:
            if similarity_score < 0.45:
                self.__compress_memory(user_id, is_leave_in_memory=False)
            else:
                self.__compress_memory(user_id)
        self.__in_memory_db_driver.add(
            user_id,
            {
                'text': query,
                'timestamp': time.time()
            }
        )

    def similarity(self, query: str, context_history: list[dict[str, Any]]) -> float:
        query_embedding = self.__encoder.encode([query])[0]
        context_history_text_list = [context_item["text"] for context_item in context_history]
        context_embeddings_tensor = self.__encoder.encode(context_history_text_list)
        similarities_tensor = torch.matmul(context_embeddings_tensor, query_embedding)
        score_list = []
        now = time.time()
        for similarity, context_item in zip(similarities_tensor, context_history):
            age = now - context_item["timestamp"]
            decay = math.exp(-age / 3600)
            score_list.append(similarity.item() * decay)
        return float(np.mean(score_list))

    def __compress_memory(self, user_id: str, is_leave_in_memory=True):
        context_history = self.__in_memory_db_driver.get(user_id)
        summary = self.__summarize([item['text'] for item in context_history])
        embeddings_tensor = self.__encoder.encode([summary])
        summary_timestamp = max(item["timestamp"] for item in context_history)
        self.__persistent_db_driver.add(
            document_list=[summary],
            embedding_list=[embedding.cpu().numpy() for embedding in embeddings_tensor],
            metadata_list=[
                {
                    "user_id": user_id,
                    "type": "summary",
                    "timestamp": summary_timestamp
                }
            ]
        )
        self.__in_memory_db_driver.clear(user_id)
        if is_leave_in_memory:
            self.__in_memory_db_driver.add(
                user_id,
                {
                    'text': summary,
                    'timestamp': summary_timestamp
                }
            )

    def get_context_history(self, user_id: str) -> list[dict[str, Any]]:
        user_context_history = self.__in_memory_db_driver.get(user_id)
        return user_context_history

    def rewrite_query(self, query: str, context_list: list[str]) -> str:
        prompt = build_prompt_for_query_rewriting(query, context_list)
        result = self.__llm.generate(prompt, "query_rewriting").strip()
        print(f"Преобразованный запрос контекст мэнэджером: '{result}'")
        return result

    def __deduplicate(self, text_list: list[str]) -> list[str]:
        seen = set()
        result = []
        for t in text_list:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result

    def enhance_context_history(self, user_id: str, full_context: dict[str, Any]) -> dict[str, Any]:
        query = full_context["query"]
        context_history = full_context["context_history"]
        query_embedding = self.__encoder.encode([query])[0]
        retrieved_document_list = self.__persistent_db_driver.search(
            embedding_tensor=query_embedding,
            top_k=self.__top_k,
            where={
                "$and": [
                    {
                        "user_id": user_id
                    },
                    {
                        "timestamp": {
                            "$gte": time.time() - 2 * 3600
                        }
                    }
                ]
            },
            order="asc"
        )
        retrieved_text_list: list[str] = [retrieved_document["text"] for retrieved_document in retrieved_document_list]
        print(f"Context manager вернул для запроса '{query}': {retrieved_text_list}")
        deduplicated_context_history = self.__deduplicate(
            [*context_history, *retrieved_text_list]
        )
        return {
            "query": query,
            "context_history": deduplicated_context_history
        }

# class DummyLLM:
#     def __init__(self):
#         self.counter = 1

#     def generate(self, prompt: str) -> str:
#         return prompt

# if __name__ == "__main__":
#     device = 'cpu'
#     train_chunk_list = get_train_chunk_list(device)
#     val_chunk_list = get_val_chunk_list(device)
#     common_chunk_list = train_chunk_list + val_chunk_list
#     in_memory_db_driver = PovaryoshkaInMemoryDatabaseDriver()
#     context_manager_persistent_db_driver = PovaryoshkaVectorDatabaseDriver(
#         collection_name="dialogue_history"
#     )
#     retriever_persistent_db_driver = PovaryoshkaVectorDatabaseDriver(
#         collection_name="documents"
#     )
#     encoder = load_encoder()
#     context_manager = PovaryoshkaContextManager(
#         encoder=encoder,
#         llm=DummyLLM(),
#         in_memory_db_driver=in_memory_db_driver,
#         persistent_db_driver=context_manager_persistent_db_driver,
#         max_history_length=3
#     )
#     retriever = PovaryoshkaRetriever(
#         encoder=encoder,
#         persistent_db_driver=retriever_persistent_db_driver
#     )
#     retriever.build_index(
#         [chunk['text'] for chunk in common_chunk_list]
#     )
#     full_context = {
#         'query': '',
#         'context_history': []
#     }
#     while True:
#         user_input = input("Введите запрос: ")
#         full_context['query'] = user_input
#         full_context = context_manager.process("user_1", full_context)
#         retriever_answer = retriever.get_chunk_list(full_context['query'])
#         print("Преобразованный запрос для ретривера:", full_context["query"])
#         print("Ответ от ретривера: ", retriever_answer)
