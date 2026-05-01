from context_manager.context_manager import PovaryoshkaContextManager
from models.llm.llm import PovaryoshkaLLM
from query_router.query_router import PovaryoshkaQueryRouter
from retriever.retriever import PovaryoshkaRetriever
from training.llm_train_loop.utils import build_prompt_for_answer_generation


class PovaryoshkaRAG:
    def __init__(
        self,
        context_manager: PovaryoshkaContextManager,
        query_router: PovaryoshkaQueryRouter,
        retriever: PovaryoshkaRetriever,
        llm: PovaryoshkaLLM
    ):
        self.__context_manager = context_manager
        self.__query_router = query_router
        self.__retriever = retriever
        self.__llm = llm

    def generate(self, user_id: str, query: str):
        self.__context_manager.add_context(user_id, query)
        context_history = self.__context_manager.get_context_history(user_id)[:-1]
        full_context = {
            'query': query,
            'context_history': context_history
        }
        max_iters = 5
        enhanced_query = query
        for _ in range(max_iters):
            enhanced_query = self.__context_manager.rewrite_query(
                full_context["query"],
                full_context["context_history"]
            )
            full_context["query"] = enhanced_query
            if self.__query_router.route_query(enhanced_query) == "retriever":
                break
            full_context = self.__context_manager.enhance_context_history(user_id, full_context)
        retrieved_chunk_list = self.__retriever.get_chunk_list(full_context['query'])
        prompt = build_prompt_for_answer_generation(
            full_context['query'],
            [retrieved_chunk["text"] for retrieved_chunk in retrieved_chunk_list]
        )
        return self.__llm.generate(prompt)
