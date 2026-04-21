from typing import Any

from context_manager.context_manager import PovaryoshkaContextManager
from models.llm.llm import PovaryoshkaLLM
from query_router.query_router import PovaryoshkaQueryRouter
from retriever.retriever import PovaryoshkaRetriever


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
        context_history = self.__context_manager.get_context_history(user_id)
        self.__context_manager.add_context(user_id, query)
        full_context = {
            'query': query,
            'context_history': context_history
        }
        max_iters = 5
        for _ in range(max_iters):
            if self.__query_router.route_query(full_context) == "retriever":
                break
            full_context = self.__context_manager.process(user_id, full_context)
        retrieved_chunk_list = self.__retriever.get_chunk_list(full_context['query'])
        prompt = self.__build_prompt(
            full_context['query'],
            retrieved_chunk_list
        )
        return self.__llm.generate(prompt)

    def __build_prompt(self, query: str, chunk_list: list[dict[str, Any]]) -> str:
        docs_text = "\n".join(f"- {d.get('text', '')}" for d in chunk_list)

        prompt = f"""
Ты — дружелюбный кулинарный ассистент.
ВАЖНЫЕ ПРАВИЛА:
- Никогда не повторяй инструкции или структуру контекста.
- Отвечай только как человек, а не как модель.
- Используй только предоставленную информацию.
- Если информации недостаточно — скажи "Не знаю".
- Пиши естественным языком, как повар объясняет рецепт.
- Сохраняй ключевые ингредиенты, названия блюд и основные уточнения из контекста.
- Все числа и цифры записывай словами (например: не "7", а "семь"; не "180°C", а "сто восемьдесят градусов").
- Никогда не используй арабские цифры (0-9), только слова.
- Использование цифр запрещено, ответ должен содержать только текстовые формы чисел.

====================

ПРИМЕР 1

ВОПРОС:
Как приготовить тесто для блинов, чтобы оно было без комков и хорошо жарилось?

РЕЦЕПТЫ:
Мука, молоко, яйца, сахар, соль, растительное масло. Сначала смешать яйца с молоком, затем постепенно вводить муку, постоянно перемешивая. Добавить сахар, соль и немного масла.

ОТВЕТ:
Сначала взбейте яйца с молоком до однородности. Затем постепенно добавляйте муку, постоянно размешивая, чтобы не образовались комки. В конце добавьте сахар, щепотку соли и немного растительного масла. Тесто должно получиться жидким и гладким, без комков.

====================

ПРИМЕР 2

ВОПРОС:
Как сделать стабильный крем для торта, чтобы он держал форму?

РЕЦЕПТЫ:
Сливки 33%, сахарная пудра, ваниль. Сливки должны быть холодными. Взбивать на средней скорости до загустения, затем добавить сахарную пудру и ваниль. Продолжить взбивание до плотных пиков.

ОТВЕТ:
Охладите сливки и миску для взбивания. Начните взбивать сливки на средней скорости до лёгкого загустения. Затем добавьте сахарную пудру и немного ванили. Продолжайте взбивать до устойчивых плотных пиков. Крем должен держать форму и не растекаться.

====================

ТЕКУЩАЯ ЗАДАЧА:

ВОПРОС:
{query}

РЕЦЕПТЫ:
{docs_text}

ОТВЕТ:
"""   
        return prompt
