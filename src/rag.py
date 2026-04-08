from llm import generate
from context_manager import ContextManager
import torch
from teacher_retriever_pool import TeacherRetrieverPool
from teacher_retriever_pool import BM25Model, DenseModel
import json


device = 'cpu'

input_filename = 'pruned_recipe_chunks_with_questions_1.json'
with open(input_filename, 'r', encoding='utf-8') as f:
    pruned_train_chunk_list = json.load(f)
print(f"Загружено {len(pruned_train_chunk_list)} чанков")
input_filename = 'pruned_recipe_chunks_with_questions_2.json'
with open(input_filename, 'r', encoding='utf-8') as f:
    pruned_eval_chunk_list = json.load(f)
print(f"Загружено {len(pruned_eval_chunk_list)} чанков")

teacher_retriever_pool = TeacherRetrieverPool(1, top_k=5)
pruned_train_chunk_list = pruned_train_chunk_list[:400]
pruned_eval_chunk_list = pruned_eval_chunk_list[:160]
pruned_common_chunk_list = pruned_train_chunk_list + pruned_eval_chunk_list
teacher_retriever_pool.add_teacher('bm25', BM25Model(pruned_common_chunk_list))
teacher_retriever_pool.add_teacher('dense', DenseModel(pruned_common_chunk_list, device='mps'))


train_ranked_chunk_list = torch.load('recipe_chunks_with_ranking_tensors_1.pth')
for train_chunk in train_ranked_chunk_list:
    train_chunk['index_tensors'] = train_chunk['index_tensors'].to(device)
    train_chunk['ranking_tensors'] = train_chunk['ranking_tensors'].to(device)
print(len(train_ranked_chunk_list))
eval_ranked_chunk_list = torch.load('recipe_chunks_with_ranking_tensors_2.pth')
for eval_chunk in eval_ranked_chunk_list:
    eval_chunk['index_tensors'] = eval_chunk['index_tensors'].to(device)
    eval_chunk['ranking_tensors'] = eval_chunk['ranking_tensors'].to(device)
print(len(eval_ranked_chunk_list))




common_document_list = [chunk['text'] for chunk in pruned_common_chunk_list]

def rewrite_query_llm(query, context_manager: ContextManager, user_id: str):
    context = context_manager.get_context(user_id)

    prompt = f"""
Ты помощник-кулинар. Твоя задача — переписать запрос пользователя для поиска по базе знаний так, чтобы он был:

- кратким,
- точным,
- максимально информативным,
- учитывал контекст диалога.

Правила:
- Не используй ссылки вида [1], [2]; вместо этого упоминай рецепты по их названиям или ингредиенты.
- Не добавляй информации, которой нет в контексте.
- Если контекст не помогает точно переписать запрос — составь максимально общий, но корректный поисковый запрос на основе того, что есть.
- Сохраняй ключевые ингредиенты, названия блюд и основные уточнения из контекста.

Пример:

Контекст:
В прошлый раз обсуждали тесто для пирога Киш Лорен с томатами и моцареллой. Пользователь интересовался ингредиентами и пропорциями для теста.

Оригинальный запрос:
Как сделать тесто для пирога с помидорами и сыром?

Новый поисковый запрос (краткий, точный):
тесто пирог Киш Лорен ингредиенты пропорции
###END###

Теперь твой запрос:

Контекст:
{context}

Оригинальный запрос:
{query}

Новый поисковый запрос (краткий, точный):
"""

    rewritten = generate(prompt, stop_tokens=["###END###"])
    return rewritten.strip()


class RAGSystem:
    def __init__(self, retriever, generator_fn):
        self.retriever = retriever
        self.generate = generator_fn
        self.context_manager = ContextManager()

    def answer(self, query: str, user_id: str, top_k=5):
        # Поиск top_k релевантных контекстов
        self.context_manager.add_context(user_id, query)
        rewritten_query = rewrite_query_llm(query, self.context_manager, user_id)
        retrieved_indices_list = self.retriever.search([rewritten_query], top_k)
        print(f"Предыдущий контекст для пользователя {user_id}: {self.context_manager.get_context(user_id)} ")

        contexts = []
        seen_texts = set()  # чтобы не дублировать одинаковый текст
        for retrieved_index_list in retrieved_indices_list:
            for i, index in enumerate(retrieved_index_list):
                text = pruned_common_chunk_list[index]['text']
                if text not in seen_texts:
                    contexts.append(f"[{i+1}] {text}")
                    seen_texts.add(text)
    
        context = "\n\n".join(contexts)

        # Структурируем prompt с явными маркерами начала и конца ответа
        prompt = f"""
Ты помощник-кулинар.

ПРАВИЛА:
- Отвечай ТОЛЬКО используя предоставленный контекст.
- Не используй ссылки вида [1], [2]; вместо этого называй рецепты по их названиям или ингредиенты.
- Если ответа нет в контексте — скажи "Не знаю".
- Не повторяй информацию несколько раз.
- Перефразируй текст из контекста, не копируй дословно.
- Сохраняй ключевые ингредиенты, названия блюд и основные уточнения из контекста.

Пример:

Контекст:
Название рецепта: Пирог Киш Лорен с томатами и моцареллой. Для теста используем муку, сливочное масло, сметану и соль. Замешиваем тесто, формируем шар и убираем в холодильник на 1 час.

Вопрос:
Как приготовить тесто для пирога?

Ответ:
Для теста смешайте муку, сливочное масло комнатной температуры, сметану и соль. Замесите тесто в шар, заверните в пленку и уберите в холодильник на 1 час.
###END###

Теперь твой вопрос:

Контекст:
{context}

Вопрос:
{query}

Ответ:
"""

        # Генерация ответа
        answer = self.generate(prompt, stop_tokens=["###END###"])
    
        return answer
