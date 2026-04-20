from typing import List
from unittest import result

from db.in_memory_database_driver import PovaryoshkaInMemoryDatabaseDriver
from models.encoder.encoder import PovaryoshkaEncoder
from db.vector_database_driver import PovaryoshkaVectorDatabaseDriver
from models.encoder.utils import load_encoder
from retriever.retriever import PovaryoshkaRetriever
from training.encoder_train_loop.utils import get_train_chunk_list, get_val_chunk_list


class PovaryoshkaContextManager:
    def __init__(
        self,
        encoder: PovaryoshkaEncoder,
        llm,
        in_memory_db_driver: PovaryoshkaInMemoryDatabaseDriver,
        persistent_db_driver: PovaryoshkaVectorDatabaseDriver,
        max_history_length: int = 10,
    ):
        self.__encoder = encoder
        self.__llm = llm
        self.__persistent_db_driver = persistent_db_driver
        self.__max_history_length = max_history_length
        self.__in_memory_db_driver = in_memory_db_driver

    def __summarize(self, text_list: List[str]) -> str:
        text = "\n".join(f"{c}" for c in text_list)
        prompt = f"""
Ты — система сжатия диалога для памяти ассистента.

Твоя задача:
Сжать диалог в краткое, точное резюме, сохранив смысл, факты и важные детали.

Правила:
- 2–5 предложений максимум
- сохраняй ключевые факты, числа, определения, шаги решений
- не добавляй новую информацию
- не выдумывай детали
- не используй фразы "пользователь спросил"
- пиши как нейтральное описание знаний из диалога

====================

ПРИМЕР 1:

ВХОД:
как приготовить блины
нужно молоко, яйца, мука и сахар
смешать до однородного теста без комков
жарить на сковороде

ВЫХОД:
Для приготовления блинов используют молоко, яйца, муку и сахар. Ингредиенты смешивают до однородного теста без комков, затем жарят на сковороде.

ПРИМЕР 2:

ВХОД:
как сделать крем для торта
используются сливки 33%, сахарная пудра и ваниль
взбивать до плотных пиков
крем должен быть густым

ВЫХОД:
Крем для торта готовят из сливок 33%, сахарной пудры и ванили. Смесь взбивают до плотной консистенции и устойчивых пиков.

ПРИМЕР 3:

ВХОД:
как испечь шарлотку
яблоки, яйца, сахар и мука
смешать тесто и добавить яблоки
выпекать при 180 градусах около 40 минут

ВЫХОД:
Шарлотку готовят из яблок, яиц, сахара и муки. Тесто смешивают с яблоками и выпекают при 180°C около 40 минут до готовности.

====================

ТЕКУЩАЯ ЗАДАЧА:

ВХОД:
{text}

ВЫХОД:
"""
        return self.__llm.generate(prompt).strip()

    def add_context(self, user_id: str, query: str):
        history = self.__in_memory_db_driver.get(user_id)
        if len(history) >= self.__max_history_length:
            self.__compress_memory(user_id)
        self.__in_memory_db_driver.add(user_id, query)

    def __compress_memory(self, user_id: str):
        history = list(self.__in_memory_db_driver.get(user_id))
        summary = self.__summarize(history)
        embeddings_tensor = self.__encoder.encode([summary])
        self.__persistent_db_driver.add(
            document_list=[summary],
            embedding_list=[embedding.cpu().numpy() for embedding in embeddings_tensor],
            metadata_list=[{"user_id": user_id, "type": "summary"}]
        )
        self.__in_memory_db_driver.clear(user_id)
        self.__in_memory_db_driver.add(user_id, summary)

    def get_context(self, user_id: str) -> List[str]:
        return list(self.__in_memory_db_driver.get(user_id))

    def __is_enough_context(self, query: str, context_list: List[str]) -> bool:
        context = "\n".join(f"- {c}" for c in context_list)
        prompt = f"""
Ты — контроллер качества контекста для системы поиска рецептов.

Твоя задача:
Определить, достаточно ли текущего контекста, чтобы ответить на вопрос пользователя.

Правила:
- Ответ только: "да" или "нет"
- Никаких объяснений
- Никаких дополнительных слов

====================

ПРИМЕР 1

ЗАПРОС:
Как сделать крем для торта?

КОНТЕКСТ:
- Сливки 33%, сахарная пудра, ваниль
- Взбить до плотных пиков

ОТВЕТ:
да

ПРИМЕР 2

ЗАПРОС:
Как приготовить тесто для блинов?

КОНТЕКСТ:
- Мука, молоко, яйца, соль, сахар
- Смешать до однородного жидкого теста

ОТВЕТ:
да

ПРИМЕР 3

ЗАПРОС:
Как приготовить сложный десерт с кремом и начинкой?

КОНТЕКСТ:
- Мука
- Молоко

ОТВЕТ:
нет

====================

ТЕКУЩАЯ ЗАДАЧА:

ЗАПРОС:
{query}

КОНТЕКСТ:
{context}

ОТВЕТ:
"""
        result = self.__llm.generate(prompt).strip().lower()
        return result.startswith("да")

    def __rewrite_query(self, query: str, context_list: List[str]) -> str:
        context = "\n".join(f"- {c}" for c in context_list)
        prompt = f"""
Ты — система улучшения поисковых запросов для поиска документов.

Твоя задача:
Переписать запрос пользователя так, чтобы он стал максимально понятным для поиска релевантных документов.

Правила:
- Добавь важные смысловые уточнения.
- Удали неопределённые слова (например: "это", "как это", "почему так").
- Сделай запрос конкретным и поисковым.
- Сохрани исходный смысл.
- Если есть контекст — используй его для уточнения смысла запроса.

====================

ПРИМЕР 1

ЗАПРОС:
как приготовить

КОНТЕКСТ:
рецепт шарлотки с яблоками, тесто, яйца, мука, сахар, яблоки

ОТВЕТ:
рецепт приготовления шарлотки с яблоками из теста с яйцами, мукой и сахаром

ПРИМЕР 2

ЗАПРОС:
как сделать крем

КОНТЕКСТ:
торт, сливочный крем, сливки 33%, сахарная пудра, ваниль

ОТВЕТ:
рецепт сливочного крема для торта из сливок 33%, сахарной пудры и ванили с взбиванием до плотных пиков

ПРИМЕР 3

ЗАПРОС:
как приготовить блины

КОНТЕКСТ:


ОТВЕТ:
рецепт приготовления тонких блинов на молоке с яйцами и мукой

====================

ТЕКУЩАЯ ЗАДАЧА:

ЗАПРОС:
{query}

КОНТЕКСТ:
{context}

ОТВЕТ:
"""
        return self.__llm.generate(prompt).strip()
    
    def __deduplicate(self, texts: List[str]) -> List[str]:
        seen = set()
        result = []
        for t in texts:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result

    def process(self, user_id: str, query: str):
        self.add_context(user_id, query)
        context = self.get_context(user_id)
        max_iters = 5
        for _ in range(max_iters):
            if self.__is_enough_context(query, context):
                break
            retrieval_query = self.__rewrite_query(query, context)
            query_embedding = self.__encoder.encode([retrieval_query])[0]
            retrieved = self.__persistent_db_driver.search(
                embedding_tensor=query_embedding,
                top_k=5,
                where={"user_id": user_id}
            )
            retrieved_texts = [r["text"] for r in retrieved]
            context = self.__deduplicate(context + retrieved_texts)
        final_query = self.__rewrite_query(query, context)
        return {
            "query": final_query,   # для retriever
            "context": context     # для LLM
        }

class DummyLLM:
    def __init__(self):
        self.counter = 1

    def generate(self, prompt: str) -> str:
        if self.counter % 4 != 0:
            message = f"да {self.counter}"
            self.counter += 1
            return message
        message = f"нет {self.counter}"
        self.counter += 1
        return message

if __name__ == "__main__":
    device = 'cpu'
    train_chunk_list = get_train_chunk_list(device)
    val_chunk_list = get_val_chunk_list(device)
    common_chunk_list = train_chunk_list + val_chunk_list
    in_memory_db_driver = PovaryoshkaInMemoryDatabaseDriver()
    context_manager_persistent_db_driver = PovaryoshkaVectorDatabaseDriver(
        collection_name="dialogue_history"
    )
    retriever_persistent_db_driver = PovaryoshkaVectorDatabaseDriver(
        collection_name="documents"
    )
    encoder = load_encoder()
    context_manager = PovaryoshkaContextManager(
        encoder=encoder,
        llm=DummyLLM(),
        in_memory_db_driver=in_memory_db_driver,
        persistent_db_driver=context_manager_persistent_db_driver,
        max_history_length=3
    )
    retriever = PovaryoshkaRetriever(
        encoder=encoder,
        persistent_db_driver=retriever_persistent_db_driver
    )
    retriever.build_index(
        [chunk['text'] for chunk in common_chunk_list]
    )
    while True:
        user_input = input("Введите запрос: ")
        result = context_manager.process("user_1", user_input)
        retriever_answer = retriever.get_chunk_list(result['context'][-1])
        print("Преобразованный запрос для ретривера:", result["query"])
        print("Ответ от ретривера: ", retriever_answer)
        print("Контекст для LLM:", result["context"])
