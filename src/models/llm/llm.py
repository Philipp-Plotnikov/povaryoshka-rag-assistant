from typing import Any, Literal

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


class PovaryoshkaLLM:
    def __init__(
            self,
            base_model_path="../models/qwen3-4b",
            answer_generation_lora_path="../models/qwen3-4b-sft-lora-adapter/final-answer-generation-lora-adapter",
            summarization_lora_path="../models/qwen3-4b-sft-lora-adapter/final-summarization-lora-adapter",
            query_rewriting_lora_path="../models/qwen3-4b-sft-lora-adapter/final-query-rewriting-lora-adapter"
        ):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # --- tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- base model ---
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map=None  # ⚠️ важно для MPS
        )

        base_model.to(self.device) # type: ignore

        # --- LoRA: answer ---
        self.model = PeftModel.from_pretrained(
            base_model,
            answer_generation_lora_path,
            adapter_name="answer"
        )

        # --- LoRA: summary ---
        self.model.load_adapter(
            summarization_lora_path,
            adapter_name="summary"
        )

        self.model.load_adapter(
            query_rewriting_lora_path,
            adapter_name="query_rewriting"
        )

        self.model.eval()

        # ⚡ ускорение
        torch.set_float32_matmul_precision("high")

    def generate(
        self,
        prompt: str,
        task: Literal["summary", "answer", "query_rewriting"]="answer",
        max_new_tokens: int = 2000,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:

        # --- переключаем LoRA ---
        if task == "summary":
            self.model.set_adapter("summary")
        elif task == "query_rewriting":
            self.model.set_adapter("query_rewriting")
        else:
            self.model.set_adapter("answer")

        # --- токенизация ---
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{prompt} /no_think"}],
            tokenize=False,
            add_generation_prompt=True,
        ) 
        model_inputs = self.tokenizer(
            [prompt],
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        try:
            # rindex finding 151668 (<tool_call>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return self.__postprocess(content, prompt)

    def __postprocess(self, text: str, prompt: str) -> str:
        # убираем echo prompt
        if prompt in text:
            text = text.replace(prompt, "")

        # убираем дубликаты строк
        lines = text.split("\n")
        seen = set()
        cleaned = []

        for line in lines:
            line = line.strip()
            if line and line not in seen:
                cleaned.append(line)
                seen.add(line)

        return "\n".join(cleaned).strip()

# from llama_cpp import Any, Llama


# class PovaryoshkaLLM:
#     def __init__(self):
#         self.__model = Llama(
#             model_path="/Users/philippplotnikov/WorkingSpace/Coding/models/Qwen3-8B-Q4_K_M-Instruct.gguf",
#             n_ctx=4096,
#             n_threads=8,
#             n_gpu_layers=40,
#             verbose=False
#         )

#     def generate(
#         self,
#         prompt: str,
#         max_new_tokens: int = 512,
#         temperature: float = 0.7,
#         top_k: int = 40,
#         top_p: float = 0.9,
#     ) -> str:

#         output = self.__model(
#             prompt,
#             max_tokens=max_new_tokens,
#             temperature=temperature,
#             top_k=top_k,
#             top_p=top_p,
#             repeat_penalty=1.2,

#             stop=[
#                 "ВОПРОС:",
#                 "ЗАПРОС:",
#                 "РЕЦЕПТЫ:",
#                 "ОТВЕТ:",
#                 "ВХОД:",
#                 "ВЫХОД:",
#                 "ТЕКУЩАЯ ЗАДАЧА:",
#                 "====================",
#                 "[Ваш ответ здесь]",
#                 "(здесь должен быть твой ответ)",
#                 "(нужно сжать диалог в краткое резюме)",
#                 "[ВАШ ВЫХОД ТУТ]",
#                 "(ответ должен быть на русском языке, отвечай дружелюбно и естественно)",
#                 "\n\n\n"
#             ]
#         )

#         text = output["choices"][0]["text"]  # type: ignore

#         text = self.__postprocess(text, prompt)

#         return text.strip()

#     def __postprocess(self, text: str, prompt: str) -> str:
#         # удаляем случайный echo prompt
#         if prompt in text:
#             text = text.replace(prompt, "")

#         # убираем возможные повторяющиеся строки
#         lines = text.split("\n")
#         cleaned = []
#         seen = set()

#         for line in lines:
#             if line.strip() and line not in seen:
#                 cleaned.append(line)
#                 seen.add(line)

#         return "\n".join(cleaned)

def build_prompt(query: str, chunk_list: list[dict[str, Any]]) -> str:
    docs_text = "\n".join(f"- {d.get('text', '')}" for d in chunk_list)
    prompt = f"""
Ты — дружелюбный кулинарный ассистент, Поварешка.
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


def build_prompt_for_summarize(text_list: list[str]):
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
- не отвечай на вопросы, твоя цель сжать диалог в краткое, точное резюме, сохранив смысл, факты и важные детали

====================

ПРИМЕР 1:

ВХОД:
как приготовить блины ?
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

ПРИМЕР 4:

ВХОД:
Какова продолжительность времени варки напитка в рецепте?

ВЫХОД:
спрашивает про продолжительность времени варки напитка в рецепте

====================

ТЕКУЩАЯ ЗАДАЧА:

ВХОД:
{text}

ВЫХОД:
"""
    return prompt


if __name__ == "__main__":
    llm = PovaryoshkaLLM()
    # prompt = build_prompt(
    #     query="Привет, хочу торт сделать вечером. Подскажи, какое количество муки используется для приготовления торта «Монастырская изба» с заварным кремом?",
    #     chunk_list=[
    #         {
    #             "text": "Название рецепта: Торт «Монастырская изба» с вишней со сливками\n\nИнгредиенты на 6 порции:\n1) мука 650 г\n2) сметана 600 г\n3) вишня 500 г\n4) маргарин 200 г\n5) сливки 1 стакан\n6) шоколад 50 г\n7) сахар 8 ст.л.\n8) разрыхлитель 5 г\n9) соль 3 г\n10) ванильный экстракт 1 пакетик\n\nИнструкции по приготовлению:\n1) Подтаявший маргарин перемешать с ванилином из пакетика, сахарным песком, сметаной, разрыхлителем и солью.\n2) Ингредиенты перемешать, массу взбить, добавить муку.\n3) Тугое тесто поместить в холодильник на час.\n4) Разделить массу на три части, раскатать пласты прямоугольной формы.\n5) Нарезать на заготовки размером десять на тридцать сантиметров.\n6) Выложить свежие или размороженные ягоды вишни без косточек в ряд.\n7) Тесто аккуратно свернуть трубочкой, края плотно закрыть, чтобы не вытек сок.\n8) Противень застелить пекарской бумагой, выложить трубочки, выпекать в течение получаса при температуре 180 градусов.\n9) Выпечка должна подрумяниться.\n10) Трубочки остудить, их должно получиться пятнадцать штук.\n11) Для приготовления сметанного крема взбить при помощи миксера сливки, добавить сметану и сахарный песок.\n12) Продукты взбивать при помощи миксера.\n13) Крем должен получиться густым.\n14) Выложить на поднос пять трубочек, смазать сметанным кремом, во второй слой выложить четыре трубочки, затем три и одну.\n15) Все слои промазывать кремом.\n16) «Избу» убрать в холод на два часа для пропитывания.\n17) Готовый торт нарезать кусочками и подать.\n"
    #         },
    #     ]
    # )
    prompt = build_prompt_for_summarize(
        text_list=[
            "Привет, сегодня мы собираемся с друзьями у меня и я хотел бы приготовить Вяленную свинину с красным перцем. Как ее приготовить ?",
            "Забыл еще спросить про морс из замороженной брусники с ванилином",
            "Какова продолжительность времени варки напитка в рецепте?",
            "Какое количество сахара используется в рецепте?"
        ]
    )
    print(llm.generate(prompt))


