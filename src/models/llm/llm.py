from llama_cpp import Llama


class PovaryoshkaLLM:
    def __init__(self):
        self.__model = Llama(
            model_path="/Users/philippplotnikov/WorkingSpace/Coding/models/Qwen3-8B-Q4_K_M-Instruct.gguf",
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=40,
            verbose=False
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> str:

        output = self.__model(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=1.2,

            # 🔥 ключевая защита от "повторения prompt"
            stop=[
                "ВОПРОС:",
                "ЗАПРОС:",
                "РЕЦЕПТЫ:",
                "ОТВЕТ:",
                "ВХОД:",
                "ВЫХОД:",
                "ТЕКУЩАЯ ЗАДАЧА:",
                "\n\n\n"
            ]
        )

        text = output["choices"][0]["text"]  # type: ignore

        # 🧹 пост-очистка от эхоповторов
        text = self.__postprocess(text, prompt)

        return text.strip()

    def __postprocess(self, text: str, prompt: str) -> str:
        # удаляем случайный echo prompt
        if prompt in text:
            text = text.replace(prompt, "")

        # убираем возможные повторяющиеся строки
        lines = text.split("\n")
        cleaned = []
        seen = set()

        for line in lines:
            if line.strip() and line not in seen:
                cleaned.append(line)
                seen.add(line)

        return "\n".join(cleaned)