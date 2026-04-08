from llama_cpp import Llama


llm_model = Llama(
    model_path="/Users/philippplotnikov/WorkingSpace/Coding/models/Qwen3-8B-Q4_K_M-Instruct.gguf",
    n_ctx=4096,        # контекст
    n_threads=8,       # под CPU
    n_gpu_layers=40,    # для M1 (ускорение через Metal)
    verbose=False
)

def generate(prompt, max_new_tokens=1024, temperature=0.4, top_k=50, top_p=0.9, repeat_penalty=1.3, stop_tokens=None):
    output = llm_model(
        prompt,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        stop=stop_tokens
    )

    # Берём только сгенерированный текст
    text = output["choices"][0]["text"] # type: ignore

    # Обрезаем до первого stop-токена, если он есть
    if stop_tokens:
        for stop in stop_tokens:
            if stop in text:
                text = text.split(stop)[0]

    return text.strip()