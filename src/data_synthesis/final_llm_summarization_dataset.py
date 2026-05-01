import json
from typing import Any

from training.llm_train_loop.utils import build_prompt_for_summarization

    
def formatting_func(example: dict[str, Any]) -> dict[str, Any]:
    user_query = example["messages"][0]["content"]
    assistant_answer = example["messages"][1]["content"]
    text_list = user_query.split("\n")
    system_prompt = build_prompt_for_summarization(text_list)
    messages = [
        {"role": "user", "content": f"{system_prompt} /no_think"},
        {"role": "assistant", "content": assistant_answer},
    ]
    return {"messages": messages}


def synthesis_train_final_llm_summarization_dataset():
    with open("../data/train_llm_summarization_dataset.json", "r", encoding="utf-8") as f:
        train_dataset = json.load(f)
    print(f"Original train dataset size: {len(train_dataset)}")
    final_train_dataset = [formatting_func(example) for example in train_dataset]
    with open("../data/train_final_llm_summarization_dataset.json", "w", encoding="utf-8") as f:
        json.dump(final_train_dataset, f, ensure_ascii=False, indent=4)


def synthesis_val_final_llm_summarization_dataset():
    with open("../data/val_llm_summarization_dataset.json", "r", encoding="utf-8") as f:
        val_dataset = json.load(f)
    print(f"Original val dataset size: {len(val_dataset)}")
    final_val_dataset = [formatting_func(example) for example in val_dataset]
    with open("../data/val_final_llm_summarization_dataset.json", "w", encoding="utf-8") as f:
        json.dump(final_val_dataset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    synthesis_train_final_llm_summarization_dataset()
    synthesis_val_final_llm_summarization_dataset()
