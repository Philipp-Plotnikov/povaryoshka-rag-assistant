import json
from typing import Any

from db.vector_database_driver import PovaryoshkaVectorDatabaseDriver
from models.encoder.utils import load_encoder
from retriever.retriever import PovaryoshkaRetriever
from training.encoder_train_loop.utils import get_train_chunk_list, get_val_chunk_list
from training.llm_train_loop.utils import build_prompt_for_answer_generation

    
def formatting_func(example: dict[str, Any], retriever: PovaryoshkaRetriever) -> dict[str, Any]:
    user_query = example["messages"][0]["content"]
    assistant_answer = example["messages"][1]["content"]
    document_list = [doc['text'] for doc in retriever.get_chunk_list(user_query)]
    system_prompt = build_prompt_for_answer_generation(
        query=user_query,
        document_list=document_list
    )
    messages = [
        {"role": "user", "content": f"{system_prompt} /no_think"},
        {"role": "assistant", "content": assistant_answer},
    ]
    return {"messages": messages}


def synthesis_train_final_llm_answer_generation_dataset(retriever: PovaryoshkaRetriever):
    with open("../data/train_llm_answer_generation_dataset.json", "r", encoding="utf-8") as f:
        train_dataset = json.load(f)
    print(f"Original train dataset size: {len(train_dataset)}")
    final_train_dataset = [formatting_func(example, retriever) for example in train_dataset]
    with open("../data/train_final_llm_answer_generation_dataset.json", "w", encoding="utf-8") as f:
        json.dump(final_train_dataset, f, ensure_ascii=False, indent=4)


def synthesis_val_final_llm_answer_generation_dataset(retriever: PovaryoshkaRetriever):
    with open("../data/val_llm_answer_generation_dataset.json", "r", encoding="utf-8") as f:
        val_dataset = json.load(f)
    print(f"Original val dataset size: {len(val_dataset)}")
    final_val_dataset = [formatting_func(example, retriever) for example in val_dataset]
    with open("../data/val_final_llm_answer_generation_dataset.json", "w", encoding="utf-8") as f:
        json.dump(final_val_dataset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    device = "cpu"
    train_chunk_list = get_train_chunk_list(device)
    val_chunk_list = get_val_chunk_list(device)
    common_chunk_list = train_chunk_list + val_chunk_list
    retriever_persistent_db_driver = PovaryoshkaVectorDatabaseDriver(
        collection_name="documents"
    )
    encoder = load_encoder()
    retriever = PovaryoshkaRetriever(
        encoder=encoder,
        persistent_db_driver=retriever_persistent_db_driver
    )
    retriever.build_index([chunk['text'] for chunk in common_chunk_list])
    synthesis_train_final_llm_answer_generation_dataset(retriever)
    synthesis_val_final_llm_answer_generation_dataset(retriever)
