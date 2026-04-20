import json


def get_train_pruned_recipe_chunks_with_questions() -> list[dict]:
    input_filename = '../data/train_pruned_recipe_chunks_with_questions_1.json'
    with open(input_filename, 'r', encoding='utf-8') as f:
        pruned_train_chunk_list = json.load(f)
    print(f"Загружено {len(pruned_train_chunk_list)} чанков из {input_filename}")
    return pruned_train_chunk_list


def get_val_pruned_recipe_chunks_with_questions() -> list[dict]:
    input_filename = '../data/val_pruned_recipe_chunks_with_questions_1.json'
    with open(input_filename, 'r', encoding='utf-8') as f:
        pruned_val_chunk_list = json.load(f)
    print(f"Загружено {len(pruned_val_chunk_list)} чанков из {input_filename}")
    return pruned_val_chunk_list
