import torch

from data_synthesis.utils import get_train_pruned_recipe_chunks_with_questions, get_val_pruned_recipe_chunks_with_questions
from models.encoder.dense_models import DenseModel
from models.encoder.encoder_teacher_pool import PovaryoshkaEncoderTeacherPool
from models.encoder.sparse_models import SparseModel


if __name__ == "__main__":
    train_ranked_recipe_chunks_filename = '../data/train_ranked_recipe_chunks_1.pth'
    val_ranked_recipe_chunks_filename = '../data/val_ranked_recipe_chunks_1.pth'
    pruned_train_chunk_list = get_train_pruned_recipe_chunks_with_questions()
    pruned_val_chunk_list = get_val_pruned_recipe_chunks_with_questions()
    common_pruned_chunk_list = pruned_train_chunk_list + pruned_val_chunk_list
    encoder_teacher_pool = PovaryoshkaEncoderTeacherPool(top_k=5)
    encoder_teacher_pool.add_teacher('bm25', SparseModel(common_pruned_chunk_list))
    encoder_teacher_pool.add_teacher('deepvk/USER2-small', DenseModel(common_pruned_chunk_list, device='mps'))

    # TODO: Think do we need here to add context ?
    train_ranked_chunk_list = []
    for train_chunk in pruned_train_chunk_list:
        # TODO: not only the first
        first_question = f"Контекст: {train_chunk['recipe_name']}\n\n{train_chunk['questions'][0]}"
        train_chunk['questions'][0] = first_question
        index_tensor, _ = encoder_teacher_pool.get_index_and_ranking_tensor([first_question])
        train_chunk['index_tensors'] = index_tensor
        train_ranked_chunk_list.append(train_chunk)
    torch.save(train_ranked_chunk_list, train_ranked_recipe_chunks_filename)

    eval_ranked_chunk_list = []
    for val_chunk in pruned_val_chunk_list:
        # TODO: not only the first
        first_question = f"Контекст: {val_chunk['recipe_name']}\n\n{val_chunk['questions'][0]}"
        val_chunk['questions'][0] = first_question
        index_tensor, _ = encoder_teacher_pool.get_index_and_ranking_tensor([first_question])
        val_chunk['index_tensors'] = index_tensor
        eval_ranked_chunk_list.append(val_chunk)
    torch.save(eval_ranked_chunk_list, val_ranked_recipe_chunks_filename)
