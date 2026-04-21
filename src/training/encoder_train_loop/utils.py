import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

from models.encoder.encoder_teacher_pool import PovaryoshkaEncoderTeacherPool


class PovaryoshkaEncoderDataset(Dataset):
    def __init__(self, ranked_chunk_list: list[dict]):
        self.ranked_chunk_list = ranked_chunk_list

    def __len__(self) -> int:
        return len(self.ranked_chunk_list)

    # TODO: randomly sample one question from questions list for each chunk, now it always takes the first one
    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        ranked_chunk = self.ranked_chunk_list[index]
        return ranked_chunk['questions'][0], ranked_chunk['index_tensors'][0]


def get_povaryoshka_encoder_collate_fn(encoder_teacher_pool: PovaryoshkaEncoderTeacherPool):
    current_epoch_number = 0

    def povaryoshka_encoder_collate_fn(
        query_data_batch: list[tuple[str, torch.Tensor]] # [(str, index_tensor[teacher_amount, top_k])]
    ) -> list[tuple[str, torch.Tensor]]: # [(str, index_tensor[current_teacher_amount, top_k])]
        encoder_teacher_amount = 1 if current_epoch_number == 0 else len(encoder_teacher_pool)
        updated_query_data_batch = []
        for query_data in query_data_batch:
            updated_query_data_batch.append(
                (query_data[0], query_data[1][:encoder_teacher_amount])
            )
        return updated_query_data_batch
    
    def set_current_epoch_number(value: int):
        nonlocal current_epoch_number
        current_epoch_number = value
    
    return povaryoshka_encoder_collate_fn, set_current_epoch_number


def copy_data_to_device(data, device):
  if torch.is_tensor(data):
    return data.to(device)
  elif isinstance(data, (list, tuple)):
    return [copy_data_to_device(elem, device) for elem in data]
  return data


def save_loss_plot(losses: list[float]):
    plt.figure()
    plt.plot(np.arange(len(losses)), losses)
    plt.title("Loss value")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig('../data/loss_plot.png', dpi=300, bbox_inches="tight")
    plt.close()


def get_train_chunk_list(device: str) -> list[dict]:
    train_ranked_chunk_list: list[dict] = torch.load('../data/train_ranked_recipe_chunks_1.pth')
    for train_chunk in train_ranked_chunk_list:
        train_chunk['index_tensors'] = train_chunk['index_tensors'].to(device)
    print(f"Train chunk list amount: {len(train_ranked_chunk_list)}")
    return train_ranked_chunk_list


def get_val_chunk_list(device: str) -> list[dict]:
    val_ranked_chunk_list: list[dict] = torch.load('../data/val_ranked_recipe_chunks_1.pth')
    for val_chunk in val_ranked_chunk_list:
        val_chunk['index_tensors'] = val_chunk['index_tensors'].to(device)
    print(f"Val chunk list amount: {len(val_ranked_chunk_list)}")
    return val_ranked_chunk_list
