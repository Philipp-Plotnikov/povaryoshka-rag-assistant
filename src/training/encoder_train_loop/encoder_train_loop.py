from typing import Callable
import torch
import copy
import datetime
import traceback

from training.encoder_train_loop.encoder_trainer import PovaryoshkaEncoderTrainer
from torch.utils.data import DataLoader, Dataset
from models.encoder.dense_models import DenseModel
from models.encoder.encoder import PovaryoshkaEncoder
from models.encoder.encoder_teacher_pool import PovaryoshkaEncoderTeacherPool
from models.encoder.sparse_models import SparseModel
from training.encoder_train_loop.utils import PovaryoshkaEncoderDataset, copy_data_to_device, save_loss_plot, get_val_chunk_list, get_povaryoshka_encoder_collate_fn, get_train_chunk_list


def run_encoder_train_loop(
    encoder_trainer: PovaryoshkaEncoderTrainer,
    set_current_epoch_number: Callable[[int], None],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: str,
    lr=1e-4,
    epoch_amount=10,
    l2_reg_alpha=0,
    optimizer_ctor=None,
    lr_scheduler_ctor=None,
    early_stopping_patience=10,
    k=5
):
    if optimizer_ctor is None:
        optimizer = torch.optim.Adam(encoder_trainer.parameters(), lr=lr, weight_decay=l2_reg_alpha)
    else:
        optimizer = optimizer_ctor(encoder_trainer.parameters(), lr=lr)

    if lr_scheduler_ctor is not None:
        lr_scheduler = lr_scheduler_ctor(optimizer)

    best_val_loss = float('inf')
    encoder_trainer = encoder_trainer.to(device)
    best_encoder_model = copy.deepcopy(encoder_trainer.get_encoder())
    loss_list = []
    for epoch_index in range(epoch_amount):
        try:
            encoder_trainer.train()
            set_current_epoch_number(epoch_index)
            epoch_start = datetime.datetime.now()
            print(f'Эпоха {epoch_index}')
            mean_train_loss = 0
            train_batches_n = 0
            for train_batch in train_dataloader:
                train_batch = copy_data_to_device(train_batch, device)
                train_loss = encoder_trainer(train_batch)
                encoder_trainer.zero_grad()
                train_loss.backward()
                optimizer.step()
                mean_train_loss += train_loss.item()
                train_batches_n += 1

            mean_train_loss /= train_batches_n
            loss_list.append(mean_train_loss)
            print('Эпоха: {} итераций, {:0.2f} сек'.format(train_batches_n,
                                                           (datetime.datetime.now() - epoch_start).total_seconds()))

            encoder_trainer.eval()
            mean_val_loss = 0
            val_batches_n = 0
            val_data = []
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_batch = copy_data_to_device(val_batch, device)
                    val_data.extend(val_batch)
                    val_loss = encoder_trainer(val_batch)
                    mean_val_loss += float(val_loss)
                    val_batches_n += 1

            mean_val_loss /= val_batches_n
            print(f'Среднее значение функции потерь на валидации {mean_val_loss}')

            if mean_val_loss < best_val_loss:
                best_epoch_index = epoch_index
                best_val_loss = mean_val_loss
                best_encoder_model = copy.deepcopy(encoder_trainer.get_encoder())
                print('Новая лучшая модель!')
                torch.save(best_encoder_model.state_dict(), '../models/encoder/povaryoshka_encoder_weights.pth')
                print("Модель сохранена!")
            elif epoch_index - best_epoch_index > early_stopping_patience:
                print(f'Модель не улучшилась за последние {early_stopping_patience} эпох, прекращаем обучение')
            print()
        except KeyboardInterrupt:
            print('Досрочно остановлено пользователем')
            break
        except Exception as ex:
            print(f'Ошибка при обучении: {ex}\n{traceback.format_exc()}')
            break
    save_loss_plot(loss_list)


def compute_recall_at_k(
    encoder_trainer: PovaryoshkaEncoderTrainer,
    test_dataloader: DataLoader,
    k=5
):
    encoder_trainer.eval()
    encoder_teacher_pool = encoder_trainer.get_encoder_teacher_pool()
    with torch.inference_mode():
        for query_batch_data in test_dataloader:
            query_list, positive_document_index_tensor = encoder_teacher_pool(query_batch_data)
            recall_at_k = encoder_trainer.compute_recall_at_k(
                query_list,
                positive_document_index_tensor.cpu().numpy().tolist(),
                k=k
            )
            print(f"Query amount: {len(query_list)}")
            print(f'Recall@{k}: {recall_at_k}')


if __name__ == "__main__":
    device = 'cpu'
    batch_size = 60
    encoder = PovaryoshkaEncoder(matryoshka_dims=[384])
    train_chunk_list = get_train_chunk_list(device)
    val_chunk_list = get_val_chunk_list(device)
    common_chunk_list = train_chunk_list + val_chunk_list
    encoder_teacher_pool = PovaryoshkaEncoderTeacherPool(top_k=5)
    encoder_teacher_pool.add_teacher('bm25', SparseModel(common_chunk_list))
    encoder_teacher_pool.add_teacher('deepvk/USER2-small', DenseModel(common_chunk_list, device='mps'))
    encoder_trainer = PovaryoshkaEncoderTrainer(
        encoder,
        encoder_teacher_pool,
        [chunk['text'] for chunk in common_chunk_list]
    )
    train_dataset = PovaryoshkaEncoderDataset(train_chunk_list)
    povaryoshka_encoder_collate_fn, set_current_epoch_number = get_povaryoshka_encoder_collate_fn(
        encoder_teacher_pool
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=povaryoshka_encoder_collate_fn,
        num_workers=0
    )
    val_dataset = PovaryoshkaEncoderDataset(val_chunk_list)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=povaryoshka_encoder_collate_fn,
        num_workers=0
    )
    test_dataloader = DataLoader(
        val_dataset,
        batch_size=173,
        shuffle=True,
        collate_fn=povaryoshka_encoder_collate_fn,
        num_workers=0
    )
    run_encoder_train_loop(
        encoder_trainer,
        set_current_epoch_number,
        train_dataloader,
        val_dataloader,
        device
    )
    compute_recall_at_k(encoder_trainer, test_dataloader)

