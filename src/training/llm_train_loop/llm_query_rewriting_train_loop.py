import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig # type: ignore
import evaluate
import nltk
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from transformers import EarlyStoppingCallback

from db.vector_database_driver import PovaryoshkaVectorDatabaseDriver
from models.encoder.utils import load_encoder
from models.llm.llm import PovaryoshkaLLM
from retriever.retriever import PovaryoshkaRetriever
from training.encoder_train_loop.utils import get_train_chunk_list, get_val_chunk_list
from training.llm_train_loop.utils import build_prompt_for_answer_generation, build_prompt_for_summarization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


current_train_run_id = f"qwen3-4b-query-rewriting-sft-lora-{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}"
log_dir = f"../logs/tensorboard/{current_train_run_id}"
os.makedirs(log_dir, exist_ok=True)

writer = SummaryWriter(log_dir)

print(f"📊 TensorBoard логи будут сохраняться в: {log_dir}")
print(f"▶️ Для просмотра выполните: tensorboard --logdir=../logs/tensorboard")

BASE_MODEL_PATH = "../models/qwen3-4b"


tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(
    "json",
    data_files={
        "train": "../data/train_final_llm_query_rewriting_dataset.json",
        "validation": "../data/val_final_llm_query_rewriting_dataset.json"
    }
)
train_dataset = dataset["train"].select(range(2)) # type: ignore
val_dataset = dataset["validation"].select(range(2)) # type: ignore


# ============================================================
# 6. MODEL
# ============================================================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    attn_implementation="sdpa",
    # attn_implementation="eager",
    # dtype=torch.float16,
    dtype=torch.bfloat16
).to("cpu") # type: ignore
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.eos_token_id = tokenizer.eos_token_id

# ============================================================
# 7. КАСТОМНЫЙ ТРЕЙНЕР С МЕТРИКАМИ ДЛЯ TENSORBOARD
# ============================================================

# Загружаем метрики
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")
exact_match_metric = evaluate.load("exact_match")

# Глобальный счётчик шагов для TensorBoard
global_step_counter = 0

def preprocess_logits_for_metrics(logits, labels):
    return torch.argmax(logits, dim=-1)



def strip_prompt(predicted_ids, labels):
    mask = (labels != -100)

    return [
        predicted_ids[i][mask[i]]
        for i in range(predicted_ids.shape[0])
    ]

def compute_metrics_for_rag(eval_pred):
    predicted_ids, labels = eval_pred

    # важно: отдельно для decode labels
    labels_for_decode = np.where(
        labels != -100,
        labels,
        tokenizer.pad_token_id
    )

    # 1. strip prompt + tool logic
    cleaned_preds = strip_prompt(predicted_ids, labels)

    # 2. decode preds
    decoded_preds = [
        tokenizer.decode(seq, skip_special_tokens=True).strip("\n")
        for seq in cleaned_preds
    ]

    # 3. decode labels
    decoded_labels = tokenizer.batch_decode(
        labels_for_decode,
        skip_special_tokens=True
    )

    # 4. filter empty
    mask = [
        (p.strip() != "") and (l.strip() != "")
        for p, l in zip(decoded_preds, decoded_labels)
    ]

    decoded_preds = [p for p, m in zip(decoded_preds, mask) if m]
    decoded_labels = [l for l, m in zip(decoded_labels, mask) if m]

    if not decoded_preds:
        return {
            "rouge1": 0,
            "rougeL": 0,
            "bert_f1": 0,
            "exact_match": 0,
            "avg_gen_len": 0
        }

    # 5. ROUGE
    rouge_result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    # 6. BERTScore
    bertscore_result = bertscore_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        lang="ru",
        model_type="distilbert-base-multilingual-cased"
    )

    # 7. Exact match
    em_result = exact_match_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        ignore_case=True,
        ignore_punctuation=True
    )

    return {
        "rouge1": rouge_result["rouge1"], # type: ignore
        "rougeL": rouge_result["rougeL"], # type: ignore
        "bert_f1": float(np.mean(bertscore_result["f1"])), # type: ignore
        "exact_match": em_result["exact_match"], # type: ignore
        "avg_gen_len": float(np.mean([len(p.split()) for p in decoded_preds]))
    }


# ============================================================
# 8. КАСТОМНЫЙ КОЛБЭК ДЛЯ TENSORBOARD
# ============================================================
class TensorBoardCallback(TrainerCallback):
    """Логирует метрики обучения в TensorBoard"""
    
    def __init__(self, writer):
        self.writer = writer
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Логируем train loss
            if "loss" in logs:
                self.writer.add_scalar("train/loss", logs["loss"], state.global_step)
            if "grad_norm" in logs:
                self.writer.add_scalar("train/grad_norm", logs["grad_norm"], state.global_step)
            if "learning_rate" in logs:
                self.writer.add_scalar("train/learning_rate", logs["learning_rate"], state.global_step)
            
            # Логируем eval метрики (если они есть в logs)
            for key, value in logs.items():
                if key.startswith("eval_"):
                    metric_name = key.replace("eval_", "eval/")
                    self.writer.add_scalar(metric_name, value, state.global_step)


# ============================================================
# 9. LORA CONFIG
# ============================================================
peft_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.2,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# ============================================================
# 10. TRAINING ARGS
# ============================================================
sft_config = SFTConfig(
    max_length=5000,
    dataset_text_field="messages",
    shuffle_dataset=True,
    output_dir=f"models/{current_train_run_id}",
    per_device_train_batch_size=2, # 4
    per_device_eval_batch_size=1, # 2
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    num_train_epochs=1, # 1
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=150,
    save_strategy="steps",
    save_steps=150,
    save_total_limit=15,
    eval_accumulation_steps=1,
    bf16=True,
    optim="adamw_torch_fused",
    # optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_steps=10, # 130
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    assistant_only_loss=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    weight_decay=0.05
)

# ============================================================
# 11. SFT TRAINER
# ============================================================
tensorboard_callback = TensorBoardCallback(writer)
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=2,          # Ждем 5 эпох без улучшения
    early_stopping_threshold=0.001      # Считаем улучшением только изменение loss более чем на 0.001
)
trainer = SFTTrainer(
    model=base_model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    args=sft_config,
    compute_metrics=compute_metrics_for_rag,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[tensorboard_callback, early_stopping_callback]
)

# ============================================================
# 12. TRAIN
# ============================================================
print("\n" + "="*60)
print("НАЧАЛО ОБУЧЕНИЯ")
print("="*60)
print(f"📊 Для просмотра графиков выполните в другом терминале:")
print(f"   tensorboard --logdir=../logs/tensorboard")
print("="*60 + "\n")

# Записываем гиперпараметры в TensorBoard
hyperparams = {
    "model": "qwen3-4b",
    "lora_r": peft_config.r,
    "lora_alpha": peft_config.lora_alpha,
    "learning_rate": sft_config.learning_rate,
    "batch_size": sft_config.per_device_train_batch_size,
    "gradient_accumulation": sft_config.gradient_accumulation_steps,
    "max_length": sft_config.max_length,
    "epochs": sft_config.num_train_epochs,
}
for key, value in hyperparams.items():
    writer.add_text("hyperparams/" + key, str(value), 0)

trainer.train()

# ============================================================
# 13. SAVE
# ============================================================
trainer.model.save_pretrained( # type: ignore
    f"../models/{current_train_run_id}/final-model"
)
tokenizer.save_pretrained(
    f"../models/{current_train_run_id}/final-tokenizer"
)

# Закрываем TensorBoard writer
writer.close()
print(f"\n✅ Обучение завершено! Логи сохранены в: {log_dir}")
print(f"▶️ Для просмотра выполните: tensorboard --logdir=../logs/tensorboard")



# llm = PovaryoshkaLLM(lora_path="../models/qwen3-4b-summarization-sft-lora-2026.04.29-12:10:25/checkpoint-600")
# while True:
#     query = input("👤 Ты: ")
#     if query.lower() in ["exit", "quit"]:
#         print("Пока 👋")
#         break
#     text_list = query.split("\n")
#     prompt = build_prompt_for_summarization(text_list)
#     answer = llm.generate(prompt)
#     print("\n🤖 Ответ:")
#     print(answer)
#     print("\n" + "="*50 + "\n")