from fastapi import FastAPI, WebSocket
import uvicorn
import numpy as np
import soundfile as sf
import io
from faster_whisper import WhisperModel
import asyncio
from rag import RAGSystem, common_document_list, pruned_common_chunk_list
from retriever import PovaryoshkaRetriever
from teacher_retriever_pool import TeacherRetrieverPool, DenseModel, BM25Model
from llm import generate
import torch

app = FastAPI()

# ===== MODELS =====
device = 'cpu'
whisper_model = WhisperModel("base", compute_type="int8", device=device)

# ===== Silero TTS =====
language = 'ru'
model_id = 'v5_ru'
sample_rate = 48000
speaker = 'xenia'

silero_tts, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id) # type: ignore
silero_tts.to(device)  # gpu or cpu

# Настройка RAG
teacher_retriever_pool = TeacherRetrieverPool(1, top_k=5)
teacher_retriever_pool.add_teacher('bm25', BM25Model(pruned_common_chunk_list))
teacher_retriever_pool.add_teacher('dense', DenseModel(pruned_common_chunk_list, device=device))

trained_retriever_model = PovaryoshkaRetriever(
    common_document_list,
    teacher_retriever_pool,
    temperature=0.1,
    dtype='float32',
    nlist=1,
    device=device,
    matryoshka_dims=[384]
)

trained_retriever_model.build_index()
rag = RAGSystem(trained_retriever_model, generate)

# ===== Функция синтеза речи =====
def synthesize_russian(text: str):
    # Генерация аудио
    audio_tensor = silero_tts.apply_tts(
        text=text,
        speaker=speaker,
        sample_rate=sample_rate
    )
    # Преобразуем Tensor → numpy
    audio_np = audio_tensor.cpu().numpy().astype("float32")

    # Конвертируем в WAV bytes
    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    await ws.accept()
    user_id = ws.client.host  # type: ignore
    buffer = []

    try:
        while True:
            data = await ws.receive_bytes()
            if not data:
                continue

            buffer.append(data)

            # Преобразуем весь буфер в аудио для Whisper
            try:
                audio_np, sr = sf.read(io.BytesIO(b"".join(buffer)), dtype="float32")
            except RuntimeError:
                audio_np = np.frombuffer(b"".join(buffer), dtype=np.int16).astype(np.float32)/32768
                sr = 16000

            buffer = []  # очистка после обработки

            # Whisper транскрибирует речь
            segments, _ = whisper_model.transcribe(audio_np, language="ru")
            text = " ".join([seg.text for seg in segments]).strip()

            if text:
                print(f"[User {user_id}]: {text}")

                # RAG ответ
                answer = await asyncio.to_thread(rag.answer, text, user_id=user_id)
                print(f"[RAG]: {answer}")

                # Silero TTS: преобразуем в WAV bytes
                wav_bytes = synthesize_russian(answer)

                # Отправляем по кускам ≤1 МБ, чтобы точно не было ошибки
                chunk_size = 1024 * 1024
                for i in range(0, len(wav_bytes), chunk_size):
                    await ws.send_bytes(wav_bytes[i:i+chunk_size])

    except Exception as e:
        print("Ошибка WebSocket:", e)
    finally:
        await ws.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_max_size=16 * 1024 * 1024) # увеличиваем лимит до 16 МБ для безопасной передачи WAV данных