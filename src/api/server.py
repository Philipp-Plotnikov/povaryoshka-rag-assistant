from fastapi import FastAPI, WebSocket
import uvicorn
import soundfile as sf
import io
from faster_whisper import WhisperModel
import asyncio
from context_manager.context_manager import PovaryoshkaContextManager
from db.in_memory_database_driver import PovaryoshkaInMemoryDatabaseDriver
from db.vector_database_driver import PovaryoshkaVectorDatabaseDriver
from models.context_sufficiency_classifier.context_sufficiency_classifier import PovaryoshkaContextSufficiencyClassifier
from models.encoder.utils import load_encoder
from models.llm.llm import PovaryoshkaLLM
from query_router.query_router import PovaryoshkaQueryRouter
from rag.rag import PovaryoshkaRAG
from retriever.retriever import PovaryoshkaRetriever
import torch
import json
import base64
import subprocess

from training.encoder_train_loop.utils import get_train_chunk_list, get_val_chunk_list


POVARYOSHKA_SERVER = FastAPI()

# ===== MODELS =====
device = 'cpu'
whisper_model = WhisperModel("base", compute_type="int8", device=device)

# ===== Silero TTS =====
language = 'ru'
model_id = 'v5_ru'
sample_rate = 48000
speaker = 'xenia'  # или 'baya', 'aidar'

# Загружаем модель Silero
silero_tts, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language=language,
    speaker=model_id
) # type: ignore
silero_tts.to(device)

# Настройка RAG
train_chunk_list = get_train_chunk_list(device)
val_chunk_list = get_val_chunk_list(device)
common_chunk_list = train_chunk_list + val_chunk_list
in_memory_db_driver = PovaryoshkaInMemoryDatabaseDriver()
context_manager_persistent_db_driver = PovaryoshkaVectorDatabaseDriver(
    collection_name="dialogue_history"
)
retriever_persistent_db_driver = PovaryoshkaVectorDatabaseDriver(
    collection_name="documents"
)
encoder = load_encoder()
llm = PovaryoshkaLLM()
context_manager = PovaryoshkaContextManager(
    encoder=encoder,
    llm=llm,
    in_memory_db_driver=in_memory_db_driver,
    persistent_db_driver=context_manager_persistent_db_driver,
    max_history_length=3
)
retriever = PovaryoshkaRetriever(
    encoder=encoder,
    persistent_db_driver=retriever_persistent_db_driver
)
context_sufficiency_classifier = PovaryoshkaContextSufficiencyClassifier(
    retriever_persistent_db_driver=retriever_persistent_db_driver
)
query_router = PovaryoshkaQueryRouter(
    encoder=encoder,
    context_sufficiency_classifier=context_sufficiency_classifier
)
retriever.build_index(
    [chunk['text'] for chunk in common_chunk_list]
)
rag = PovaryoshkaRAG(
    context_manager=context_manager,
    query_router=query_router,
    retriever=retriever,
    llm=llm
)

# ===== Функция синтеза речи =====
def synthesize_russian_ogg(text: str) -> bytes:
    audio_tensor = silero_tts.apply_tts(
        text=text,
        speaker=speaker,
        sample_rate=sample_rate
    )
    audio_np = audio_tensor.cpu().numpy().astype("float32")

    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio_np, sample_rate, format="WAV")
    wav_buf.seek(0)

    # 🔥 WAV → OGG/Opus
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-i", "pipe:0",
            "-c:a", "libopus",
            "-b:a", "32k",      # битрейт (можно 16k–64k)
            "-vbr", "on",
            "-f", "ogg",
            "pipe:1"
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    ogg_audio, _ = process.communicate(wav_buf.read())
    return ogg_audio

CHUNK_SIZE = 64 * 1024  # 64 KB

async def send_audio_stream(ws: WebSocket, audio_bytes: bytes, utterance_id: int, text: str):
    total_size = len(audio_bytes)

    # 🔹 1. старт
    start_msg = {
        "type": "audio_start",
        "utterance_id": utterance_id,
        "format": "ogg_opus",
        "total_size": total_size,
        "chunk_size": CHUNK_SIZE,
        "text": text  # опционально
    }
    await ws.send_text(json.dumps(start_msg))

    # 🔹 2. чанки (бинарные)
    for i in range(0, total_size, CHUNK_SIZE):
        chunk = audio_bytes[i:i + CHUNK_SIZE]
        await ws.send_bytes(chunk)

    # 🔹 3. конец
    end_msg = {
        "type": "audio_end",
        "utterance_id": utterance_id
    }
    await ws.send_text(json.dumps(end_msg))

# ===== Обработчик WebSocket =====
@POVARYOSHKA_SERVER.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket):
    await websocket.accept()
    
    # Информация о клиенте
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"[{client_host}] Подключился")
    
    # Состояние сессии
    session = {
        "user_id": client_host,
        "last_utterance_id": -1,
        "is_processing": False
    }
    
    try:
        while True:
            # Принимаем сообщение
            message = await websocket.receive_text()
            
            # Парсим JSON
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                print(f"[{client_host}] Ошибка: получен невалидный JSON")
                continue
            
            # Обрабатываем только сообщения с аудио
            if data.get("type") != "audio_final":
                print(f"[{client_host}] Неизвестный тип сообщения: {data.get('type')}")
                continue
            
            # Получаем метаданные
            utterance_id = data.get("utterance_id", 0)
            audio_b64 = data.get("audio")
            duration = data.get("duration", 0)
            
            if not audio_b64:
                print(f"[{client_host}] Нет аудио в сообщении")
                continue
            
            # Декодируем аудио из base64
            try:
                wav_bytes = base64.b64decode(audio_b64)
            except Exception as e:
                print(f"[{client_host}] Ошибка декодирования base64: {e}")
                continue
            
            print(f"[{client_host}] Получена фраза #{utterance_id}, длительность: {duration:.2f}s, размер: {len(wav_bytes)} bytes")
            
            # Транскрибируем аудио
            try:
                # Читаем WAV из bytes
                audio_np, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
                
                # Whisper транскрипция
                segments, _ = whisper_model.transcribe(audio_np, language="ru")
                text = " ".join([seg.text for seg in segments]).strip()
                
                if not text:
                    print(f"[{client_host}] Фраза #{utterance_id}: текст не распознан")
                    # Отправляем уведомление, что ничего не распознано
                    response = {
                        "type": "error",
                        "utterance_id": utterance_id,
                        "message": "Речь не распознана"
                    }
                    await websocket.send_text(json.dumps(response))
                    continue
                    
                print(f"[{client_host}] Фраза #{utterance_id}: \"{text}\"")
                
            except Exception as e:
                print(f"[{client_host}] Ошибка транскрипции: {e}")
                continue
            
            # Получаем ответ от RAG (запускаем в отдельном потоке, чтобы не блокировать)
            try:
                session["is_processing"] = True
                answer = await asyncio.to_thread(rag.generate, user_id=client_host, query=text)
                print(f"[{client_host}] RAG ответ #{utterance_id}: \"{answer[:100]}...\"")
            except Exception as e:
                print(f"[{client_host}] Ошибка RAG: {e}")
                answer = "Извините, произошла ошибка при обработке запроса."
            finally:
                session["is_processing"] = False
            
            # Синтезируем речь
            try:
                ogg_audio = synthesize_russian_ogg(answer)

                await send_audio_stream(
                    websocket,
                    ogg_audio,
                    utterance_id,
                    answer
                )

                print(f"[{client_host}] Отправлен OGG поток #{utterance_id}, размер: {len(ogg_audio)} bytes")

            except Exception as e:
                print(f"[{client_host}] Ошибка TTS: {e}")
                response = {
                    "type": "error",
                    "utterance_id": utterance_id,
                    "message": "Ошибка синтеза речи"
                }
                await websocket.send_text(json.dumps(response))
    
    except Exception as e:
        print(f"[{client_host}] Общая ошибка WebSocket: {e}")
    finally:
        print(f"[{client_host}] Отключился")
        await websocket.close()

# ===== Здоровье и статус =====
@POVARYOSHKA_SERVER.get("/health")
async def health_check():
    return {
        "status": "ok",
        "models": {
            "whisper": "base",
            "tts": "silero_v5_ru",
            "rag": "active"
        }
    }

@POVARYOSHKA_SERVER.get("/status")
async def get_status():
    return {
        "server": "running",
        "device": device,
        "sample_rate": sample_rate,
        "tts_speaker": speaker
    }

if __name__ == "__main__":
    uvicorn.run(
        POVARYOSHKA_SERVER, 
        host="0.0.0.0", 
        port=8080,
        ws_max_size=16 * 1024 * 1024  # 16 MB для WebSocket
    )
