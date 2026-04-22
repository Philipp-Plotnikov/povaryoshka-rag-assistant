from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
import uvicorn
import re
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

# ======================
# MODELS
# ======================
device = "cpu"
whisper_model = WhisperModel("base", compute_type="int8", device=device)

language = "ru"
model_id = "v5_ru"
sample_rate = 48000
speaker = "xenia"

silero_tts, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_tts",
    language=language,
    speaker=model_id
) # type: ignore
silero_tts.to(device)

# ======================
# RAG SETUP
# ======================
train_chunk_list = get_train_chunk_list(device)
val_chunk_list = get_val_chunk_list(device)
common_chunk_list = train_chunk_list + val_chunk_list

in_memory_db_driver = PovaryoshkaInMemoryDatabaseDriver()
context_manager_persistent_db_driver = PovaryoshkaVectorDatabaseDriver("dialogue_history")
retriever_persistent_db_driver = PovaryoshkaVectorDatabaseDriver("documents")

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

retriever.build_index([c["text"] for c in common_chunk_list])

rag = PovaryoshkaRAG(
    context_manager=context_manager,
    query_router=query_router,
    retriever=retriever,
    llm=llm
)

# ======================
# AUDIO UTILS
# ======================
CHUNK_SIZE = 16 * 1024


def split_text(text: str, max_len: int = 350) -> list[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    # сначала разбиваем по "естественным границам"
    rough_chunks = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for part in rough_chunks:
        # если одно предложение уже слишком длинное — режем его
        if len(part) > max_len:
            words = part.split(" ")
            for w in words:
                if len(current) + len(w) + 1 <= max_len:
                    current += w + " "
                else:
                    flush()
                    current = w + " "
        else:
            if len(current) + len(part) + 1 <= max_len:
                current += part + " "
            else:
                flush()
                current = part + " "
    flush()
    return chunks

def synthesize_russian_wav(text: str):
    chunks = split_text(text)

    audio_parts = []

    for chunk in chunks:
        audio_tensor = silero_tts.apply_tts(
            text=chunk,
            speaker=speaker,
            sample_rate=sample_rate
        )
        audio_parts.append(audio_tensor.cpu())

    full_audio = torch.cat(audio_parts)
    return full_audio.numpy().astype("float32")

def synthesize_ogg(text: str) -> bytes:
    audio_np = synthesize_russian_wav(text)

    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio_np, sample_rate, format="WAV")
    wav_buf.seek(0)

    process = subprocess.Popen(
        [
            "ffmpeg",
            "-i", "pipe:0",
            "-c:a", "libopus",
            "-b:a", "32k",
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


# ======================
# SAFE STREAMING
# ======================
async def send_audio_stream(ws: WebSocket, audio: bytes, utterance_id: int):
    try:
        await ws.send_text(json.dumps({
            "type": "audio_start",
            "utterance_id": utterance_id,
            "text": "Ответ готовится, сейчас начнется воспроизведение аудио"
        }))
        for i in range(0, len(audio), CHUNK_SIZE):
            await ws.send_bytes(audio[i:i + CHUNK_SIZE])

        await ws.send_text(json.dumps({
            "type": "audio_end",
            "utterance_id": utterance_id
        }))

    except Exception as e:
        print("stream error:", e)


# ======================
# WEB SOCKET
# ======================
@POVARYOSHKA_SERVER.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket):
    await websocket.accept()

    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"[{client_host}] connected")

    try:
        while True:

            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                print(f"[{client_host}] disconnected")
                return

            try:
                data = json.loads(raw)
            except:
                continue

            if data.get("type") != "audio_final":
                continue

            utterance_id = data.get("utterance_id", 0)
            audio_b64 = data.get("audio")

            if not audio_b64:
                continue

            # ======================
            # decode audio
            # ======================
            try:
                wav_bytes = base64.b64decode(audio_b64)
                audio_np, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            except Exception as e:
                print("decode error:", e)
                continue

            # ======================
            # whisper (non-blocking)
            # ======================
            try:
                segments, _ = await asyncio.to_thread(
                    whisper_model.transcribe,
                    audio_np,
                    language="ru"
                )

                user_text = " ".join(s.text for s in segments).strip()

                if not user_text:
                    continue

            except Exception as e:
                print("whisper error:", e)
                continue

            print(f"[{client_host}] user: {user_text}")

            # ======================
            # RAG
            # ======================
            try:
                answer = await asyncio.to_thread(
                    rag.generate,
                    user_id=client_host,
                    query=user_text
                )
            except Exception as e:
                print("rag error:", e)
                answer = "Ошибка обработки запроса"

            # ======================
            # TTS (NON BLOCKING)
            # ======================
            try:
                ogg = await asyncio.to_thread(synthesize_ogg, answer)
            except Exception as e:
                print("tts error:", e)
                continue

            # ======================
            # SEND AUDIO
            # ======================
            await send_audio_stream(
                websocket,
                ogg,
                utterance_id
            )

    except Exception as e:
        print(f"[{client_host}] fatal error:", e)

    finally:
        try:
            await websocket.close()
        except:
            pass


# ======================
# STATUS
# ======================
@POVARYOSHKA_SERVER.get("/health")
async def health():
    return {"status": "ok"}


@POVARYOSHKA_SERVER.get("/status")
async def status():
    return {
        "server": "running",
        "device": device,
        "tts": "silero"
    }


# ======================
# RUN
# ======================
if __name__ == "__main__":
    uvicorn.run(
        POVARYOSHKA_SERVER,
        host="0.0.0.0",
        port=8080,
        ws_max_size=16 * 1024 * 1024
    )
