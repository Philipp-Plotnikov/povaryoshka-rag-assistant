import asyncio
import sounddevice as sd
import numpy as np
import websockets
import io
import soundfile as sf

SERVER_URI = "ws://localhost:8000/ws/voice"
CHUNK_DURATION = 0.5
SAMPLERATE = 16000
CHANNELS = 1
VOLUME_THRESHOLD = 0.015
MIN_AUDIO_SEC = 0.3  # минимальная длина для отправки
FINAL_SILENCE = 1.2  # пауза для конца фразы

async def stream_microphone():
    async with websockets.connect(SERVER_URI) as ws:
        print("🎤 Начало стрима микрофона. Говорите...")

        loop = asyncio.get_running_loop()
        buffer = []
        silence_time = 0.0

        def callback(indata, frames, time, status):
            nonlocal buffer, silence_time
            volume = np.abs(indata).mean()
            
            if volume < VOLUME_THRESHOLD:
                silence_time += CHUNK_DURATION
            else:
                silence_time = 0.0
                buffer.append(indata.copy())

            # конец фразы — отправляем аудио на сервер
            if silence_time >= FINAL_SILENCE and buffer:
                audio_np = np.concatenate(buffer, axis=0)
                if len(audio_np)/SAMPLERATE >= MIN_AUDIO_SEC:
                    buf = io.BytesIO()
                    sf.write(buf, audio_np, SAMPLERATE, format='WAV')
                    wav_bytes = buf.getvalue()
                    print(f"[Отправка] {len(audio_np)/SAMPLERATE:.2f}s, volume={volume:.4f}, size={len(wav_bytes)} bytes")
                    asyncio.run_coroutine_threadsafe(ws.send(wav_bytes), loop)
                buffer = []
                silence_time = 0.0

        with sd.InputStream(
            samplerate=SAMPLERATE,
            channels=CHANNELS,
            dtype='float32',
            blocksize=int(CHUNK_DURATION*SAMPLERATE),
            callback=callback
        ):
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)

                    # проверяем тип данных
                    if isinstance(msg, str):
                        # текстовый ответ
                        print("🤖 RAG (текст):", msg)
                    else:
                        # бинарные данные — RAW PCM16 (assuming server sends raw)
                        # if server sends WAV, sf.read(io.BytesIO(msg)) is enough
                        # but if it fails, it might be RAW or a specific format.
                        # Let's try to wrap it in a try-except or check if it's RAW.
                        try:
                            audio, sr = sf.read(io.BytesIO(msg), dtype="float32")
                        except sf.LibsndfileError:
                            # Try reading as RAW PCM16 if WAV fails
                            audio, sr = sf.read(io.BytesIO(msg), samplerate=SAMPLERATE, 
                                               channels=CHANNELS, format='RAW', 
                                               subtype='PCM_16', dtype="float32")
                        print(f"🎧 Воспроизвожу аудио, {len(audio)/sr:.2f}s")
                        sd.play(audio, sr)
                except asyncio.TimeoutError:
                    pass

if __name__ == "__main__":
    asyncio.run(stream_microphone())