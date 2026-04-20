import asyncio
import sounddevice as sd
import numpy as np
import websockets
import io
import soundfile as sf
import json
import base64


SERVER_URI = "ws://localhost:8080/ws/voice"
CHUNK_DURATION = 0.5
SAMPLERATE = 16000
CHANNELS = 1
VOLUME_THRESHOLD = 0.008
MIN_AUDIO_SEC = 0.5
FINAL_SILENCE = 2.0

async def stream_microphone():
    # Флаг для предотвращения захвата собственного эха
    is_bot_speaking = False

    async with websockets.connect(SERVER_URI, max_size=16 * 1024 * 1024) as ws:
        print("🎤 Подключено к серверу. Говорите...")
        
        loop = asyncio.get_running_loop()
        buffer = []
        silence_time = 0.0
        utterance_id = 0
        is_speaking = False

        receiving_audio = False
        audio_chunks = []
        expected_size = 0
        received_size = 0

        def callback(indata, frames, time, status):
            nonlocal buffer, silence_time, is_speaking, utterance_id, is_bot_speaking
            
            # 🔥 ГЛАВНЫЙ ФИКС: Если бот говорит, игнорируем вход с микрофона
            if is_bot_speaking:
                return

            volume = np.abs(indata).mean()
            
            if volume < VOLUME_THRESHOLD:
                if is_speaking:
                    silence_time += CHUNK_DURATION
                    
                    if silence_time >= FINAL_SILENCE and buffer:
                        audio_np = np.concatenate(buffer, axis=0)
                        audio_duration = len(audio_np) / SAMPLERATE
                        
                        if audio_duration >= MIN_AUDIO_SEC:
                            buf = io.BytesIO()
                            sf.write(buf, audio_np, SAMPLERATE, format='WAV')
                            wav_bytes = buf.getvalue()
                            audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
                            
                            message = {
                                "type": "audio_final",
                                "utterance_id": utterance_id,
                                "audio": audio_b64,
                                "duration": audio_duration,
                                "format": "wav",
                                "sample_rate": SAMPLERATE
                            }
                            
                            print(f"\n📤 Отправка фразы #{utterance_id}: {audio_duration:.2f}s")
                            asyncio.run_coroutine_threadsafe(ws.send(json.dumps(message)), loop)
                            utterance_id += 1
                        
                        buffer = []
                        silence_time = 0.0
                        is_speaking = False
            else:
                if not is_speaking:
                    is_speaking = True
                    buffer = []
                    print("\n🎙️ Начало речи...", end="", flush=True)
                
                silence_time = 0.0
                buffer.append(indata.copy())

        with sd.InputStream(
            samplerate=SAMPLERATE,
            channels=CHANNELS,
            dtype='float32',
            blocksize=int(CHUNK_DURATION * SAMPLERATE),
            callback=callback
        ):
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.1)

                    if isinstance(msg, bytes):
                        if receiving_audio:
                            audio_chunks.append(msg)
                            received_size += len(msg)
                        continue

                    data = json.loads(msg)

                    if data["type"] == "audio_start":
                        receiving_audio = True
                        audio_chunks = []
                        received_size = 0
                        expected_size = data["total_size"]
                        print(f"\n🤖 Ответ: {data.get('text', '')[:50]}...")

                    elif data["type"] == "audio_end":
                        receiving_audio = False
                        full_audio = b"".join(audio_chunks)

                        try:
                            audio_np, sr = sf.read(io.BytesIO(full_audio), dtype="float32")
                            
                            # 🔥 Блокируем микрофон перед проигрыванием
                            is_bot_speaking = True
                            sd.play(audio_np, sr)
                            sd.wait() # Ждем окончания воспроизведения
                            
                            # 🔥 Сбрасываем все состояния, чтобы эхо не попало в новый буфер
                            buffer = []
                            is_speaking = False
                            silence_time = 0.0
                            is_bot_speaking = False
                            
                        except Exception as e:
                            print(f"❌ Ошибка воспроизведения: {e}")
                            is_bot_speaking = False

                        print("\n🎤 Слушаю...")

                    elif data["type"] == "error":
                        print(f"\n❌ Ошибка: {data.get('message')}")

                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break

if __name__ == "__main__":
    try:
        asyncio.run(stream_microphone())
    except KeyboardInterrupt:
        print("\n👋 Завершено")
