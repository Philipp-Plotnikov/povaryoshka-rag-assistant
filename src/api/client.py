import asyncio
import sounddevice as sd
import numpy as np
import websockets
import io
import soundfile as sf
import json
import base64


SERVER_URI = "ws://localhost:8080/ws/voice"

SAMPLERATE = 16000
CHANNELS = 1
CHUNK_DURATION = 0.4
CHUNK_SIZE = int(SAMPLERATE * CHUNK_DURATION)

VOLUME_THRESHOLD = 0.008
FINAL_SILENCE = 1.5
MIN_AUDIO_SEC = 0.5


# =========================
# STATE
# =========================
class State:
    def __init__(self):
        self.buffer = []
        self.is_speaking = False
        self.silence_time = 0.0
        self.utterance_id = 0
        self.bot_speaking = False


state = State()

# очередь вместо run_coroutine_threadsafe
audio_queue = asyncio.Queue()


# =========================
# AUDIO CALLBACK (THREAD SAFE)
# =========================
def audio_callback(indata, frames, time, status):
    if state.bot_speaking:
        return

    volume = np.abs(indata).mean()

    if volume < VOLUME_THRESHOLD:
        if state.is_speaking:
            state.silence_time += CHUNK_DURATION

            if state.silence_time >= FINAL_SILENCE and state.buffer:
                audio_np = np.concatenate(state.buffer, axis=0)
                duration = len(audio_np) / SAMPLERATE

                if duration >= MIN_AUDIO_SEC:
                    # безопасно в очередь
                    audio_queue.put_nowait((audio_np, duration, state.utterance_id))
                    state.utterance_id += 1

                state.buffer = []
                state.silence_time = 0.0
                state.is_speaking = False

    else:
        if not state.is_speaking:
            state.is_speaking = True
            state.buffer = []
            print("\n🎙️ speaking...")

        state.silence_time = 0.0
        state.buffer.append(indata.copy())


# =========================
# SEND AUDIO
# =========================
async def send_audio(ws, audio_np, duration, utterance_id):
    buf = io.BytesIO()
    sf.write(buf, audio_np, SAMPLERATE, format="WAV")
    wav_bytes = buf.getvalue()

    msg = {
        "type": "audio_final",
        "utterance_id": utterance_id,
        "audio": base64.b64encode(wav_bytes).decode(),
        "duration": duration,
        "sample_rate": SAMPLERATE
    }

    print(f"\n📤 send #{utterance_id} ({duration:.2f}s)")
    await ws.send(json.dumps(msg))


# =========================
# AUDIO PLAYBACK
# =========================
def play_audio(audio_np, sr):
    state.bot_speaking = True
    sd.play(audio_np, sr)
    sd.wait()
    state.bot_speaking = False
    print("\n🎤 listening...")


async def play_audio_async(audio_bytes):
    audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    await asyncio.to_thread(play_audio, audio_np, sr)


# =========================
# RECEIVER
# =========================
async def receiver(ws):
    audio_buffer = []
    receiving = False

    async for msg in ws:

        if isinstance(msg, bytes):
            if receiving:
                audio_buffer.append(msg)
            continue

        data = json.loads(msg)

        if data["type"] == "audio_start":
            receiving = True
            audio_buffer = []
            print(f"\n🤖 {data.get('text', '')[:60]}...")

        elif data["type"] == "audio_end":
            receiving = False

            full_audio = b"".join(audio_buffer)

            try:
                await play_audio_async(full_audio)
            except Exception as e:
                print("playback error:", e)

        elif data["type"] == "error":
            print("❌ error:", data.get("message"))


# =========================
# AUDIO SENDER WORKER
# =========================
async def audio_sender(ws):
    while True:
        audio_np, duration, utterance_id = await audio_queue.get()
        await send_audio(ws, audio_np, duration, utterance_id)


# =========================
# MAIN
# =========================
async def main():
    print("🎤 connecting...")

    async with websockets.connect(
        SERVER_URI,
        max_size=16 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=20
    ) as ws:

        print("✅ connected")

        stream = sd.InputStream(
            samplerate=SAMPLERATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=audio_callback
        )

        # параллельные задачи
        async with asyncio.TaskGroup() as tg:
            tg.create_task(receiver(ws))
            tg.create_task(audio_sender(ws))

            with stream:
                await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 stopped")