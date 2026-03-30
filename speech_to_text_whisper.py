import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# Settings
DURATION = 10        # seconds to record
SAMPLE_RATE = 16000 # whisper expects 16kHz
OUTPUT_FILE = "temp_audio.wav"

print("🎤 Speak now...")

# Record audio from microphone
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='int16'
)
sd.wait()  # wait until recording is done

# Save to temp wav file
wav.write(OUTPUT_FILE, SAMPLE_RATE, audio)

# Load Whisper model and transcribe
model = whisper.load_model("base")
result = model.transcribe(OUTPUT_FILE)

print("\n✅ Recognized Text:")
print(result["text"])