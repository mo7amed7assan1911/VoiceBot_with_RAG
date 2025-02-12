import pyaudio
import numpy as np
import wave
import time

# Audio settings
CHUNK = 1024  # Buffer size
FORMAT = pyaudio.paInt16  # 16-bit format
CHANNELS = 1  # Mono
RATE = 44100  # Sampling rate
RECORD_SECONDS = 4  # Recording duration
SOUND_THRESHOLD = 5000  # Adjust based on noise level

p = pyaudio.PyAudio()

# Open microphone stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening for high sound levels...")

def is_sound_high(data):
    """Check if the sound exceeds the threshold."""
    audio_data = np.frombuffer(data, dtype=np.int16)
    return np.max(np.abs(audio_data)) > SOUND_THRESHOLD

def record_audio():
    """Record audio for a fixed duration when a high sound is detected."""
    print("High sound detected! Recording for 4 seconds...")

    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    filename = f"recorded_{int(time.time())}.wav"
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Recording saved as {filename}")
    print("Waiting for new high sound...")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        if is_sound_high(data):
            record_audio()

except KeyboardInterrupt:
    print("\nStopping script...")
    stream.stop_stream()
    stream.close()
    p.terminate()
