import streamlit as st
import pyaudio
import wave

# Initialize session state for recording
if 'is_recording' not in st.session_state:
    st.session_state['is_recording'] = False

# Function to record audio
def record_audio(filename="output.wav", duration=5, chunk=1024, format=pyaudio.paInt16, channels=1, rate=44100):
    audio = pyaudio.PyAudio()

    # Open stream
    stream = audio.open(format=format, channels=channels,
                        rate=rate, input=True,
                        frames_per_buffer=chunk)

    st.write("Recording...")
    frames = []

    try:
        for _ in range(0, int(rate / chunk * duration)):
            if not st.session_state['is_recording']:
                break
            data = stream.read(chunk)
            frames.append(data)
    finally:
        st.write("Finished recording.")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save the recording to a file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

        return filename

# Title of the app
st.title("Audio Recorder")

output_file = "recorded_audio.wav"

# Button to start recording
if st.button("Start Recording"):
    st.session_state['is_recording'] = True
    record_audio(filename=output_file, duration=60)  # Set max duration to 60 seconds

# Button to stop recording
if st.button("Stop Recording"):
    st.session_state['is_recording'] = False

# Button to play the recorded audio
if st.button("Play Audio"):
    if output_file:
        audio_file = open(output_file, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')
    else:
        st.error("No audio file found. Please record audio first.")

# Provide a placeholder for future extensions
st.info("You can extend this app to process or analyze the recorded audio.")
