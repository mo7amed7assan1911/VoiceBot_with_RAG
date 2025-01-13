import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from dotenv import load_dotenv
from text_to_text_with_RAG import text_to_text_with_RAG
from Speach_to_text_Providers.stt_manager import SpeechToTextManager
from Text_to_Speach_Providers.tts_manager import TextToSpeachManager
from config.settings import (
    VECTOR_DB_PATH,
    KNOWLEDGE_BASE_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    TTT_PROVIDER_NAME,
    TTT_MODEL_NAME,
    STT_MODEL_NAME,
    STT_PROVIDER_NAME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
)

# Load environment variables
load_dotenv()

# Global Configurations
SAMPLE_RATE = 44100  # Audio sample rate (Hz)
DURATION = 5  # Recording duration (seconds)
AUDIO_FILE = "recorded_audio.wav"  # Temporary audio file for processing
OUTPUT_VOICE_PATH = "output_voices/speech.mp3"

# Initialize RAG, STT, and TTS systems
rag = text_to_text_with_RAG(
    vector_db_path=VECTOR_DB_PATH,
    knowledge_base_path=KNOWLEDGE_BASE_PATH,
    metadata_path=METADATA_PATH,
    embedding_model_name=EMBEDDING_MODEL_NAME,
    llm_provider=TTT_PROVIDER_NAME,
    model_name=TTT_MODEL_NAME,
    max_tokens=DEFAULT_MAX_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
)
stt_manager = SpeechToTextManager(mode=STT_PROVIDER_NAME, model_name=STT_MODEL_NAME)
tss_manager = TextToSpeachManager(mode="elevenlabs")

# Utility Functions
def record_audio(duration: int, sample_rate: int) -> np.ndarray:
    """
    Record audio for the specified duration.

    Args:
        duration (int): Duration of the recording in seconds.
        sample_rate (int): Sample rate for recording.

    Returns:
        np.ndarray: Recorded audio data.
    """
    st.info(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()  # Wait for the recording to finish
    st.success("Recording completed!")
    return audio


def save_audio(audio: np.ndarray, filename: str, sample_rate: int) -> None:
    """
    Save audio data to a WAV file.

    Args:
        audio (np.ndarray): Recorded audio data.
        filename (str): Path to save the audio file.
        sample_rate (int): Sample rate of the audio.
    """
    write(filename, sample_rate, (audio * 32767).astype(np.int16))  # Convert float32 to int16
    st.success(f"Audio saved to {filename}")


def process_audio(audio_file: str) -> None:
    """
    Process the audio file: transcribe, generate response, and synthesize speech.

    Args:
        audio_file (str): Path to the audio file.
    """
    if not audio_file:
        st.warning("No audio file found. Please record audio first.")
        return

    # Transcription
    st.info("Transcribing audio...")
    transcription = stt_manager.transcribe(audio_file)
    st.text_area("Transcription Output", transcription, height=200)

    # Generate RAG response
    st.info("Generating response using RAG...")
    response, relevant_chunks = rag.process_user_message(transcription)
    st.text_area("Model Response", response, height=200)

    # Text-to-Speech Synthesis
    st.info("Synthesizing audio response...")
    tss_manager.synthesis(response, output_path=OUTPUT_VOICE_PATH)
    st.audio(OUTPUT_VOICE_PATH, format="audio/mp3")
    st.success("Response synthesized and played successfully!")

# Streamlit App Layout
st.title("Streamlit Voice Assistant with RAG")
st.markdown("Record audio, process it with RAG, and generate audio responses!")

# Record Audio
if st.button("Record Audio"):
    audio_data = record_audio(DURATION, SAMPLE_RATE)
    save_audio(audio_data, AUDIO_FILE, SAMPLE_RATE)

# Play Recorded Audio
if st.button("Play Audio"):
    if AUDIO_FILE:
        st.audio(AUDIO_FILE)
    else:
        st.warning("No audio file found. Please record audio first.")

# Transcribe and Process Audio
if st.button("Transcribe and Generate Response"):
    process_audio(AUDIO_FILE)
