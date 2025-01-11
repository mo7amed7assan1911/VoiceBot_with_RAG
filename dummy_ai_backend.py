from flask import Flask, request, send_file, Response
import numpy as np
from flask_cors import CORS
import io
import time
import threading
import sys
import os
import logging
from logging.handlers import RotatingFileHandler
import ctypes
import socket
from AI import RAGVoiceBot_v2
from pydub import AudioSegment
from scipy.signal import butter, filtfilt
import noisereduce as nr
import wave
import pygame

# Set ffprobe path for pydub

class AudioProcessor:
    def __init__(self):
        
        
        self.voice_bot = RAGVoiceBot_v2(
            knowldge_path='D:\GitHub projects\Mic_Server_Test\Backend\knowledge_base',
            groq_token_path='D:\GitHub projects\Mic_Server_Test\Backend\groq_token.txt',
            vector_db_path='vector_db'
        )
        
        self.processed_audio = None
        self.is_processing = False
        self.processing_thread = None
    
    def low_pass_filter(data, cutoff=3000, fs=44100, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btpyte='low', analog=False)
        return filtfilt(b, a, data)
    
    def process_audio(self, audio_data):
        """Start processing audio in a separate thread"""
        self.is_processing = True
        self.processed_audio = None

        float_array = np.frombuffer(audio_data, dtype=np.float32)

        # Normalize float32 samples to int16 range
        int16_array = np.int16(float_array * 32767)

        # Convert to bytes in int16 format
        int16_bytes = int16_array.tobytes()

        wav_audio = io.BytesIO()
        
        # Write to wav file with wave library
        with wave.open(wav_audio, 'wb') as obj:
            obj.setnchannels(1)       # Set to 1 for mono or 2 for stereo
            obj.setsampwidth(2)       # 2 bytes per sample for int16 format
            obj.setframerate(44100)   # Set frame rate (sample rate) to 44100 Hz
            obj.writeframes(int16_bytes)
            
        # audio_segment = AudioSegment.from_raw(io.BytesIO(reduced_noise), sample_width=4, frame_rate=48000, channels=1)
        
        # Export as wav format in memory
        # wav_audio = io.BytesIO()
        # audio_segment.export(wav_audio, format="wav")
        # wav_audio.seek(0)
        # print('finished saving')
        wav_audio.seek(0)
        # with open('output_wav.wav', 'wb') as f:
        #     f.write(wav_audio.getvalue())
        logger.info(f"\nConverted audio data directly to WAV format for transcription.")
        
        self.processing_thread = threading.Thread(
            target=self._process_audio_thread,
            args=(wav_audio,)
        )
        self.processing_thread.start()

    def _process_audio_thread(self, samples):
        """Simulate processing steps and store result"""
        try:
            # Simulate Speech-to-Text
            transcription, response, audio_data = self.voice_bot.process_audio_file(samples)
            print(f'model transcription: {transcription}')
            print(f'Modle response LLM: {response}')
            print('='*50)

            # with open("output_voices\speech.mp3", 'rb') as file:
            #     audio_bytes = file.read()
            self.processed_audio = audio_data

            # self.processed_audio = tts_output.getvalue()
            # print(f'Processed audio size: {len(self.processed_audio)}')
        except Exception as e:
            logger.error(f"Error in processing thread: {e}")
            self.processed_audio = None

        finally:
            self.is_processing = False

    def get_audio(self):
        """Return processed audio if available"""
        
        return self.processed_audio
    
    def is_busy(self):
        """Check if still processing"""
        return self.is_processing


def is_port_available(port):
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except socket.error:
            return False


def find_available_port(start_port=5000, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    return None


def setup_logging():
    """Setup logging configuration"""
    # Get appropriate directory for logs
    if hasattr(sys, '_MEIPASS'):  # Running as exe
        log_dir = os.path.join(os.path.expanduser("~"), "Documents", "AudioServer")
    else:  # Running as script
        log_dir = os.path.dirname(os.path.abspath(__file__))

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "AudioServer.log")

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=1024 * 1024,  # 1MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger('AudioServer')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def is_admin():
    """Check if the program is running with admin privileges"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


# Setup logging
logger = setup_logging()

# Create Flask app
app = Flask(__name__)
CORS(app)

# Create global audio processor instance
audio_processor = AudioProcessor()


@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # Get audio data from request
        audio_data = request.files['audio'].read()

        # Start processing in background
        audio_processor.process_audio(audio_data)

        return "Processing started", 200

    except Exception as e:
        logger.error(f"Error receiving audio: {e}")
        return str(e), 500


@app.route('/get_audio', methods=['GET'])
def get_audio():
    try:
        # If still processing, return 204 No Content
        if audio_processor.is_busy():
            return Response("Audio processing not complete", status=503)

        # Get processed audio in int16 format (e.g., 44100 Hz, mono)
        audio_data = audio_processor.get_audio()
        if audio_data is None:
            return Response("Audio processing not complete", status=503)
        # # Convert int16 PCM to float32 PCM for Unity
        # int16_samples = np.frombuffer(audio_data, dtype=np.int16)
        # float32_samples = int16_samples.astype(np.float32) / 32767  # Normalize to [-1, 1] range
        # float32_bytes = float32_samples.tobytes()

        # # Prepare in-memory file for sending
        mem_file = io.BytesIO(audio_data)
        mem_file.seek(0)

        # Clear the processed audio to prevent multiple sends
        audio_processor.processed_audio = None

        return send_file(
            io.BytesIO(audio_data),
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='response.raw'
        )

    except Exception as e:
        logger.error(f"Error sending audio: {e}")
        return str(e), 500


def main():
    # Check if running as admin and warn if not
    if not is_admin():
        logger.warning("Running without administrator privileges. Some features might not work.")

    # Find available port
    port = find_available_port()
    if port is None:
        logger.error("Could not find an available port. Please close other applications and try again.")
        input("Press Enter to exit...")
        return

    host = '0.0.0.0'

    logger.info("=" * 50)
    logger.info("Audio Server Starting")
    logger.info("=" * 50)
    logger.info(f"Server URL: http://{host}:{port}")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 50)

    try:
        app.run(host=host, port=port)
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        logger.info("Server shutting down...")
        input("Press Enter to exit...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        input("Press Enter to exit...")