import streamlit as st
import requests
import time
from io import BytesIO
from audio_recorder_streamlit import audio_recorder

from Rag_System import text_to_text_with_RAG
from Speech_to_text_Providers.stt_manager import SpeechToTextManager
from Text_to_Speech_Providers.tts_manager import TextToSpeachManager

from dotenv import load_dotenv
load_dotenv()

from config.settings import (
    VECTOR_DB_PATH,
    KNOWLEDGE_BASE_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    TTT_PROVIDER_NAME,
    TTT_MODEL_NAME,
    REPHRASER_MODEL_NAME,
    STT_MODEL_NAME,
    STT_PROVIDER_NAME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
)

# Initialize RAG system
if "rag_system" not in st.session_state:
    st.session_state.rag_system = text_to_text_with_RAG(
        vector_db_path=VECTOR_DB_PATH,
        knowledge_base_path=KNOWLEDGE_BASE_PATH,
        metadata_path=METADATA_PATH,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        llm_provider=TTT_PROVIDER_NAME,
        model_name=TTT_MODEL_NAME,
        rephraser_model_name=REPHRASER_MODEL_NAME,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE
    )
    st.session_state.stt_manager = SpeechToTextManager(mode=STT_PROVIDER_NAME, model_name=STT_MODEL_NAME)
    st.session_state.tts_manager = TextToSpeachManager(mode='elevenlabs')
    
    # st.session_state.messages = []  # Initialize chat history

    print("✅ System initialized once per session!")

# Shortcuts
rag = st.session_state.rag_system
stt_manager = st.session_state.stt_manager
tts_manager = st.session_state.tts_manager


st.title("🤖 Chat with RAG using Voice")

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

audio_bytes = audio_recorder()

if audio_bytes:
    with st.spinner('Transcribing your audio...'):
        transcript = stt_manager.transcribe(audio_bytes)
        st.success(f"Transcript: {transcript}")
    
    if transcript:
        # st.session_state.messages.append({'role': 'User', 'content': transcript})
        with st.spinner('Generating Response ...'):
            try:
                response, _ = rag.process_user_message(transcript)

            except Exception as e:
                st.error(f"An error occurred: {e}")

        with st.chat_message('assistant'):
                    st.write(response)
                    
        with st.spinner('Generating audio ...'):
            try:
                tts_audio = tts_manager.synthesis(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
            
        # st.audio(tts_audio, format='audio/mp3')




# user_query = st.chat_input('Type your message ...')
# if user_query:
#     st.session_state.messages.append({'role': 'User', 'content': user_query})
    
#     with st.spinner('Generating Response ...'):
#         try:
#             response, _ = rag.process_user_message(user_query)
#             # st.success(f"Response: {response}")
            
#             with st.chat_message('assistant'):
#                 st.write(response)
            
#         except Exception as e:
#             st.error(f"An error occurred: {e}")
            
#     with st.spinner('Generating audio ...'):
#         try:
#             tts_audio = tts_manager.synthesis(response)
#             st.audio(tts_audio, format='audio/mp3')
#         except Exception as e:
#             st.error(f"An error occurred: {e}")    