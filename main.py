from Rag_System import text_to_text_with_RAG
from Speech_to_text_Providers.stt_manager import SpeechToTextManager
from Text_to_Speech_Providers.tts_manager import TextToSpeachManager

from elevenlabs import stream

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


def main():
    streaming_mode = True
    
    rag = text_to_text_with_RAG(
        vector_db_path=VECTOR_DB_PATH,
        knowledge_base_path=KNOWLEDGE_BASE_PATH,
        metadata_path=METADATA_PATH,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        llm_provider=TTT_PROVIDER_NAME,
        model_name=TTT_MODEL_NAME,
        rephraser_model_name=REPHRASER_MODEL_NAME,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        streaming_mode=streaming_mode
    )
    
    
    print('='*50)
    print('System intialized successfully!')
    print("You can now start asking questions. Say `/bye` to exit.")
    print('='*50)
    
    stt_manager = SpeechToTextManager(mode=STT_PROVIDER_NAME, model_name=STT_MODEL_NAME)
    tts_manager = TextToSpeachManager(mode='elevenlabs')

    # transcript = stt_manager.transcribe("./input_test_voices/audio.m4a")
    # tts_manager.synthesis(transcript)
    # print(f"Transcript: {transcript}")
    
    while True:
        user_query = input("User: ")
        if user_query.lower() == '/bye':
            print('System: Good bye!')
            break
        
        try:
            if streaming_mode:
                streaming, relevant_chunks = rag.process_user_message(user_query)
                print('='*50)
                print("Relevant Chunks:")
                for chunk in relevant_chunks:
                    print("-"*50)
                    print("- ", chunk)
                    
                    
                print('='*50)
                print('Answer: ')
                for chunk in streaming:
                    print(chunk, end='', flush=True)
                print()
                print('='*50)
                
                # def text_stream():
                #     yield "Hi there, I'm Eleven "
                #     yield "I'm a text to speech API "
                    
                # audio_stream = tts_manager.synthesis(text=streaming, streaming_mode=True)
                # stream(audio_stream)
                    
            else:
                response, _ = rag.process_user_message(user_query)
                print(f"\nResponse:\n{response}\n")
            # tss_manager.synthesis(response, output_path='output_voices/speech.mp3')

    
        except Exception as e:
            print(f"An error occurred: {e}\n")

if __name__ == "__main__":
    main()