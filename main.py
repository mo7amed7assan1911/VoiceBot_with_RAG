from text_to_text_with_RAG import text_to_text_with_RAG
from Speach_to_text_Providers.stt_manager import SpeechToTextManager

from dotenv import load_dotenv
load_dotenv()

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


def main():
    
    rag = text_to_text_with_RAG(
        vector_db_path=VECTOR_DB_PATH,
        knowledge_base_path=KNOWLEDGE_BASE_PATH,
        metadata_path=METADATA_PATH,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        llm_provider=TTT_PROVIDER_NAME,
        model_name=TTT_MODEL_NAME,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE
    )
    
    
    print('='*50)
    print('System intialized successfully!')
    print("You can now start asking questions. Say `/bye` to exit.")
    print('='*50)
    
    stt_manager = SpeechToTextManager(mode=STT_PROVIDER_NAME, model_name=STT_MODEL_NAME)
    transcript = stt_manager.transcribe("./input_test_voices/audio.m4a")
    
    print(f"Transcript: {transcript}")
    
    while True:
        user_query = input("User: ")
        if user_query.lower() == '/bye':
            print('System: Good bye!')
            break
        
        try:
            response, relevant_chunks = rag.process_user_message(user_query)
            print("Relevant Chunks:")
            for chunk in relevant_chunks:
                print(f"- {chunk}")
            print(f"Response:\n{response}\n")
        except Exception as e:
            print(f"An error occurred: {e}\n")


if __name__ == "__main__":
    main()