import os
import time
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from faster_whisper import WhisperModel
from gtts import gTTS
from groq import Groq
from playsound import playsound 
import pyglet
import io

class RAGVoiceBot:
    def __init__(self, vector_db_path, knowldge_path, groq_token_path, whisper_size='base', model_name='llama-3.1-70b-versatile'):

        self.load_groq_token(groq_token_path)
        self.client = Groq()
        
        self.embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.whisper_model = self.initialize_STT_model(whisper_size)
        self.llm = self.initialize_llm(model_name)
        self.qa_pipeline = self.initialize_qa_pipeline(vector_db_path, knowldge_path, k=5)

    def load_groq_token(self, groq_token_path):
        with open(groq_token_path, 'r') as f:
            os.environ["GROQ_API_KEY"] = f.readline().strip()

    def initialize_STT_model(self, whisper_size):
        num_cores = os.cpu_count() // 2
        return WhisperModel(
            whisper_size,
            device="cpu",
            # compute_type="int8",
            cpu_threads=num_cores
        )
        

    def initialize_llm(self, model_name):
        return ChatGroq(
            model=model_name
        )

    def initialize_qa_pipeline(self, vector_db_path, knowldge_path, k):
        if os.path.exists(vector_db_path):
            print('Loading exisiting vector database')
            vec_db = FAISS.load_local(vector_db_path, self.embedding_model, allow_dangerous_deserialization=True)
        else:
            print('Creating new vector database')
            vec_db = self.create_vector_database_texts(knowldge_path)
            vec_db.save_local(vector_db_path)

            
        # The RAG pipeline
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=vec_db.as_retriever(search_kwargs={'k': 5}),
            # return_source_documents=True
        )

    def create_vector_database_docs_pdfs(self, knowldge_path):
        loader = PyPDFLoader(knowldge_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        return FAISS.from_documents(docs, self.embedding_model)
    
    def create_vector_database_texts(self, knowldge_path):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_chunks = []
        
        for filename in os.listdir(knowldge_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(knowldge_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_content = file.read()
                    
                    chunks = text_splitter.split_text(text_content)
                    all_chunks.extend(chunks)
                    
        return FAISS.from_texts(all_chunks, self.embedding_model)
    
    def speach_to_text(self, wav_audio):
        # print('we now opened the speech_to_text method .........')
        # segments, _ = self.whisper_model.transcribe(audio_path, language='ar')
        # return ''.join(segment.text for segment in segments)
        # with open(audio_path, "rb") as file:
        # audio_file 
        audio_file = ("audio.wav", wav_audio)
        transcription2 = self.client.audio.transcriptions.create(
            file=audio_file, 
            model="whisper-large-v3-turbo",
            prompt="Specify context or spelling",
            response_format="json",
            # language='ar',
            temperature=0.0,
        )
        
        return transcription2.text

    def generate_response(self, prompt):
        response_data = self.qa_pipeline.invoke(prompt)
        # print('Sources:\n')
        # print(response_data)
        return response_data['result']

    def text_to_speech(self, text, output_path=r'D:\GitHub projects\Mic_Server_Test\Backend\output_voices\speech.mp3'):
        start = time.time()
        tts = gTTS(text, lang='ar', slow=False)
        
        tts.save(output_path)
        print(f'Text to sound time: {time.time() - start}')

        # output_path = os.path.join(os.getcwd(), output_path)
        
        # music = pyglet.media.load("D:\GitHub projects\Mic_Server_Test\Backend\output_voices\speech.mp3", streaming=False)
        # music.play()
        # os.remove(output_path)
        return tts
                
    def process_audio_file(self, audio_path):
        start = time.time()
        transcription = self.speach_to_text(audio_path)
        print(f'Sound to text time: {time.time() - start}')

        start = time.time()
        
#         arabic_instruction_100_words = (
#     "يرجى تقديم الإجابة باللغة العربية فقط وباختصار لا يتجاوز 100 كلمة. إذا لم يتم العثور على المعلومات في السياق الحالي، يرجى محاولة الإجابة باستخدام المعرفة العامة: "
# )
        
        arabic_instruction = (
    "يرجى تقديم الإجابة باللغة العربية فقط. إذا لم يتم العثور على المعلومات في السياق الحالي، يرجى محاولة الإجابة باختصار باستخدام المعرفة العامة: "
)
        prompt = f"{arabic_instruction}\n{transcription}"
        # print(prompt)
        response = self.generate_response(prompt)
        print(f'Model response time: {time.time() - start}')
        
        
        tts_output = self.text_to_speech(response)
        return transcription, response, tts_output



if __name__ == '__main__':
    audio_path = r"C:\Users\zmlka\Documents\Sound Recordings\Recording (3).m4a"

    voice_bot = RAGVoiceBot(
        knowldge_path='./knowledge_base',
        groq_token_path='groq_token.txt',
        vector_db_path='vector_db'
    )

    transcription, response = voice_bot.process_audio_file(audio_path)

    print('='*50)

    print("USER:", transcription)
    print("Assistant:", response)