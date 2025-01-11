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
from elevenlabs import save, play
from elevenlabs.client import ElevenLabs
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
        # audio_file = ("audio.wav", wav_audio)
        transcription2 = self.client.audio.transcriptions.create(
            file=wav_audio.getvalue(), 
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
    



class RAGVoiceBot_v2:
    def __init__(self, vector_db_path, knowldge_path, groq_token_path, whisper_size='base', model_name='llama-3.1-70b-versatile'):

        self.load_groq_token(groq_token_path)
        self.grok_client = Groq()

        self.elevenlabs_client = ElevenLabs(api_key='sk_4b8af34b2615328298ba8718fc90797eafbcc39d08382917')
        self.embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.whisper_model = self.initialize_STT_model(whisper_size)
        self.llm = self.initialize_llm(model_name)
        self.vector_db = self.intialize_vectore_db(vector_db_path, knowldge_path)
        # self.qa_pipeline = self.initialize_qa_pipeline(vector_db_path, knowldge_path, k=3)

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

    def intialize_vectore_db(self, vector_db_path, knowldge_path):
        if os.path.exists(vector_db_path):
            print('Loading exisiting vector database')
            vec_db = FAISS.load_local(vector_db_path, self.embedding_model, allow_dangerous_deserialization=True)
        else:
            print('Creating new vector database')
            vec_db = self.create_vector_database_texts(knowldge_path)
            vec_db.save_local(vector_db_path)

        return vec_db
    
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
            retriever=vec_db.as_retriever(search_kwargs={'k': k}),
            # return_source_documents=True
        )

    
    def get_relevant_chunks(self, query, k=5):
        docs = self.vector_db.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def construct_prompt(self, chunks, query):
        context = "\n".join([f"- {chunk}" for chunk in chunks])
    
        rag_template = f"""You are given a user query, some textual context and rules, all inside xml tags. You have to answer the query based on the context while respecting the rules.

<context>
 {context}
</context>

<rules>
- If you don't know, just say so.
- If you are not sure, ask for clarification.
- Answer in the same language as the user query.
- If the context appears unreadable or of poor quality, tell the user then answer as best as you can.
- If the answer is not in the context but you think you know the answer, explain that to the user then answer with your own knowledge.
- Answer directly and without using xml tags.
</rules>

<user_query>
{query}
</user_query>
"""


        return rag_template

    def get_model_response(self, full_prompt, model_name='llama-3.1-70b-versatile'):
        response = self.grok_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a professional conference assistant fluent in Arabic and English. Respond concisely and professionally ONLY IN ARABIC"},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=250,
            temperature=0.4,
        )
    
        return response.choices[0].message.content


    def process_user_message(self, query):
        relevant_chunks = self.get_relevant_chunks(query)
        full_prompt_after_rag = self.construct_prompt(relevant_chunks, query)
        response = self.get_model_response(full_prompt_after_rag, model_name='llama-3.1-70b-versatile')
    
        return response
    
    def create_vector_database_docs_pdfs(self, knowldge_path):
        loader = PyPDFLoader(knowldge_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        return FAISS.from_documents(docs, self.embedding_model)
    
    def create_vector_database_texts(self, knowldge_path):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
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
        print('we now opened the speech_to_text method .........')
        # segments, _ = self.whisper_model.transcribe(audio_path, language='ar')
        # return ''.join(segment.text for segment in segments)
        # with open(audio_path, "rb") as file:
        # audio_file 
        audio_file = ("audio.wav", wav_audio)
        transcription = self.grok_client.audio.transcriptions.create(
            file=audio_file, 
            model="whisper-large-v3-turbo",
            prompt="Specify context or spelling",
            response_format="json",
            # language='ar',
            temperature=0.0,
        )

        # with open(wav_audio, "rb") as file:
        #     # Create a transcription of the audio file
        #     transcription = self.grok_client.audio.transcriptions.create(
        #       file=(wav_audio, file.read()), # Required audio file
        #       model="whisper-large-v3", # Required model to use for transcription
        #       prompt="Specify context or spelling",  # Optional
        #       response_format="json",  # Optional
        #       # language="ar",  # Optional
        #       temperature=0.0  # Optional
        #     )
            # Print th
        
        return transcription.text.strip()

    def generate_response(self, prompt):
        response_data = self.qa_pipeline.invoke(prompt)
        print('Final prompt to LLM after RAG:\n')
        print(response_data['query'])
        return response_data['result']

    def text_to_speech(self, text, output_path="output_voices\speech.mp3"):
        print('***********'*100)
        start = time.time()
        tts = gTTS(text, lang='ar', slow=False)
        
        tts.save(output_path)
        print(f'Text to sound time: {time.time() - start}')

        # output_path = os.path.join(os.getcwd(), output_path)
        
        # music = pyglet.media.load("D:\GitHub projects\Mic_Server_Test\Backend\output_voices\speech.mp3", streaming=False)
        # music.play()
        # os.remove(output_path)
        return tts

    def text_to_sound(self, text):
        audio = self.elevenlabs_client.generate(
            text=text,
            voice="IK7YYZcSpmlkjKrQxbSn",
            model="eleven_multilingual_v2",
            stream=True
        )
        
        start = time.time()
        audio_data = bytearray()
        for chunk in audio:
            audio_data.extend(chunk)
        print(f"time of conversion to bytes: {time.time() - start}")

        return bytes(audio_data)
        # save(audio, "output_voices\speech.mp3")
        # return audio
        # play(audio)
        
    def process_audio_file(self, audio_path):
        # start = time.time()
        # transcription = self.speach_to_text(audio_path)
        # print(f'Sound to text time: {time.time() - start}')

        # start = time.time()
        
        # arabic_instruction = "يرجى تقديم الإجابة باللغة العربية فقط  واجعل الأجابة مختصرة في سطر قدر الامكان الامكان: "
        # # arabic_instruction = "response in just 2 lines: "
        # prompt = f"{arabic_instruction}\n{transcription}"
        # print(prompt)
        # response = self.generate_response(prompt)
        # print(f'Model response time: {time.time() - start}')
        
        
        # tts_output = self.text_to_speech(response)
        # return transcription, response, tts_output

        start = time.time()
        transcription = self.speach_to_text(audio_path)
        print(f'Sound to text time: {time.time() - start}')
        print(f'Transcritption: {transcription}')
        print('='*50)
        
        start = time.time()
        response = self.process_user_message(transcription)
        
        print(f'Model response: {response}')
        print(f'Model response time: {time.time() - start}')
        print('='*50)
        
        start = time.time()
        tts_output = self.text_to_sound(response)
        print(f"time of TTS: {time.time() - start}")

        return transcription, response, tts_output

if __name__ == '__main__':
    audio_path = r"C:\Users\zmlka\Documents\Sound Recordings\Recording (3).m4a"

    print('Initializing all things')
    voice_bot = RAGVoiceBot(
        knowldge_path='./knowledge_base',
        groq_token_path='groq_token.txt',
        vector_db_path='vector_db'
    )

    print('='*50)
    transcription, response, tts_output = voice_bot.process_audio_file(audio_path)

    print('='*50)

    # print("USER:", transcription)
    # print("Assistant:", response)