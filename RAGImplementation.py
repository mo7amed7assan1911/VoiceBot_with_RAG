from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI

class RAGImplementation:
    def __init__(self, vector_db_path, knowledge_path, model_name='llama-3.1-70b-versatile'):
        self.embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.vector_store = self.load_vector_store(vector_db_path)
        self.llm = self.initialize_llm(model_name)
        self.qa_pipeline = self.initialize_qa_pipeline(knowledge_path)

    def load_vector_store(self, vector_db_path):
        return FAISS.load_local(vector_db_path, self.embedding_model)

    def initialize_llm(self, model_name):
        return OpenAI(model_name=model_name)

    def initialize_qa_pipeline(self, knowledge_path):
        loader = PyPDFLoader(knowledge_path)
        documents = loader.load()
        return RetrievalQA.from_documents(documents, self.vector_store, self.llm)

    def answer_query(self, query):
        return self.qa_pipeline.run(query)