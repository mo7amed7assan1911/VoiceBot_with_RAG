import os
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


class VectoreDatabaseManager:
    def __init__(self,
                vector_db_path="data/vector_db",
                knowledge_base_path="data/knowledge",
                metadata_path="data/metadata.json",
                embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        
        self.vector_db_path = vector_db_path
        self.knowldge_base_path = knowledge_base_path
        self.metadata_path = metadata_path
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self, metadata):
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    def _get_current_metadata(self):
        metadata = {}
        for root, _, files in os.walk(self.knowldge_base_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                last_modified = os.path.getmtime(file_path)
                metadata[file_path] = last_modified
        return metadata
    
    
    def _create_vector_database_texts(self):
        """Create a new vector database from knowledge files."""
        print('Creating new vector database ...')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        all_chunks = []

        for filename in os.listdir(self.knowldge_base_path):
            file_path = os.path.join(self.knowldge_base_path, filename)
            
            if filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_content = file.read()
                    chunks = text_splitter.split_text(text_content)
                    all_chunks.extend(chunks)
                    
            elif filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                chunks = text_splitter.split_documents(documents)
                all_chunks.extend(chunks)

        return FAISS.from_texts(all_chunks, self.embedding_model)
    
    
    def get_vector_databaase(self):
        old_metadata = self._load_metadata()
        current_metadata = self._get_current_metadata()
        
        if old_metadata != current_metadata: # if there are files changed or added or removed
            vector_db = self._create_vector_database_texts()
            vector_db.save_local(self.vector_db_path)
            self._save_metadata(current_metadata)
        
        elif os.path.exists(self.vector_db_path):
            print('Loading exisiting vector database ...')
            vector_db = FAISS.load_local(self.vector_db_path, self.embedding_model, allow_dangerous_deserialization=True)
        
        else: # if there is no vector database
            vector_db = self._create_vector_database_texts()
            vector_db.save_local(self.vector_db_path)
            self._save_metadata(current_metadata)
            
        return vector_db