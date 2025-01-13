from vector_db_manager import VectoreDatabaseManager

class text_to_text_with_RAG:
    def __init__(self, 
                vector_db_path='vector_db',
                knowledge_base_path='./knowledge_base',
                metadata_path='metabase.json',
                embedding_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                llm_provider='groq',
                model_name='llama-3.3-70b-versatile',
                max_tokens=250,
                temperature=0.4):
        """
        Initialize the RAG application.

        Args:
            vector_db_path (str): Path to the vector database.
            knowledge_path (str): Path to the knowledge files.
            metadata_path (str): Path to the metadata file.
            llm_provider (str): Name of the model provider (e.g., "groq", "huggingface", "openai").
            model_name (str): Name of the model to use.
            embedding_model (str): Name of the embedding model for vector database.
            max_tokens (int): Maximum tokens for the response.
            temperature (float): Sampling temperature for response generation.
        """
        
        # Initialize the vector database manager
        vdb_manager = VectoreDatabaseManager(
            vector_db_path=vector_db_path,
            knowledge_base_path=knowledge_base_path,
            metadata_path=metadata_path,
            embedding_model_name=embedding_model_name)
        
        self.vector_db = vdb_manager.get_vector_databaase()
        
        
        # Initialize the model provider
        if llm_provider == "groq":
            from LLM_Providers.groq_provider import GroqProvider
            self.model = GroqProvider(model_name, max_tokens, temperature)

        elif llm_provider == "huggingface":
            from LLM_Providers.huggingface_provider import HuggingFaceProvider
            self.model =  HuggingFaceProvider(model_name, max_tokens, temperature)
            
        elif llm_provider == "openai":
            from LLM_Providers.openai_provider import OpenAIProvider
            self.model = OpenAIProvider(model_name, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported provider: {llm_provider}")


    def _get_relevant_chunks(self, query, k=5):
        
        docs = self.vector_db.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def _construct_prompt(self, chunks, query):
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


        return rag_template.strip()

    def process_user_message(self, query):
        relevant_chunks = self._get_relevant_chunks(query)
        full_prompt_after_rag = self._construct_prompt(relevant_chunks, query)
        response = self.model.get_response(full_prompt_after_rag)
    
        return response, relevant_chunks