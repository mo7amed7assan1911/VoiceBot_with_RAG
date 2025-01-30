from vector_db_manager import VectoreDatabaseManager

class text_to_text_with_RAG:
    def __init__(self, 
                vector_db_path='vector_db',
                knowledge_base_path='./knowledge_base',
                metadata_path='metabase.json',
                embedding_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                llm_provider='groq',
                model_name='llama-3.3-70b-versatile',
                rephraser_model_name='llama-3.1-8b-instant',
                max_tokens=250,
                temperature=0.4,
                streaming_mode=False):
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
        self.streaming_mode = streaming_mode
        
        # Initialize the vector database manager
        vdb_manager = VectoreDatabaseManager(
            vector_db_path=vector_db_path,
            knowledge_base_path=knowledge_base_path,
            metadata_path=metadata_path,
            embedding_model_name=embedding_model_name)
        
        self.vector_db = vdb_manager.get_vector_databaase()
        self.retriver = self.vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.8}
        )
                # Initialize the model provider
        if llm_provider == "groq":
            from Text_to_Text_Providers.groq_provider import GroqProvider
            self.model = GroqProvider(model_name, max_tokens, temperature)
            self.rephraser_model = GroqProvider(rephraser_model_name, max_tokens=1000, temperature=0)

        elif llm_provider == "huggingface":
            from Text_to_Text_Providers.huggingface_provider import HuggingFaceProvider
            self.model =  HuggingFaceProvider(model_name, max_tokens, temperature)
            self.rephraser_model = HuggingFaceProvider(rephraser_model_name, max_tokens=1000, temperature=0)
            
        elif llm_provider == "openai":
            from Text_to_Text_Providers.openai_provider import OpenAIProvider
            self.model = OpenAIProvider(model_name, max_tokens, temperature)
            self.rephraser_model = OpenAIProvider(rephraser_model_name, max_tokens=1000, temperature=0)
        else:
            raise ValueError(f"Unsupported provider: {llm_provider}")


    def _get_relevant_chunks_from_vdb(self, query, k=5):

        docs_with_scores = self.vector_db.similarity_search_with_score(query, k=k)
        # relevance_score_fn = self.vector_db._cosine_relevance_score_fn
        # print(docs_with_scores)
        # print('docs with scores')
        # for doc, score in docs_with_scores:
        #     print('='*50)
        #     print(f'content:, {doc.page_content}')
        #     print()
        #     print(f'score: {score}')
        print('='*50)
        # relevant_chunks = [doc.page_content for doc, score in docs_with_scores if score <= score_threshold]
        relevant_chunks = [doc.page_content for doc, _ in docs_with_scores]
        
        return relevant_chunks

    def _construct_prompt_of_rag(self, context, query):
        rag_template = f"""You are an advanced AI answering user queries using provided context. Follow these rules strictly:

        <rules>
        - If you can't find any relevant information, DIRECTLY Answer from your knowledge.
        - Use only the relevant parts of the context to answer.
        - If the context is unreadable or unclear, provide the best possible answer.
        - Answer concisely and in the same language as the query.
        </rules>

        <context>
        {context}
        </context>

        <user_query>
        {query}
        </user_query>

        Answer:
        """

        return rag_template.strip()

    
    def _construct_rephraser_prompt_template(self, chunks, query):
        context = "\n".join([f"- {chunk}" for chunk in chunks])
    
        rephraser_template = f"""You are an advanced language model tasked with extracting only the information relevant to a specific query.\
            Your job is to analyze the provided context and return only the portions that directly address the user's question.\
            Do not include unrelated or ambiguous content. If no relevant information is found, respond with: 'Answer from your knowledge'.
        
        ONLY RETURN THE INFORMATION THAT DIRECTLY ADDRESSES THE USER'S QUESTION. DON'T ANSWER THE QUESTION.
        Example Workflow
            Input:
                Query: "What is the capital of France?"
                Relevant Chunks:
                - France is a country in Europe.
                - Paris is the capital city of France.
                - The Eiffel Tower is located in Paris.

            Refined Helpful Chunks:
                - Paris is the capital city of France.
                
            Relevant information: Paris is the capital city of France.
        
        # Rules:
        - Don't summarize the relevant paragraphs, just return them as they are.
        - Don't answer the question, just return the relevant information.
        - If you can't find any relevant information, respond ONLY with: 'Answer from your knowledge'.
        
        
        <context>
        {context}
        </context>
        
        <user_query>
        {query}
        </user_query>
        
        Relevant information: 
        """

        # rephraser_template = f"""
        # Take each paragraph of these paragraphs\n:{context}\n
        # Just return only paragraphs that help answring this question: {query}.
        
        # # Rules:
        # - Don't summarize the relevant paragraphs, just return them as they are.
        # - Don't answer the question, just return the relevant information.
        # - If you can't find any relevant information, respond ONLY with: 'Answer from your knowledge'."""
        
        return rephraser_template.strip()
    
    def _rephraser(self, relevant_chunks, query):
        rephraser_prompt = self._construct_rephraser_prompt_template(relevant_chunks, query)
        response = self.rephraser_model.get_response(rephraser_prompt)
        return response
    
    def process_user_message(self, query):
        relevant_chunks = self._get_relevant_chunks_from_vdb(query)
        # print('Relevant Chunks:')
        # for chunk in relevant_chunks:
        #     print('- ', chunk)
        #     print('='*50)
            
        # rephrased_context = self._rephraser(relevant_chunks, query)
        
        # print("="*50)
        # print('Rephrased Context: ', rephrased_context)
        # print("="*50)
        
        full_prompt_after_rag = self._construct_prompt_of_rag(relevant_chunks, query)
        
        if self.streaming_mode:
            stream = self.model.get_stream(full_prompt_after_rag)
            return stream, relevant_chunks
        
        else:
            final_response = self.model.get_response(full_prompt_after_rag)
            return final_response, relevant_chunks
        