import os
from dotenv import load_dotenv
load_dotenv('../.env')

# Paths
VECTOR_DB_PATH = "data/vector_db"  # Path to the vector database
KNOWLEDGE_BASE_PATH = "data/knowledge_base"  # Path to the knowledge base files
METADATA_PATH = "data/metadata.json"  # Path to the metadata file

# Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Default embedding model

# Model Provider Configurations
LLM_PROVIDER_NAME = "groq"  # Options: "groq", "huggingface", "openai"
MODEL_NAME = "llama-3.3-70b-versatile"  # Default model name for the provider

# Default Response Generation Settings
DEFAULT_MAX_TOKENS = 250  # Maximum tokens for the response
DEFAULT_TEMPERATURE = 0.4  # Sampling temperature for the response

CHUNK_SIZE=500  # Chunk size for text splitting
CHUNK_OVERLAP=100  # Chunk overlap for text splitting