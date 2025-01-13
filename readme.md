# **Voice Assistant System with Modular Design**

This project demonstrates a modular and extensible **Voice Assistant System** that integrates **Speech-to-Text (STT)**, **Text-to-Text Retrieval-Augmented Generation (RAG)**, and **Text-to-Speech (TTS)** components without relying on external frameworks. The design emphasizes flexibility and scalability, allowing seamless integration or removal of different providers for each subsystem.

**Highlights**:
* Framework-free implementation with a focus on modularity and extensibility.
* Dynamic vector database management for efficient updates.
* Easily scalable due to its OOP-driven architecture.

---
## **System Design**
![System Design](./images/System_Design.svg)

---

## **System Features**

1. **Framework-Free Implementation**:
   - All functionalities, including STT, RAG, and TTS, are **implemented without relying on external frameworks**, ensuring lightweight and highly customizable code.

2. **Modular Design**:
   - **Three Independent Managers**:
     - **Speech-to-Text Manager**: Modular design for transcribing user audio.
     - **Text-to-Text Manager | Rag_System**: Retrieves and processes user queries using contextual knowledge.
     - **Text-to-Speech Manager**: Converts generated responses to audio with any provider you need.
   - Adding or removing providers for any subsystem is straightforward and requires minimal changes.

3. **Dynamic Vector Database Management**:
   - **Efficient Change Detection**: The `VectorDatabaseManager` continuously monitors the knowledge base for updates (e.g., added/removed/modified files). The vector database is rebuilt **only when necessary**, optimizing performance and resource usage.

4. **Extensible Providers**:
   - STT, RAG, and TTS managers support multiple providers with provider-specific implementations:
     - **Speech-to-Text Providers**:
       - **Fast Whisper** (Local)
       - **Groq API**
       - **Hugging Face API**
     - **Text-to-Text Providers**:
       - **Groq API**
       - **Hugging Face Models**
       - **OpenAI Models**
     - **Text-to-Speech Providers**:
       - **ElevenLabs**
       - **gTTS**

5. **End-to-End Workflow**:
   - Audio input → Speech-to-Text → RAG Query → Text-to-Speech → Audio output.
   - Both audio responses and textual outputs are provided for user queries.

---
