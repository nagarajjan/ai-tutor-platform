# ai-tutor-platform
this was compiled using python 3.11.9

import streamlit as st
import os
import tempfile
import json
import chromadb
from pathlib import Path
from io import BytesIO

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# OpenAI Client
from openai import OpenAI as OpenAIClient

# Streamlit Components
from streamlit_mic_recorder import mic_recorder
from streamlit.runtime.scriptrunner import get_script_run_ctx

# --- Initialize OpenAI and ChromaDB clients ---
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OpenAI API key not found. Please add it to `.streamlit/secrets.toml`.")
    st.stop()

# Configure LlamaIndex global settings
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"])
Settings.embed_model = OpenAIEmbedding(api_key=st.secrets["OPENAI_API_KEY"])
openai_client = OpenAIClient(api_key=st.secrets["OPENAI_API_KEY"])

# --- Session-specific file storage ---
@st.cache_resource
def get_session_storage_path():
    """Create a persistent session storage path."""
    ctx = get_script_run_ctx()
    if ctx is None:
        session_id = "default"
    else:
        session_id = ctx.session_id
    
    storage_path = Path("./session_data") / session_id
    storage_path.mkdir(parents=True, exist_ok=True)
    return storage_path

# --- LlamaIndex and ChromaDB Functions ---
@st.cache_resource
def ingest_documents(uploaded_file, collection_name="tutor_collection"):
    """
    Ingests documents from an uploaded file, creates an index, and stores it in ChromaDB.
    Returns the index for the RAG agent.
    """
    try:
        session_path = get_session_storage_path()
        file_path = session_path / uploaded_file.name
        
        # Write file to persistent session storage
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize and configure ChromaDB
        db = chromadb.PersistentClient(path="./chroma_data")
        chroma_collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Read documents from the session-specific file
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        
        # Create the index from the documents
        index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
        return index
    except Exception as e:
        st.error(f"Error during document ingestion: {e}")
        return None

def setup_rag_agent(index):
    """Creates a ReAct agent directly with query engine tools."""
    query_engine = index.as_query_engine(similarity_top_k=3)
    query_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="knowledge_base_retriever",
            description="Provides factual answers by retrieving information from ingested files."
        )
    )
    # Instantiate ReActAgent directly with tools
    return ReActAgent(tools=[query_tool], verbose=False)

def generate_questions(text, question_type="objective"):
    """Generates questions using the LLM with robust JSON handling."""
    if question_type == "objective":
        prompt = f"""Based on the following content, generate 3 multiple-choice questions with 4 options each, and provide the correct answer. The response MUST be a valid JSON array.
        
        Content: {text}
        
        Example Format:
        [
          {{"question": "What is the capital of France?", "options": ["Berlin", "Madrid", "Paris", "Rome"], "answer": "Paris"}},
          ...
        ]
        """
    else:  # Descriptive questions
        prompt = f"""Based on the following content, generate 3 descriptive questions that require a paragraph-long answer. The response MUST be a valid JSON array of strings.
        
        Content: {text}
        
        Example Format:
        [
          "Explain the process of photosynthesis.",
          ...
        ]
        """
    try:
        response = Settings.llm.complete(prompt)
        response_text = response.text.strip()
        
        # Attempt to clean up common LLM formatting mistakes
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "", 1).strip()
        if response_text.endswith("```"):
            response_text = response_text.removesuffix("```").strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            st.error("Raw LLM output failed to parse as JSON. See details below.")
            st.code(response_text)
            st.exception(e)
            return []
    except Exception as e:
        st.error(f"An error occurred during question generation: {e}")
        return []

def lecturer_mode(index):
    """Generates a comprehensive overview of the document's content."""
    prompt = "Provide a comprehensive and well-structured overview of the entire document, as if you are a lecturer explaining it to a class."
    query_engine = index.as_query_engine(similarity_top_k=10)
    response = query_engine.query(prompt)
    return str(response)

# --- Streamlit UI ---
st.set_page_config(page_title="📚 AI-Powered Tutor with ChromaDB", layout="wide")
st.title("📚 AI-Powered Tutor with ChromaDB")

# Sidebar for file upload and configuration
st.sidebar.title("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload a PDF to start:", type="pdf")
if uploaded_file:
    with st.spinner("Ingesting document..."):
        index = ingest_documents(uploaded_file)
        if index:
            agent = setup_rag_agent(index)
            st.session_state.agent = agent
            st.session_state.index = index
            st.session_state.uploaded_file = uploaded_file
            st.sidebar.success("File ingested successfully!")
        else:
            st.sidebar.error("Failed to ingest file.")
else:
    if 'agent' in st.session_state:
        del st.session_state.agent
        del st.session_state.index

if 'agent' not in st.session_state:
    st.info("Please upload a file in the sidebar to begin.")
    st.stop()

# --- Interactive Sections ---

st.header("🗣️ Voice Tutor")
st.markdown("Use the microphone to ask a question and hear the response.")
col1, col2 = st.columns(2)
with col1:
    audio_data = mic_recorder(start_prompt="Start speaking", stop_prompt="Stop")
with col2:
    if audio_data:
        st.audio(audio_data['bytes'])
        with st.spinner("Transcribing..."):
            try:
                audio_buffer = BytesIO(audio_data['bytes'])
                audio_buffer.name = 'temp_audio.wav'
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_buffer
                )
                st.session_state.voice_prompt = transcript.text
            except Exception as e:
                st.error(f"Whisper transcription failed: {e}")
                st.session_state.voice_prompt = ""

if "voice_prompt" in st.session_state and st.session_state.voice_prompt:
    st.markdown(f"**Your question:** *{st.session_state.voice_prompt}*")
    with st.spinner("Thinking..."):
        response_text = str(st.session_state.agent.chat(st.session_state.voice_prompt))
        st.write(f"**AI response:** {response_text}")
        
        with st.spinner("Generating audio..."):
            speech_response = openai_client.audio.speech.create(model="tts-1", voice="nova", input=response_text)
            st.audio(speech_response.content)
    del st.session_state.voice_prompt

# Text-based chat interface
st.header("✍️ Text-Based Chat")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("Thinking..."):
        response = st.session_state.agent.chat(prompt)
        st.session_state.messages.append({"role": "assistant", "content": str(response)})
    with st.chat_message("assistant"):
        st.markdown(str(response))

# Lecturer mode
st.header("🧑‍🏫 Lecturer Mode")
if st.button("Start Lecture"):
    with st.spinner("Generating lecture..."):
        lecture_text = lecturer_mode(st.session_state.index)
        st.write(f"**Lecture:** {lecture_text}")
        with st.spinner("Generating audio..."):
            speech_response = openai_client.audio.speech.create(model="tts-1", voice="nova", input=lecture_text)
            st.audio(speech_response.content)

# Assessment generation section
st.header("📝 Generate Assessments")
question_type = st.radio("Choose question type:", ("objective", "descriptive"), key="q_type")

if st.button("Generate Questions"):
    if 'uploaded_file' not in st.session_state:
        st.warning("Please upload a document first.")
    else:
        with st.spinner(f"Generating {question_type} questions..."):
            try:
                session_path = get_session_storage_path()
                file_path = session_path / st.session_state.uploaded_file.name
                
                reader = SimpleDirectoryReader(input_files=[file_path])
                documents = reader.load_data()
                content_for_questions = " ".join([doc.text for doc in documents])[:4000]
                generated_questions = generate_questions(content_for_questions, question_type)
            except FileNotFoundError:
                st.warning("Please re-upload the document.")
                generated_questions = []
            
            if generated_questions:
                st.subheader("Generated Questions")
                for i, q in enumerate(generated_questions):
                    if question_type == "objective":
                        st.markdown(f"**{i+1}.** {q['question']}")
                        st.radio("Options:", q['options'], key=f"q_{i}")
                        with st.expander("Show Answer"):
                            st.write(q['answer'])
                    else:
                        st.markdown(f"**{i+1}.** {q}")
            else:
                st.warning("Could not generate questions. Try a different document.")




