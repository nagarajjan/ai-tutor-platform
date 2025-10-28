import streamlit as st
import os
import tempfile
import json
import chromadb
from pathlib import Path
from io import BytesIO
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.ollama import Ollama

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

# --- Global configuration ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# --- LLM selection and LlamaIndex setup ---
llm_choice = st.sidebar.radio(
    "Select your LLM provider:",
    ('OpenAI', 'Ollama'),
    key='llm_selector',
    help="Ollama requires a running Ollama server."
)
ollama_model = "llama3.1:8b"

if llm_choice == "OpenAI":
    if "OPENAI_API_KEY" not in os.environ:
        st.error("OpenAI API key not found. Please add it to secrets.toml.")
        st.stop()
    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding()
    openai_client = OpenAIClient()
    use_audio = True
else: # Ollama
    Settings.llm = Ollama(model=ollama_model, request_timeout=600.0)
    Settings.embed_model = OpenAIEmbedding()
    use_audio = False

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
def ingest_documents(uploaded_files, collection_name="tutor_collection"):
    """
    Ingests multiple documents from a list of uploaded files, creates an index, 
    and stores it in ChromaDB. Returns the index for the RAG agent.
    """
    if not uploaded_files:
        st.error("No files were uploaded for ingestion.")
        return None

    try:
        session_path = get_session_storage_path()
        file_paths = []

        for uploaded_file in uploaded_files:
            file_path = session_path / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        db = chromadb.PersistentClient(path="./chroma_data")
        chroma_collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        documents = SimpleDirectoryReader(input_files=file_paths).load_data()
        index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
        return index
    except Exception as e:
        st.error(f"Error during document ingestion: {e}")
        return None

def setup_rag_agent(index):
    """
    Creates and returns a chat engine with a ReAct agent.
    """
    query_engine = index.as_query_engine(similarity_top_k=3)
    query_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="knowledge_base_retriever",
            description="Provides factual answers by retrieving information from ingested files."
        )
    )
    react_agent = ReActAgent(tools=[query_tool], verbose=False)
    chat_engine = SimpleChatEngine.from_defaults(agent_runner=react_agent)
    return chat_engine

def generate_questions_and_answers(text, question_type="objective", num_questions=3):
    """
    Generates questions and their answers using the LLM with robust JSON handling.
    """
    if question_type == "objective":
        prompt = f"""Based on the following content, generate {num_questions} multiple-choice questions with 4 options each, and provide the correct answer. The response MUST be a valid JSON array.
        
        Content: {text}
        
        Example Format:
        [
          {{"question": "What is the capital of France?", "options": ["Berlin", "Madrid", "Paris", "Rome"], "answer": "Paris"}},
          ...
        ]
        """
    else:  # Descriptive questions
        prompt = f"""Based on the following content, generate {num_questions} descriptive questions and a corresponding answer for each. The response MUST be a valid JSON array of objects.
        
        Content: {text}
        
        Example Format:
        [
          {{"question": "Explain the process of photosynthesis.", "answer": "Photosynthesis is the process used by plants, algae and certain bacteria to turn sunlight, water and carbon dioxide into food energy."}},
          ...
        ]
        """
    try:
        response = Settings.llm.complete(prompt)
        response_text = response.text.strip()
        
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "", 1).strip()
        if response_text.endswith("```"):
            response_text = response_text.removesuffix("```").strip()

        return json.loads(response_text)
    except json.JSONDecodeError as e:
        st.error("LLM output failed to parse as JSON. Please try again.")
        st.code(response_text)
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

# --- Session state callbacks ---
def generate_questions_callback():
    if st.session_state.index:
        with st.spinner("Generating questions..."):
            summary_prompt = "Provide a concise summary of the key information from the uploaded documents."
            summary_engine = st.session_state.index.as_query_engine()
            summary = summary_engine.query(summary_prompt)
            questions_list = generate_questions_and_answers(
                str(summary), 
                st.session_state.question_type_radio,
                st.session_state.num_questions
            )
            st.session_state.questions = questions_list
            st.session_state.show_answers = [False] * len(questions_list)

def show_answer_callback(index):
    st.session_state.show_answers[index] = not st.session_state.show_answers[index]

# --- Streamlit UI ---
st.set_page_config(page_title="📚 AI-Powered Tutor with ChromaDB", layout="wide")
st.title("📚 AI-Powered Tutor with ChromaDB")

# Sidebar for file upload and configuration
st.sidebar.title("Configuration")
st.sidebar.markdown("---")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDFs to start:", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button("Process documents"):
        with st.spinner("Ingesting documents..."):
            index = ingest_documents(uploaded_files)
            if index:
                agent = setup_rag_agent(index)
                st.session_state.agent = agent
                st.session_state.index = index
                st.session_state.uploaded_files = uploaded_files
                st.sidebar.success("Files ingested successfully!")
                # Reset question state when new docs are processed
                st.session_state.questions = []
                st.session_state.show_answers = []
            else:
                st.sidebar.error("Failed to ingest files.")
else:
    if 'agent' in st.session_state:
        del st.session_state.agent
        del st.session_state.index
        st.session_state.questions = []
        st.session_state.show_answers = []

if 'agent' not in st.session_state:
    st.info("Please upload files in the sidebar and click 'Process documents' to begin.")
    st.stop()

# --- Interactive Sections ---
if use_audio:
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
                    audio_buffer = Bytesio(audio_data['bytes'])
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
else:
    st.header("🗣️ Voice Tutor (Disabled)")
    st.info("Voice options are only available when using OpenAI.")

# Text-based chat interface
st.header("✍️ Text-Based Chat")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.agent.chat(prompt)
            st.markdown(str(response))
            st.session_state.messages.append({"role": "assistant", "content": str(response)})

# Tutor tools
st.header("🧑‍🏫 Tutor Tools")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Generate Questions")
    question_type = st.radio("Select question type:", ("objective", "descriptive"), key="question_type_radio")
    num_questions = st.number_input("Number of questions:", min_value=1, max_value=10, value=3, key="num_questions")

    if st.button("Generate Questions", key="generate_questions_btn", on_click=generate_questions_callback):
        pass # The button click is handled by the callback

    if st.session_state.questions:
        for i, q in enumerate(st.session_state.questions):
            with st.expander(f"Question {i+1}: {q['question']}"):
                if question_type == "objective":
                    # Added check for 'options' key
                    if 'options' in q and q['options']:
                        st.radio("Options:", q['options'], key=f"q_{i}", index=None)
                    else:
                        st.warning("Ollama did not provide options for this question.")
                    
                    if st.button(f"Show Answer (MC {i})", key=f"show_mc_answer_{i}", on_click=show_answer_callback, args=(i,)):
                        pass
                    if st.session_state.show_answers[i]:
                        st.success(f"**Answer:** {q.get('answer', 'Answer not available.')}")
                else: # Descriptive
                    if st.button(f"Show Answer (Descriptive {i})", key=f"show_descriptive_answer_{i}", on_click=show_answer_callback, args=(i,)):
                        pass
                    if st.session_state.show_answers[i]:
                        with st.spinner("Retrieving answer..."):
                            rag_answer = st.session_state.agent.chat(q['question'])
                            st.info(str(rag_answer))

with col2:
    st.subheader("Lecturer Mode")
    if st.button("Activate Lecturer Mode", key="lecturer_mode_btn"):
        with st.spinner("Preparing lecture..."):
            explanation = lecturer_mode(st.session_state.index)
            st.markdown(explanation)
