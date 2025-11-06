import streamlit as st
import os
import tempfile
import json
import chromadb
from pathlib import Path
import pandas as pd
from datetime import datetime
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

# Certificate Authority Name (configurable via secrets.toml or env var)
CERTIFICATE_AUTHORITY = st.secrets.get("CERTIFICATE_AUTHORITY_NAME", "AI Business Solutions Inc.")
PASSING_MARK = st.secrets.get("QUIZ_PASSING_MARK", 70) # 70% passing mark

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

# --- Database for student results ---
RESULTS_DB_PATH = Path("./student_results.csv")

def log_pass_result(name, date_time, passing_mark, score):
    """Stores the student's passing result to a CSV database."""
    new_record = {
        "Name": name,
        "Date_Time": date_time.isoformat(),
        "Passing_Mark_Percentage": passing_mark,
        "Achieved_Score": score,
        "Certificate_Authority": CERTIFICATE_AUTHORITY
    }
    
    if not RESULTS_DB_PATH.exists():
        df = pd.DataFrame([new_record])
        df.to_csv(RESULTS_DB_PATH, index=False)
    else:
        df = pd.read_csv(RESULTS_DB_PATH)
        df_new_row = pd.DataFrame([new_record])
        df = pd.concat([df, df_new_row], ignore_index=True)
        df.to_csv(RESULTS_DB_PATH, index=False)

def display_certificate(name, date_time, score):
    """Generates and displays a simple certificate UI."""
    st.balloons()
    st.markdown(
        f"""
        <div style="border: 2px solid black; padding: 20px; border-radius: 10px; text-align: center;">
            <h1>Certificate of Achievement</h1>
            <p>This is to certify that</p>
            <h2>{name}</h2>
            <p>has successfully passed the AI Business Efficiency test with a score of **{score}%**.</p>
            <p>Awarded on: {date_time.strftime("%B %d, %Y")}</p>
            <p><b>{CERTIFICATE_AUTHORITY}</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- LlamaIndex and ChromaDB Functions ---
@st.cache_resource
def ingest_documents(uploaded_files, collection_name="tutor_collection"):
    if not uploaded_files: return None
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
    query_engine = index.as_query_engine(similarity_top_k=3)
    query_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(name="knowledge_base_retriever", description="Provides factual answers")
    )
    react_agent = ReActAgent(tools=[query_tool], verbose=False)
    chat_engine = SimpleChatEngine.from_defaults(agent_runner=react_agent)
    return chat_engine

def generate_questions_and_answers(text, num_questions=3):
    prompt = f"""Based on the following content, generate {num_questions} multiple-choice questions with 4 options each, and provide the correct answer. The response MUST be a valid JSON array.
    Content: {text}
    Example Format:
    [
      {{"question": "What is the capital of France?", "options": ["Berlin", "Madrid", "Paris", "Rome"], "answer": "Paris"}},
      ...
    ]
    """
    try:
        response = Settings.llm.complete(prompt)
        response_text = response.text.strip()
        if response_text.startswith("```json"): response_text = response_text.replace("```json", "", 1).strip()
        if response_text.endswith("```"): response_text = response_text.removesuffix("```").strip()
        data = json.loads(response_text)
        for item in data:
            if not all(k in item for k in ("question", "options", "answer")):
                raise ValueError("JSON structure is incorrect. Missing key.")
        return data
    except Exception as e:
        st.error(f"An error occurred during question generation: {e}")
        st.code(response_text)
        return []

def lecturer_mode(index):
    prompt = "Provide a comprehensive and well-structured overview of the entire document, as if you are a lecturer explaining it to a class."
    query_engine = index.as_query_engine(similarity_top_k=10)
    response = query_engine.query(prompt)
    return str(response)

# --- Session state callbacks ---

# Callback to switch tabs
def switch_tab(tab_name):
    st.session_state.active_tab = tab_name

def generate_questions_callback():
    if st.session_state.index:
        with st.spinner("Generating questions..."):
            summary_prompt = "Provide a concise summary of the key information from the uploaded documents."
            summary_engine = st.session_state.index.as_query_engine()
            summary = summary_engine.query(summary_prompt)
            questions_list = generate_questions_and_answers(
                str(summary), 
                num_questions=st.session_state.num_questions_input_box
            )
            st.session_state.questions = questions_list
            st.session_state.user_answers = [None] * len(questions_list) 
            st.session_state.quiz_submitted = False
            st.session_state.quiz_score = 0
            st.session_state.show_answers = False 
            st.session_state.details_submitted = False
            # Switch to the quiz tab immediately after generation
            # Note: Streamlit reruns after this function, the tab selection happens below in the main UI logic.

def submit_quiz_callback():
    if None in st.session_state.user_answers:
        # st.warning here won't show because the page immediately reruns into the result state
        st.session_state.submit_error = "Please answer all questions before submitting."
        return False
    
    st.session_state.submit_error = "" # Clear error if successful

    score = 0
    total_questions = len(st.session_state.questions)
    for i, qa in enumerate(st.session_state.questions):
        user_ans = str(st.session_state.user_answers[i]).strip().lower()
        correct_ans = str(qa['answer']).strip().lower()
        if user_ans == correct_ans:
            score += 1
    
    score_percentage = (score / total_questions) * 100 if total_questions > 0 else 0
    st.session_state.quiz_score = score_percentage
    st.session_state.quiz_submitted = True
    # System reruns here, which is fine

def submit_candidate_details():
    if not st.session_state.user_full_name or not st.session_state.user_email:
        st.session_state.details_error = "Please enter your full name and email address."
        return
    st.session_state.details_error = "" # Clear error

    st.session_state.candidate_info = {
        "name": st.session_state.user_full_name,
        "email": st.session_state.user_email,
        "phone": st.session_state.user_phone
    }
    st.session_state.details_submitted = True


# --- Streamlit UI ---
st.set_page_config(page_title="📚 Business Operation Efficiency", layout="wide")
st.title("📚 AI-Powered Operation Efficiency")

# --- Initialize all necessary session state variables robustly at the start ---
if 'agent' not in st.session_state: st.session_state.agent = None
if 'index' not in st.session_state: st.session_state.index = None
if 'questions' not in st.session_state: st.session_state.questions = []
if 'user_answers' not in st.session_state: st.session_state.user_answers = []
if 'quiz_submitted' not in st.session_state: st.session_state.quiz_submitted = False
if 'quiz_score' not in st.session_state: st.session_state.quiz_score = 0
if 'show_answers' not in st.session_state: st.session_state.show_answers = False
if 'details_submitted' not in st.session_state: st.session_state.details_submitted = False
if 'candidate_info' not in st.session_state: st.session_state.candidate_info = {}
if 'messages' not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": "Hello! Please upload a document to get started."}]
if 'details_error' not in st.session_state: st.session_state.details_error = ""
if 'submit_error' not in st.session_state: st.session_state.submit_error = ""


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
                st.sidebar.success("Files ingested successfully! Go to the 'Quiz Generator' tab and click 'Generate New Quiz'.")
                # Reset states when new docs are processed
                st.session_state.questions = []
                st.session_state.user_answers = []
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score = 0
                st.session_state.show_answers = False
                st.session_state.details_submitted = False
                st.session_state.candidate_info = {}
            else:
                st.sidebar.error("Failed to ingest files.")

# Main content area tabs
tab1, tab2, tab3 = st.tabs(["Chat Tutor", "Quiz Generator & Grader", "Lecturer Mode"])

with tab1:
    st.header("Chat with your documents")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    col1, col2 = st.columns(2) 
    with col1:
        prompt = st.chat_input("Ask a question about the document...")
    
    with col2:
        if use_audio:
            audio_bytes = mic_recorder(start_prompt="Record", stop_prompt="Stop", just_once=True, use_container_width=True, format="webm")
            if audio_bytes and st.session_state.agent:
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_audio_file:
                    temp_audio_file.write(audio_bytes)
                    temp_audio_file_path = temp_audio_file.name
                audio_file = open(temp_audio_file_path, "rb")
                transcription = openai_client.audio.transcriptions.create(model="whisper-1", file=audio_file)
                prompt = transcription.text
                audio_file.close()
                os.remove(temp_audio_file_path)

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        if st.session_state.agent:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.agent.chat(prompt)
                    st.markdown(response.response)
                    st.session_state.messages.append({"role": "assistant", "content": response.response})
        else:
            with st.chat_message("assistant"): st.info("Please upload and process documents first.")

with tab2:
    st.header(f"Generate and Grade Quizzes (Passing Mark: {PASSING_MARK}%)")
    
    if st.session_state.index is None:
        st.info("Please upload and process documents in the sidebar first to enable quiz generation.")
    else:
        st.markdown("---")
        st.subheader("Quiz Settings")
        col_quiz_settings_1, col_quiz_settings_2 = st.columns(2)
        with col_quiz_settings_1:
            st.number_input("Number of Questions", min_value=1, max_value=20, value=5, key="num_questions_input_box")
        with col_quiz_settings_2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.button("Generate New Quiz", on_click=generate_questions_callback)
        st.markdown("---")
        
        # Display errors if any exist
        if st.session_state.details_error:
            st.error(st.session_state.details_error)
            st.session_state.details_error = "" # Clear error after display
        if st.session_state.submit_error:
            st.error(st.session_state.submit_error)
            st.session_state.submit_error = "" # Clear error after display


        # Main Quiz Logic
        if len(st.session_state.questions) > 0:
            if not st.session_state.details_submitted:
                st.subheader("Candidate Details")
                with st.form("candidate_details_form"):
                    st.text_input("Full Name (Required)", key="user_full_name")
                    st.text_input("Email Address (Required)", key="user_email")
                    st.text_input("Phone Number", key="user_phone")
                    st.form_submit_button("Submit Details & Start Quiz", on_click=submit_candidate_details)
            else:
                if st.session_state.quiz_submitted:
                    st.subheader("Quiz Results")
                    score_pct = st.session_state.quiz_score
                    if score_pct >= PASSING_MARK:
                        st.success(f"Congratulations, {st.session_state.candidate_info['name']}! You passed with a score of {score_pct:.1f}%!")
                        now = datetime.now()
                        log_pass_result(st.session_state.candidate_info['name'], now, PASSING_MARK, score_pct)
                        display_certificate(st.session_state.candidate_info['name'], now, score_pct)
                    else:
                        st.error(f"Sorry, {st.session_state.candidate_info['name']}. You scored {score_pct:.1f}%, which is below the passing mark of {PASSING_MARK}%.")
                    
                    st.checkbox("Show Correct Answers", key="show_answers")

                st.subheader("Questions")
                with st.form("quiz_form"):
                    for i, qa in enumerate(st.session_state.questions):
                        with st.container(border=True):
                            st.markdown(f"**Q{i+1}: {qa['question']}**")
                            
                            default_index = None
                            if st.session_state.quiz_submitted:
                                try:
                                    if st.session_state.show_answers:
                                        default_index = qa['options'].index(qa['answer'])
                                    elif st.session_state.user_answers[i] in qa['options']:
                                         default_index = qa['options'].index(st.session_state.user_answers[i])
                                except ValueError:
                                    pass
                            else:
                                if st.session_state.user_answers[i] is not None and st.session_state.user_answers[i] in qa['options']:
                                    default_index = qa['options'].index(st.session_state.user_answers[i])

                            selected_option = st.radio(
                                "Select an option:",
                                qa['options'],
                                key=f"user_answer_{i}",
                                index=default_index,
                                disabled=st.session_state.quiz_submitted
                            )
                            
                            # Update user answer immediately for subsequent reruns
                            if selected_option is not None:
                                st.session_state.user_answers[i] = selected_option
                            
                            if st.session_state.quiz_submitted and st.session_state.show_answers:
                                if selected_option == qa['answer']:
                                    st.success("Correct!")
                                else:
                                    st.error(f"Incorrect. Correct answer: {qa['answer']}")

                    # The submit button MUST remain stable within the form structure
                    # Use a dynamic label for better UX
                    submit_label = "Submit Quiz for Grading" if not st.session_state.quiz_submitted else "Quiz Submitted (Disabled)"
                    st.form_submit_button(submit_label, on_click=submit_quiz_callback, disabled=st.session_state.quiz_submitted)

        else:
            st.info("Use the settings above to generate a new quiz from the document content.")


with tab3:
    st.header("Lecturer Mode: Document Overview")
    if st.session_state.index:
        if st.button("Generate Overview"):
            with st.spinner("Generating lecture..."):
                overview_text = lecturer_mode(st.session_state.index)
                st.markdown(overview_text)
    else:
        st.info("Please upload documents in the sidebar to use lecturer mode.")
