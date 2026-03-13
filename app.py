import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from gtts import gTTS
import tempfile

st.set_page_config(page_title="AI Document Assistant")

st.title("AI Document Assistant")

# -----------------------------
# Cache LLM
# -----------------------------
@st.cache_resource
def load_llm():
    pipe = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


# -----------------------------
# Cache Embeddings
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Upload File
# -----------------------------
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Load document
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = load_embeddings()

    # Vector database
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Load LLM
    llm = load_llm()

    # QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the document"):

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # AI response
        answer = qa.run(prompt)

        with st.chat_message("assistant"):
            st.markdown(answer)

            # Text to speech
            tts = gTTS(answer)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )