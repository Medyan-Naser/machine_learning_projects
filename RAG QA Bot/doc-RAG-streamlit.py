import streamlit as st
import os
import logging
import uuid
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import (
    UnstructuredFileLoader, UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader, UnstructuredExcelLoader,
    UnstructuredMarkdownLoader, UnstructuredCSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import ollama
import shutil

# --- Logging ---
logging.basicConfig(level=logging.INFO)

# --- Constants ---
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "rag-store"


# --- File Loading ---
def load_document(file_path, file_type):
    try:
        if file_type == "pdf":
            loader = UnstructuredPDFLoader(file_path)
        elif file_type in ["doc", "docx"]:
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_type in ["xls", "xlsx"]:
            loader = UnstructuredExcelLoader(file_path)
        elif file_type == "csv":
            loader = UnstructuredCSVLoader(file_path)
        elif file_type in ["md", "markdown"]:
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path)

        data = loader.load()
        logging.info(f"{file_type.upper()} loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {str(e)}")
        st.error(f"Failed to load {file_type.upper()} file.")
        return None


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


def create_vector_db(file_paths):
    if not file_paths:
        return None

    ollama.pull(EMBEDDING_MODEL)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    temp_dir = f"./temp_chroma_{uuid.uuid4().hex}"
    os.makedirs(temp_dir, exist_ok=True)

    all_chunks = []
    for path in file_paths:
        ext = path.split(".")[-1].lower()
        data = load_document(path, ext)
        if data:
            chunks = split_documents(data)
            all_chunks.extend(chunks)

    if not all_chunks:
        return None

    vector_db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding,
        collection_name=VECTOR_STORE_NAME,
        persist_directory=temp_dir,
    )
    logging.info("Vector database created.")
    return vector_db


def create_retriever(vector_db):
    return vector_db.as_retriever()


def create_chain(retriever, llm):
    template = """Answer the question based ONLY on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Document RAG Assistant", layout="centered")
    st.title("📄 Document Chat Assistant")

    # Initialize session state
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "file_paths" not in st.session_state:
        st.session_state.file_paths = []

    # LLM initialization
    llm = ChatOllama(model=MODEL_NAME)

    # File upload
    uploaded_files = st.file_uploader(
        "Upload your files (any type):",
        type=None,
        accept_multiple_files=True,
    )

    if uploaded_files:
        new_files = []
        for file in uploaded_files:
            if file.name not in [f.name for f in st.session_state.uploaded_files]:
                tmp = NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}")
                tmp.write(file.read())
                tmp_path = tmp.name
                tmp.close()
                st.session_state.file_paths.append(tmp_path)
                st.session_state.uploaded_files.append(file)
                new_files.append(file.name)
        if new_files:
            st.success(f"Added {len(new_files)} new file(s): {', '.join(new_files)}")

    # --- PROCESS BUTTON ---
    if st.button("⚙️ Process Files for RAG"):
        if not st.session_state.file_paths:
            st.warning("Please upload at least one file first.")
        else:
            with st.spinner("Processing files..."):
                st.session_state.vector_db = create_vector_db(st.session_state.file_paths)
            if st.session_state.vector_db:
                st.success("✅ Files processed successfully! You can now ask questions.")
            else:
                st.error("❌ Failed to process files.")

    # --- Chat Interface ---
    st.divider()
    user_input = st.text_input("💬 Ask a question:")

    if user_input:
        with st.spinner("Generating response..."):
            try:
                if st.session_state.vector_db:
                    retriever = create_retriever(st.session_state.vector_db)
                    chain = create_chain(retriever, llm)
                    response = chain.invoke(input=user_input)
                else:
                    response = llm.invoke(user_input)

                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Upload files and press 'Process Files' to use RAG, or chat directly without it.")


if __name__ == "__main__":
    main()
