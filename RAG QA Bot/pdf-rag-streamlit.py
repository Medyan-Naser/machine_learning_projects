import streamlit as st
import os
import logging
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import ollama
import uuid
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"

def ingest_pdf(file_path):
    """Load PDF documents."""
    try:
        loader = UnstructuredPDFLoader(file_path=file_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading PDF: {str(e)}")
        st.error("Failed to load PDF.")
        return None

def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks

def create_vector_db(pdf_files):
    """Create a temporary vector database from uploaded PDFs."""
    if not pdf_files:
        return None

    ollama.pull(EMBEDDING_MODEL)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    temp_dir = f"./temp_chroma_{uuid.uuid4().hex}"
    os.makedirs(temp_dir, exist_ok=True)

    all_chunks = []
    for pdf_file in pdf_files:
        data = ingest_pdf(pdf_file)
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
    logging.info("Temporary vector database created.")
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

def main():
    st.title("Document Assistant")

    # Initialize session state
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

    # Initialize LLM
    llm = ChatOllama(model=MODEL_NAME)

    # Upload PDFs (multiple)
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    # Add new files to session_state
    new_files = []
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in [f.name for f in st.session_state.uploaded_files]:
                # Save temporary file
                tmp = NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(file.read())
                tmp_path = tmp.name
                tmp.close()
                new_files.append(tmp_path)
                st.session_state.uploaded_files.append(file)

    # Process only when new files are added
    if new_files:
        with st.spinner("Processing new PDF files..."):
            if st.session_state.vector_db:
                # Merge new files into existing vector DB
                new_db = create_vector_db(new_files)
                # Note: For simplicity we just overwrite with combined new DB
                st.session_state.vector_db = new_db
            else:
                st.session_state.vector_db = create_vector_db(new_files)

            st.success("PDF(s) processed and ready for questions!")

    # User question
    user_input = st.text_input("Enter your question:", "")

    if user_input:
        with st.spinner("Generating response..."):
            try:
                if st.session_state.vector_db:
                    retriever = create_retriever(st.session_state.vector_db)
                    chain = create_chain(retriever, llm)
                    response = chain.invoke(input=user_input)
                else:
                    response = llm.chat(user_input)

                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("You can ask a question even without uploading a PDF.")

if __name__ == "__main__":
    main()
