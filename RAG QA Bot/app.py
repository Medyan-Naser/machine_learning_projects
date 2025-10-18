import streamlit as st
from qa_bot import load_documents, create_vectorstore, init_qa
import os

st.set_page_config(page_title="Doc Chat AI", layout="wide")

st.title("📄 Chat with Your Documents")

uploaded_files = st.file_uploader("Upload files (PDF, TXT, etc.)", accept_multiple_files=True)

if uploaded_files:
    file_paths = []
    for uploaded_file in uploaded_files:
        path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(path)

    st.success(f"{len(file_paths)} files uploaded successfully!")

    # Load and split documents
    documents = load_documents(file_paths)
    st.info(f"Documents split into {len(documents)} chunks.")

    # Create vector store
    vectorstore = create_vectorstore(documents)

    # Initialize QA chain
    qa = init_qa(vectorstore)
    st.success("QA bot is ready! Ask your questions below.")

    # Chat input
    question = st.text_input("Ask a question about your documents:")
    if question:
        answer = qa.run(question)
        st.write("**Answer:**")
        st.write(answer)
