import os
# from langchain.document_loaders import PyPDFLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_ollama.chat_models import ChatOllama

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama.chat_models import ChatOllama




DB_DIR = "db"

# Embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# LLM
llm = ChatOllama(model="llama3")

def load_documents(file_paths):
    docs = []
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path)
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def create_vectorstore(documents):
    os.makedirs(DB_DIR, exist_ok=True)
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=DB_DIR)
    vectorstore.persist()
    return vectorstore

def init_qa(vectorstore):
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
