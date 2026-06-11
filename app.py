import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢")
st.title("🏢 Zyro Dynamics HR Help Desk")
st.write("Ask any question regarding company policies, handbooks, or benefits.")

# 1. Fetch the API key from Streamlit Secrets
groq_api_key = st.secrets.get("GROQ_API_KEY")

@st.cache_resource
def init_rag():
    corpus_path = "./zyro-dynamics-hr-corpus"
    if not os.path.exists(corpus_path):
        os.makedirs(corpus_path)
    
    loader = PyPDFDirectoryLoader(corpus_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=75)
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 2. Pass the API key here
    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model="llama-3.3-70b-versatile", 
        temperature=0.1
    )
    return retriever, llm

# 3. Initialization with safety check
if not groq_api_key:
    st.error("GROQ_API_KEY is missing! Please set it in Streamlit Secrets.")
    st.stop()

try:
    retriever, llm = init_rag()
    # Add your chat logic here (st.chat_input, etc.)
    st.success("System Initialized!")
except Exception as e:
    st.error(f"Initialization Error: {e}")
