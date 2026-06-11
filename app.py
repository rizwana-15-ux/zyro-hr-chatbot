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

# Fetch the API key from Streamlit Secrets
groq_api_key = st.secrets.get("GROQ_API_KEY")

@st.cache_resource
def init_rag():
    corpus_path = "./zyro-dynamics-hr-corpus"
    
    # 1. Safety check for folder
    if not os.path.exists(corpus_path):
        st.error(f"Folder '{corpus_path}' not found! Please create it in your repository.")
        st.stop()
    
    loader = PyPDFDirectoryLoader(corpus_path)
    documents = loader.load()
    
    # 2. Safety check for documents
    if not documents:
        st.error(f"No PDF files found in '{corpus_path}'. Please upload your HR documents there.")
        st.stop()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=75)
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Initialize Groq
    if not groq_api_key:
        st.error("GROQ_API_KEY not set in Secrets!")
        st.stop()
        
    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model="llama-3.3-70b-versatile", 
        temperature=0.1
    )
    return retriever, llm

# Run the app
try:
    retriever, llm = init_rag()
    st.success("System Initialized! Ready to answer.")
    # Add your chat input logic here
    user_query = st.chat_input("Ask a question...")
    if user_query:
        st.write(f"You asked: {user_query}")
        # Add RAG retrieval and generation logic here
except Exception as e:
    st.error(f"An error occurred: {e}")
