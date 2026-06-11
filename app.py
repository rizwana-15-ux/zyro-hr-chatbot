import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢")
st.title("🏢 Zyro Dynamics HR Help Desk")
st.write("Ask any question regarding company policies, handbooks, or benefits.")

@st.cache_resource
def init_rag():
    corpus_path = "./zyro-dynamics-hr-corpus"
    if not os.path.exists(corpus_path):
        os.makedirs(corpus_path)
    loader = PyPDFDirectoryLoader(corpus_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=75)
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=512)
    return retriever, llm

try:
    retriever, llm = init_rag()
    
    OOS_PROMPT = ChatPromptTemplate.from_template("Analyze the user's question. If it is related to workplace rules, company policies, human resources, leaves, travel expenses, benefits, behavior, or employee handbooks, reply with 'SAFE'. Otherwise reply with 'UNSAFE'.\n\nQuestion: {question}\nClassification:")
    RAG_PROMPT = ChatPromptTemplate.from_template("You are a helpful HR Assistant for Zyro Dynamics. Answer the question accurately using ONLY the provided context. If you don't know, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("What is the policy on casual leaves?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        guard_chain = OOS_PROMPT | llm | StrOutputParser()
        safety_check = guard_chain.invoke({"question": prompt}).strip().upper()
        
        with st.chat_message("assistant"):
            if "SAFE" in safety_check and "UNSAFE" not in safety_check:
                docs = retriever.invoke(prompt)
                context_str = format_docs(docs)
                
                rag_pipeline = RAG_PROMPT | llm | StrOutputParser()
                answer = rag_pipeline.invoke({"context": context_str, "question": prompt})
                
                st.markdown(answer)
                with st.expander("📚 View Sources Cited"):
                    for doc in docs:
                        source_name = os.path.basename(doc.metadata.get('source', 'Policy Doc'))
                        st.write(f"- *{source_name}* (Page {doc.metadata.get('page', 0) + 1})")
            else:
                answer = "I can only answer HR-related questions from Zyro Dynamics policy documents."
                st.markdown(answer)
                
        st.session_state.messages.append({"role": "assistant", "content": answer})
except Exception as e:
    st.error(f"Configuration Error: Ensure your GROQ_API_KEY is set up correctly. Details: {e}")
