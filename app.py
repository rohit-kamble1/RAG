from langchain_openai import OpenAI
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os, tempfile
import streamlit as st
from pinecone import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
import warnings 
# Settings the warnings to be ignored 
warnings.filterwarnings('ignore')

pinecone_api_key = os.environ.get('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
index_name = "testvector"
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
# Streamlit app
st.subheader('Generative Q&A with LangChain')

with st.sidebar:
    source_doc = st.file_uploader("Upload source document", type="pdf", label_visibility="collapsed")
query = st.text_input("Enter your query")
if st.button("Submit"):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(source_doc.read())
    loader = PyPDFLoader(tmp_file.name)
    pages = loader.load()
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    texts = text_splitter.split_documents(pages)     
    # Generate embeddings for the pages, insert into Pinecone vector database, and expose the index in a retriever interface
    embeddings = HuggingFaceHubEmbeddings()
    Pinecone(api_key=pinecone_api_key)
    db = PineconeVectorStore.from_documents(texts, embeddings, index_name = index_name)
    retriever = db.as_retriever()
    
   # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
    llm=HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN), 
    chain_type="map_reduce", retriever=retriever, return_source_documents=True)
    result = qa({"query": query})
    st.write(result['result'])

