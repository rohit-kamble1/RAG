from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os, tempfile
import streamlit as st
import pinecone
from pinecone import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore

#google_api_key = os.environ.get('GOOGLE_API_KEY')
google_api_key = "AIzaSyDkEntqJsGZk4LcucJwt_Y09Pc0OmzO1wA"
pinecone_api_key = "e1c3c436-e2c9-4201-990e-9b7962700209"
#pinecone_api_key = os.environ.get('PINECONE_API_KEY')
index_name = "testvector"
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
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(pages)     
    # Generate embeddings for the pages, insert into Pinecone vector database, and expose the index in a retriever interface
    
    #st.write(pinecone_api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key = "AIzaSyDkEntqJsGZk4LcucJwt_Y09Pc0OmzO1wA")
    Pinecone(api_key='e1c3c436-e2c9-4201-990e-9b7962700209')
    client = pinecone.Pinecone(api_key= 'e1c3c436-e2c9-4201-990e-9b7962700209')
    db = PineconeVectorStore.from_documents(texts, embeddings, index_name = index_name)
    retriever = db.as_retriever()

   # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(model="gemini-pro", google_api_key = "AIzaSyDkEntqJsGZk4LcucJwt_Y09Pc0OmzO1wA", temperature=0.5, convert_system_message_to_human=True), chain_type="stuff", retriever=retriever, return_source_documents=True)
    result = qa({"query": query})
    retriever.get_relevant_documents(query)
    
    st.write(result['result'])

