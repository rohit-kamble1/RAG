# from langchain_openai import OpenAI
# from langchain_openai import OpenAIEmbeddings

from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# from langchain_community.embeddings import HuggingFaceHubEmbeddings
# from langchain_community.llms import HuggingFaceEndpoint

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os, tempfile
import streamlit as st
from pinecone import Pinecone
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
import warnings 
# Settings the warnings to be ignored 
warnings.filterwarnings('ignore')

#pinecone_api_key = os.environ.get('PINECONE_API_KEY')
index_name = "testvector"


def loadFile(source_doc):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(source_doc.read())
    loader = PyPDFLoader(tmp_file.name)
    pages = loader.load()
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    texts = text_splitter.split_documents(pages)    
    return texts

addSelectBox = st.sidebar.selectbox(
    "Select LLM Model:",
    ("Mistral AI", "Gemini Pro", "OpenAI")
)

# if addSelectBox == "Mistral AI":
#     HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
#     repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
#     #repo_id = "meta-llama/Meta-Llama-3-8B"
#     # Streamlit app
#     st.subheader('Generative Q&A with LangChain')
#     source_doc = st.file_uploader("Upload source document", type="pdf", label_visibility="collapsed")

#     query = st.text_input("Enter your query")
#     with st.sidebar:
#         temperature = st.number_input("Define value of temperature which controls the randomness of model output:",
#                                   min_value=0.0, max_value=2.0, step= 0.1)
#     if st.button("Submit"):
#         texts = loadFile(source_doc)  
#         # Generate embeddings for the pages, insert into Pinecone vector database, and expose the index in a retriever interface
#         embeddings = HuggingFaceHubEmbeddings()
#         Pinecone(api_key=pinecone_api_key)    #initialize pinecone
#         db = PineconeVectorStore.from_documents(texts, embeddings, index_name = index_name)
#         retriever = db.as_retriever()
        
    # create a chain to answer questions 
        # qa = RetrievalQA.from_chain_type(
        # llm=HuggingFaceEndpoint(
        # repo_id=repo_id, max_length=128, temperature=temperature, token=HUGGINGFACEHUB_API_TOKEN), 
        # chain_type="map_reduce", retriever=retriever, return_source_documents=True)
        # result = qa({"query": query})
        # st.write(result['result'])

if addSelectBox == "Gemini Pro":
    #GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")   
    st.header("Generative Q&A with LangChain")
    source_doc = st.file_uploader("Upload source document", type="pdf", label_visibility="visible")
    query = st.text_input("Enter your query")

    with st.sidebar:
        temperature = st.number_input("Define value of temperature which controls the randomness of model output:",
                                  min_value=0.0, max_value=2.0, step= 0.1)
    if st.button("Submit"):
        texts = loadFile(source_doc)      
        # Generate embeddings for the pages, insert into Pinecone vector database, and expose the index in a retriever interface
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key = 'AIzaSyDkEntqJsGZk4LcucJwt_Y09Pc0OmzO1wA')
        Pinecone(api_key="e1c3c436-e2c9-4201-990e-9b7962700209") #initialize pinecone
        db = PineconeVectorStore.from_documents(texts, embeddings, index_name = index_name)
        retriever = db.as_retriever()
        
        # create a chain to answer questions 
        qa = RetrievalQA.from_chain_type(
        llm= GoogleGenerativeAI(
        model="gemini-pro", GOOGLE_API_KEY="AIzaSyDkEntqJsGZk4LcucJwt_Y09Pc0OmzO1wA", temperature=temperature, convert_system_message_to_human=True), 
        chain_type="map_reduce", retriever=retriever, return_source_documents=True)
        result = qa({"query": query})
        st.write(result['result'])

# if addSelectBox == "OpenAI":
#
#     OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
#     st.header("Generative Q&A with LangChain")
#     query = st.text_input("Enter your query.")
#     with st.sidebar:
#         source_doc = st.file_uploader("Upload source document", type="pdf", label_visibility="collapsed")
#         temperature = st.number_input("Define value of temperature which controls the randomness of model output:",
#                                   min_value=0.0, max_value=2.0, step= 0.1)
#     if st.button("Submit"):
#         texts = loadFile(source_doc)      
#         # Generate embeddings for the pages, insert into Pinecone vector database, and expose the index in a retriever interface
#         embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key = OPENAI_API_KEY)
#         Pinecone(api_key=pinecone_api_key)  #initialize pinecone
#         db = PineconeVectorStore.from_documents(texts, embeddings, index_name = index_name)
#         retriever = db.as_retriever()
        
#     # create a chain to answer questions 
#         qa = RetrievalQA.from_chain_type(
#         llm= OpenAI(
#         model="gpt-3.5-turbo-instruct", openai_api_key=OPENAI_API_KEY, temperature=temperature, convert_system_message_to_human=True), 
#         chain_type="map_reduce", retriever=retriever, return_source_documents=True)
#         result = qa({"query": query})
#         st.write(result['result'])
    

