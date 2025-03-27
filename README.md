# Generative Q&A with LangChain

This is a Streamlit web application that enables users to upload PDF documents and interact with a Generative AI model (Mistral AI, Gemini Pro, or OpenAI) for Q&A.

## Features
- Supports multiple LLMs: OpenAI (GPT-3.5 Turbo), Gemini Pro, and Mistral AI.
- Upload PDF documents for context-based querying.
- Uses LangChain for document processing and retrieval.
- Pinecone vector database for efficient information retrieval.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/rohit-kamble1/RAG
   cd <your-repo-directory>

2. Install Dependencies
   ```sh
   pip install -r requirements.txt

3. Set up environment variables:
   
- OPENAI_API_KEY
- GOOGLE_API_KEY
- HUGGINGFACEHUB_API_TOKEN
- PINECONE_API_KEY

4. Run the Streamlit app:
   ```sh
   streamlit run app.py
   
5. Usage
- Select the LLM model from the sidebar.
- Upload a PDF document.
- Enter your query in the text input field.
- Adjust the temperature for model output randomness.
- Click the "Submit" button to generate an answer.

6. Requirements
- Python 3.8+
- Internet connection (for API calls)
- API keys for OpenAI, Google Gemini, and Pinecone

7. Acknowledgments
- Built with LangChain
- Uses Streamlit for UI
- Powered by Pinecone for vector storage.



