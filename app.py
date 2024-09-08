import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pinecone

# Load environment variables
load_dotenv()

# Configure Google Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Google Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Pinecone API key and environment from .env file
api_key = os.getenv("PINECONE_API_KEY")
index_name = "tarunaim"
environment = "us-west1-gcp"

# Initialize Pinecone client
pinecone.init(api_key=api_key, environment=environment)

# Access the existing Pinecone index
index = pinecone.Index(index_name)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def get_vector_store(text_chunks):
    # Use LangChain's Pinecone wrapper to store documents and embeddings
    vector_store = Pinecone.from_texts(text_chunks, embeddings, index_name=index_name)
    return vector_store

def user_input(user_question, k=2):
    # Perform similarity search using LangChain's Pinecone wrapper
    vector_store = Pinecone(index, embeddings.embed_query)  # Embedding query handler
    matching_results = vector_store.similarity_search(user_question, k=k)
    
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": matching_results, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat PDF", page_icon="📄")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                
                # Store documents in Pinecone via LangChain wrapper
                get_vector_store(text_chunks)
                
                st.success("Processing complete")

if __name__ == "__main__":
    main()
