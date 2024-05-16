import streamlit as st
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma  # Import Chroma from langchain.vectorstores
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import pandas as pd
import torch
import langdetect
from langdetect import DetectorFactory, detect, detect_langs
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# Build prompt
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use five sentences maximum. Keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 

{context}

Question: {question}

Helpful Answer:
"""
prompt = PromptTemplate.from_template(template)

model_name = "sentence-transformers/all-mpnet-base-v2"
# model_name = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize Chroma vector store
vectordb = Chroma(persist_directory="./data/chroma/", embedding_function=embeddings)  # Specify the path to your Chroma database


llm = Ollama(model="mistral", callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))
# Define the chain
prompt = PromptTemplate.from_template(template)
chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt})

# Streamlit application
def main():
    st.title("Fitness chatbot")
    
    # Input text box for user input
    user_input = st.text_input("You:", "")

    if user_input:
        # Get response from the chat model
        response = chain({"query": user_input})

        # Display the response
        st.write("Response:", response['result'])

        # Optionally display source documents
        if 'source_documents' in response:
            st.write("Source Documents:")
            for doc in response['source_documents']:
                st.write(doc)

if __name__ == "__main__":
    main()