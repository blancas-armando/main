import os
from dotenv import load_dotenv

import openai
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(temperature=0.3, verbose=True)
embeddings = OpenAIEmbeddings()

loader = PyPDFLoader('frb_earnings_release_q4_2022.pdf')
pages = loader.load_and_split()
store = Chroma.from_documents(pages, embeddings, collection_name='earnings_release')

vectorstore_info = VectorStoreInfo(
    name='frb_earnings_release_q4_2022',
    description='frb earnings report for q4 2022',
    vectorstore=store
)

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    tookit=toolkit,
    verbose=True
)

st.title('Page Title')
prompt=st.text_input('Input your prompt here')

if prompt:
    response = agent_executor.run(prompt)
    st.write(response)

    with st.expander(''):
        search = store.similarity_search_with_score(prompt)
        st.write(search[0][0].page_content)