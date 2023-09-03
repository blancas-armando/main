# Import os to set API key
import os
from dotenv import load_dotenv

import openai
import streamlit as st
from PIL import Image

from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)


def main():
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    llm = OpenAI(temperature=0.1, verbose=True)
    embeddings = OpenAIEmbeddings()

    loader = PyPDFLoader(r'projects\llm\langchain_doc_reader\jpmc_annualreport_2022.pdf')

    pages = loader.load_and_split()
    store = Chroma.from_documents(pages, embeddings, collection_name='jpmc_annualreport')

    vectorstore_info = VectorStoreInfo(
        name="annual_report_2022",
        description="JPMC annual report 2022 PDF",
        vectorstore=store
    )
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

    agent_executor = create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )
    
    image = Image.open(r'projects\llm\langchain_doc_reader\jpmorganchase_logo.png')
    with st.sidebar:
        st.image(image)
        st.write('This is a personal project and does not reflect JPMorgan Chase & Co.')

    st.title(':bookmark_tabs: 2022 Annual Report Reader')
    prompt = st.text_input('Input your question here')

    if prompt:
        response = agent_executor.run(prompt)
        st.write(response)
        with st.expander('Document Similarity Search'):
            search = store.similarity_search_with_score(prompt)  
            st.write(search[0][0].page_content)


if __name__ == '__main__':
    main()