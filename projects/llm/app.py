import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ['OPENAI_API_KEY'] = apikey

st.title('Job Interview Prep')
prompt = st.text_input('Input the role you are applying for here')

# Prompt Template
title_template = PromptTemplate(
    input_variables= ['topic'],
    template='Write me interview questions for a {role} position'
)

# LLMs
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(
    llm=llm,
    prompt=title_template,
    verbose=True
)
# Show to screen if prompt is provided
if prompt:
    response = title_chain.run(prompt)
    st.write(response)

