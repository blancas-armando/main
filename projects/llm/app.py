import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

os.environ['OPENAI_API_KEY'] = apikey

st.title('Job Interview Prep')
prompt = st.text_input('Input the role you are applying for here')

# Prompt Template
interview_question_template = PromptTemplate(
    input_variables= ['role'],
    template='Write me the most important interview questions for a {role} position'
)

interview_prep_template = PromptTemplate(
    input_variables= ['role'],
    template='Write me an interview prep guideline for a {role} position'
)

# LLMs
llm = OpenAI(temperature=0.9)
interview_question_chain = LLMChain(llm=llm, prompt=interview_question_template, verbose=True)
interview_prep_chain = LLMChain(llm=llm, prompt=interview_prep_template, verbose=True)
sequential_chain = SimpleSequentialChain(chains=[interview_question_chain, interview_prep_chain], verbose=True)


# Show to screen if prompt is provided
if prompt:
    response = sequential_chain.run(prompt)
    st.write(response)

