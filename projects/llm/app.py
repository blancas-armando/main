import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

os.environ['OPENAI_API_KEY'] = apikey

st.title('Job Interview Prep')
prompt = st.text_input('Input the role you are applying for here')

# Prompt Template
interview_question_template = PromptTemplate(
    input_variables= ['role'],
    template='Write me the most important interview questions for a {role} position'
)

interview_prep_template = PromptTemplate(
    input_variables= ['question'],
    template='Write me an interview prep guideline for these questions {question}'
)

# LLMs
llm = OpenAI(temperature=0.9)
interview_question_chain = LLMChain(
    llm=llm, 
    prompt=interview_question_template, 
    verbose=True, 
    output_key='role'
    )
interview_prep_chain = LLMChain(
    llm=llm, 
    prompt=interview_prep_template, 
    verbose=True, 
    output_key='questions'
    )

sequential_chain = SequentialChain(
    chains=[interview_question_chain, interview_prep_chain], 
    input_variables=['role'],
    output_variables=['role', 'questions'] 
    verbose=True
    )


# Show to screen if prompt is provided
if prompt:
    response = sequential_chain.run({'role':prompt})
    st.write(response['role'])
    st.write(response['questions'])
