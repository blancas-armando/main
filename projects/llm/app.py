import os
from apikey import apikey_str

import streamlit as st
from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY'] = apikey_str

st.title('testing!')

